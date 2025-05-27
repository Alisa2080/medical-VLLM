import argparse
import os
import torch
import random
import math
import warnings
import numpy as np
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from PIL import Image
from RandStainNA.randstainna import RandStainNA
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform, ImageDataset 
from modules.masking_generator import PathologyMaskingGenerator
from vision_datasets.dataset_folder import ImageFolder

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        return Image.BILINEAR

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)

class ToPILImage:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            from PIL import Image
            return Image.fromarray(img.astype('uint8'))
        return img

class RandomResizedCropAndInterpolationWithTwoPic:
    def __init__(self, size, second_size=None, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear', second_interpolation='lanczos'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if second_size is not None:
            if isinstance(second_size, tuple):
                self.second_size = second_size
            else:
                self.second_size = (second_size, second_size)
        else:
            self.second_size = None
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        self.second_interpolation = _pil_interp(second_interpolation)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped and resized.

        Returns:
            PIL Image: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation), \
                   F.resized_crop(img, i, j, h, w, self.second_size, self.second_interpolation)

    def __repr__(self):
        if isinstance(self.interpolation, (tuple, list)):
            interpolate_str = ' '.join([_pil_interpolation_to_str[x] for x in self.interpolation])
        else:
            interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4) for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4) for r in self.ratio))
        format_string += ', interpolation={0}'.format(interpolate_str)
        if self.second_size is not None:
            format_string += ', second_size={0}'.format(self.second_size)
            format_string += ', second_interpolation={0}'.format(_pil_interpolation_to_str[self.second_interpolation])
        format_string += ')'
        return format_string

class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        fov_transform = transforms.RandomChoice([
            transforms.RandomResizedCrop(
                size=args.input_size,
                scale=(0.1, 0.3),  # 10× 放大，相当于裁剪 10% 区域再放大
                interpolation=_pil_interp(args.train_interpolation)
            ),
            transforms.RandomResizedCrop(
                size=args.input_size,
                scale=(0.05, 0.15),  # 20× 放大
                interpolation=_pil_interp(args.train_interpolation)
            ),
            transforms.RandomResizedCrop(
                size=args.input_size,
                scale=(0.025, 0.1),  # 40× 放大
                interpolation=_pil_interp(args.train_interpolation)
            ),
            RandomResizedCropAndInterpolationWithTwoPic(  # 原有双视野
                size=args.input_size,
                second_size=args.second_input_size,
                scale=(args.min_crop_scale, 1.0),
                interpolation=args.train_interpolation,
                second_interpolation=args.second_interpolation,
            )
        ])

        self.common_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.05,
                    contrast=0.05,
                    saturation=0.02,
                    hue=0.01
                )
            ], p=0.5),
            RandStainNA(
                yaml_file=r"/gz-data/Vision_Encoder/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
                std_hyper=0.05,
                probability=0.6,
                distribution="normal",
                is_train=True,
            ),
            ToPILImage(),
            fov_transform,      # 多视野裁剪
        ])

        # 对图像块进行的转换操作
        self.patch_transform = transforms.Compose([
            # 将 PIL 图像或 numpy.ndarray 转换为 torch.Tensor
            transforms.ToTensor(),
            # 对图像进行归一化处理，使用指定的均值和标准差
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])
                # 对视觉令牌进行的转换操作
        self.visual_token_transform = transforms.Compose([
            # 将 PIL 图像或 numpy.ndarray 转换为 torch.Tensor
            transforms.ToTensor(),
        ])

        # 初始化病理特定的掩码生成器
        masking_strategy = getattr(args, 'masking_strategy', 'pathology_aware')
        foreground_bias = getattr(args, 'foreground_bias', 0.8)
        complexity_adaptive = getattr(args, 'complexity_adaptive', True)
        curriculum_masking = getattr(args, 'curriculum_masking', False)

        self.masking_generator = PathologyMaskingGenerator(
            input_size=args.window_size,
            num_masking_patches=args.num_mask_patches,
            min_num_patches=args.min_mask_patches_per_block,
            max_num_patches=args.max_mask_patches_per_block,
            strategy=masking_strategy,
            foreground_bias=foreground_bias,
            complexity_adaptive=complexity_adaptive,
            curriculum_masking=curriculum_masking,
            current_epoch=0,
            total_epochs=getattr(args, 'epochs', 100)
        )

        # self.masked_position_generator = MaskingGenerator(
        #     args.window_size, num_masking_patches=args.num_mask_patches,
        #     max_num_patches=args.max_mask_patches_per_block,
        #     min_num_patches=args.min_mask_patches_per_block,
        # )
        self.store_original_image = masking_strategy in ['pathology_aware', 'entropy_based', 'gradient_based']
    
    def update_epoch(self, epoch: int):
        """更新当前epoch，用于课程学习"""
        if hasattr(self.masked_position_generator, 'update_epoch'):
            self.masked_position_generator.update_epoch(epoch)


    # def __call__(self, image):
    #     result = self.common_transform(image)
    #     if isinstance(result, tuple) and len(result) == 2:
    #         for_patches, for_visual_tokens = result
    #     else:
    #         for_patches = result
    #         for_visual_tokens = result # 当 common_transform 返回单个图像时，两路使用相同的图像
    #     mask_2d = self.masked_position_generator()  # (H, W)
    #     mask_1d = mask_2d.flatten()  # (H*W,)
        
    #     return (
    #         self.patch_transform(for_patches),
    #         self.visual_token_transform(for_visual_tokens),
    #         mask_1d,
    #     )
    
    def __call__(self, image):
        # 如果需要原始图像信息用于掩码生成，先保存
        original_image = image if self.store_original_image else None
        
        result = self.common_transform(image)
        if isinstance(result, tuple) and len(result) == 2:
            for_patches, for_visual_tokens = result
        else:
            for_patches = result
            for_visual_tokens = result
        
        # 传递原始图像给掩码生成器
        mask_2d = self.masked_position_generator(original_image)  # (H, W)
        mask_1d = mask_2d.flatten()  # (H*W,)
        
        return (
            self.patch_transform(for_patches),
            self.visual_token_transform(for_visual_tokens),
            mask_1d,
        )


    def __repr__(self):
        repr = "(DataAugmentationForBEiT,\n"
        repr += "  common_transform = %s,\n" % str(self.common_transform)
        repr += "  patch_transform = %s,\n" % str(self.patch_transform)
        repr += "  visual_tokens_transform = %s,\n" % str(self.visual_token_transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr

def build_beit_pretraining_dataset(args):
    transform = DataAugmentationForBEiT(args)
    print("Data Aug = %s" % str(transform))
    
    return ImageFolder(args.data_path, transform=transform)

############################################### Dataset and Transforms for Tokenizer Training #########################################################

def build_vqkd_dataset(is_train, args):
    # 训练集数据增强
    if is_train:
        t = []
        if args.color_jitter > 0.:
            t.append(transforms.ColorJitter(args.color_jitter, args.color_jitter, args.color_jitter)) # 用于随机调整图像的亮度、对比度、饱和度和色调。
        # 随机裁剪图像并调整大小到 args.input_size，裁剪比例范围由 args.min_crop_scale 控制。
        t.append(transforms.RandomResizedCrop(args.input_size, scale=(args.min_crop_scale, 1.0), interpolation=_pil_interp(args.train_interpolation))) 
        # 0.5 的概率随机水平翻转图像。
        t.append(transforms.RandomHorizontalFlip(0.5))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)
    
    
    #测试集数据增强
    else:
        t = []
        if args.input_size < 384:
            args.crop_pct = 224 / 256
        else:
            args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=_pil_interp(args.train_interpolation)),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        transform = transforms.Compose(t)
    
    print(f"{'Train' if is_train else 'Test'} Data Aug: {str(transform)}")

    if args.data_set == 'image_folder':
        if is_train:
            return ImageFolder(args.data_path, transform=transform)
        else:
            if args.eval_data_path == '':
                return ImageFolder(args.data_path, transform=transform)
            else:
                return ImageFolder(args.eval_data_path, transform=transform)

    else:
        raise NotImplementedError()


############################################### Dataset and Transforms for Ft #########################################################

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        index_file = args.image_folder_class_index_file
        dataset = ImageFolder(root, transform=transform, index_file=index_file)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop_pct is None:
            if args.input_size < 384:
                args.crop_pct = 224 / 256
            else:
                args.crop_pct = 1.0
        size = int(args.input_size / args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
