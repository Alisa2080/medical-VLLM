# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import argparse
import os
import torch
import random
from RandStainNA.randstainna import RandStainNA
from torchvision import datasets, transforms
import numpy as np
from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from transforms import RandomResizedCropAndInterpolationWithTwoPic, _pil_interp
from timm.data import create_transform, ImageDataset 
from masking_generator import MaskingGenerator
from dataset.dataset_folder import ImageFolder


# 添加这个类在RandStainNA和RandomResizedCropAndInterpolationWithTwoPic之间转换数据类型
class ToPILImage:
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            from PIL import Image
            return Image.fromarray(img.astype('uint8'))
        return img
    

class DataAugmentationForBEiT(object):
    def __init__(self, args):
        imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

        fov_transform = transforms.RandomChoice([
            transforms.RandomResizedCrop(
                size=args.input_size,
                scale=(0.1, 0.1),  # 10× 放大，相当于裁剪 10% 区域再放大
                interpolation=_pil_interp(args.train_interpolation)
            ),
            transforms.RandomResizedCrop(
                size=args.input_size,
                scale=(0.05, 0.05),  # 20× 放大
                interpolation=_pil_interp(args.train_interpolation)
            ),
            transforms.RandomResizedCrop(
                size=args.input_size,
                scale=(0.025, 0.025),  # 40× 放大
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
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1
                )
            ], p=0.8),
            RandStainNA(
                yaml_file=r"/gz-data/beit2/RandStainNA/CRC_LAB_randomTrue_n0.yaml",
                std_hyper=-0.3,
                probability=1.0,
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

        # 初始化掩码位置生成器，用于生成图像块的掩码
        self.masked_position_generator = MaskingGenerator(
            args.window_size, num_masking_patches=args.num_mask_patches,
            max_num_patches=args.max_mask_patches_per_block,
            min_num_patches=args.min_mask_patches_per_block,
        )

    def __call__(self, image):
        result = self.common_transform(image)
        if isinstance(result, tuple) and len(result) == 2:
            for_patches, for_visual_tokens = result
        else:
            for_patches = result
            for_visual_tokens = result # 当 common_transform 返回单个图像时，两路使用相同的图像
        return (
            self.patch_transform(for_patches),
            self.visual_token_transform(for_visual_tokens),
            self.masked_position_generator()
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
