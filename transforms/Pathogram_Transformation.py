import torch
import torchvision.transforms as transforms
from PIL import Image
from RandStainNA.randstainna import RandStainNA
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from typing import Dict, Any, Optional, Union, List, Tuple
import random

class ToPILImage:
    """确保输入是PIL图像"""
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # 如果是tensor，转换为PIL
            if img.dim() == 3 and img.shape[0] in [1, 3]:
                # CHW格式
                img = transforms.ToPILImage()(img)
            else:
                raise ValueError(f"Unsupported tensor shape: {img.shape}")
        elif hasattr(img, 'mode'):
            # 已经是PIL图像
            return img
        else:
            # numpy array等其他格式
            return Image.fromarray(img)
        return img

class MultiScaleRandomResizedCrop:
    """
    多尺度随机裁剪，模拟不同倍镜下的病理视野
    """
    def __init__(self, 
                 size: Union[int, Tuple[int, int]], 
                 scale_ranges: List[Tuple[float, float]],
                 interpolation: int = Image.BICUBIC):
        """
        Args:
            size: 输出图像大小
            scale_ranges: 多个裁剪比例范围，例如 [(0.1, 0.3), (0.05, 0.15), (0.025, 0.1)]
            interpolation: 插值方法
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale_ranges = scale_ranges
        self.interpolation = interpolation
    
    def __call__(self, img):
        # 随机选择一个尺度范围
        scale_range = random.choice(self.scale_ranges)
        
        # 使用选定的尺度范围进行随机裁剪
        crop_transform = transforms.RandomResizedCrop(
            size=self.size,
            scale=scale_range,
            interpolation=self.interpolation
        )
        
        return crop_transform(img)

class PathologyAugmentation:
    """
    针对病理图像设计的数据增强类
    统一使用 image_size 参数
    """
    
    def __init__(self, config: Dict[str, Any], is_training: bool = True):
        """
        Args:
            config: 增强配置字典
            is_training: 是否为训练模式
        """
        self.config = config
        self.is_training = is_training
        self.enabled = config.get("enable_pathology_augmentation", True)
        
        # 统一从 image_size 获取图像尺寸，向后兼容 input_size
        self.image_size = self._get_image_size_from_config(config)
        
        if not self.enabled:
            # 如果禁用增强，只做基础的resize和normalize
            self._build_minimal_transform()
        else:
            if is_training:
                self._build_training_transform()
            else:
                self._build_validation_transform()
    
    def _get_image_size_from_config(self, config: Dict[str, Any]) -> int:
        """
        统一获取图像尺寸，优先使用 image_size，向后兼容 input_size
        
        Args:
            config: 配置字典
            
        Returns:
            int: 图像尺寸
        """
        # 优先使用 image_size
        if "image_size" in config:
            return config["image_size"]
        
        # 向后兼容：使用 input_size
        if "input_size" in config:
            print("Warning: Using deprecated 'input_size', please use 'image_size' instead")
            return config["input_size"]
        
        # 默认值
        print("Warning: Neither 'image_size' nor 'input_size' found in config, using default 384")
        return 384
    
    def _build_minimal_transform(self):
        """构建最小化的转换（用于文本预训练或禁用增强时）"""
        imagenet_default = self.config.get("imagenet_default_mean_and_std", True)
        
        mean = IMAGENET_DEFAULT_MEAN if imagenet_default else IMAGENET_INCEPTION_MEAN
        std = IMAGENET_DEFAULT_STD if imagenet_default else IMAGENET_INCEPTION_STD
        
        self.transform = transforms.Compose([
            ToPILImage(),
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def _build_training_transform(self):
        """构建训练时的增强变换"""
        imagenet_default = self.config.get("imagenet_default_mean_and_std", True)
        
        mean = IMAGENET_DEFAULT_MEAN if imagenet_default else IMAGENET_INCEPTION_MEAN
        std = IMAGENET_DEFAULT_STD if imagenet_default else IMAGENET_INCEPTION_STD
        
        # 构建训练时的变换列表
        train_transforms = []
        
        # 1. 确保输入是PIL图像
        train_transforms.append(ToPILImage())
        
        # 2. 几何变换
        if self.config.get("random_vertical_flip_prob", 0) > 0:
            train_transforms.append(
                transforms.RandomVerticalFlip(p=self.config["random_vertical_flip_prob"])
            )
        
        if self.config.get("random_horizontal_flip_prob", 0) > 0:
            train_transforms.append(
                transforms.RandomHorizontalFlip(p=self.config["random_horizontal_flip_prob"])
            )
        
        # 3. 颜色增强
        color_jitter_prob = self.config.get("color_jitter_probability", 0)
        if color_jitter_prob > 0:
            color_jitter = transforms.ColorJitter(
                brightness=self.config.get("color_jitter_brightness", 0.05),
                contrast=self.config.get("color_jitter_contrast", 0.05),
                saturation=self.config.get("color_jitter_saturation", 0.02),
                hue=self.config.get("color_jitter_hue", 0.01)
            )
            train_transforms.append(transforms.RandomApply([color_jitter], p=color_jitter_prob))
        
        # 4. RandStainNA 染色增强
        if self.config.get("randstainna_enabled", False):
            try:
                randstainna = RandStainNA(
                    yaml_file=self.config.get("randstainna_yaml_file", ""),
                    std_hyper=self.config.get("randstainna_std_hyper", 0.05),
                    probability=self.config.get("randstainna_probability", 0.6),
                    distribution=self.config.get("randstainna_distribution", "normal"),
                    is_train=True
                )
                train_transforms.append(randstainna)
            except Exception as e:
                print(f"Warning: RandStainNA initialization failed: {e}")
        
        # 5. 空间变换 - 多尺度裁剪或标准裁剪
        if self.config.get("multi_scale_cropping", False):
            crop_scale_ranges = self.config.get("crop_scale_ranges", [(0.08, 1.0)])
            train_transforms.append(
                MultiScaleRandomResizedCrop(
                    size=self.image_size,
                    scale_ranges=crop_scale_ranges,
                    interpolation=self._get_interpolation()
                )
            )
        else:
            # 标准随机裁剪
            min_scale = self.config.get("min_crop_scale", 0.08)
            max_scale = self.config.get("max_crop_scale", 1.0)
            train_transforms.append(
                transforms.RandomResizedCrop(
                    size=self.image_size,
                    scale=(min_scale, max_scale),
                    interpolation=self._get_interpolation()
                )
            )
        
        # 6. 转换为tensor并标准化
        train_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        self.transform = transforms.Compose(train_transforms)
    
    def _build_validation_transform(self):
        """构建验证时的转换"""
        imagenet_default = self.config.get("imagenet_default_mean_and_std", True)
        
        mean = IMAGENET_DEFAULT_MEAN if imagenet_default else IMAGENET_INCEPTION_MEAN
        std = IMAGENET_DEFAULT_STD if imagenet_default else IMAGENET_INCEPTION_STD
        
        # 验证时使用简单的确定性变换
        self.transform = transforms.Compose([
            ToPILImage(),
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    def _get_interpolation(self):
        """根据配置获取插值方法"""
        interp_str = self.config.get("train_interpolation", "bicubic")
        if interp_str == "bicubic":
            return Image.BICUBIC
        elif interp_str == "bilinear":
            return Image.BILINEAR
        elif interp_str == "lanczos":
            return Image.LANCZOS
        else:
            return Image.BICUBIC
    
    def __call__(self, image):
        """
        应用增强变换
        
        Args:
            image: 输入图像（PIL, tensor或numpy array）
            
        Returns:
            torch.Tensor: 增强后的图像tensor
        """
        return self.transform(image)
    
    def __repr__(self):
        return f"PathologyAugmentation(enabled={self.enabled}, is_training={self.is_training}, image_size={self.image_size}, transform={self.transform})"

class MultiViewPathologyAugmentation:
    """
    多视图病理图像增强，用于对比学习或多视图训练
    统一使用 image_size 参数
    """
    
    def __init__(self, config: Dict[str, Any], num_views: int = 2):
        """
        Args:
            config: 增强配置字典
            num_views: 生成的视图数量
        """
        self.num_views = num_views
        # 确保每个视图的增强器都使用统一的 image_size
        self.augmentations = [
            PathologyAugmentation(config, is_training=True) 
            for _ in range(num_views)
        ]
        
        # 获取图像尺寸用于显示
        first_aug = self.augmentations[0] if self.augmentations else None
        self.image_size = first_aug.image_size if first_aug else 384
    
    def __call__(self, image):
        """
        生成多个增强视图
        
        Args:
            image: 输入图像
            
        Returns:
            List[torch.Tensor]: 多个增强后的图像tensor列表
        """
        return [aug(image) for aug in self.augmentations]
    
    def __repr__(self):
        return f"MultiViewPathologyAugmentation(num_views={self.num_views}, image_size={self.image_size})"