import random
import cv2
from PIL import Image
from skimage import filters, measure, morphology
from skimage.color import rgb2gray, rgb2hsv
from scipy import ndimage
import warnings
import math
import random
from typing import Optional, Tuple, Dict, Any
import numpy as np
import torch
warnings.filterwarnings('ignore')

# class MaskingGenerator:
#     def __init__(
#             self,
#             input_size,
#             num_masking_patches,
#             min_num_patches=16,
#             max_num_patches=75,
#             min_aspect=0.3,
#             max_aspect=None,
#             block_wise=True,
#             ):

#         if not isinstance(input_size, tuple):
#             input_size = (input_size, ) * 2
#         self.height, self.width = input_size

#         self.num_patches = self.height * self.width
#         self.num_masking_patches = num_masking_patches
#         self.min_num_patches = min_num_patches
#         self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
#         self.block_wise = block_wise

#         max_aspect = max_aspect or 1 / min_aspect
#         self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

#     def __repr__(self):
#         repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
#             self.height, self.width, self.min_num_patches, self.max_num_patches,
#             self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
#         return repr_str

#     def get_shape(self):
#         return self.height, self.width

#     def _mask(self, mask, max_mask_patches):
#         delta = 0
#         for attempt in range(10):
#             target_area = random.uniform(self.min_num_patches, max_mask_patches)
#             aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
#             h = int(round(math.sqrt(target_area * aspect_ratio)))
#             w = int(round(math.sqrt(target_area / aspect_ratio)))
#             if w < self.width and h < self.height:
#                 top = random.randint(0, self.height - h)
#                 left = random.randint(0, self.width - w)

#                 num_masked = mask[top: top + h, left: left + w].sum()
#                 # Overlap
#                 if 0 < h * w - num_masked <= max_mask_patches:
#                     for i in range(top, top + h):
#                         for j in range(left, left + w):
#                             if mask[i, j] == 0:
#                                 mask[i, j] = 1
#                                 delta += 1

#                 if delta > 0:
#                     break
#         return delta
    
#     def _generate_block_mask(self, target_masks):
#         """生成块状掩码，适合病理图像"""
#         mask = torch.zeros(self.height, self.width, dtype=torch.bool)
#         masked_count = 0
#         max_attempts = 100
        
#         while masked_count < target_masks and max_attempts > 0:
#             # 计算剩余需要掩码的数量
#             remaining = target_masks - masked_count
            
#             # 动态调整块大小
#             target_area = min(remaining, 
#                             random.uniform(self.min_num_patches, 
#                                          min(self.max_num_patches, remaining * 1.5)))
            
#             aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
#             h = max(1, min(self.height, int(round(math.sqrt(target_area * aspect_ratio)))))
#             w = max(1, min(self.width, int(round(math.sqrt(target_area / aspect_ratio)))))
            
#             # 确保块不会超出边界
#             if h <= self.height and w <= self.width:
#                 top = random.randint(0, self.height - h)
#                 left = random.randint(0, self.width - w)
                
#                 # 计算这个区域中未掩码的patch数量
#                 region = mask[top:top+h, left:left+w]
#                 free_patches = (~region).sum().item()
                
#                 if free_patches > 0:
#                     # 只掩码未被掩码的patch，避免重复
#                     mask[top:top+h, left:left+w] = True
#                     masked_count = mask.sum().item()
            
#             max_attempts -= 1
        
#         # 精确调整到目标数量
#         current_count = mask.sum().item()
#         if current_count > target_masks:
#             # 随机移除多余的掩码
#             masked_positions = mask.nonzero(as_tuple=False)
#             remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
#             for idx in remove_indices:
#                 pos = masked_positions[idx]
#                 mask[pos[0], pos[1]] = False
#         elif current_count < target_masks:
#             # 随机添加不足的掩码
#             unmasked_positions = (~mask).nonzero(as_tuple=False)
#             if len(unmasked_positions) > 0:
#                 add_count = min(target_masks - current_count, len(unmasked_positions))
#                 add_indices = torch.randperm(len(unmasked_positions))[:add_count]
#                 for idx in add_indices:
#                     pos = unmasked_positions[idx]
#                     mask[pos[0], pos[1]] = True
        
#         return mask
    
#     def _generate_random_mask(self, target_masks):
#         """生成随机掩码，适合高掩码率情况"""
#         prob = target_masks / self.num_patches
#         mask = torch.rand(self.height, self.width) < prob
        
#         # 精确调整到目标数量
#         current_count = mask.sum().item()
#         if current_count != target_masks:
#             if current_count > target_masks:
#                 masked_positions = mask.nonzero(as_tuple=False)
#                 remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
#                 for idx in remove_indices:
#                     pos = masked_positions[idx]
#                     mask[pos[0], pos[1]] = False
#             else:
#                 unmasked_positions = (~mask).nonzero(as_tuple=False)
#                 if len(unmasked_positions) > 0:
#                     add_count = min(target_masks - current_count, len(unmasked_positions))
#                     add_indices = torch.randperm(len(unmasked_positions))[:add_count]
#                     for idx in add_indices:
#                         pos = unmasked_positions[idx]
#                         mask[pos[0], pos[1]] = True
        
#         return mask
    
#     def __call__(self):
#         target_masks = self.num_masking_patches
#         mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        
#         if self.block_wise:
#             # 改进的块状掩码生成
#             mask = self._generate_block_mask(target_masks)
#         else:
#             # 随机掩码生成（适用于高掩码率）
#             mask = self._generate_random_mask(target_masks)
        
#         return mask
        
# if __name__ == '__main__':
#     import pdb
#     generator = MaskingGenerator(input_size=14, num_masking_patches=118, min_num_patches=16,)
#     for i in range(10):
#         mask = generator()
#         if mask.sum() != 118:
#             pdb.set_trace()
#             print(mask)
#             print(mask.sum())

class PathologyImageAnalyzer:
    """病理图像分析工具类，用于前景/背景分割和组织区域分析"""
    
    def __init__(self, 
                 foreground_threshold: float = 0.8,
                 min_foreground_area: int = 100,
                 blur_kernel_size: int = 5):
        self.foreground_threshold = foreground_threshold
        self.min_foreground_area = min_foreground_area
        self.blur_kernel_size = blur_kernel_size
    
    def detect_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        """
        检测病理图像的前景区域
        
        Args:
            image: RGB图像数组 (H, W, 3)
            
        Returns:
            foreground_mask: 二值掩码，True为前景，False为背景 (H, W)
        """
        if len(image.shape) == 3:
            # 方法1: 基于HSV颜色空间的前景检测
            hsv = rgb2hsv(image)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # 病理图像中，背景通常饱和度低且亮度高
            # 前景（组织）通常饱和度较高或亮度较低
            foreground_by_saturation = saturation > 0.1
            foreground_by_value = value < 0.95
            
            # 方法2: 基于灰度的OTSU阈值
            gray = rgb2gray(image)
            # 对于病理图像，通常较暗的区域是前景
            threshold = filters.threshold_otsu(gray)
            foreground_by_otsu = gray < threshold * 1.1  # 稍微放宽阈值
            
            # 方法3: 基于RGB强度的检测
            # 病理图像背景通常是白色或接近白色
            rgb_intensity = np.mean(image, axis=2)
            foreground_by_intensity = rgb_intensity < self.foreground_threshold
            
            # 综合多种方法
            foreground_mask = (foreground_by_saturation | 
                             foreground_by_value | 
                             foreground_by_otsu | 
                             foreground_by_intensity)
        else:
            # 灰度图像
            threshold = filters.threshold_otsu(image)
            foreground_mask = image < threshold * 1.1
        
        # 形态学操作去除噪声
        foreground_mask = morphology.remove_small_objects(
            foreground_mask, min_size=self.min_foreground_area
        )
        foreground_mask = morphology.binary_closing(
            foreground_mask, morphology.disk(3)
        )
        
        return foreground_mask.astype(bool)
    
    def compute_information_entropy(self, image: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """
        计算图像块的信息熵
        
        Args:
            image: 输入图像 (H, W, 3) 或 (H, W)
            patch_size: patch大小
            
        Returns:
            entropy_map: 信息熵图 (H//patch_size, W//patch_size)
        """
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # 将图像分割成patches
        h, w = gray.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        
        entropy_map = np.zeros((h_patches, w_patches))
        
        for i in range(h_patches):
            for j in range(w_patches):
                patch = gray[i*patch_size:(i+1)*patch_size, 
                           j*patch_size:(j+1)*patch_size]
                
                # 计算直方图
                hist, _ = np.histogram(patch, bins=256, range=(0, 1), density=True)
                hist = hist[hist > 0]  # 移除零值
                
                # 计算熵
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropy_map[i, j] = entropy
        
        return entropy_map
    
    def compute_gradient_intensity(self, image: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """
        计算图像块的梯度强度
        
        Args:
            image: 输入图像 (H, W, 3) 或 (H, W)
            patch_size: patch大小
            
        Returns:
            gradient_map: 梯度强度图 (H//patch_size, W//patch_size)
        """
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # 计算梯度
        grad_x = filters.sobel_h(gray)
        grad_y = filters.sobel_v(gray)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # 将梯度图分割成patches并计算平均梯度强度
        h, w = gradient_magnitude.shape
        h_patches = h // patch_size
        w_patches = w // patch_size
        
        gradient_map = np.zeros((h_patches, w_patches))
        
        for i in range(h_patches):
            for j in range(w_patches):
                patch = gradient_magnitude[i*patch_size:(i+1)*patch_size, 
                                         j*patch_size:(j+1)*patch_size]
                gradient_map[i, j] = np.mean(patch)
        
        return gradient_map
    
    def compute_tissue_complexity(self, image: np.ndarray) -> float:
        """
        计算整个图像的组织复杂度
        
        Args:
            image: 输入图像 (H, W, 3) 或 (H, W)
            
        Returns:
            complexity_score: 复杂度分数 [0, 1]
        """
        foreground_mask = self.detect_foreground_mask(image)
        foreground_ratio = np.mean(foreground_mask)
        
        if len(image.shape) == 3:
            gray = rgb2gray(image)
        else:
            gray = image
        
        # 计算全局信息熵
        hist, _ = np.histogram(gray[foreground_mask], bins=256, range=(0, 1), density=True)
        hist = hist[hist > 0]
        global_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # 计算梯度变化
        grad_x = filters.sobel_h(gray)
        grad_y = filters.sobel_v(gray)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        avg_gradient = np.mean(gradient_magnitude[foreground_mask])
        
        # 综合复杂度分数
        complexity_score = (
            0.4 * foreground_ratio +
            0.3 * (global_entropy / 8.0) +  # 归一化到[0,1]
            0.3 * min(avg_gradient * 10, 1.0)  # 归一化到[0,1]
        )
        
        return min(complexity_score, 1.0)

def pil_to_numpy(image):
    """将PIL图像转换为numpy数组"""
    if isinstance(image, Image.Image):
        return np.array(image)
    return image

def numpy_to_pil(image):
    """将numpy数组转换为PIL图像"""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)
    return image

class PathologyMaskingGenerator:
    """
    针对病理图像优化的掩码生成器
    """
    def __init__(
        self,
        input_size: Tuple[int, int] = (24, 24),
        num_masking_patches: int = 230,
        min_num_patches: int = 16,
        max_num_patches: int = 75,
        min_aspect: float = 0.3,
        max_aspect: Optional[float] = None,
        strategy: str = "pathology_aware",
        foreground_bias: float = 0.8,
        complexity_adaptive: bool = True,
        curriculum_masking: bool = False,
        current_epoch: int = 0,
        total_epochs: int = 100,
        **kwargs
    ):
        """
        Args:
            input_size: 输入图像尺寸 (height, width)
            num_masking_patches: 目标掩码patch数量
            strategy: 掩码策略 ["random", "block", "pathology_aware", "entropy_based", "gradient_based"]
            foreground_bias: 前景区域掩码偏向权重 [0, 1]
            complexity_adaptive: 是否根据图像复杂度调整掩码率
            curriculum_masking: 是否使用课程学习掩码
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.base_num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = max_num_patches
        self.strategy = strategy
        self.foreground_bias = foreground_bias
        self.complexity_adaptive = complexity_adaptive
        self.curriculum_masking = curriculum_masking
        self.current_epoch = current_epoch
        self.total_epochs = total_epochs
        
        # 纵横比相关
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        
        # 病理图像分析器
        self.analyzer = PathologyImageAnalyzer()
        
        # 课程学习参数
        self.curriculum_progress = min(current_epoch / max(total_epochs * 0.8, 1), 1.0)
        
    def update_epoch(self, epoch: int):
        """更新当前epoch，用于课程学习"""
        self.current_epoch = epoch
        self.curriculum_progress = min(epoch / max(self.total_epochs * 0.8, 1), 1.0)
    
    def _compute_adaptive_mask_count(self, image_complexity: float) -> int:
        """根据图像复杂度调整掩码数量"""
        if not self.complexity_adaptive:
            return self.base_num_masking_patches
        
        # 复杂图像减少掩码，简单图像增加掩码
        complexity_factor = 1.0 - 0.3 * image_complexity  # [0.7, 1.0]
        
        # 课程学习调整
        if self.curriculum_masking:
            # 训练初期减少掩码，后期逐渐增加
            curriculum_factor = 0.6 + 0.4 * self.curriculum_progress  # [0.6, 1.0]
            complexity_factor *= curriculum_factor
        
        adaptive_count = int(self.base_num_masking_patches * complexity_factor)
        return max(self.min_num_patches, min(adaptive_count, self.num_patches // 2))
    
    def _compute_adaptive_block_size(self) -> Tuple[int, int]:
        """根据课程学习调整块大小"""
        if not self.curriculum_masking:
            return self.min_num_patches, self.max_num_patches
        
        # 训练初期使用小块，后期使用大块
        progress = self.curriculum_progress
        
        # 最小块大小从8逐渐增加到min_num_patches
        adaptive_min = max(8, int(self.min_num_patches * (0.5 + 0.5 * progress)))
        # 最大块大小从min_num_patches逐渐增加到max_num_patches
        adaptive_max = min(
            self.max_num_patches,
            int(self.min_num_patches + (self.max_num_patches - self.min_num_patches) * progress)
        )
        
        return adaptive_min, adaptive_max
    
    def _generate_pathology_aware_mask(
        self, 
        foreground_mask: np.ndarray,
        entropy_map: Optional[np.ndarray] = None,
        gradient_map: Optional[np.ndarray] = None,
        target_masks: int = None
    ) -> torch.Tensor:
        """
        生成病理感知的掩码
        
        Args:
            foreground_mask: 前景掩码 (H, W)
            entropy_map: 信息熵图 (H, W) 可选
            gradient_map: 梯度图 (H, W) 可选
            target_masks: 目标掩码数量
        """
        if target_masks is None:
            target_masks = self.base_num_masking_patches
        
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        masked_count = 0
        max_attempts = 200
        
        # 计算权重图
        weight_map = np.ones((self.height, self.width))
        
        # 前景偏向
        if foreground_mask is not None:
            foreground_weight = self.foreground_bias * 2.0 + 0.1  # [0.1, 1.7]
            background_weight = 0.1
            weight_map = np.where(foreground_mask, foreground_weight, background_weight)
        
        # 信息熵偏向
        if entropy_map is not None:
            entropy_normalized = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-8)
            weight_map *= (1.0 + entropy_normalized)
        
        # 梯度偏向
        if gradient_map is not None:
            gradient_normalized = (gradient_map - gradient_map.min()) / (gradient_map.max() - gradient_map.min() + 1e-8)
            weight_map *= (1.0 + gradient_normalized)
        
        # 归一化权重图
        weight_map = weight_map / weight_map.sum()
        
        # 获取自适应块大小
        adaptive_min, adaptive_max = self._compute_adaptive_block_size()
        
        while masked_count < target_masks and max_attempts > 0:
            remaining = target_masks - masked_count
            
            # 根据权重图选择掩码块的中心位置
            flat_weights = weight_map.flatten()
            try:
                center_idx = np.random.choice(len(flat_weights), p=flat_weights)
                center_h = center_idx // self.width
                center_w = center_idx % self.width
            except:
                # 如果权重图有问题，回退到随机选择
                center_h = random.randint(0, self.height - 1)
                center_w = random.randint(0, self.width - 1)
            
            # 确定块大小
            target_area = min(remaining, 
                            random.uniform(adaptive_min, 
                                         min(adaptive_max, remaining * 1.5)))
            
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = max(1, min(self.height, int(round(math.sqrt(target_area * aspect_ratio)))))
            w = max(1, min(self.width, int(round(math.sqrt(target_area / aspect_ratio)))))
            
            # 以选定的中心点为基础确定块的位置
            top = max(0, min(self.height - h, center_h - h // 2))
            left = max(0, min(self.width - w, center_w - w // 2))
            
            # 检查这个区域中未掩码的patch数量
            region = mask[top:top+h, left:left+w]
            free_patches = (~region).sum().item()
            
            if free_patches > 0:
                # 掩码这个区域
                mask[top:top+h, left:left+w] = True
                masked_count = mask.sum().item()
                
                # 更新权重图，降低已掩码区域周围的权重
                weight_map[max(0, top-5):min(self.height, top+h+5),
                          max(0, left-5):min(self.width, left+w+5)] *= 0.1
                
                # 重新归一化
                if weight_map.sum() > 0:
                    weight_map = weight_map / weight_map.sum()
                else:
                    weight_map = np.ones_like(weight_map) / weight_map.size
            
            max_attempts -= 1
        
        # 精确调整到目标数量
        current_count = mask.sum().item()
        if current_count > target_masks:
            masked_positions = mask.nonzero(as_tuple=False)
            remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
            for idx in remove_indices:
                pos = masked_positions[idx]
                mask[pos[0], pos[1]] = False
        elif current_count < target_masks:
            unmasked_positions = (~mask).nonzero(as_tuple=False)
            if len(unmasked_positions) > 0:
                add_count = min(target_masks - current_count, len(unmasked_positions))
                
                # 根据权重选择要添加的位置
                if foreground_mask is not None:
                    weights = []
                    for pos in unmasked_positions:
                        h, w = pos[0].item(), pos[1].item()
                        weight = 2.0 if foreground_mask[h, w] else 0.1
                        weights.append(weight)
                    weights = torch.tensor(weights)
                    weights = weights / weights.sum()
                    
                    add_indices = torch.multinomial(weights, add_count, replacement=False)
                else:
                    add_indices = torch.randperm(len(unmasked_positions))[:add_count]
                
                for idx in add_indices:
                    pos = unmasked_positions[idx]
                    mask[pos[0], pos[1]] = True
        
        return mask
    
    def _generate_entropy_based_mask(self, entropy_map: np.ndarray, target_masks: int) -> torch.Tensor:
        """基于信息熵的掩码生成"""
        # 将熵图转换为patch级别
        patch_entropy = torch.from_numpy(entropy_map).float()
        
        # 高熵区域有更高的被掩码概率
        entropy_weights = torch.softmax(patch_entropy.flatten() * 2.0, dim=0)
        
        # 根据权重选择掩码位置
        mask_indices = torch.multinomial(entropy_weights, target_masks, replacement=False)
        
        mask = torch.zeros(self.height * self.width, dtype=torch.bool)
        mask[mask_indices] = True
        mask = mask.reshape(self.height, self.width)
        
        return mask
    
    def _generate_gradient_based_mask(self, gradient_map: np.ndarray, target_masks: int) -> torch.Tensor:
        """基于梯度强度的掩码生成"""
        # 将梯度图转换为patch级别
        patch_gradient = torch.from_numpy(gradient_map).float()
        
        # 高梯度区域有更高的被掩码概率
        gradient_weights = torch.softmax(patch_gradient.flatten() * 3.0, dim=0)
        
        # 根据权重选择掩码位置
        mask_indices = torch.multinomial(gradient_weights, target_masks, replacement=False)
        
        mask = torch.zeros(self.height * self.width, dtype=torch.bool)
        mask[mask_indices] = True
        mask = mask.reshape(self.height, self.width)
        
        return mask
    
    def _generate_standard_block_mask(self, target_masks: int) -> torch.Tensor:
        """标准的块状掩码生成（优化版）"""
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        masked_count = 0
        max_attempts = 100
        
        adaptive_min, adaptive_max = self._compute_adaptive_block_size()
        
        while masked_count < target_masks and max_attempts > 0:
            remaining = target_masks - masked_count
            target_area = min(remaining, 
                            random.uniform(adaptive_min, 
                                         min(adaptive_max, remaining * 1.5)))
            
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = max(1, min(self.height, int(round(math.sqrt(target_area * aspect_ratio)))))
            w = max(1, min(self.width, int(round(math.sqrt(target_area / aspect_ratio)))))
            
            if h <= self.height and w <= self.width:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                
                region = mask[top:top+h, left:left+w]
                free_patches = (~region).sum().item()
                
                if free_patches > 0:
                    mask[top:top+h, left:left+w] = True
                    masked_count = mask.sum().item()
            
            max_attempts -= 1
        
        # 精确调整
        current_count = mask.sum().item()
        if current_count > target_masks:
            masked_positions = mask.nonzero(as_tuple=False)
            remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
            for idx in remove_indices:
                pos = masked_positions[idx]
                mask[pos[0], pos[1]] = False
        elif current_count < target_masks:
            unmasked_positions = (~mask).nonzero(as_tuple=False)
            if len(unmasked_positions) > 0:
                add_count = min(target_masks - current_count, len(unmasked_positions))
                add_indices = torch.randperm(len(unmasked_positions))[:add_count]
                for idx in add_indices:
                    pos = unmasked_positions[idx]
                    mask[pos[0], pos[1]] = True
        
        return mask
    
    def __call__(self, image=None) -> torch.Tensor:
        """
        生成掩码
        
        Args:
            image: 输入图像（PIL或numpy），用于病理感知掩码生成
            
        Returns:
            mask: 掩码张量 (H, W)
        """
        # 确定目标掩码数量
        if image is not None and self.complexity_adaptive:
            image_np = pil_to_numpy(image)
            complexity = self.analyzer.compute_tissue_complexity(image_np)
            target_masks = self._compute_adaptive_mask_count(complexity)
        else:
            target_masks = self.base_num_masking_patches
        
        target_masks = min(target_masks, self.num_patches - 1)  # 确保不超过总patch数
        
        if self.strategy == "random":
            # 随机掩码
            prob = target_masks / self.num_patches
            mask = torch.rand(self.height, self.width) < prob
            
            # 精确调整到目标数量
            current_count = mask.sum().item()
            if current_count != target_masks:
                if current_count > target_masks:
                    masked_positions = mask.nonzero(as_tuple=False)
                    remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
                    for idx in remove_indices:
                        pos = masked_positions[idx]
                        mask[pos[0], pos[1]] = False
                else:
                    unmasked_positions = (~mask).nonzero(as_tuple=False)
                    if len(unmasked_positions) > 0:
                        add_count = min(target_masks - current_count, len(unmasked_positions))
                        add_indices = torch.randperm(len(unmasked_positions))[:add_count]
                        for idx in add_indices:
                            pos = unmasked_positions[idx]
                            mask[pos[0], pos[1]] = True
            
        elif self.strategy == "block":
            mask = self._generate_standard_block_mask(target_masks)
            
        elif self.strategy in ["pathology_aware", "entropy_based", "gradient_based"]:
            if image is None:
                # 如果没有提供图像，回退到块状掩码
                mask = self._generate_standard_block_mask(target_masks)
            else:
                image_np = pil_to_numpy(image)
                
                # 分析图像
                foreground_mask = self.analyzer.detect_foreground_mask(image_np)
                entropy_map = None
                gradient_map = None
                
                if self.strategy in ["pathology_aware", "entropy_based"]:
                    entropy_map = self.analyzer.compute_information_entropy(image_np, patch_size=1)
                    # 将entropy_map调整到与mask相同的尺寸
                    if entropy_map.shape != (self.height, self.width):
                        from scipy import ndimage
                        entropy_map = ndimage.zoom(entropy_map, 
                                                 (self.height / entropy_map.shape[0], 
                                                  self.width / entropy_map.shape[1]), 
                                                 order=1)
                
                if self.strategy in ["pathology_aware", "gradient_based"]:
                    gradient_map = self.analyzer.compute_gradient_intensity(image_np, patch_size=1)
                    # 将gradient_map调整到与mask相同的尺寸
                    if gradient_map.shape != (self.height, self.width):
                        from scipy import ndimage
                        gradient_map = ndimage.zoom(gradient_map,
                                                  (self.height / gradient_map.shape[0],
                                                   self.width / gradient_map.shape[1]),
                                                  order=1)
                
                if self.strategy == "pathology_aware":
                    mask = self._generate_pathology_aware_mask(
                        foreground_mask, entropy_map, gradient_map, target_masks
                    )
                elif self.strategy == "entropy_based":
                    mask = self._generate_entropy_based_mask(entropy_map, target_masks)
                elif self.strategy == "gradient_based":
                    mask = self._generate_gradient_based_mask(gradient_map, target_masks)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")
        
        return mask
    
    def __repr__(self):
        return (f"PathologyMaskingGenerator(strategy={self.strategy}, "
                f"input_size=({self.height}, {self.width}), "
                f"num_masking_patches={self.base_num_masking_patches}, "
                f"foreground_bias={self.foreground_bias}, "
                f"complexity_adaptive={self.complexity_adaptive}, "
                f"curriculum_masking={self.curriculum_masking})")
