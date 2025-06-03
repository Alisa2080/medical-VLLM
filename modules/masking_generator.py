import random
import cv2
import math
import warnings
import numpy as np
import torch
from PIL import Image
import logging
from skimage import filters, measure, morphology
from skimage.color import rgb2gray, rgb2hsv
from scipy import ndimage
from typing import Optional, Tuple, Dict, Any

warnings.filterwarnings('ignore')
# 设置日志记录器
logger = logging.getLogger(__name__)

class MaskingGenerator:
    def __init__(
            self,
            input_size,
            num_masking_patches,
            min_num_patches=16,
            max_num_patches=75,
            min_aspect=0.3,
            max_aspect=None,
            block_wise=True,
            ):

        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        self.block_wise = block_wise

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta
    
    def _generate_block_mask(self, target_masks):
        """生成块状掩码，适合病理图像"""
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        masked_count = 0
        max_attempts = 100
        
        while masked_count < target_masks and max_attempts > 0:
            # 计算剩余需要掩码的数量
            remaining = target_masks - masked_count
            
            # 动态调整块大小
            target_area = min(remaining, 
                            random.uniform(self.min_num_patches, 
                                         min(self.max_num_patches, remaining * 1.5)))
            
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = max(1, min(self.height, int(round(math.sqrt(target_area * aspect_ratio)))))
            w = max(1, min(self.width, int(round(math.sqrt(target_area / aspect_ratio)))))
            
            # 确保块不会超出边界
            if h <= self.height and w <= self.width:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                
                # 计算这个区域中未掩码的patch数量
                region = mask[top:top+h, left:left+w]
                free_patches = (~region).sum().item()
                
                if free_patches > 0:
                    # 只掩码未被掩码的patch，避免重复
                    mask[top:top+h, left:left+w] = True
                    masked_count = mask.sum().item()
            
            max_attempts -= 1
        
        # 精确调整到目标数量
        current_count = mask.sum().item()
        if current_count > target_masks:
            # 随机移除多余的掩码
            masked_positions = mask.nonzero(as_tuple=False)
            remove_indices = torch.randperm(len(masked_positions))[:current_count - target_masks]
            for idx in remove_indices:
                pos = masked_positions[idx]
                mask[pos[0], pos[1]] = False
        elif current_count < target_masks:
            # 随机添加不足的掩码
            unmasked_positions = (~mask).nonzero(as_tuple=False)
            if len(unmasked_positions) > 0:
                add_count = min(target_masks - current_count, len(unmasked_positions))
                add_indices = torch.randperm(len(unmasked_positions))[:add_count]
                for idx in add_indices:
                    pos = unmasked_positions[idx]
                    mask[pos[0], pos[1]] = True
        
        return mask
    
    def _generate_random_mask(self, target_masks):
        """生成随机掩码，适合高掩码率情况"""
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
        
        return mask
    
    def __call__(self):
        target_masks = self.num_masking_patches
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        
        if self.block_wise:
            # 改进的块状掩码生成
            mask = self._generate_block_mask(target_masks)
        else:
            # 随机掩码生成（适用于高掩码率）
            mask = self._generate_random_mask(target_masks)
        
        return mask


class PathologyImageAnalyzer:
    """病理图像分析工具类，用于前景/背景分割和组织区域分析"""
    # 配置常量
    DEFAULT_FOREGROUND_THRESHOLD = 0.8
    DEFAULT_MIN_FOREGROUND_AREA = 100
    DEFAULT_BLUR_KERNEL_SIZE = 5
    ENTROPY_BINS = 128
    MORPHOLOGY_DISK_SIZE = 3
    CANNY_SIGMA = 1.0
    CANNY_LOW_THRESHOLD = 0.1
    CANNY_HIGH_THRESHOLD = 0.3
    
    def __init__(self, 
                 foreground_threshold: float = DEFAULT_FOREGROUND_THRESHOLD,
                 min_foreground_area: int = DEFAULT_MIN_FOREGROUND_AREA,
                 blur_kernel_size: int = DEFAULT_BLUR_KERNEL_SIZE):
        self.foreground_threshold = foreground_threshold
        self.min_foreground_area = min_foreground_area
        self.blur_kernel_size = blur_kernel_size
        # 添加调试计数器
        self._debug_counter = 0
        
        # 添加复杂度统计信息，用于调试
        self.complexity_history = []
        self.max_history_size = 1000
    
    def detect_foreground_mask(self, image: np.ndarray) -> np.ndarray:
        """
        检测病理图像的前景区域
        
        Args:
            image: RGB图像数组 (H, W, 3)
            
        Returns:
            foreground_mask: 二值掩码，True为前景，False为背景 (H, W)
        """
        # 确保输入图像格式正确
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            # 转换为 float64 确保计算精度
            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0
            elif image.dtype != np.float64:
                image = image.astype(np.float64)
                
            # 方法1: 基于HSV颜色空间的前景检测
            hsv = rgb2hsv(image)
            saturation = hsv[:, :, 1]
            value = hsv[:, :, 2]
            
            # 病理图像中，背景通常饱和度低且亮度高
            # 前景（组织）通常饱和度较高或亮度较低
            foreground_by_saturation = saturation > 0.15  # 调整阈值
            foreground_by_value = value < 0.9  # 调整阈值
            
            # 方法2: 基于灰度的OTSU阈值
            gray = rgb2gray(image)
            # 对于病理图像，通常较暗的区域是前景
            threshold = filters.threshold_otsu(gray)
            foreground_by_otsu = gray < threshold * 1.2  # 调整阈值
            
            # 方法3: 基于RGB强度的检测
            # 病理图像背景通常是白色或接近白色
            rgb_intensity = np.mean(image, axis=2)
            foreground_by_intensity = rgb_intensity < self.foreground_threshold
            
            # 方法4: 基于颜色方差的检测
            color_variance = np.var(image, axis=2)
            foreground_by_variance = color_variance > 0.01  # 有颜色变化的区域
            
            # 综合多种方法
            foreground_mask = (foreground_by_saturation | 
                             foreground_by_value | 
                             foreground_by_otsu | 
                             foreground_by_intensity |
                             foreground_by_variance)
        else:
            # 灰度图像
            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0
            elif image.dtype != np.float64:
                image = image.astype(np.float64)
                
            threshold = filters.threshold_otsu(image)
            foreground_mask = image < threshold * 1.2
        
        # 形态学操作去除噪声
        try:
            foreground_mask = morphology.remove_small_objects(
                foreground_mask, min_size=self.min_foreground_area
            )
            foreground_mask = morphology.binary_closing(
                foreground_mask, morphology.disk(3)
            )
        except Exception as e:
            print(f"[WARNING] Morphology operations failed: {e}")
        
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
            gray = image.copy()
        
        # 确保图像是 float64 类型
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float64) / 255.0
        elif gray.dtype != np.float64:
            gray = gray.astype(np.float64)
        
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
                hist, _ = np.histogram(patch, bins=64, range=(0, 1), density=True)
                hist = hist[hist > 0]  # 移除零值
                
                # 计算熵
                if len(hist) > 0:
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                    entropy_map[i, j] = entropy
                else:
                    entropy_map[i, j] = 0.0
        
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
            gray = image.copy()
        
        # 确保图像是 float64 类型
        if gray.dtype == np.uint8:
            gray = gray.astype(np.float64) / 255.0
        elif gray.dtype != np.float64:
            gray = gray.astype(np.float64)
        
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
        
        # 配置常量
        MAX_ENTROPY_FOR_NORMALIZATION = 7.0  # 128 bins的理论最大熵
        GRADIENT_SCALING_FACTORS = {
            'mean': 15.0,
            'std': 20.0,
            'p90': 10.0
        }
        TEXTURE_SCALING_FACTOR = 8.0
        EDGE_SCALING_FACTORS = {
            'density': 50.0,
            'ratio': 20.0
        }
        CONTRAST_SCALING_FACTORS = {
            'std': 25.0,
            'mean': 30.0
        }
        COLOR_SCALING_FACTOR = 10.0
        
        # 复杂度组件权重
        COMPLEXITY_WEIGHTS = {
            'foreground': 0.15,
            'entropy': 0.15,
            'gradient_mean': 0.12,
            'gradient_std': 0.10,
            'gradient_p90': 0.08,
            'texture': 0.10,
            'edge_density': 0.08,
            'edge_ratio': 0.07,
            'local_contrast': 0.08,
            'local_contrast_mean': 0.05,
            'color': 0.02
        }
        self._debug_counter += 1
        
        try:
            # 确保输入图像格式正确
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # 数据类型转换
            if image.dtype == np.uint8:
                image = image.astype(np.float64) / 255.0
            elif image.dtype != np.float64:
                image = image.astype(np.float64)
            
            # 检测前景区域
            foreground_mask = self.detect_foreground_mask(image)
            foreground_ratio = np.mean(foreground_mask)
            
            if len(image.shape) == 3:
                gray = rgb2gray(image)
            else:
                gray = image.copy()
            
            # 1. 计算信息熵 - 增强差异化
            if np.any(foreground_mask):
                foreground_pixels = gray[foreground_mask]
                # 使用更多的bins来提高敏感度
                hist, _ = np.histogram(foreground_pixels, bins=self.ENTROPY_BINS, range=(0, 1), density=True)
                hist = hist[hist > 0]
                if len(hist) > 1:  # 至少需要2个非零bins才能有意义的熵
                    entropy = -np.sum(hist * np.log2(hist + 1e-10))
                    entropy_normalized = min(entropy / MAX_ENTROPY_FOR_NORMALIZATION, 1.0)
                else:
                    entropy_normalized = 0.0
            else:
                entropy_normalized = 0.0
            
            # 2. 计算梯度复杂度 - 增强敏感度
            grad_x = filters.sobel_h(gray)
            grad_y = filters.sobel_v(gray)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            if np.any(foreground_mask):
                gradient_mean = np.mean(gradient_magnitude[foreground_mask])
                gradient_std = np.std(gradient_magnitude[foreground_mask])
                gradient_percentile_90 = np.percentile(gradient_magnitude[foreground_mask], 90)
            else:
                gradient_mean = np.mean(gradient_magnitude)
                gradient_std = np.std(gradient_magnitude)
                gradient_percentile_90 = np.percentile(gradient_magnitude, 90)
            
            # 3. 计算纹理复杂度 - 改进版本
            try:
                # 使用多尺度局部标准差
                kernel_small = np.ones((3, 3)) / 9.0
                kernel_large = np.ones((5, 5)) / 25.0
                
                local_mean_small = ndimage.convolve(gray, kernel_small, mode='constant')
                local_variance_small = ndimage.convolve(gray**2, kernel_small, mode='constant') - local_mean_small**2
                local_std_small = np.sqrt(np.maximum(local_variance_small, 0))
                
                local_mean_large = ndimage.convolve(gray, kernel_large, mode='constant')
                local_variance_large = ndimage.convolve(gray**2, kernel_large, mode='constant') - local_mean_large**2
                local_std_large = np.sqrt(np.maximum(local_variance_large, 0))
                
                if np.any(foreground_mask):
                    texture_complexity_small = np.mean(local_std_small[foreground_mask])
                    texture_complexity_large = np.mean(local_std_large[foreground_mask])
                else:
                    texture_complexity_small = np.mean(local_std_small)
                    texture_complexity_large = np.mean(local_std_large)
                
                texture_complexity = (texture_complexity_small + texture_complexity_large) / 2.0
                
            except Exception as e:
                # 回退到简单的全局标准差
                if np.any(foreground_mask):
                    texture_complexity = np.std(gray[foreground_mask])
                else:
                    texture_complexity = np.std(gray)
            
            # 4. 计算边缘密度 - 增强版
            try:
                edges = filters.canny(gray, sigma=self.CANNY_SIGMA, low_threshold=self.CANNY_LOW_THRESHOLD, high_threshold=self.CANNY_HIGH_THRESHOLD)
                if np.any(foreground_mask):
                    edge_density = np.mean(edges[foreground_mask])
                    edge_count = np.sum(edges[foreground_mask])
                    edge_ratio = edge_count / np.sum(foreground_mask) if np.sum(foreground_mask) > 0 else 0
                else:
                    edge_density = np.mean(edges)
                    edge_ratio = np.mean(edges)
            except Exception as e:
                edge_density = 0.0
                edge_ratio = 0.0
            
            # 5. 计算局部对比度
            try:
                laplacian = filters.laplace(gray)
                if np.any(foreground_mask):
                    local_contrast = np.std(laplacian[foreground_mask])
                    local_contrast_mean = np.mean(np.abs(laplacian[foreground_mask]))
                else:
                    local_contrast = np.std(laplacian)
                    local_contrast_mean = np.mean(np.abs(laplacian))
            except Exception as e:
                local_contrast = 0.0
                local_contrast_mean = 0.0
            
            # 6. 计算颜色复杂度（仅对RGB图像）
            color_complexity = 0.0
            if len(image.shape) == 3:
                try:
                    # 计算RGB通道间的标准差
                    if np.any(foreground_mask):
                        color_std = np.std(image[foreground_mask], axis=0)
                        color_complexity = np.mean(color_std)
                        
                        # 计算色彩饱和度变化
                        hsv = rgb2hsv(image)
                        saturation_std = np.std(hsv[foreground_mask, 1])
                        color_complexity = (color_complexity + saturation_std) / 2.0
                    else:
                        color_std = np.std(image.reshape(-1, 3), axis=0)
                        color_complexity = np.mean(color_std)
                except Exception as e:
                    color_complexity = 0.0
            
            # 7. 添加随机因子防止完全相同的复杂度值
            random_factor = np.random.normal(0, 0.02)  # 小的随机噪声
            
            # 综合复杂度分数 - 重新设计权重和缩放
            # 使用非线性映射确保更好的分布
            components = {
                'foreground': min(foreground_ratio * 2.0, 1.0),
                'entropy': entropy_normalized,
                'gradient_mean': min(gradient_mean * GRADIENT_SCALING_FACTORS['mean'], 1.0),
                'gradient_std': min(gradient_std * GRADIENT_SCALING_FACTORS['std'], 1.0),
                'gradient_p90': min(gradient_percentile_90 * GRADIENT_SCALING_FACTORS['p90'], 1.0),
                'texture': min(texture_complexity * TEXTURE_SCALING_FACTOR, 1.0),
                'edge_density': min(edge_density * EDGE_SCALING_FACTORS['density'], 1.0),
                'edge_ratio': min(edge_ratio * EDGE_SCALING_FACTORS['ratio'], 1.0),
                'local_contrast': min(local_contrast * CONTRAST_SCALING_FACTORS['std'], 1.0),
                'local_contrast_mean': min(local_contrast_mean * CONTRAST_SCALING_FACTORS['mean'], 1.0),
                'color': min(color_complexity * COLOR_SCALING_FACTOR, 1.0),
            }
            
            # 加权组合
            complexity_score = sum(
                COMPLEXITY_WEIGHTS[key] * components[key] 
                for key in COMPLEXITY_WEIGHTS.keys()
            )
            
            # 应用非线性变换增加差异化
            complexity_score = np.power(complexity_score, 0.75)  # 稍微压缩高值
            
            # 添加随机因子
            complexity_score = complexity_score + random_factor
            
            # 确保结果在[0, 1]范围内
            complexity_score = max(0.0, min(1.0, complexity_score))
            
            # 记录复杂度历史用于分析
            if len(self.complexity_history) >= self.max_history_size:
                self.complexity_history.pop(0)
            self.complexity_history.append(complexity_score)
            
            # 调试信息（每200次记录一次，减少频率）
            if self._debug_counter % 200 == 0:
                complexity_std = np.std(self.complexity_history) if len(self.complexity_history) > 1 else 0.0
                complexity_range = np.max(self.complexity_history) - np.min(self.complexity_history) if len(self.complexity_history) > 1 else 0.0
                logger.debug(f"TISSUE COMPLEXITY Sample #{self._debug_counter}: "
                           f"Foreground: {foreground_ratio:.3f}, "
                           f"Entropy: {entropy_normalized:.3f}, "
                           f"Gradient mean: {gradient_mean:.3f}, "
                           f"Final complexity: {complexity_score:.3f}, "
                           f"History std: {complexity_std:.3f}")
                
                # 检查复杂度分布是否合理
                if complexity_std < 0.01 and len(self.complexity_history) > 50:
                    logger.warning(f"Low complexity variance detected! std={complexity_std:.4f}")
            return complexity_score
            
        except Exception as e:
            if self._debug_counter % 200 == 0:
                logger.error(f"Complexity computation failed: {e}")
                
            # 返回一个随机复杂度避免全部相同
            return np.random.uniform(0.3, 0.7)


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
    针对病理图像优化的掩码生成器 - 修复版
    """
    # 配置常量
    CURRICULUM_END_RATIO = 0.8  # 课程学习在80%的epoch后完成
    MIN_BLOCK_SIZE_SCALE = 0.5  # 最小块大小相对于min_num_patches的比例
    COMPLEXITY_ADJUSTMENT_RANGE = 0.4  # 复杂度调整范围 ±40%
    MICRO_VARIATION_RANGE = 21  # 微调范围: -10 到 +10
    BLOCK_MASK_RATIO = 0.7  # 病理感知掩码中块状掩码的比例
    DEBUG_PRINT_INTERVAL = 500  # 调试信息打印间隔
    VALIDATION_PRINT_INTERVAL = 100  # 验证信息打印间隔
    COMPLEXITY_DEBUG_INTERVAL = 100  # 复杂度调试信息打印间隔

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
            num_masking_patches: 最大掩码patch数量（阈值）
            strategy: 掩码策略 ["random", "block", "pathology_aware", "entropy_based", "gradient_based"]
            foreground_bias: 前景区域掩码偏向权重 [0, 1]
            complexity_adaptive: 是否根据图像复杂度调整掩码率
            curriculum_masking: 是否使用课程学习掩码
        """
        if not isinstance(input_size, tuple):
            input_size = (input_size, input_size)
        
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.max_masking_patches = num_masking_patches # 最大掩码数量
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
        self.curriculum_progress = min(current_epoch / max(total_epochs * self.CURRICULUM_END_RATIO, 1), 1.0)

        # 课程学习的掩码范围
        if self.curriculum_masking:
            # 训练初期最少掩码数量（约为最大值的30%）
            self.curriculum_min_masks = max(min_num_patches, int(self.max_masking_patches * 0.3))
        else:
            self.curriculum_min_masks = self.max_masking_patches
        
        logger.info(f"PathologyMaskingGenerator initialized: "
                   f"Strategy={self.strategy}, "
                   f"Max masking patches={self.max_masking_patches}, "
                   f"Curriculum masking={self.curriculum_masking}, "
                   f"Complexity adaptive={self.complexity_adaptive}")
        
        if self.curriculum_masking:
            logger.info(f"  Curriculum range: {self.curriculum_min_masks} -> {self.max_masking_patches}")
        else:
            logger.info(f"  All strategies will use close to {self.max_masking_patches} masks")

        if self.complexity_adaptive:
           logger.info(f"Complexity adaptation: ±{int(self.COMPLEXITY_ADJUSTMENT_RANGE * 100)}% based on image complexity")
        
    def update_epoch(self, epoch: int):
        """更新当前epoch，用于课程学习"""
        self.current_epoch = epoch
        self.curriculum_progress = min(epoch / max(self.total_epochs * self.CURRICULUM_END_RATIO, 1), 1.0)

        if self.curriculum_masking and epoch % 10 == 0:
            current_base = self._get_curriculum_base_masks()
            logger.info(f"[CURRICULUM] Epoch {epoch}: progress={self.curriculum_progress:.3f}, "
                        f"base_masks={current_base} (range: {self.curriculum_min_masks}-{self.max_masking_patches})")

        elif not self.curriculum_masking and epoch == 0:
            logger.info(f"[NO CURRICULUM] All epochs will use close to {self.max_masking_patches} masks")

    def _get_curriculum_base_masks(self) -> int:
        """根据课程学习进度计算基础掩码数量"""
        if not self.curriculum_masking:
            return self.max_masking_patches
        
        # 线性插值：从 curriculum_min_masks 到 max_masking_patches
        base_masks = int(
            self.curriculum_min_masks + 
            (self.max_masking_patches - self.curriculum_min_masks) * self.curriculum_progress
        )
        
        return max(self.curriculum_min_masks, min(base_masks, self.max_masking_patches))
    
    def _compute_adaptive_block_size(self) -> Tuple[int, int]:
        """根据课程学习调整块大小"""
        if not self.curriculum_masking:
            return self.min_num_patches, self.max_num_patches
        
        # 训练初期使用小块，后期使用大块
        progress = self.curriculum_progress
        
        # 最小块大小从8逐渐增加到min_num_patchesint
        min_block_start = max(8, int(self.min_num_patches * self.MIN_BLOCK_SIZE_SCALE))
        adaptive_min = max(min_block_start, int(self.min_num_patches * (self.MIN_BLOCK_SIZE_SCALE + (1 - self.MIN_BLOCK_SIZE_SCALE) * progress)))
        # 最大块大小从min_num_patches逐渐增加到max_num_patches
        adaptive_max = min(
            self.max_num_patches,
            int(self.min_num_patches + (self.max_num_patches - self.min_num_patches) * progress)
        )
        
        return adaptive_min, adaptive_max
    
    def _generate_block_mask_with_limit(self, target_masks: int) -> torch.Tensor:
        """生成块状掩码，但不超过目标数量"""
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        masked_count = 0
        max_attempts = 100
        
        adaptive_min, adaptive_max = self._compute_adaptive_block_size()
        
        while masked_count < target_masks and max_attempts > 0:
            remaining = target_masks - masked_count
            if remaining <= 0:
                break
            
            # 动态调整目标区域大小，不超过剩余需要的数量
            max_block_size = min(adaptive_max, remaining)
            target_area = random.uniform(adaptive_min, max_block_size)
            
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = max(1, min(self.height, int(round(math.sqrt(target_area * aspect_ratio)))))
            w = max(1, min(self.width, int(round(math.sqrt(target_area / aspect_ratio)))))
            
            if h <= self.height and w <= self.width:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                
                # 计算这个区域会新增多少掩码
                region_slice = mask[top:top+h, left:left+w]
                new_mask_count = (~region_slice).sum().item()
                
                # 只有在不会超出目标时才应用
                if masked_count + new_mask_count <= target_masks:
                    mask[top:top+h, left:left+w] = True
                    masked_count = mask.sum().item()
                elif new_mask_count > 0:
                    # 如果整个块会超出，尝试部分应用
                    available_slots = target_masks - masked_count
                    if available_slots > 0:
                        # 随机选择块内的一些位置
                        unmasked_in_block = (~region_slice).nonzero(as_tuple=False)
                        if len(unmasked_in_block) > 0:
                            select_count = min(available_slots, len(unmasked_in_block))
                            selected = torch.randperm(len(unmasked_in_block))[:select_count]
                            for idx in selected:
                                rel_pos = unmasked_in_block[idx]
                                abs_h, abs_w = top + rel_pos[0], left + rel_pos[1]
                                mask[abs_h, abs_w] = True
                            masked_count = mask.sum().item()
            
            max_attempts -= 1
        
        return mask
    
    def _compute_adaptive_mask_count_with_base(self, image_complexity: float, base_masks: int) -> int:
        """
        基于给定基础掩码数量进行复杂度调整 - 修复版
        
        Args:
            image_complexity: 图像复杂度分数 [0, 1]
            base_masks: 基础掩码数量（可能来自课程学习或最大掩码数量）
            
        Returns:
            adaptive_count: 调整后的掩码数量
        """
        if not self.complexity_adaptive:
            return base_masks
        
        # 增强复杂度调整范围：±40%
        # image_complexity = 0.5 时，complexity_factor = 1.0 (无调整)
        # image_complexity = 0.0 时，complexity_factor = 1.4 (增加40%)
        # image_complexity = 1.0 时，complexity_factor = 0.6 (减少40%)
        complexity_factor = 1.0 - 0.8 * (image_complexity - 0.5)  # [0.6, 1.4]
        complexity_factor = max(0.6, min(1.4, complexity_factor))  # 严格限制范围
        
        # 应用复杂度调整
        adaptive_count = int(base_masks * complexity_factor)
        
        # 确保结果在合理范围内
        adaptive_count = max(
            self.min_num_patches,  # 不低于最小patch数
            min(adaptive_count, self.max_masking_patches)  # 不超过最大掩码数量
        )
        
        # 额外的边界检查：确保不超过总patch数
        adaptive_count = min(adaptive_count, self.num_patches - 1)

        # 添加调试信息
        if hasattr(self, '_complexity_debug_counter'):
            self._complexity_debug_counter += 1
        else:
            self._complexity_debug_counter = 1
        
        # 每100次调用记录一次复杂度调整信息
        if self._complexity_debug_counter % 100 == 0:
            logger.debug(f"COMPLEXITY DEBUG #{self._complexity_debug_counter}: "
                        f"Image complexity={image_complexity:.3f}, "
                        f"Complexity factor={complexity_factor:.3f}, "
                        f"Base masks={base_masks}, "
                        f"Adaptive masks={adaptive_count}, "
                        f"Adjustment={adaptive_count - base_masks:+d}")
        
        return adaptive_count
    
    def __call__(self, image=None) -> torch.Tensor:
        """
        生成掩码 
        
        Args:
            image: 输入图像（PIL或numpy），用于病理感知掩码生成
            
        Returns:
            mask: 掩码张量 (H, W)
        """
        import time
        batch_randomizer = hash(str(time.time()) + str(id(self))) % 1000
        
        # 确定目标掩码数量 - 修复策略区分逻辑
        if self.curriculum_masking:
            # 课程学习启用：根据epoch进度调整掩码数量
            base_masks = self._get_curriculum_base_masks()
            
            if image is not None and self.complexity_adaptive:
                # 在课程学习基础上进行复杂度调整
                image_np = pil_to_numpy(image)
                complexity = self.analyzer.compute_tissue_complexity(image_np)
                target_masks = self._compute_adaptive_mask_count_with_base(complexity, base_masks)
            else:
                # 即使没有复杂度自适应，也要加入小幅随机变化
                variation = max(1, int(base_masks * 0.1))  # 10%的变化范围
                target_masks = random.randint(
                    max(1, base_masks - variation),
                    min(self.num_patches - 1, base_masks + variation)
                )
        else:
            # 课程学习禁用：所有策略都使用接近最大掩码数量
            if image is not None and self.complexity_adaptive:
                # 复杂度自适应，但基于最大掩码数量
                image_np = pil_to_numpy(image)
                complexity = self.analyzer.compute_tissue_complexity(image_np)
                target_masks = self._compute_adaptive_mask_count_with_base(complexity, self.max_masking_patches)
            else:
                # 使用最大掩码数量，允许较大幅度随机变化
                variation = int(self.max_masking_patches * 0.15)  # 15%的变化范围
                target_masks = random.randint(
                    max(1, self.max_masking_patches - variation),
                    min(self.num_patches - 1, self.max_masking_patches + variation)
                )
        
         # 添加基于batch_randomizer的微调
        micro_variation = (batch_randomizer % self.MICRO_VARIATION_RANGE) - 10  # -10 到 +10 的微调
        target_masks = max(1, min(target_masks + micro_variation, self.num_patches - 1))

        # 添加调试信息
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 1
        
        # 每500次调用打印一次详细调试信息
        if self._debug_counter % 500 == 0:
            curriculum_base = self._get_curriculum_base_masks() if self.curriculum_masking else "disabled"
            logger.debug(f"Masking Generator Call #{self._debug_counter}: "
                        f"Strategy={self.strategy}, "
                        f"Curriculum masking={self.curriculum_masking}, "
                        f"Target masks={target_masks}, "
                        f"Max masks={self.max_masking_patches}")
            if self.curriculum_masking:
                logger.debug(f"Curriculum base masks={curriculum_base}, "
                           f"Curriculum progress={self.curriculum_progress:.3f}")
        
        # 生成掩码
        if self.strategy == "random":
            # 随机策略：精确生成 target_masks 个掩码
            mask_indices = torch.randperm(self.num_patches)[:target_masks]
            mask = torch.zeros(self.num_patches, dtype=torch.bool)
            mask[mask_indices] = True
            mask = mask.reshape(self.height, self.width)
            
        elif self.strategy == "block":
            # 块状策略：生成不超过 target_masks 的块状掩码
            mask = self._generate_block_mask_with_limit(target_masks)
            
        elif self.strategy in ["pathology_aware", "entropy_based", "gradient_based"]:
            if image is None:
                # 如果没有提供图像，回退到块状掩码
                mask = self._generate_block_mask_with_limit(target_masks)
            else:
                # 病理感知掩码生成
                mask = self._generate_pathology_aware_mask_with_limit(image, target_masks)
        else:
            raise ValueError(f"Unknown masking strategy: {self.strategy}")
        
        # 最终验证和统计
        final_mask_count = mask.sum().item()
        
        # 每100次调用验证一次
        if self._debug_counter % 100 == 0:
                logger.debug(f"VALIDATION Call #{self._debug_counter}: Generated {final_mask_count} masks, "
                        f"target was {target_masks}, max allowed: {self.max_masking_patches}")
        
        # 确保不超过最大限制
        if final_mask_count > self.max_masking_patches:
            logger.warning(f"Generated masks ({final_mask_count}) exceed max limit ({self.max_masking_patches})")
            mask = self._limit_mask_count(mask, self.max_masking_patches)
            final_mask_count = mask.sum().item()
            logger.info(f"CORRECTED Final mask count: {final_mask_count}")
        
        return mask
    
    def _generate_pathology_aware_mask_with_limit(self, image, target_masks: int) -> torch.Tensor:
        """生成病理感知的掩码，结合块状逻辑和权重选择"""
        image_np = pil_to_numpy(image)
        
        # 分析图像
        foreground_mask = self.analyzer.detect_foreground_mask(image_np)
        if foreground_mask.shape != (self.height, self.width):
            from scipy import ndimage
            foreground_mask = ndimage.zoom(foreground_mask, 
                                         (self.height / foreground_mask.shape[0], 
                                          self.width / foreground_mask.shape[1]), 
                                         order=0).astype(bool)
        
        entropy_map = None
        gradient_map = None
        
        # 计算熵图
        if self.strategy in ["pathology_aware", "entropy_based"]:
            patch_size = max(1, image_np.shape[0] // self.height, image_np.shape[1] // self.width)
            entropy_map = self.analyzer.compute_information_entropy(image_np, patch_size=patch_size)
            if entropy_map.shape != (self.height, self.width):
                from scipy import ndimage
                entropy_map = ndimage.zoom(entropy_map, 
                                         (self.height / entropy_map.shape[0], 
                                          self.width / entropy_map.shape[1]), 
                                         order=1)
        
        # 计算梯度图
        if self.strategy in ["pathology_aware", "gradient_based"]:
            patch_size = max(1, image_np.shape[0] // self.height, image_np.shape[1] // self.width)
            gradient_map = self.analyzer.compute_gradient_intensity(image_np, patch_size=patch_size)
            if gradient_map.shape != (self.height, self.width):
                from scipy import ndimage
                gradient_map = ndimage.zoom(gradient_map,
                                          (self.height / gradient_map.shape[0],
                                           self.width / gradient_map.shape[1]),
                                          order=1)
        
        # 混合策略：70%块状 + 30%权重随机
        block_target = int(target_masks * self.BLOCK_MASK_RATIO)
        weighted_target = target_masks - block_target
        
        # 第一阶段：生成病理感知的块状掩码
        mask = self._generate_weighted_block_mask(foreground_mask, entropy_map, gradient_map, block_target)
        
        # 第二阶段：使用权重随机选择填充剩余
        current_count = mask.sum().item()
        if current_count < target_masks and weighted_target > 0:
            mask = self._add_weighted_random_masks(mask, foreground_mask, entropy_map, gradient_map, 
                                                 target_masks - current_count)
        
        return mask
    
    def _generate_weighted_block_mask(self, foreground_mask, entropy_map, gradient_map, target_masks: int) -> torch.Tensor:
        """生成基于权重的块状掩码"""
        # 计算权重图
        weight_map = np.ones((self.height, self.width), dtype=np.float32)
        
        if foreground_mask is not None:
            foreground_weight = self.foreground_bias * 2.0 + 0.1
            background_weight = 0.1
            weight_map = np.where(foreground_mask, foreground_weight, background_weight)
        
        if entropy_map is not None:
            entropy_min, entropy_max = entropy_map.min(), entropy_map.max()
            if entropy_max > entropy_min:
                entropy_normalized = (entropy_map - entropy_min) / (entropy_max - entropy_min)
                weight_map *= (1.0 + entropy_normalized)
        
        if gradient_map is not None:
            grad_min, grad_max = gradient_map.min(), gradient_map.max()
            if grad_max > grad_min:
                gradient_normalized = (gradient_map - grad_min) / (grad_max - grad_min)
                weight_map *= (1.0 + gradient_normalized)
        
        # 归一化权重图
        if weight_map.sum() > 0:
            weight_map = weight_map / weight_map.sum()
        else:
            weight_map = np.ones_like(weight_map) / weight_map.size
        
        # 使用权重选择块的中心点，然后生成块状掩码
        mask = torch.zeros(self.height, self.width, dtype=torch.bool)
        masked_count = 0
        max_attempts = 50
        
        adaptive_min, adaptive_max = self._compute_adaptive_block_size()
        
        while masked_count < target_masks and max_attempts > 0:
            remaining = target_masks - masked_count
            if remaining <= 0:
                break
            
            # 根据权重选择块的中心位置
            flat_weights = weight_map.flatten()
            try:
                center_idx = np.random.choice(len(flat_weights), p=flat_weights)
                center_h = center_idx // self.width
                center_w = center_idx % self.width
            except:
                center_h = random.randint(0, self.height - 1)
                center_w = random.randint(0, self.width - 1)
            
            # 确定块大小
            max_block_size = min(adaptive_max, remaining)
            target_area = random.uniform(adaptive_min, max_block_size)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = max(1, min(self.height, int(round(math.sqrt(target_area * aspect_ratio)))))
            w = max(1, min(self.width, int(round(math.sqrt(target_area / aspect_ratio)))))
            
            # 计算块的位置（以center为中心）
            top = max(0, min(self.height - h, center_h - h // 2))
            left = max(0, min(self.width - w, center_w - w // 2))
            
            # 计算这个区域会新增多少掩码
            region_slice = mask[top:top+h, left:left+w]
            new_mask_count = (~region_slice).sum().item()
            
            # 只有在不会超出目标时才应用
            if masked_count + new_mask_count <= target_masks:
                mask[top:top+h, left:left+w] = True
                masked_count = mask.sum().item()
            
            max_attempts -= 1
        
        return mask
    
    def _add_weighted_random_masks(self, mask, foreground_mask, entropy_map, gradient_map, additional_masks: int) -> torch.Tensor:
        """在现有掩码基础上添加权重随机掩码"""
        if additional_masks <= 0:
            return mask
        
        # 获取未掩码的位置
        unmasked_positions = (~mask).nonzero(as_tuple=False)
        if len(unmasked_positions) == 0:
            return mask
        
        # 计算权重图
        weight_map = np.ones((self.height, self.width), dtype=np.float32)
        
        if foreground_mask is not None:
            foreground_weight = self.foreground_bias * 2.0 + 0.1
            background_weight = 0.1
            weight_map = np.where(foreground_mask, foreground_weight, background_weight)
        
        if entropy_map is not None:
            entropy_min, entropy_max = entropy_map.min(), entropy_map.max()
            if entropy_max > entropy_min:
                entropy_normalized = (entropy_map - entropy_min) / (entropy_max - entropy_min)
                weight_map *= (1.0 + entropy_normalized)
        
        if gradient_map is not None:
            grad_min, grad_max = gradient_map.min(), gradient_map.max()
            if grad_max > grad_min:
                gradient_normalized = (gradient_map - grad_min) / (grad_max - grad_min)
                weight_map *= (1.0 + gradient_normalized)
        
        # 计算未掩码位置的权重
        unmasked_weights = []
        for pos in unmasked_positions:
            unmasked_weights.append(weight_map[pos[0], pos[1]])
        
        unmasked_weights = np.array(unmasked_weights)
        if unmasked_weights.sum() > 0:
            unmasked_weights = unmasked_weights / unmasked_weights.sum()
            
            # 根据权重选择额外的掩码位置
            try:
                select_count = min(additional_masks, len(unmasked_positions))
                selected_indices = np.random.choice(
                    len(unmasked_positions),
                    size=select_count,
                    replace=False,
                    p=unmasked_weights
                )
                
                for idx in selected_indices:
                    pos = unmasked_positions[idx]
                    mask[pos[0], pos[1]] = True
            except Exception as e:
                logger.warning(f"Weighted random selection failed: {e}")
                # 回退到简单随机选择
                select_count = min(additional_masks, len(unmasked_positions))
                selected_indices = torch.randperm(len(unmasked_positions))[:select_count]
                for idx in selected_indices:
                    pos = unmasked_positions[idx]
                    mask[pos[0], pos[1]] = True
        
        return mask
    
    def _limit_mask_count(self, mask: torch.Tensor, max_count: int) -> torch.Tensor:
        """限制掩码数量不超过最大值"""
        current_count = mask.sum().item()
        if current_count <= max_count:
            return mask
        
        # 随机移除多余的掩码
        masked_positions = mask.nonzero(as_tuple=False)
        remove_count = current_count - max_count
        remove_indices = torch.randperm(len(masked_positions))[:remove_count]
        
        for idx in remove_indices:
            pos = masked_positions[idx]
            mask[pos[0], pos[1]] = False
        
        return mask
    
    def __repr__(self):
        return (f"PathologyMaskingGenerator(strategy={self.strategy}, "
                f"input_size=({self.height}, {self.width}), "
                f"max_masking_patches={self.max_masking_patches}, "
                f"foreground_bias={self.foreground_bias}, "
                f"complexity_adaptive={self.complexity_adaptive}, "
                f"curriculum_masking={self.curriculum_masking})")