# Vision_Encoder/wsi_core/patch_quality_assessor.py
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
from skimage import feature, measure, filters
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import json
from tqdm import tqdm

class PatchQualityAssessor:
    """
    离线计算patch质量得分的工具类
    """
    def __init__(self, 
                 min_tissue_ratio: float = 0.1,
                 blur_threshold: float = 100.0,
                 diversity_weight: float = 0.3,
                 information_weight: float = 0.4,
                 tissue_weight: float = 0.3):
        self.min_tissue_ratio = min_tissue_ratio
        self.blur_threshold = blur_threshold
        self.diversity_weight = diversity_weight
        self.information_weight = information_weight
        self.tissue_weight = tissue_weight
    
    def assess_patch_quality(self, patch_path: str) -> Dict[str, float]:
        """
        评估单个patch的质量
        
        Returns:
            Dict containing various quality metrics
        """
        try:
            img = cv2.imread(patch_path)
            if img is None:
                return {'total_score': 0.0, 'tissue_ratio': 0.0, 'sharpness': 0.0, 'information_content': 0.0}
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 1. 组织区域比例评估
            tissue_ratio = self._calculate_tissue_ratio(img_rgb)
            
            # 2. 图像清晰度评估
            sharpness = self._calculate_sharpness(img)
            
            # 3. 信息内容评估
            information_content = self._calculate_information_content(img_rgb)
            
            # 4. 颜色多样性评估
            color_diversity = self._calculate_color_diversity(img_rgb)
            
            # 5. 纹理复杂度评估
            texture_complexity = self._calculate_texture_complexity(img)
            
            # 综合得分计算
            total_score = (
                tissue_ratio * self.tissue_weight +
                min(sharpness / self.blur_threshold, 1.0) * 0.2 +
                information_content * self.information_weight +
                color_diversity * self.diversity_weight +
                texture_complexity * 0.1
            )
            
            return {
                'total_score': float(total_score),
                'tissue_ratio': float(tissue_ratio),
                'sharpness': float(sharpness),
                'information_content': float(information_content),
                'color_diversity': float(color_diversity),
                'texture_complexity': float(texture_complexity)
            }
            
        except Exception as e:
            print(f"Error assessing patch {patch_path}: {e}")
            return {'total_score': 0.0, 'tissue_ratio': 0.0, 'sharpness': 0.0, 'information_content': 0.0}
    
    def _calculate_tissue_ratio(self, img_rgb: np.ndarray) -> float:
        """计算组织区域比例"""
        # 转换到HSV空间进行组织检测
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        
        # 白色区域检测（背景）
        white_mask = cv2.inRange(img_hsv, (0, 0, 200), (180, 30, 255))
        
        # 黑色区域检测（空洞或伪影）
        black_mask = cv2.inRange(img_hsv, (0, 0, 0), (180, 255, 50))
        
        # 组织区域 = 总区域 - 白色区域 - 黑色区域
        background_mask = cv2.bitwise_or(white_mask, black_mask)
        tissue_pixels = np.sum(background_mask == 0)
        total_pixels = img_rgb.shape[0] * img_rgb.shape[1]
        
        return tissue_pixels / total_pixels
    
    def _calculate_sharpness(self, img: np.ndarray) -> float:
        """计算图像清晰度（Laplacian方差）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var
    
    def _calculate_information_content(self, img_rgb: np.ndarray) -> float:
        """计算信息内容（基于熵）"""
        # 转换为灰度图
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        
        # 计算直方图
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        hist = hist[hist > 0]  # 移除零值
        
        # 归一化
        hist = hist / hist.sum()
        
        # 计算熵
        entropy = -np.sum(hist * np.log2(hist))
        
        # 归一化到0-1范围
        return entropy / 8.0  # log2(256) = 8
    
    def _calculate_color_diversity(self, img_rgb: np.ndarray) -> float:
        """计算颜色多样性"""
        # 重塑图像为像素列表
        pixels = img_rgb.reshape(-1, 3)
        
        # 使用K-means聚类来评估颜色多样性
        n_colors = min(8, len(np.unique(pixels.view(np.ndarray), axis=0)))
        if n_colors < 2:
            return 0.0
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # 计算聚类中心之间的平均距离
        centers = kmeans.cluster_centers_
        distances = []
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist = np.linalg.norm(centers[i] - centers[j])
                distances.append(dist)
        
        return np.mean(distances) / 255.0 if distances else 0.0
    
    def _calculate_texture_complexity(self, img: np.ndarray) -> float:
        """计算纹理复杂度（LBP特征）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 计算LBP特征
        lbp = feature.local_binary_pattern(gray, P=8, R=1, method='uniform')
        
        # 计算LBP直方图的熵
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy / np.log2(10)  # 归一化

def assess_wsi_patches(wsi_patch_folder: str, 
                      output_file: str,
                      assessor: PatchQualityAssessor = None) -> pd.DataFrame:
    """
    评估一个WSI文件夹中所有patch的质量
    """
    if assessor is None:
        assessor = PatchQualityAssessor()
    
    patch_files = []
    for f in os.listdir(wsi_patch_folder):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
            patch_files.append(os.path.join(wsi_patch_folder, f))
    
    results = []
    slide_id = os.path.basename(wsi_patch_folder)
    
    print(f"Assessing {len(patch_files)} patches for slide {slide_id}")
    
    for patch_path in tqdm(patch_files, desc=f"Processing {slide_id}"):
        patch_name = os.path.basename(patch_path)
        
        # 从文件名提取坐标
        coord_str = os.path.splitext(patch_name)[0]
        try:
            x, y = map(int, coord_str.split('_'))
        except:
            print(f"Warning: Cannot parse coordinates from {patch_name}")
            continue
        
        quality_metrics = assessor.assess_patch_quality(patch_path)
        
        result = {
            'slide_id': slide_id,
            'patch_name': patch_name,
            'patch_path': patch_path,
            'x_coord': x,
            'y_coord': y,
            **quality_metrics
        }
        results.append(result)
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    return df