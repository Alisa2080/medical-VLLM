# Vision_Encoder/dataset/WSIBagDatasetMTL_Smart.py
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from .WSIBagDatasetMTL import WSIBagDatasetMIL
import random

class SmartWSIBagDatasetMIL(WSIBagDatasetMIL):
    """
    支持智能patch采样的WSI数据集
    """
    def __init__(self,
                 slide_list_csv: str,
                 patches_root_dir: str,
                 quality_scores_dir: str,  # 质量评估结果目录
                 sampling_strategy: str = 'quality_based',  # 采样策略
                 max_patches_per_wsi: int = 1000,  # 每个WSI最大patch数量
                 min_patches_per_wsi: int = 100,   # 每个WSI最小patch数量
                 quality_threshold: float = 0.3,   # 质量阈值
                 diversity_sampling: bool = True,   # 是否进行多样性采样
                 **kwargs):
        
        super().__init__(slide_list_csv, patches_root_dir, **kwargs)
        
        self.quality_scores_dir = quality_scores_dir
        self.sampling_strategy = sampling_strategy
        self.max_patches_per_wsi = max_patches_per_wsi
        self.min_patches_per_wsi = min_patches_per_wsi
        self.quality_threshold = quality_threshold
        self.diversity_sampling = diversity_sampling
        
        # 加载质量评估数据
        self._load_quality_scores()
        
        # 重新处理slide数据，应用智能采样
        self._apply_smart_sampling()
    
    def _load_quality_scores(self):
        """加载所有WSI的质量评估数据"""
        self.quality_data = {}
        
        print("Loading patch quality scores...")
        for slide_info in self.slide_data:
            slide_id = slide_info['slide_id']
            quality_file = os.path.join(self.quality_scores_dir, f"{slide_id}_quality.csv")
            
            if os.path.exists(quality_file):
                df = pd.read_csv(quality_file)
                self.quality_data[slide_id] = df
            else:
                print(f"Warning: Quality file not found for {slide_id}, using random sampling")
                self.quality_data[slide_id] = None
    
    def _apply_smart_sampling(self):
        """应用智能采样策略"""
        print("Applying smart sampling strategies...")
        
        for i, slide_info in enumerate(self.slide_data):
            slide_id = slide_info['slide_id']
            original_patches = slide_info['patch_paths']
            quality_df = self.quality_data.get(slide_id)
            
            if quality_df is not None:
                # 基于质量进行采样
                sampled_patches = self._sample_patches_by_quality(
                    original_patches, quality_df, slide_id
                )
            else:
                # 随机采样作为fallback
                n_samples = min(len(original_patches), self.max_patches_per_wsi)
                sampled_patches = random.sample(original_patches, n_samples)
            
            # 更新patch路径
            self.slide_data[i]['patch_paths'] = sampled_patches
            self.slide_data[i]['n_patches_original'] = len(original_patches)
            self.slide_data[i]['n_patches_sampled'] = len(sampled_patches)
        
        # 打印采样统计
        self._print_sampling_stats()
    
    def _sample_patches_by_quality(self, patch_paths: List[str], 
                                 quality_df: pd.DataFrame, 
                                 slide_id: str) -> List[str]:
        """基于质量得分进行patch采样"""
        
        # 创建patch路径到质量得分的映射
        path_to_quality = {}
        for _, row in quality_df.iterrows():
            patch_name = row['patch_name']
            # 找到对应的完整路径
            matching_paths = [p for p in patch_paths if patch_name in os.path.basename(p)]
            if matching_paths:
                path_to_quality[matching_paths[0]] = row
        
        # 过滤出有质量得分的patch
        valid_patches = list(path_to_quality.keys())
        
        if not valid_patches:
            print(f"Warning: No quality scores found for patches in {slide_id}")
            n_samples = min(len(patch_paths), self.max_patches_per_wsi)
            return random.sample(patch_paths, n_samples)
        
        if self.sampling_strategy == 'quality_based':
            return self._quality_based_sampling(valid_patches, path_to_quality)
        elif self.sampling_strategy == 'stratified':
            return self._stratified_sampling(valid_patches, path_to_quality)
        elif self.sampling_strategy == 'diversity_aware':
            return self._diversity_aware_sampling(valid_patches, path_to_quality)
        else:
            return self._hybrid_sampling(valid_patches, path_to_quality)
    
    def _quality_based_sampling(self, patches: List[str], 
                              quality_map: Dict) -> List[str]:
        """基于质量得分的采样"""
        # 过滤低质量patch
        high_quality_patches = [
            p for p in patches 
            if quality_map[p]['total_score'] >= self.quality_threshold
        ]
        
        if len(high_quality_patches) < self.min_patches_per_wsi:
            # 如果高质量patch不够，降低阈值
            sorted_patches = sorted(patches, 
                                  key=lambda x: quality_map[x]['total_score'], 
                                  reverse=True)
            return sorted_patches[:max(self.min_patches_per_wsi, 
                                     min(len(sorted_patches), self.max_patches_per_wsi))]
        
        # 基于质量得分进行加权采样
        n_samples = min(len(high_quality_patches), self.max_patches_per_wsi)
        
        if n_samples >= len(high_quality_patches):
            return high_quality_patches
        
        # 计算采样权重
        scores = np.array([quality_map[p]['total_score'] for p in high_quality_patches])
        weights = scores / scores.sum()
        
        # 加权随机采样
        selected_indices = np.random.choice(
            len(high_quality_patches), 
            size=n_samples, 
            replace=False, 
            p=weights
        )
        
        return [high_quality_patches[i] for i in selected_indices]
    
    def _stratified_sampling(self, patches: List[str], 
                           quality_map: Dict) -> List[str]:
        """分层采样：确保不同质量层级都有代表"""
        scores = [quality_map[p]['total_score'] for p in patches]
        
        # 将patch分为3个质量层级
        score_33 = np.percentile(scores, 33)
        score_67 = np.percentile(scores, 67)
        
        low_quality = [p for p in patches if quality_map[p]['total_score'] < score_33]
        mid_quality = [p for p in patches if score_33 <= quality_map[p]['total_score'] < score_67]
        high_quality = [p for p in patches if quality_map[p]['total_score'] >= score_67]
        
        # 分配采样数量：高质量50%，中质量35%，低质量15%
        n_total = min(len(patches), self.max_patches_per_wsi)
        n_high = int(n_total * 0.5)
        n_mid = int(n_total * 0.35)
        n_low = n_total - n_high - n_mid
        
        # 从每个层级采样
        sampled = []
        sampled.extend(random.sample(high_quality, min(n_high, len(high_quality))))
        sampled.extend(random.sample(mid_quality, min(n_mid, len(mid_quality))))
        sampled.extend(random.sample(low_quality, min(n_low, len(low_quality))))
        
        return sampled
    
    def _diversity_aware_sampling(self, patches: List[str], 
                                quality_map: Dict) -> List[str]:
        """多样性感知采样：平衡质量和空间分布"""
        # 首先按质量过滤
        quality_filtered = [
            p for p in patches 
            if quality_map[p]['total_score'] >= self.quality_threshold
        ]
        
        if len(quality_filtered) < self.min_patches_per_wsi:
            quality_filtered = sorted(patches, 
                                    key=lambda x: quality_map[x]['total_score'], 
                                    reverse=True)[:self.max_patches_per_wsi]
        
        # 提取空间坐标
        coords = []
        for patch in quality_filtered:
            row = quality_map[patch]
            coords.append([row['x_coord'], row['y_coord']])
        
        coords = np.array(coords)
        
        # 使用贪婪算法进行空间多样性采样
        n_samples = min(len(quality_filtered), self.max_patches_per_wsi)
        
        if n_samples >= len(quality_filtered):
            return quality_filtered
        
        # 选择第一个patch（质量最高的）
        quality_scores = [quality_map[p]['total_score'] for p in quality_filtered]
        selected_indices = [np.argmax(quality_scores)]
        selected_coords = [coords[selected_indices[0]]]
        
        # 贪婪选择剩余的patch
        for _ in range(n_samples - 1):
            max_min_dist = -1
            best_idx = -1
            
            for i, patch in enumerate(quality_filtered):
                if i in selected_indices:
                    continue
                
                # 计算到已选择patch的最小距离
                min_dist = min([
                    np.linalg.norm(coords[i] - selected_coord) 
                    for selected_coord in selected_coords
                ])
                
                # 结合距离和质量得分
                combined_score = min_dist * 0.7 + quality_scores[i] * 0.3
                
                if combined_score > max_min_dist:
                    max_min_dist = combined_score
                    best_idx = i
            
            if best_idx != -1:
                selected_indices.append(best_idx)
                selected_coords.append(coords[best_idx])
        
        return [quality_filtered[i] for i in selected_indices]
    
    def _hybrid_sampling(self, patches: List[str], 
                        quality_map: Dict) -> List[str]:
        """混合采样策略：结合多种方法"""
        n_total = min(len(patches), self.max_patches_per_wsi)
        
        # 50%使用质量采样
        n_quality = int(n_total * 0.5)
        quality_sampled = self._quality_based_sampling(patches, quality_map)[:n_quality]
        
        # 30%使用多样性采样
        n_diversity = int(n_total * 0.3)
        remaining_patches = [p for p in patches if p not in quality_sampled]
        if remaining_patches:
            diversity_sampled = self._diversity_aware_sampling(remaining_patches, quality_map)[:n_diversity]
        else:
            diversity_sampled = []
        
        # 20%随机采样
        n_random = n_total - len(quality_sampled) - len(diversity_sampled)
        final_remaining = [p for p in patches if p not in quality_sampled and p not in diversity_sampled]
        if final_remaining and n_random > 0:
            random_sampled = random.sample(final_remaining, min(n_random, len(final_remaining)))
        else:
            random_sampled = []
        
        return quality_sampled + diversity_sampled + random_sampled
    
    def _print_sampling_stats(self):
        """打印采样统计信息"""
        total_original = sum([s['n_patches_original'] for s in self.slide_data])
        total_sampled = sum([s['n_patches_sampled'] for s in self.slide_data])
        
        print(f"\nSmart Sampling Statistics:")
        print(f"Total WSIs: {len(self.slide_data)}")
        print(f"Original patches: {total_original}")
        print(f"Sampled patches: {total_sampled}")
        print(f"Sampling ratio: {total_sampled/total_original:.2%}")
        
        for slide_info in self.slide_data[:5]:  # 显示前5个WSI的统计
            print(f"  {slide_info['slide_id']}: {slide_info['n_patches_original']} -> {slide_info['n_patches_sampled']}")