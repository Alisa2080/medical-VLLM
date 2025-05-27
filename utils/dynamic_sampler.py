# Vision_Encoder/dataset/dynamic_sampler.py
import torch
import numpy as np
from typing import Dict, List
import random

class DynamicPatchSampler:
    """
    训练过程中的动态patch采样器
    """
    def __init__(self, 
                 initial_samples_per_wsi: int = 500,
                 curriculum_epochs: int = 20,
                 final_samples_per_wsi: int = 1000):
        self.initial_samples = initial_samples_per_wsi
        self.curriculum_epochs = curriculum_epochs
        self.final_samples = final_samples_per_wsi
        self.current_epoch = 0
        
    def update_epoch(self, epoch: int):
        """更新当前epoch"""
        self.current_epoch = epoch
    
    def get_current_sample_size(self) -> int:
        """根据当前epoch计算采样数量"""
        if self.current_epoch >= self.curriculum_epochs:
            return self.final_samples
        
        # 线性增长
        progress = self.current_epoch / self.curriculum_epochs
        current_samples = int(
            self.initial_samples + 
            (self.final_samples - self.initial_samples) * progress
        )
        return current_samples
    
    def adaptive_sampling(self, 
                         patch_paths: List[str], 
                         quality_scores: Dict[str, float],
                         attention_weights: Dict[str, float] = None,
                         loss_history: List[float] = None) -> List[str]:
        """
        自适应采样：基于质量得分、注意力权重和损失历史
        """
        current_sample_size = self.get_current_sample_size()
        
        if len(patch_paths) <= current_sample_size:
            return patch_paths
        
        # 计算采样权重
        sampling_weights = []
        for patch_path in patch_paths:
            weight = quality_scores.get(patch_path, 0.5)  # 默认权重
            
            # 如果有注意力权重，结合使用
            if attention_weights and patch_path in attention_weights:
                weight = 0.6 * weight + 0.4 * attention_weights[patch_path]
            
            # 根据损失历史调整（如果损失较高，偏向采样更多高质量patch）
            if loss_history and len(loss_history) > 5:
                recent_loss = np.mean(loss_history[-5:])
                if recent_loss > np.mean(loss_history):
                    weight = weight ** 1.5  # 增强高质量patch的权重
            
            sampling_weights.append(weight)
        
        # 归一化权重
        sampling_weights = np.array(sampling_weights)
        sampling_weights = sampling_weights / sampling_weights.sum()
        
        # 加权采样
        selected_indices = np.random.choice(
            len(patch_paths),
            size=current_sample_size,
            replace=False,
            p=sampling_weights
        )
        
        return [patch_paths[i] for i in selected_indices]