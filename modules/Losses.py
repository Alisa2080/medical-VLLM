import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

class LDAMLoss(nn.Module):
    """
    LDAM Loss: Learning Deep Class-Balanced Models for Long-Tailed Recognition
    """
    def __init__(self, samples_per_cls: List[int], num_classes: int, C_factor: float = 1.0, scale: float = 30.0):
        """
        Args:
            samples_per_cls: 每个类别的样本数量列表 [N_0, N_1, ..., N_k-1]
            num_classes: 类别总数
            C_factor: margin计算的缩放因子
            scale: logits的缩放因子（类似于softmax temperature的倒数）
        """
        super(LDAMLoss, self).__init__()
        self.num_classes = num_classes
        self.C_factor = C_factor
        self.scale = scale
        
        # 计算每个类别的margin
        # margin = C_factor / (N_c^(1/4))
        margins = []
        for n_c in samples_per_cls:
            margin = C_factor / (n_c ** 0.25)
            margins.append(margin)
        
        self.margins = torch.tensor(margins, dtype=torch.float32)
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] 模型预测的logits
            targets: [batch_size] 真实标签
        
        Returns:
            loss: LDAM损失值
        """
        device = logits.device
        if self.margins.device != device:
            self.margins = self.margins.to(device)
        
        batch_size = logits.shape[0]
        
        # 为每个样本创建对应的margin
        batch_margins = self.margins[targets]  # [batch_size]
        
        # 修改logits：对于真实类别，减去对应的margin
        modified_logits = logits.clone()
        for i in range(batch_size):
            true_class = targets[i]
            modified_logits[i, true_class] -= batch_margins[i]
        
        # 应用缩放
        scaled_logits = modified_logits * self.scale
        
        # 计算交叉熵损失
        loss = F.cross_entropy(scaled_logits, targets)
        
        return loss

class ClassBalancedLoss(nn.Module):
    """
    类别平衡损失函数
    Class-Balanced Loss Based on Effective Number of Samples
    """
    def __init__(self, samples_per_cls: List[int], num_classes: int, beta: float = 0.999, 
                 loss_type: str = "focal", gamma: float = 2.0, alpha: Optional[List[float]] = None):
        """
        Args:
            samples_per_cls: 每个类别的样本数量列表
            num_classes: 类别总数
            beta: 有效样本数计算的衰减因子
            loss_type: 基础损失类型 ("cross_entropy", "focal")
            gamma: focal loss的gamma参数
            alpha: focal loss的alpha参数
        """
        super(ClassBalancedLoss, self).__init__()
        self.num_classes = num_classes
        self.beta = beta
        self.loss_type = loss_type
        self.gamma = gamma
        
        # 计算有效样本数 EN_c = (1 - beta^N_c) / (1 - beta)
        effective_nums = []
        for n_c in samples_per_cls:
            en = (1.0 - (beta ** n_c)) / (1.0 - beta) if beta != 1.0 else n_c
            effective_nums.append(en)
        
        # 计算权重：权重与有效样本数成反比
        weights = [1.0 / en for en in effective_nums]
        # 归一化权重
        total_weight = sum(weights)
        weights = [w / total_weight * num_classes for w in weights]
        
        self.weights = torch.tensor(weights, dtype=torch.float32)
        
        if alpha is None and loss_type == "focal":
            self.alpha = torch.ones(num_classes) * 0.25
        else:
            self.alpha = torch.tensor(alpha) if alpha else None
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        
        Returns:
            loss: 类别平衡损失值
        """
        device = logits.device
        if self.weights.device != device:
            self.weights = self.weights.to(device)
        
        if self.loss_type == "cross_entropy":
            # 使用类别权重的交叉熵损失
            loss = F.cross_entropy(logits, targets, weight=self.weights)
            
        elif self.loss_type == "focal":
            # 类别平衡的Focal Loss
            if self.alpha is not None and self.alpha.device != device:
                self.alpha = self.alpha.to(device)
            
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
            if self.alpha is not None:
                alpha_weights = self.alpha[targets]
                focal_loss = alpha_weights * focal_loss
            
            # 应用类别平衡权重
            cb_weights = self.weights[targets]
            loss = (cb_weights * focal_loss).mean()
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return loss

class CBCrossEntropyLoss(ClassBalancedLoss):
    """类别平衡交叉熵损失的简化版本"""
    def __init__(self, samples_per_cls: List[int], num_classes: int, beta: float = 0.999):
        super().__init__(samples_per_cls, num_classes, beta, loss_type="cross_entropy")

class CBFocalLoss(ClassBalancedLoss):
    """类别平衡Focal损失的简化版本"""
    def __init__(self, samples_per_cls: List[int], num_classes: int, beta: float = 0.999, 
                 gamma: float = 2.0, alpha: Optional[List[float]] = None):
        super().__init__(samples_per_cls, num_classes, beta, loss_type="focal", gamma=gamma, alpha=alpha)