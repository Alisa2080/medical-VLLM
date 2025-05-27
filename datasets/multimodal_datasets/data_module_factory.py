from typing import Dict, Any
import pytorch_lightning as pl
from transformers import AutoTokenizer
from .datamodules import CaptionDataModule, VQADataModule, NLVR2DataModule, RetrievalDataModule
from .transforms import create_image_transform  
# 替换引用未实现的类为实际实现的数据模块
from .datamodules import CaptionDataModule, VQADataModule, BaseDataModule

class DataModuleFactory:
    """数据模块工厂类，根据配置创建适合不同任务的数据模块"""
    
    @staticmethod
    def create(config: Dict[str, Any]) -> pl.LightningDataModule:
        """
        根据配置创建数据模块
        
        Args:
            config: 包含数据集配置的字典
            
        Returns:
            LightningDataModule: PyTorch Lightning数据模块
        """
        dataset_name = config.get("dataset", "").lower()
        
        # 创建适合的数据模块
        if dataset_name == "caption":
            return CaptionDataModule(config)
        elif dataset_name == "vqa":
            return VQADataModule(config)
        # 添加其他数据集类型
        elif dataset_name == "nlvr2":
            return NLVR2DataModule(config)
        elif dataset_name == "irtr":
            return RetrievalDataModule(config)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
