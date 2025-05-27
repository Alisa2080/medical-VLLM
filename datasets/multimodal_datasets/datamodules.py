import os
import torch
from typing import Optional, Dict, Any
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizer

from .transforms import create_image_transform
from .datasets import (
    CaptionDataset,
    VQADataset,
    NLVR2Dataset,
    RetrievalDataset
)

class BaseDataModule(pl.LightningDataModule):
    """所有数据模块的基类，提供通用功能"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.data_root = config.get("data_root", "")
        
        # 图像变换配置
        self.image_size = config.get("image_size", 384)
        train_transform_key = config.get("train_transform_keys", ["square_transform_randaug"])[0]
        val_transform_key = config.get("val_transform_keys", ["square_transform"])[0]
        
        # 创建训练和验证的图像变换
        self.train_transform = create_image_transform(self.image_size, is_train=True)
        self.val_transform = create_image_transform(self.image_size, is_train=False)
        
        # 创建分词器
        tokenizer_path = config.get("tokenizer", "")
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.vocab_size = self.tokenizer.vocab_size
        else:
            self.tokenizer = None
            self.vocab_size = config.get("vocab_size", 30522)  # 默认BERT词汇表大小
        
        # 模型训练配置
        self.max_text_len = config.get("max_text_len", 196)
        self.image_only = config.get("image_only", False)
        self.text_only = config.get("text_only", False)
        
        # 初始化数据集
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """下载和准备数据，只在全局执行一次"""
        pass
        
    def setup(self, stage: Optional[str] = None):
        """
        为训练/验证/测试设置数据集。
        子类必须实现此方法以创建特定的数据集。
        """
        raise NotImplementedError("子类必须实现setup方法")
        
    def train_dataloader(self):
        """返回训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate_fn if hasattr(self.train_dataset, "collate_fn") else None,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def val_dataloader(self):
        """返回验证数据加载器"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, "collate_fn") else None,
            persistent_workers=True if self.num_workers > 0 else False
        )
        
    def test_dataloader(self):
        """返回测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate_fn if hasattr(self.test_dataset, "collate_fn") else None,
            persistent_workers=True if self.num_workers > 0 else False
        )

class CaptionDataModule(BaseDataModule):
    """用于图像描述任务的数据模块"""
    
    def __init__(self, config):
        super().__init__(config)
        # 特定于图像描述任务的配置
        self.caption_json_path = config.get("caption_json_path", "captions.json")
        
    def setup(self, stage: Optional[str] = None):
        """设置训练、验证和测试数据集"""
        if stage == "fit" or stage is None:
            # 创建训练数据集
            self.train_dataset = CaptionDataset(
                data_root=os.path.join(self.data_root, "train"),
                json_file=self.caption_json_path,
                transform=self.train_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=True
            )
            rank_zero_info(f"训练集大小: {len(self.train_dataset)}")
            
            # 创建验证数据集
            self.val_dataset = CaptionDataset(
                data_root=os.path.join(self.data_root, "val"),
                json_file=self.caption_json_path,
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=False
            )
            rank_zero_info(f"验证集大小: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            # 创建测试数据集
            self.test_dataset = CaptionDataset(
                data_root=os.path.join(self.data_root, "test"),
                json_file=self.caption_json_path,
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=False
            )
            rank_zero_info(f"测试集大小: {len(self.test_dataset)}")

class VQADataModule(BaseDataModule):
    """用于视觉问答任务的数据模块"""
    
    def __init__(self, config):
        super().__init__(config)
        # 视觉问答任务的特定配置
        self.questions_json = config.get("vqa_questions_json", "questions.json")
        self.answers_json = config.get("vqa_answers_json", "answers.json")
        self.vqav2_label_size = config.get("vqav2_label_size", 3129)
        
    def setup(self, stage: Optional[str] = None):
        """设置训练、验证和测试数据集"""
        if stage == "fit" or stage is None:
            # 创建训练数据集
            self.train_dataset = VQADataset(
                data_root=os.path.join(self.data_root, "train"),
                questions_json=self.questions_json,
                answers_json=self.answers_json,
                transform=self.train_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                label_size=self.vqav2_label_size,
                is_train=True
            )
            rank_zero_info(f"VQA训练集大小: {len(self.train_dataset)}")
            
            # 创建验证数据集
            self.val_dataset = VQADataset(
                data_root=os.path.join(self.data_root, "val"),
                questions_json=self.questions_json,
                answers_json=self.answers_json,
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                label_size=self.vqav2_label_size,
                is_train=False
            )
            rank_zero_info(f"VQA验证集大小: {len(self.val_dataset)}")
        
        if stage == "test" or stage is None:
            # 创建测试数据集
            self.test_dataset = VQADataset(
                data_root=os.path.join(self.data_root, "test"),
                questions_json=self.questions_json,
                answers_json=None,  # 测试集可能没有答案
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                label_size=self.vqav2_label_size,
                is_train=False
            )
            rank_zero_info(f"VQA测试集大小: {len(self.test_dataset)}")

    def collate(self, batch):
        """
        Custom collate function to handle missing images in VQA batches.
        """
        batch_size = len(batch)
        keys = batch[0].keys()
        final_batch = {}

        # Determine image shape from the first valid image
        img_shape = None
        for item in batch:
            if item["image"] is not None:
                img_shape = item["image"].shape  # (C, H, W)
                break
        # If no valid image found in batch, create a dummy shape
        if img_shape is None:
            img_shape = (3, self.image_size, self.image_size)  # Default C, H, W

        # Create placeholder for images and the mask
        images = []
        has_image_mask = torch.zeros(batch_size, dtype=torch.bool)

        for i, item in enumerate(batch):
            if item["image"] is not None:
                images.append(item["image"])
                has_image_mask[i] = True
            else:
                # Use a zero tensor as placeholder
                images.append(torch.zeros(img_shape))

        # Stack images (real and placeholders)
        final_batch["image"] = torch.stack(images)
        final_batch["has_image_mask"] = has_image_mask

        # Handle other keys using default_collate or specific padding
        for key in keys:
            if key == "image":  # Already handled
                continue
            elif key in ["text_ids", "text_masks", "vqa_targets", "vqa_scores"]:
                # These are tensors, use default_collate for stacking/padding
                final_batch[key] = default_collate([item[key] for item in batch])
            elif key == "index":  # Keep index if needed
                final_batch[key] = [item[key] for item in batch]
            # Add handling for other keys if necessary

        return final_batch

class NLVR2DataModule(BaseDataModule):
    """用于NLVR2自然语言视觉推理任务的数据模块"""
    
    def __init__(self, config):
        super().__init__(config)
        # NLVR2特定配置
        
    def setup(self, stage: Optional[str] = None):
        """设置训练、验证和测试数据集"""
        if stage == "fit" or stage is None:
            # NLVR2训练和验证
            self.train_dataset = NLVR2Dataset(
                data_root=os.path.join(self.data_root, "train"),
                transform=self.train_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=True
            )
            
            self.val_dataset = NLVR2Dataset(
                data_root=os.path.join(self.data_root, "dev"),
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=False
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = NLVR2Dataset(
                data_root=os.path.join(self.data_root, "test"),
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=False
            )

class RetrievalDataModule(BaseDataModule):
    """用于图像-文本检索任务的数据模块"""
    
    def __init__(self, config):
        super().__init__(config)
        # 检索任务特定配置
        self.is_f30k = "f30k" in config.get("datasets", [""])[0]
        
    def setup(self, stage: Optional[str] = None):
        dataset_cls = RetrievalDataset
        
        if stage == "fit" or stage is None:
            self.train_dataset = dataset_cls(
                data_root=self.data_root,
                split="train",
                transform=self.train_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=True,
                is_f30k=self.is_f30k
            )
            
            self.val_dataset = dataset_cls(
                data_root=self.data_root,
                split="val",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=False,
                is_f30k=self.is_f30k
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = dataset_cls(
                data_root=self.data_root,
                split="test",
                transform=self.val_transform,
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                is_train=False,
                is_f30k=self.is_f30k
            )
