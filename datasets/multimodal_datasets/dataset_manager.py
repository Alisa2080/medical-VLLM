import torch
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizer
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class VLMoBaseDataset(Dataset):
    """多模态基础数据集类，处理图像和文本数据"""
    
    def __init__(
        self, 
        data_root: str,
        transform: transforms.Compose,
        tokenizer: PreTrainedTokenizer,
        max_text_len: int = 196,
    ):
        self.data_root = data_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        
    def _process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
        
    def _process_text(self, text):
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids[0],
            "attention_mask": encoding.attention_mask[0],
        }
    
    def collate(self, batch):
        """批处理函数，对批次数据进行整理"""
        raise NotImplementedError("子类必须实现collate方法")
