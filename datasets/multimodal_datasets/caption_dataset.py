import json
import os
import random
from typing import Dict, List, Tuple, Union
import torch
from .dataset_manager import VLMoBaseDataset

class CaptionDataset(VLMoBaseDataset):
    """图像描述数据集，针对自回归生成任务优化"""
    
    def __init__(
        self,
        data_root: str,
        json_file: str,
        transform,
        tokenizer,
        max_text_len: int = 196,
        is_train: bool = True,
    ):
        super().__init__(data_root, transform, tokenizer, max_text_len)
        
        # 加载标注
        with open(os.path.join(data_root, json_file), 'r') as f:
            self.annotations = json.load(f)
            
        self.is_train = is_train
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # 处理图像
        img_path = os.path.join(self.data_root, item['image_path'])
        image = self._process_image(img_path)
        
        # 处理文本
        caption = item['caption']
        text_data = self._process_text(caption)
        
        return {
            "image": image,
            "input_ids": text_data["input_ids"],
            "attention_mask": text_data["attention_mask"],
            "labels": text_data["input_ids"].clone(), # 为自回归训练设置标签
        }
    
    def collate(self, batch):
        """针对自回归生成任务优化的批处理函数"""
        images = torch.stack([item["image"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        
        # 将标签中的填充位置设为-100，使损失函数忽略这些位置
        labels = labels.masked_fill(attention_mask == 0, -100)
        
        return {
            "pixel_values": images,
            "decoder_input_ids": input_ids,
            "decoder_attention_mask": attention_mask,
            "labels": labels,
        }

# TODO: 用户需要修改其 VQA 和 Text QA 数据集类的 __getitem__ 方法
# 以包含 'task_type' 字段 ('vqa' 或 'text_qa')
# 并为各自任务中不存在的字段添加 None 占位符。
# 示例：
# class YourVQADataset(Dataset):
#     def __getitem__(self, index):
#         # ... load data ...
#         return {
#             'image': image_tensor,
#             'text_ids': question_ids,
#             'text_masks': question_mask,
#             'vqa_labels': vqa_target_score,
#             'task_type': 'vqa',
#             'decoder_input_ids': None, # Placeholder
#             'labels': None,            # Placeholder
#         }
#
# class YourTextQADataset(Dataset):
#     def __getitem__(self, index):
#         # ... load data ...
#         return {
#             'text_ids': question_ids,
#             'text_masks': question_mask,
#             'decoder_input_ids': decoder_input_ids_tensor,
#             'labels': labels_tensor, # Padded with -100 later
#             'task_type': 'text_qa',
#             'image': None,             # Placeholder
#             'vqa_labels': None,        # Placeholder
#         }
