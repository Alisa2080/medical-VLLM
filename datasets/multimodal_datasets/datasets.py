import os
import json
import torch
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from pytorch_lightning.utilities import rank_zero_info

class CaptionDataset(Dataset):
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
        super().__init__()
        
        self.data_root = data_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.is_train = is_train
        
        # 加载标注
        json_path = os.path.join(data_root, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            rank_zero_info(f"已加载{len(self.annotations)}个样本从{json_path}")
        else:
            # 如果文件不存在，则查找替代文件或使用空列表
            rank_zero_info(f"警告: {json_path}不存在，尝试查找替代文件...")
            self.annotations = self._find_alternative_annotations(data_root)
    
    def _find_alternative_annotations(self, data_root):
        """查找目录中可用的标注文件"""
        json_files = list(Path(data_root).glob("*.json"))
        if not json_files:
            rank_zero_info(f"在{data_root}中找不到任何json文件，返回空列表")
            return []
        
        # 使用找到的第一个json文件
        with open(str(json_files[0]), 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        rank_zero_info(f"使用替代文件{json_files[0]}，包含{len(annotations)}个样本")
        return annotations
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # 处理图像
        if isinstance(item, dict):
            img_path = os.path.join(self.data_root, item.get("image_path", item.get("image", "")))
            caption = item.get("caption", "")
        else:
            # 如果是列表等其他格式，尝试适应
            img_path = os.path.join(self.data_root, item[0] if isinstance(item, list) else item)
            caption = item[1] if isinstance(item, list) and len(item) > 1 else ""
        
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            rank_zero_info(f"图像加载错误: {img_path}, {e}")
            # 创建空白图像作为备用
            image = torch.zeros(3, self.transform.transforms[-1].size[0], self.transform.transforms[-1].size[1])
        
        # 处理文本
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(), # 为自回归训练设置标签
        }
    
    def collate_fn(self, batch):
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

class VQADataset(Dataset):
    """视觉问答数据集"""
    
    def __init__(
        self,
        data_root: str,
        questions_json: str,
        answers_json: Optional[str],
        transform,
        tokenizer,
        max_text_len: int = 196,
        label_size: int = 3129,
        is_train: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.label_size = label_size
        self.is_train = is_train
        
        # 加载问题
        with open(os.path.join(data_root, questions_json), 'r') as f:
            self.questions = json.load(f)
        
        # 加载答案（如果可用）
        self.answers = None
        if answers_json and os.path.exists(os.path.join(data_root, answers_json)):
            with open(os.path.join(data_root, answers_json), 'r') as f:
                self.answers = json.load(f)
    
    def __len__(self):
        return len(self.questions)
    
    def get_image(self, index):
        question_item = self.questions[index]
        img_path = os.path.join(self.data_root, "images", question_item["image_id"])
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            rank_zero_info(f"图像加载错误: {img_path}, {e}")
            image = None  # Return None for missing or invalid images
        return {"image": image}
    
    def __getitem__(self, idx):
        question_item = self.questions[idx]
        
        # 获取图像
        image_data = self.get_image(idx)
        
        # 处理问题文本
        question = question_item["question"]
        encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        
        # 准备答案标签
        vqa_targets = torch.zeros(self.label_size)
        if self.is_train and self.answers:
            answers = self.answers.get(str(question_item["question_id"]))
            if answers:
                for answer, score in answers.items():
                    answer_id = int(answer) if answer.isdigit() else hash(answer) % self.label_size
                    vqa_targets[answer_id] = score
        
        return {
            **image_data,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vqa_targets": vqa_targets,
            "vqa_scores": vqa_targets.sum().item() if self.is_train else 0,
            "question_id": question_item["question_id"],
        }
    
    def collate_fn(self, batch):
        """VQA任务的批处理函数"""
        images = torch.stack([item["image"] if item["image"] is not None else torch.zeros(3, self.transform.transforms[-1].size[0], self.transform.transforms[-1].size[1]) for item in batch])
        has_image_mask = torch.tensor([item["image"] is not None for item in batch], dtype=torch.bool)
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        if self.is_train:
            vqa_targets = torch.stack([item["vqa_targets"] for item in batch])
            vqa_scores = torch.tensor([item["vqa_scores"] for item in batch])
        else:
            vqa_targets = None
            vqa_scores = None
        
        question_ids = [item["question_id"] for item in batch]
        
        return {
            "pixel_values": images,
            "has_image_mask": has_image_mask,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vqa_targets": vqa_targets,
            "vqa_scores": vqa_scores,
            "question_ids": question_ids,
        }

class NLVR2Dataset(Dataset):
    """NLVR2自然语言视觉推理数据集"""
    
    def __init__(
        self,
        data_root: str,
        transform,
        tokenizer,
        max_text_len: int = 196,
        is_train: bool = True,
    ):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.is_train = is_train
        
        # 加载NLVR2数据
        split = "train" if is_train else "dev"
        jsonl_path = os.path.join(data_root, f"{split}.jsonl")
        
        self.items = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.items.append(json.loads(line))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # NLVR2有两张图片和一段文本
        img_path_1 = os.path.join(self.data_root, "images", item["directory"], item["identifier"] + "-img0.png")
        img_path_2 = os.path.join(self.data_root, "images", item["directory"], item["identifier"] + "-img1.png")
        
        try:
            image_1 = Image.open(img_path_1).convert("RGB")
            image_1 = self.transform(image_1)
            
            image_2 = Image.open(img_path_2).convert("RGB")
            image_2 = self.transform(image_2)
        except Exception as e:
            rank_zero_info(f"图像加载错误: {e}")
            # 创建空白图像作为备用
            size = self.transform.transforms[-1].size[0]
            image_1 = torch.zeros(3, size, size)
            image_2 = torch.zeros(3, size, size)
        
        # 处理文本
        text = item["sentence"]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        
        # 处理标签
        label = torch.tensor(1 if item["label"] == "True" else 0, dtype=torch.long)
        
        return {
            "image_1": image_1,
            "image_2": image_2,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
            "identifier": item["identifier"],
        }
    
    def collate_fn(self, batch):
        """NLVR2任务的批处理函数"""
        images_1 = torch.stack([item["image_1"] for item in batch])
        images_2 = torch.stack([item["image_2"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        identifiers = [item["identifier"] for item in batch]
        
        return {
            "pixel_values_1": images_1,
            "pixel_values_2": images_2,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "identifiers": identifiers,
        }

class RetrievalDataset(Dataset):
    """图像-文本检索数据集"""
    
    def __init__(
        self,
        data_root: str,
        split: str,
        transform,
        tokenizer,
        max_text_len: int = 196,
        is_train: bool = True,
        is_f30k: bool = False,
    ):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.is_train = is_train
        self.is_f30k = is_f30k
        
        # 决定数据集路径
        dataset_name = "f30k" if is_f30k else "coco"
        annotations_dir = os.path.join(data_root, dataset_name)
        
        # 加载图像标注
        with open(os.path.join(annotations_dir, f"{split}_imgs.json"), 'r') as f:
            self.images = json.load(f)
        
        # 加载文本标注
        with open(os.path.join(annotations_dir, f"{split}_captions.json"), 'r') as f:
            self.captions = json.load(f)
        
        # 处理图像和文本的对应关系
        self.img2txt = {}
        self.txt2img = {}
        for img_id, img_item in enumerate(self.images):
            self.img2txt[img_id] = []
        
        for txt_id, txt_item in enumerate(self.captions):
            img_id = txt_item["image_id"]
            self.img2txt[img_id].append(txt_id)
            self.txt2img[txt_id] = img_id
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        txt_item = self.captions[idx]
        img_id = self.txt2img[idx]
        img_item = self.images[img_id]
        
        # 处理图像
        img_path = os.path.join(self.data_root, img_item["image"])
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception as e:
            rank_zero_info(f"图像加载错误: {img_path}, {e}")
            # 创建空白图像作为备用
            image = torch.zeros(3, self.transform.transforms[-1].size[0], self.transform.transforms[-1].size[1])
        
        # 处理文本
        caption = txt_item["caption"]
        encoding = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        
        input_ids = encoding.input_ids[0]
        attention_mask = encoding.attention_mask[0]
        
        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_id": img_id,
            "txt_id": idx,
        }
    
    def collate_fn(self, batch):
        """检索任务的批处理函数"""
        images = torch.stack([item["image"] for item in batch])
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        
        img_ids = torch.tensor([item["img_id"] for item in batch])
        txt_ids = torch.tensor([item["txt_id"] for item in batch])
        
        return {
            "pixel_values": images,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
        }
