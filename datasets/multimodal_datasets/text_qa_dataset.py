import torch
from torch.utils.data import Dataset
import json
import os
from transformers import BertTokenizer
from typing import Callable, Optional, Dict, List, Tuple, Any

class TextQADataset(Dataset):
    """
    Dataset for Text-based Question Answering (Generative Task).
    Loads questions and answers, tokenizes them for sequence generation.
    """
    def __init__(
        self,
        split: str, # e.g., 'train', 'val', 'test'
        tokenizer: BertTokenizer,
        max_text_len: int = 256,
        max_answer_len: int = 256,
        data_dir: str = "annotations.json",
    ):
        """
        Args:
            data_dir: Root directory containing the annotation file.
            split: Dataset split ('train', 'val', 'test').
            tokenizer: Tokenizer instance.
            max_text_len: Max length for tokenized question (encoder input).
            max_answer_len: Max length for tokenized answer (decoder target).
            annotation_file_name: Name of the JSON file with annotations.
                                  Expected format: List of dicts, each with
                                  {'question': str, 'answer': str}
        """
        self.split = split
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_answer_len = max_answer_len
        self.data_dir = data_dir

        try:
            with open(self.data_dir, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
            # Filter annotations based on split if necessary
            # Example: self.annotations = [ann for ann in all_annotations if ann['split'] == split]
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {self.data_dir}")
            self.annotations = []
        except Exception as e:
            print(f"Error loading annotations from {self.data_dir}: {e}")
            self.annotations = []

        self.bos_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        question = ann["Question"]
        answer = ann["Response"]

        # --- Tokenize Question (Encoder Input) ---
        question_encoding = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        text_ids = question_encoding["input_ids"].squeeze(0) # Remove batch dim
        attention_mask = question_encoding["attention_mask"].squeeze(0)

        # --- Tokenize Answer (Decoder Target) ---
        answer = "" if answer is None else answer
        answer_encoding = self.tokenizer(
            answer + self.tokenizer.sep_token, # Append EOS token
            padding="max_length",
            truncation=True,
            max_length=self.max_answer_len,
            return_tensors="pt",
        )
        labels = answer_encoding["input_ids"].squeeze(0)
        labels[labels == self.pad_token_id] = -100 # For CrossEntropyLoss

        # --- Create Decoder Input IDs ---
        decoder_input_ids = torch.full_like(labels, self.pad_token_id)
        # 添加开头标记
        decoder_input_ids[0] = self.bos_token_id
        valid_labels_mask = labels != -100
        valid_labels_len = valid_labels_mask.sum().item() # 171
        copy_len = min(valid_labels_len, self.max_answer_len - 1)
        if copy_len > 0:
             decoder_input_ids[1 : 1 + copy_len] = labels[:copy_len]

        return {
            "text_ids": text_ids,           # Question for encoder
            "attention_mask": attention_mask, # Question mask for encoder
            "decoder_input_ids": decoder_input_ids, # Shifted answer for decoder input
            "labels": labels,               # Original answer (with EOS, padded with -100)
        }

def collate_fn_text_qa(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate function for TextQADataset.
    """
    text_ids = torch.stack([item["text_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    decoder_input_ids = torch.stack([item["decoder_input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    decoder_attention_mask = (decoder_input_ids != pad_token_id).bool()

    return {
        "text_ids": text_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": labels,
    }
