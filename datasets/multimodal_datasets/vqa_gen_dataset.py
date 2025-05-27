import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import os
from transformers import BertTokenizerFast
from typing import Callable, Optional, Dict, List, Tuple, Any

class VQAGenDataset(Dataset):
    """
    Dataset for Visual Question Answering (Generative Task).
    Loads images, questions, and answers, tokenizes them for sequence generation.
    """
    def __init__(
        self,
        data_dir: str,
        split: str, # e.g., 'train', 'val', 'test'
        tokenizer: BertTokenizerFast,
        transform: Callable,
        max_text_len: int = 40,
        max_answer_len: int = 64,
        image_dir_name: str = "images", # Subdirectory containing images
        annotation_file_name: str = "vqa_gen_annotations.json" # Example annotation file name
    ):
        """
        Args:
            data_dir: Root directory containing image subdir and annotation file.
            split: Dataset split ('train', 'val', 'test').
            tokenizer: Tokenizer instance.
            transform: Image transform function.
            max_text_len: Max length for tokenized question.
            max_answer_len: Max length for tokenized answer (target sequence).
            image_dir_name: Name of the subdirectory containing images.
            annotation_file_name: Name of the JSON file with annotations.
                                  Expected format: List of dicts, each with
                                  {'image_id': str, 'question': str, 'answer': str}
        """
        self.data_dir = data_dir
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_text_len = max_text_len
        self.max_answer_len = max_answer_len
        self.image_root = os.path.join(data_dir, image_dir_name)
        annotation_path = os.path.join(data_dir, annotation_file_name)

        try:
            with open(annotation_path, 'r') as f:
                self.annotations = json.load(f)
            # Filter annotations based on split if necessary (depends on JSON structure)
            # Example: self.annotations = [ann for ann in all_annotations if ann['split'] == split]
        except FileNotFoundError:
            print(f"Error: Annotation file not found at {annotation_path}")
            self.annotations = []
        except Exception as e:
            print(f"Error loading annotations from {annotation_path}: {e}")
            self.annotations = []

        self.bos_token_id = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
        self.eos_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        question = ann['question']
        answer = ann['answer']

        # --- Load Image ---
        image_path = os.path.join(self.image_root, f"{image_id}.jpg") # Assuming .jpg format
        try:
            image = Image.open(image_path).convert("RGB")
            pixel_values = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}, returning None pixel_values.")
            # Handle missing images appropriately, e.g., return dummy data or skip
            # For simplicity, returning zeros here, but skipping might be better
            pixel_values = torch.zeros((3, self.transform.transforms[0].size, self.transform.transforms[0].size)) # Adjust size based on transform
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            pixel_values = torch.zeros((3, self.transform.transforms[0].size, self.transform.transforms[0].size))

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
        # Add EOS token to the target answer sequence
        answer_encoding = self.tokenizer(
            answer + self.tokenizer.eos_token, # Append EOS token
            padding="max_length",
            truncation=True,
            max_length=self.max_answer_len, # Max length for the target sequence
            return_tensors="pt",
        )
        labels = answer_encoding["input_ids"].squeeze(0)
        # Replace padding token ID in labels with -100 for CrossEntropyLoss
        labels[labels == self.pad_token_id] = -100

        # --- Create Decoder Input IDs ---
        # Shift labels right and add BOS token
        decoder_input_ids = torch.full_like(labels, self.pad_token_id)
        decoder_input_ids[0] = self.bos_token_id
        # Copy labels (excluding the last token if truncated, and ignoring padding)
        valid_labels_mask = labels != -100
        valid_labels_len = valid_labels_mask.sum().item()
        copy_len = min(valid_labels_len, self.max_answer_len - 1) # Ensure we don't exceed length
        if copy_len > 0:
             decoder_input_ids[1 : 1 + copy_len] = labels[:copy_len] # Shifted labels

        # Set remaining positions after the shifted content to padding ID
        # This is already handled by initializing with pad_token_id

        return {
            "pixel_values": pixel_values,
            "text_ids": text_ids,           # Question for encoder
            "attention_mask": attention_mask, # Question mask for encoder
            "decoder_input_ids": decoder_input_ids, # Shifted answer for decoder input
            "labels": labels,               # Original answer (with EOS, padded with -100) for loss calculation
        }

def collate_fn_vqa_gen(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate function for VQAGenDataset. Pads sequences to the max length in the batch.
    """
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    text_ids = torch.stack([item["text_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    decoder_input_ids = torch.stack([item["decoder_input_ids"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    # Create decoder_attention_mask based on decoder_input_ids
    decoder_attention_mask = (decoder_input_ids != pad_token_id).long()

    return {
        "pixel_values": pixel_values,
        "text_ids": text_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_attention_mask": decoder_attention_mask, # Add decoder mask
        "labels": labels,
    }
