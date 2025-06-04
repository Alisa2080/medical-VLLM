import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from functools import partial

# Import necessary components
from datasets.multimodal_datasets.text_qa_dataset import TextQADataset, collate_fn_text_qa

class TextQADataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Text-based Generative QA.
    """
    def __init__(self, config, batch_size,**kwargs):
        super().__init__()
        self.config = config
        self.data_dir = config.data_root
        self.num_workers = config.num_workers
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained(
            config.tokenizer_path, local_files_only=True
        )
        self.max_text_len = config.max_text_len
        self.max_answer_len = config.max_answer_len


        self.collate_fn = partial(collate_fn_text_qa, pad_token_id=self.tokenizer.pad_token_id)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Load datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = TextQADataset(
                data_dir=self.data_dir,
                split="train", 
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                max_answer_len=self.max_answer_len,
                
            )
            self.val_dataset = TextQADataset(
                data_dir=self.data_dir,
                split="val", 
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                max_answer_len=self.max_answer_len,
                
            )
        if stage == "test" or stage is None:
            self.test_dataset = TextQADataset(
                data_dir=self.data_dir,
                split="test", # Adjust split name if needed
                tokenizer=self.tokenizer,
                max_text_len=self.max_text_len,
                max_answer_len=self.max_answer_len,
                # Add annotation_file_name if different from default
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
