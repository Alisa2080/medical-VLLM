import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from functools import partial

# Import necessary components
from datasets.multimodal_datasets.vqa_gen_dataset import VQAGenDataset, collate_fn_vqa_gen
from datasets.multimodal_datasets.transforms import create_image_transform # Assuming transforms are defined here

class VQAGenDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for Generative VQA.
    """
    def __init__(self, _config, dist=False):
        super().__init__()
        self.config = _config
        self.data_dir = _config["data_root"]
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["batch_size"]
        self.tokenizer = BertTokenizerFast.from_pretrained(
            _config["tokenizer"], local_files_only=True
        )
        self.max_text_len = _config["max_text_len"]
        self.max_answer_len = _config["max_answer_len"]

        # Get image transforms
        train_transform_keys = _config["train_transform_keys"]
        val_transform_keys = _config["val_transform_keys"]
        self.train_transform = create_image_transform(self.config, train_transform_keys)
        self.val_transform = create_image_transform(self.config, val_transform_keys)

        self.collate_fn = partial(collate_fn_vqa_gen, pad_token_id=self.tokenizer.pad_token_id)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Load datasets."""
        if stage == "fit" or stage is None:
            self.train_dataset = VQAGenDataset(
                data_dir=self.data_dir,
                split="train", # Adjust split name if needed
                tokenizer=self.tokenizer,
                transform=self.train_transform,
                max_text_len=self.max_text_len,
                max_answer_len=self.max_answer_len,
                # Add image_dir_name and annotation_file_name if different from defaults
            )
            self.val_dataset = VQAGenDataset(
                data_dir=self.data_dir,
                split="val", # Adjust split name if needed
                tokenizer=self.tokenizer,
                transform=self.val_transform,
                max_text_len=self.max_text_len,
                max_answer_len=self.max_answer_len,
                # Add image_dir_name and annotation_file_name if different from defaults
            )
        if stage == "test" or stage is None:
            self.test_dataset = VQAGenDataset(
                data_dir=self.data_dir,
                split="test", # Adjust split name if needed
                tokenizer=self.tokenizer,
                transform=self.val_transform, # Use val transform for test
                max_text_len=self.max_text_len,
                max_answer_len=self.max_answer_len,
                # Add image_dir_name and annotation_file_name if different from defaults
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
