import torch
import pytorch_lightning as pl
from typing import Callable, List, Optional
from torch.utils.data import DataLoader
from dataset.WSIBagDatasetMTL import WSIBagDatasetMIL


class WSIDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_csv: str,
                 patches_root_dir: str,
                 model_input_size: int,
                 train_batch_size: int = 1, # Outer DataLoader batch_size is 1 WSI
                 val_batch_size: int = 1,   # Outer DataLoader batch_size is 1 WSI
                 num_workers: int = 0,      # For the outer DataLoader
                 train_csv: Optional[str] = None,
                 val_csv: Optional[str] = None,
                 test_csv: Optional[str] = None,
                 label_column: str = "label",
                 slide_id_column: str = "slide_id",
                 **kwargs): # Catches other args like pin_memory if needed for DataLoader
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args to self.hparams

        self.train_dataset: Optional[WSIBagDatasetMIL] = None
        self.val_dataset: Optional[WSIBagDatasetMIL] = None
        self.test_dataset: Optional[WSIBagDatasetMIL] = None

    def setup(self, stage: Optional[str] = None):
        # Called on every GPU if DDP
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_data_source = self.hparams.train_csv if self.hparams.train_csv else self.hparams.data_csv
            self.train_dataset = WSIBagDatasetMIL(
                slide_list_csv=train_data_source,
                patches_root_dir=self.hparams.patches_root_dir,
                model_input_size=self.hparams.model_input_size,
                label_column=self.hparams.label_column,
                slide_id_column=self.hparams.slide_id_column
            )
            if self.hparams.val_csv:
                self.val_dataset = WSIBagDatasetMIL(
                    slide_list_csv=self.hparams.val_csv,
                    patches_root_dir=self.hparams.patches_root_dir,
                    model_input_size=self.hparams.model_input_size,
                    label_column=self.hparams.label_column,
                    slide_id_column=self.hparams.slide_id_column
                )
            print(f"Setup train dataset with {len(self.train_dataset)} WSIs.")
            if self.val_dataset:
                print(f"Setup val dataset with {len(self.val_dataset)} WSIs.")


        if stage == 'test' or stage is None:
            if self.hparams.test_csv:
                self.test_dataset = WSIBagDatasetMIL(
                    slide_list_csv=self.hparams.test_csv,
                    patches_root_dir=self.hparams.patches_root_dir,
                    model_input_size=self.hparams.model_input_size,
                    label_column=self.hparams.label_column,
                    slide_id_column=self.hparams.slide_id_column
                )
                if self.test_dataset:
                    print(f"Setup test dataset with {len(self.test_dataset)} WSIs.")


    def train_dataloader(self) -> DataLoader:
        if not self.train_dataset:
            raise RuntimeError("Train dataset not initialized. Call setup('fit') first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size, # Should be 1 for WSI processing
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=self.hparams.get('pin_memory', True if self.hparams.num_workers > 0 else False),
            drop_last=True, # Important if using grad_accum > 1 and want consistent batch counts
            persistent_workers=True if self.hparams.num_workers > 0 else False # Added
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if not self.val_dataset:
            return None # Or an empty DataLoader if preferred
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.val_batch_size, # Should be 1
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.get('pin_memory', True if self.hparams.num_workers > 0 else False),
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False # Added
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_dataset:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.val_batch_size, # Typically same as val
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=self.hparams.get('pin_memory', True if self.hparams.num_workers > 0 else False),
            drop_last=False,
            persistent_workers=True if self.hparams.num_workers > 0 else False # Added
        )