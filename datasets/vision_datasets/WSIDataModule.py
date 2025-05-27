import torch
import pytorch_lightning as pl
from typing import Callable, List, Optional
from torch.utils.data import DataLoader
from vision_datasets.WSIBagDatasetMTL import WSIBagDatasetMIL
import pandas as pd

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

        self.samples_per_cls: Optional[List[int]] = None
        self.num_classes: Optional[int] = None
        self.class_distribution: Optional[dict] = None

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

            self._compute_class_distribution(train_data_source)

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

            self._print_class_distribution()

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

    def _compute_class_distribution(self, train_csv_path: str):
        """计算训练集的类别分布"""
        try:
            df = pd.read_csv(train_csv_path)
            
            # 统计各类别的样本数
            label_counts = df[self.hparams.label_column].value_counts().sort_index()
            
            self.num_classes = len(label_counts)
            self.class_distribution = label_counts.to_dict()
            
            # 创建samples_per_cls列表，按照类别标签顺序
            self.samples_per_cls = []
            for class_idx in range(self.num_classes):
                if class_idx in self.class_distribution:
                    self.samples_per_cls.append(self.class_distribution[class_idx])
                else:
                    self.samples_per_cls.append(0)
                    print(f"Warning: Class {class_idx} has 0 samples in training set")
            
            print(f"Computed class distribution from {train_csv_path}")
            
        except Exception as e:
            print(f"Error computing class distribution: {e}")
            # 设置默认值
            self.num_classes = 2
            self.samples_per_cls = [1, 1]  # 默认平衡
            self.class_distribution = {0: 1, 1: 1}

    def _print_class_distribution(self):
        """打印类别分布信息"""
        if self.class_distribution is None:
            return
        
        print("\n" + "="*50)
        print("Training Set Class Distribution:")
        print("="*50)
        
        total_samples = sum(self.samples_per_cls)
        
        for class_idx, count in enumerate(self.samples_per_cls):
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"Class {class_idx}: {count:4d} samples ({percentage:5.1f}%)")
        
        print(f"Total: {total_samples} samples")
        
        # 计算不平衡比率
        if len(self.samples_per_cls) >= 2:
            max_samples = max(self.samples_per_cls)
            min_samples = min([s for s in self.samples_per_cls if s > 0])
            imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
            print(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
        
        print("="*50 + "\n")

    def get_class_info(self):
        """获取类别信息，供其他模块使用"""
        return {
            'samples_per_cls': self.samples_per_cls,
            'num_classes': self.num_classes,
            'class_distribution': self.class_distribution
        }

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