# Vision_Encoder/dataset/WSIDataModule_Smart.py
from typing import Optional
from .WSIDataModule import WSIDataModule
from .WSIDataModule_Smart import SmartWSIBagDatasetMIL

class SmartWSIDataModule(WSIDataModule):
    """
    支持智能采样的数据模块
    """
    def __init__(self, 
                 quality_scores_dir: str,
                 sampling_strategy: str = 'hybrid',
                 max_patches_per_wsi: int = 1000,
                 min_patches_per_wsi: int = 300,
                 quality_threshold: float = 0.3,
                 **kwargs):
        super().__init__(**kwargs)
        self.quality_scores_dir = quality_scores_dir
        self.sampling_strategy = sampling_strategy
        self.max_patches_per_wsi = max_patches_per_wsi
        self.min_patches_per_wsi = min_patches_per_wsi
        self.quality_threshold = quality_threshold

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            train_data_source = self.hparams.train_csv if self.hparams.train_csv else self.hparams.data_csv

            # 先计算类别分布
            self._compute_class_distribution(train_data_source)
            
            self.train_dataset = SmartWSIBagDatasetMIL(
                slide_list_csv=train_data_source,
                patches_root_dir=self.hparams.patches_root_dir,
                quality_scores_dir=self.quality_scores_dir,
                sampling_strategy=self.sampling_strategy,
                max_patches_per_wsi=self.max_patches_per_wsi,
                min_patches_per_wsi=self.min_patches_per_wsi,
                quality_threshold=self.quality_threshold,
                model_input_size=self.hparams.model_input_size,
                label_column=self.hparams.label_column,
                slide_id_column=self.hparams.slide_id_column
            )
            
            if self.hparams.val_csv:
                self.val_dataset = SmartWSIBagDatasetMIL(
                    slide_list_csv=self.hparams.val_csv,
                    patches_root_dir=self.hparams.patches_root_dir,
                    quality_scores_dir=self.quality_scores_dir,
                    sampling_strategy=self.sampling_strategy,
                    max_patches_per_wsi=self.max_patches_per_wsi,
                    min_patches_per_wsi=self.min_patches_per_wsi,
                    quality_threshold=self.quality_threshold,
                    model_input_size=self.hparams.model_input_size,
                    label_column=self.hparams.label_column,
                    slide_id_column=self.hparams.slide_id_column
                )
            
            print(f"Setup train dataset with {len(self.train_dataset)} WSIs.")
            if self.val_dataset:
                print(f"Setup val dataset with {len(self.val_dataset)} WSIs.")
            
            # 打印类别分布信息
            self._print_class_distribution()
        
        if stage == 'test' or stage is None:
            if self.hparams.test_csv:
                self.test_dataset = SmartWSIBagDatasetMIL(
                    slide_list_csv=self.hparams.test_csv,
                    patches_root_dir=self.hparams.patches_root_dir,
                    quality_scores_dir=self.quality_scores_dir,
                    sampling_strategy=self.sampling_strategy,
                    max_patches_per_wsi=self.max_patches_per_wsi,
                    min_patches_per_wsi=self.min_patches_per_wsi,
                    quality_threshold=self.quality_threshold,
                    model_input_size=self.hparams.model_input_size,
                    label_column=self.hparams.label_column,
                    slide_id_column=self.hparams.slide_id_column
                )
                if self.test_dataset:
                    print(f"Setup test dataset with {len(self.test_dataset)} WSIs.")