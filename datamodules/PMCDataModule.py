from vlmo.datasets import PMCDataset # 确保 PMCDataset 在 datasets 目录下或可被导入
from .datamodule_base import BaseDataModule


class PMCDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PMCDataset

    @property
    def dataset_name(self):
        return "PMC" 