import functools

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules


class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        """
        初始化MTDataModule类的实例。

        参数:
        _config (dict): 包含数据集配置信息的字典。
        dist (bool): 是否使用分布式训练，默认为False。
        """
        # 从配置中获取数据集的键
        datamodule_keys = _config["datasets"]
        # 确保至少有一个数据集被配置
        assert len(datamodule_keys) > 0

        # 调用父类的构造函数
        super().__init__()

        # 保存数据集的键
        self.dm_keys = datamodule_keys
        # 根据配置创建数据集模块的字典
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        # 从字典中提取数据集模块的列表
        self.dms = [v for k, v in self.dm_dicts.items()]

        # 使用第一个数据集模块的批量大小
        self.batch_size = self.dms[0].batch_size
        # 使用第一个数据集模块的词汇表大小
        self.vocab_size = self.dms[0].vocab_size
        # 使用第一个数据集模块的工作进程数
        self.num_workers = self.dms[0].num_workers

        # 保存是否使用分布式训练的标志
        self.dist = dist

    def prepare_data(self):
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        for dm in self.dms:
            dm.setup(stage)

        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        self.tokenizer = self.dms[0].tokenizer

        self.collate = functools.partial(
            self.dms[0].train_dataset.collate, mlm_collator=self.dms[0].mlm_collator,
        )

        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        return loader

    def val_dataloader(self, batch_size=None):
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        return loader
