import functools
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.distributed import DistributedSampler

from . import _datamodules


class MTDataModule(LightningDataModule):
    """
    多任务数据模块，支持动态vocab_size获取
    """
    
    def __init__(self, _config, dist=False):
        """
        初始化MTDataModule类的实例。

        参数:
        _config (dict): 包含数据集配置信息的字典，必须包含tokenizer路径。
        dist (bool): 是否使用分布式训练，默认为False。
        """
        # 验证必要的配置
        if "tokenizer" not in _config:
            raise ValueError("MTDataModule requires 'tokenizer' in config for dynamic vocab_size determination")
        
        # 从配置中获取数据集的键
        datamodule_keys = _config.get("datasets", [])
        # 确保至少有一个数据集被配置
        if not datamodule_keys:
            raise ValueError("MTDataModule requires at least one dataset in 'datasets' configuration")

        # 调用父类的构造函数
        super().__init__()

        # 保存数据集的键
        self.dm_keys = datamodule_keys
        
        rank_zero_info(f"MTDataModule initializing with datasets: {datamodule_keys}")
        rank_zero_info(f"Using tokenizer: {_config['tokenizer']}")
        
        # 根据配置创建数据集模块的字典
        # 每个数据模块都会通过BaseDataModule动态获取vocab_size
        try:
            self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        except KeyError as e:
            available_keys = list(_datamodules.keys())
            raise ValueError(f"Unknown dataset key: {e}. Available datasets: {available_keys}")
        
        # 从字典中提取数据集模块的列表
        self.dms = [v for k, v in self.dm_dicts.items()]

        # 使用第一个数据集模块的配置
        first_dm = self.dms[0]
        self.batch_size = first_dm.batch_size
        self.vocab_size = first_dm.vocab_size  # 这是通过tokenizer动态获取的
        self.num_workers = first_dm.num_workers
        self.tokenizer = first_dm.tokenizer

        # 验证所有数据模块使用相同的vocab_size（确保tokenizer一致性）
        for i, dm in enumerate(self.dms):
            if dm.vocab_size != self.vocab_size:
                raise ValueError(
                    f"Vocab size mismatch between datasets. Dataset '{datamodule_keys[0]}' "
                    f"has vocab_size={self.vocab_size}, but dataset '{datamodule_keys[i]}' "
                    f"has vocab_size={dm.vocab_size}. All datasets must use the same tokenizer."
                )
            if dm.tokenizer.vocab_size != self.tokenizer.vocab_size:
                rank_zero_info(f"Warning: Tokenizer vocab_size mismatch detected between datasets")

        # 保存是否使用分布式训练的标志
        self.dist = dist
        
        rank_zero_info(f"MTDataModule initialized successfully:")
        rank_zero_info(f"  - Datasets: {datamodule_keys}")
        rank_zero_info(f"  - Dynamic vocab_size: {self.vocab_size}")
        rank_zero_info(f"  - Batch size: {self.batch_size}")
        rank_zero_info(f"  - Workers: {self.num_workers}")
        rank_zero_info(f"  - Distributed: {self.dist}")

    def prepare_data(self):
        """准备数据 - 在所有数据模块上调用"""
        for dm in self.dms:
            dm.prepare_data()

    def setup(self, stage):
        """设置数据集"""
        for dm in self.dms:
            dm.setup(stage)

        # 组合所有数据集
        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])
        
        # 确保tokenizer一致性
        self.tokenizer = self.dms[0].tokenizer

        # 设置collate函数
        self.collate = functools.partial(
            self.dms[0].train_dataset.collate, 
            mlm_collator=self.dms[0].mlm_collator,
        )

        # 设置分布式采样器
        if self.dist:
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None
            
        rank_zero_info(f"MTDataModule setup completed:")
        rank_zero_info(f"  - Combined train dataset size: {len(self.train_dataset)}")
        rank_zero_info(f"  - Combined val dataset size: {len(self.val_dataset)}")
        rank_zero_info(f"  - Combined test dataset size: {len(self.test_dataset)}")
        rank_zero_info(f"  - Vocab size consistency verified: {self.vocab_size}")

    def train_dataloader(self):
        """返回训练数据加载器"""
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            shuffle=self.train_sampler is None,  # 只有在没有sampler时才shuffle
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        return loader

    def val_dataloader(self, batch_size=None):
        """返回验证数据加载器"""
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        return loader

    def test_dataloader(self):
        """返回测试数据加载器"""
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False
        )
        return loader