from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from pytorch_lightning.utilities import rank_zero_info
import pytorch_lightning as pl

from . import _datamodules


class TextPretrainDataModule(pl.LightningDataModule):
    """
    专门用于纯文本预训练的轻量级DataModule
    相比MTDataModule，这个类专注于文本数据，避免了不必要的图像处理开销
    支持动态vocab_size获取
    """
    
    def __init__(self, _config, dist=False):
        """
        初始化TextPretrainDataModule
        
        Args:
            _config: 配置字典，必须包含tokenizer路径
            dist: 是否使用分布式训练
        """
        super().__init__()
        
        # 验证配置
        if "tokenizer" not in _config:
            raise ValueError("TextPretrainDataModule requires 'tokenizer' in config for dynamic vocab_size")
        
        # 获取数据集键
        datamodule_keys = _config.get("datasets", ["wikibk", "pmc"])
        if not datamodule_keys:
            raise ValueError("TextPretrainDataModule requires at least one dataset")
        
        self.dm_keys = datamodule_keys
        self.dist = dist
        
        rank_zero_info(f"TextPretrainDataModule initializing with datasets: {datamodule_keys}")
        
        # 创建数据模块字典 - 这些将会使用BaseDataModule的动态vocab_size逻辑
        try:
            self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        except KeyError as e:
            available_keys = list(_datamodules.keys())
            raise ValueError(f"Unknown dataset key: {e}. Available datasets: {available_keys}")
        
        # 提取数据模块列表
        self.dms = [v for k, v in self.dm_dicts.items()]
        
        # 使用第一个数据模块的配置（所有数据模块应该具有相同的vocab_size）
        first_dm = self.dms[0]
        self.batch_size = first_dm.batch_size
        self.vocab_size = first_dm.vocab_size  # 这是动态获取的
        self.num_workers = first_dm.num_workers
        self.tokenizer = first_dm.tokenizer
        
        # 验证所有数据模块的vocab_size一致性
        for i, dm in enumerate(self.dms[1:], 1):
            if dm.vocab_size != self.vocab_size:
                raise ValueError(
                    f"Vocab size mismatch: dataset {datamodule_keys[0]} has vocab_size={self.vocab_size}, "
                    f"but dataset {datamodule_keys[i]} has vocab_size={dm.vocab_size}. "
                    f"All datasets must use the same tokenizer."
                )
        
        rank_zero_info(f"TextPretrainDataModule initialized successfully:")
        rank_zero_info(f"  - Datasets: {datamodule_keys}")
        rank_zero_info(f"  - Dynamic vocab_size: {self.vocab_size}")
        rank_zero_info(f"  - Batch size: {self.batch_size}")
        rank_zero_info(f"  - Distributed: {self.dist}")

    def prepare_data(self):
        """准备数据，在所有数据模块上调用"""
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
        
        # 使用第一个数据模块的collate函数
        import functools
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
            
        rank_zero_info(f"TextPretrainDataModule setup completed:")
        rank_zero_info(f"  - Train dataset size: {len(self.train_dataset)}")
        rank_zero_info(f"  - Val dataset size: {len(self.val_dataset)}")
        rank_zero_info(f"  - Test dataset size: {len(self.test_dataset)}")

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


class TextPretrainLargeDataModule(TextPretrainDataModule):
    """
    大规模文本预训练数据模块
    支持更多数据集和更大的批次大小
    """
    
    def __init__(self, _config, dist=False):
        # 为大规模训练设置默认数据集
        default_large_datasets = ["wikibk", "pmc", "pubmed", "arxiv"]
        if "datasets" not in _config:
            _config["datasets"] = default_large_datasets
            rank_zero_info(f"TextPretrainLargeDataModule: Using default large datasets: {default_large_datasets}")
        
        # 调用父类初始化
        super().__init__(_config, dist)
        
        rank_zero_info(f"TextPretrainLargeDataModule initialized for large-scale training")
        rank_zero_info(f"  - Using {len(self.dm_keys)} datasets: {self.dm_keys}")
        rank_zero_info(f"  - Dynamic vocab_size: {self.vocab_size}")