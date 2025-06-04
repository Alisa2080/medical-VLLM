import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
    AutoTokenizer,
)
from pytorch_lightning.utilities import rank_zero_info

def get_pretrained_tokenizer(from_pretrained):
    """
    从预训练模型加载分词器，支持动态vocab_size获取。

    如果使用分布式训练，只有rank为0的进程会首先加载分词器，
    其他进程会等待rank为0的进程加载完成后再加载。

    参数:
    from_pretrained (str): 预训练模型的名称或路径。

    返回:
    tokenizer: 加载好的分词器实例
    """
    # 检查是否使用分布式训练
    rank_zero_info(f"torch.distributed.is_initialized(): {torch.distributed.is_initialized()}")
    rank_zero_info(f"加载文本tokenizer，tokenizer的路径为: {from_pretrained}")
    
    if torch.distributed.is_initialized():
        # 如果当前进程的rank为0
        if torch.distributed.get_rank() == 0:
            rank_zero_info("Rank 0: Loading tokenizer...")
            try:
                # 优先使用AutoTokenizer，兼容更多模型
                AutoTokenizer.from_pretrained(from_pretrained)
            except Exception as e:
                rank_zero_info(f"AutoTokenizer failed, trying BertTokenizer: {e}")
                BertTokenizer.from_pretrained(from_pretrained)
        
        # 所有进程在此处同步，确保rank为0的进程加载完成后其他进程再继续
        torch.distributed.barrier()
    
    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
    except Exception as e:
        rank_zero_info(f"AutoTokenizer failed, trying BertTokenizer: {e}")
        tokenizer = BertTokenizer.from_pretrained(from_pretrained)
    
    vocab_size = tokenizer.vocab_size
    rank_zero_info(f"Successfully loaded tokenizer with vocab_size: {vocab_size}")
    
    return tokenizer


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        """
        初始化BaseDataModule类的实例。

        参数:
        _config (dict): 包含数据模块配置信息的字典。
        """
        # 调用父类的构造函数
        super().__init__()

        # 从配置中获取数据目录
        self.data_dir = _config["data_root"]

        # 从配置中获取数据加载器的工作进程数
        self.num_workers = _config["num_workers"]
        # 从配置中获取每个GPU的批量大小
        self.batch_size = _config["batch_size"]
        # 评估时的批量大小与训练时相同
        self.eval_batch_size = self.batch_size

        # 从配置中获取图像的大小
        self.image_size = _config["image_size"]
        # 从配置中获取最大文本长度
        self.max_text_len = _config["max_text_len"]
        # 从配置中获取随机抽取错误图像的数量
        self.draw_false_image = _config["draw_false_image"]
        # 从配置中获取随机抽取错误文本的数量
        self.draw_false_text = _config["draw_false_text"]
        # 从配置中获取是否仅使用图像的标志
        self.image_only = _config["image_only"]
        # 从配置中获取是否仅使用文本的标志
        self.text_only = _config["text_only"]

        # 如果训练转换键列表为空，则使用默认的训练转换键
        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )

        # 如果验证转换键列表为空，则使用默认的验证转换键
        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        # 从配置中获取分词器的名称或路径
        tokenizer = _config["tokenizer"]
        # 从预训练模型加载分词器
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        # 获取分词器的词汇表大小
        self.vocab_size = self.tokenizer.vocab_size

        # 根据配置选择数据整理器
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        # 初始化用于掩码语言模型的数据整理器
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        # 设置数据加载器的设置标志为False
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def make_no_false_test_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
            persistent_workers=True if self.num_workers > 0 else False  # 只有当有工作进程时才启用
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
            persistent_workers=True if self.num_workers > 0 else False  # 只有当有工作进程时才启用
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
            persistent_workers=True if self.num_workers > 0 else False
        )
        return loader
