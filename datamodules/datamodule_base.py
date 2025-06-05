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
from transforms.Pathogram_Transformation import PathologyAugmentation

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

        # 统一获取图像尺寸 - 支持多种参数名以确保兼容性
        self.image_size = self._get_image_size_from_config(_config)
        
        # 从配置中获取最大文本长度
        self.max_text_len = _config.get("max_text_len", 196)
        # 从配置中获取随机抽取错误图像的数量
        self.draw_false_image = _config.get("draw_false_image", 0)
        # 从配置中获取随机抽取错误文本的数量
        self.draw_false_text = _config.get("draw_false_text", 0)
        # 从配置中获取是否仅使用图像的标志
        self.image_only = _config.get("image_only", False)
        # 从配置中获取是否仅使用文本的标志
        self.text_only = _config.get("text_only", False)

        # 从配置中获取分词器的名称或路径
        tokenizer_path = _config.get("tokenizer")
        if not tokenizer_path:
            raise ValueError("Configuration must contain 'tokenizer' field for dynamic vocab_size determination")
        
        rank_zero_info(f"BaseDataModule: Loading tokenizer from {tokenizer_path}")
        # 从预训练模型加载分词器（动态获取vocab_size）
        self.tokenizer = get_pretrained_tokenizer(tokenizer_path)
        
        # 动态获取分词器的词汇表大小
        self.vocab_size = self.tokenizer.vocab_size
        rank_zero_info(f"BaseDataModule: Dynamic vocab_size determined as {self.vocab_size}")
        
        # 验证vocab_size的合理性
        if self.vocab_size <= 0:
            raise ValueError(f"Invalid vocab_size: {self.vocab_size}. Tokenizer may not be loaded correctly.")

        # 根据配置选择数据整理器
        collator = (
            DataCollatorForWholeWordMask
            if _config.get("whole_word_masking", False)
            else DataCollatorForLanguageModeling
        )

        # 初始化用于掩码语言模型的数据整理器
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, 
            mlm=True, 
            mlm_probability=_config.get("mlm_prob", 0.15)
        )
        
        # 创建图像增强器
        self._setup_image_augmentations(_config)
        
        # 设置数据加载器的设置标志为False
        self.setup_flag = False
        
        rank_zero_info(f"BaseDataModule initialized successfully:")
        rank_zero_info(f"  - Tokenizer: {tokenizer_path}")
        rank_zero_info(f"  - Vocab size: {self.vocab_size}")
        rank_zero_info(f"  - Image size: {self.image_size}")
        rank_zero_info(f"  - Max text length: {self.max_text_len}")
        rank_zero_info(f"  - MLM probability: {_config.get('mlm_prob', 0.15)}")
        rank_zero_info(f"  - Whole word masking: {_config.get('whole_word_masking', False)}")
        rank_zero_info(f"  - Text only: {self.text_only}")
        rank_zero_info(f"  - Image only: {self.image_only}")

    def _get_image_size_from_config(self, _config):
        """
        统一获取图像尺寸，支持多种参数名以确保向后兼容性
        
        Args:
            _config: 配置字典
            
        Returns:
            int: 图像尺寸
        """
        # 优先使用顶级的 image_size
        if "image_size" in _config:
            return _config["image_size"]
        
        # 检查 image_augmentation 中的 image_size
        image_aug_config = _config.get("image_augmentation", {})
        if "image_size" in image_aug_config:
            rank_zero_info("Using image_size from image_augmentation config")
            return image_aug_config["image_size"]
        
        # 向后兼容：检查 image_augmentation 中的 input_size
        if "input_size" in image_aug_config:
            rank_zero_info("Warning: Using deprecated 'input_size' from image_augmentation, please use 'image_size' instead")
            return image_aug_config["input_size"]
        
        # 默认值
        rank_zero_info("Warning: No image_size found in config, using default 384")
        return 384

    def _setup_image_augmentations(self, _config):
        """设置图像增强器，确保 image_size 一致性"""
        # 获取图像增强配置
        image_aug_config = _config.get("image_augmentation", {}).copy()
        
        # 确保图像增强配置中包含正确的 image_size
        if "image_size" not in image_aug_config:
            image_aug_config["image_size"] = self.image_size
            rank_zero_info(f"Added image_size={self.image_size} to image_augmentation config")
        elif image_aug_config["image_size"] != self.image_size:
            rank_zero_info(f"Warning: image_augmentation.image_size ({image_aug_config['image_size']}) differs from main image_size ({self.image_size}), using main image_size")
            image_aug_config["image_size"] = self.image_size
        
        # 向后兼容：如果还有 input_size，确保与 image_size 一致
        if "input_size" in image_aug_config:
            if image_aug_config["input_size"] != self.image_size:
                rank_zero_info(f"Warning: Updating input_size from {image_aug_config['input_size']} to {self.image_size} for consistency")
            image_aug_config["input_size"] = self.image_size
        
        # 创建训练和验证的增强器
        self.train_image_augmentation = PathologyAugmentation(
            config=image_aug_config,
            is_training=True
        )
        
        self.val_image_augmentation = PathologyAugmentation(
            config=image_aug_config,
            is_training=False
        )
        
        rank_zero_info(f"Image augmentation setup:")
        rank_zero_info(f"  - Enabled: {image_aug_config.get('enable_pathology_augmentation', True)}")
        rank_zero_info(f"  - Image size: {self.image_size}")
        rank_zero_info(f"  - RandStainNA: {image_aug_config.get('randstainna_enabled', False)}")
        rank_zero_info(f"  - Multi-scale cropping: {image_aug_config.get('multi_scale_cropping', False)}")

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            image_augmentation=self.train_image_augmentation,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            image_augmentation=self.val_image_augmentation,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                data_dir=self.data_dir,
                image_augmentation=self.val_image_augmentation,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
            )

    def make_no_false_val_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            data_dir=self.data_dir,
            image_augmentation=self.val_image_augmentation,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def make_no_false_test_dset(self, image_only=False):
        return self.dataset_cls_no_false(
            data_dir=self.data_dir,
            image_augmentation=self.val_image_augmentation,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            data_dir=self.data_dir,
            image_augmentation=self.val_image_augmentation,
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
            persistent_workers=True if self.num_workers > 0 else False
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
            persistent_workers=True if self.num_workers > 0 else False
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