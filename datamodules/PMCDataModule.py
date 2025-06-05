from datasets.multimodal_datasets.PMCDataset import PMCDataset
from .datamodule_base import BaseDataModule
from pytorch_lightning.utilities import rank_zero_info


class PMCDataModule(BaseDataModule):
    def __init__(self, _config, *args, **kwargs):
        """
        初始化PMCDataModule，确保所有必要配置正确传递
        
        Args:
            _config: 配置字典，应包含图像增强相关参数
        """
        # 验证并确保配置完整性
        _config = self._validate_and_complete_config(_config.copy())
        
        rank_zero_info(f"PMCDataModule initializing with configuration:")
        rank_zero_info(f"  - Data root: {_config.get('data_root', 'Not set')}")
        rank_zero_info(f"  - Image size: {_config.get('image_size', 'Not set')}")
        rank_zero_info(f"  - Tokenizer: {_config.get('tokenizer', 'Not set')}")
        rank_zero_info(f"  - Image augmentation enabled: {_config.get('image_augmentation', {}).get('enable_pathology_augmentation', True)}")
        rank_zero_info(f"  - RandStainNA enabled: {_config.get('image_augmentation', {}).get('randstainna_enabled', False)}")
        
        # 调用父类构造函数，传递完整配置
        super().__init__(_config, *args, **kwargs)
    
    def _validate_and_complete_config(self, _config):
        """
        验证并完善配置，确保所有必要字段存在
        
        Args:
            _config: 原始配置字典
            
        Returns:
            dict: 完善后的配置字典
        """
        # 确保顶级 image_size 存在
        if "image_size" not in _config:
            # 尝试从 image_augmentation 中获取
            image_aug_config = _config.get("image_augmentation", {})
            if "image_size" in image_aug_config:
                _config["image_size"] = image_aug_config["image_size"]
                rank_zero_info(f"PMCDataModule: Using image_size={_config['image_size']} from image_augmentation")
            elif "input_size" in image_aug_config:
                _config["image_size"] = image_aug_config["input_size"]
                rank_zero_info(f"PMCDataModule: Using image_size={_config['image_size']} from image_augmentation.input_size (deprecated)")
            else:
                _config["image_size"] = 384
                rank_zero_info(f"PMCDataModule: No image_size found, using default 384")
        
        # 确保 image_augmentation 配置存在且完整
        if "image_augmentation" not in _config:
            rank_zero_info("PMCDataModule: Creating default image_augmentation config")
            _config["image_augmentation"] = {
                "enable_pathology_augmentation": False,  # 对于文本预训练默认关闭
                "image_size": _config["image_size"],
            }
        else:
            # 确保 image_augmentation 中的 image_size 与顶级一致
            image_aug_config = _config["image_augmentation"]
            if "image_size" not in image_aug_config:
                image_aug_config["image_size"] = _config["image_size"]
                rank_zero_info(f"PMCDataModule: Added image_size={_config['image_size']} to image_augmentation")
            elif image_aug_config["image_size"] != _config["image_size"]:
                rank_zero_info(f"PMCDataModule: Updating image_augmentation.image_size from {image_aug_config['image_size']} to {_config['image_size']}")
                image_aug_config["image_size"] = _config["image_size"]
            
            # 向后兼容：如果有 input_size，确保与 image_size 一致
            if "input_size" in image_aug_config:
                if image_aug_config["input_size"] != _config["image_size"]:
                    rank_zero_info(f"PMCDataModule: Updating image_augmentation.input_size from {image_aug_config['input_size']} to {_config['image_size']}")
                image_aug_config["input_size"] = _config["image_size"]
        
        # 确保其他必要字段存在
        default_values = {
            "tokenizer": r"E:\article_code\Bert_tokenizer",  # 提供默认值，但应该在上层配置中设置
            "max_text_len": 196,
            "batch_size": 32,
            "num_workers": 4,
            "data_root": r"F:\dataset\Medical_TEXT",  # 确保有默认的data_root
            "mlm_prob": 0.15,
            "whole_word_masking": False,
            "draw_false_image": 0,
            "draw_false_text": 0,
            "image_only": False,
            "text_only": False,
        }
        
        for key, default_value in default_values.items():
            if key not in _config:
                _config[key] = default_value
                if key in ["tokenizer", "data_root"]:
                    rank_zero_info(f"PMCDataModule: Warning - using default {key}: {default_value}")
                else:
                    rank_zero_info(f"PMCDataModule: Using default {key}: {default_value}")
        
        return _config

    @property
    def dataset_cls(self):
        # 在这里添加调试信息
        rank_zero_info(f"PMCDataModule.dataset_cls called, returning: {PMCDataset}")
        rank_zero_info(f"PMCDataset type: {type(PMCDataset)}")
        return PMCDataset
    
    @property
    def dataset_name(self):
        return "PMC"