from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from sacred import Experiment

ex = Experiment("VLMo")

@dataclass
class VLMoBaseConfig:
    """
    VLMo模型的基础配置类，包含所有任务共享的通用参数
    """
    # ==================== 基础设置 ====================
    seed: int = 42
    exp_name: str = "vlmo_base"
    
    # ==================== 模型架构配置 ====================
    model_arch: Dict[str, Any] = field(default_factory=lambda: {
        "img_size": 384,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 512,
        "depth": 6,
        "num_heads": 8,
        "num_kv_heads": 4,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_eps": 1e-6,
        "layer_scale_init_values": 0.01,
        "init_std": 0.02,
        # MoE参数
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "mlp_ratio": 4.0,
        "norm_topk_prob": True,
        "moe_hidden_act": "silu",
        # RoPE参数
        "rope_base": 10000,
        # Token Type参数
        "num_token_types": 2,
        "padding_idx": 0,
    })
    
    # ==================== 图像增强配置 ====================
    image_augmentation: Dict[str, Any] = field(default_factory=lambda: {
        # 基础图像增强参数
        "enable_pathology_augmentation": True,
        "image_size": 384,  # 统一使用 image_size
        "min_crop_scale": 0.08,
        "max_crop_scale": 1.0,
        "train_interpolation": "bicubic",
        "imagenet_default_mean_and_std": True,
        
        # 病理图像特定增强
        "color_jitter_brightness": 0.05,
        "color_jitter_contrast": 0.05,
        "color_jitter_saturation": 0.02,
        "color_jitter_hue": 0.01,
        "color_jitter_probability": 0.5,
        
        # RandStainNA 配置
        "randstainna_enabled": True,
        "randstainna_yaml_file": r"E:\article_code\Vision_Encoder\RandStainNA\CRC_LAB_randomTrue_n0.yaml",
        "randstainna_std_hyper": 0.05,
        "randstainna_probability": 0.6,
        "randstainna_distribution": "normal",
        
        # 几何变换
        "random_vertical_flip_prob": 0.5,
        "random_horizontal_flip_prob": 0.0,  # 病理图像通常不需要水平翻转
        
        # 多尺度裁剪策略
        "multi_scale_cropping": True,
        "crop_scale_ranges": [
            (0.1, 0.3),    # 高倍镜视野
            (0.05, 0.15),  # 中倍镜视野
            (0.025, 0.1),  # 低倍镜视野
        ],
    })
    
    # ==================== 优化器配置 ====================
    optim_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    opt_eps: float = 1e-8
    opt_betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    
    # ==================== 学习率调度配置 ====================
    warmup_lr: float = 1e-6
    min_lr: float = 1e-5
    warmup_epochs: int = 2
    warmup_steps: int = -1
    decay_power: Union[str, float] = "cosine"
    end_lr: float = 0.0
    
    # ==================== 训练配置 ====================
    max_epochs: int = 10
    max_steps: Optional[int] = None
    grad_accum_steps: int = 1
    gradient_clip_val: float = 3.0
    
    # ==================== MoE损失配置 ====================
    moe_balance_loss_weight: float = 0.01
    moe_router_z_loss_weight: float = 0.001
    
    # ==================== Checkpoint 控制配置 ====================
    
    checkpoint_save_every_n_train_steps: Optional[int] = None  # 每 N 个训练步保存一次
    checkpoint_save_every_n_epochs: Optional[int] = None       # 每 N 个 epoch 保存一次
    save_top_k: int = 1                                        # 基于监控指标保存的最佳检查点数量，-1表示保存所有符合条件的
    
    
    checkpoint_save_last: bool = True                          # 是否保存最后一个检查点 (last.ckpt)
    checkpoint_monitor: str = "val/the_metric"                 # 监控的指标名称
    checkpoint_mode: str = "max"                               # 监控模式 ("min" 或 "max")
    checkpoint_save_weights_only: bool = False                 # 是否只保存模型权重
    checkpoint_auto_insert_metric_name: bool = True            # 是否自动在文件名中插入指标名称
    
    # 检查点文件名模板配置
    checkpoint_filename_template: str = "{epoch:02d}-{step:06d}"  # 基础文件名模板
    
    # ==================== 环境配置 ====================
    data_root: str = r"F:\dataset\Medical_TEXT"
    output_dir: str = r"E:\article_code\output\vlmo\output"  # 明确这是检查点目录
    log_dir: str = r"E:\article_code\output\vlmo\log"        # 明确这是日志目录
    resume_from: Optional[str] = None
    num_workers: int = 3
    precision: str = "16-mixed"
    
    # ==================== 训练控制 ====================
    fast_dev_run: bool = False
    val_check_interval: float = 1.0
    resume_during_training: bool = True


    
    # ==================== GPU配置 ====================
    num_gpus: int = 1
    num_nodes: int = 1
    
    # ==================== 图像尺寸（统一参数） ====================
    image_size: int = 384  # 添加顶级 image_size 参数

@dataclass 
class TextPretrainConfig(VLMoBaseConfig):
    """
    纯文本预训练任务的配置类
    """
    # ==================== 任务标识 ====================
    task_type: str = "text_pretrain"
    exp_name: str = "vlmo_text_pretrain"
    datamodule_name: str = "text_pretrain"
    
    # ==================== 数据配置 ====================
    batch_size: int = 2
    datasets: List[str] = field(default_factory=lambda: ["pmc"])
    
    # ==================== 文本特定配置 ====================
    max_text_len: int = 512
    tokenizer: str = r"E:\article_code\Bert_tokenizer"
    mlm_prob: float = 0.15
    whole_word_masking: bool = True
    
    # ==================== 文本预训练特定的 Checkpoint 配置 ====================
    checkpoint_save_every_n_epochs: Optional[int] = 1          # 文本预训练每个epoch保存一次
    checkpoint_save_every_n_train_steps: Optional[int] = None  # 不按步数保存
    save_top_k: int = 3                                        # 保存最好的3个检查点
    checkpoint_monitor: str = "val/textmlm_accuracy"           # 监控文本MLM准确率
    checkpoint_mode: str = "max"                               # 最大化准确率
    
    # 目录配置
    output_dir: str = r"E:\article_code\output\vlmo\text_pretrain\checkpoints"
    log_dir: str = r"E:\article_code\output\vlmo\text_pretrain\logs"
    
    # ==================== 权重加载配置 ====================
    weight_path: str = r"E:\article_code\output\beit2\finetuning_pl\mil_checkpoints\version_0\last.ckpt"
    
    # ==================== 学习率配置 ====================
    learning_rate: float = 2e-4
    lr_mult_head: float = 1.0
    
    # ==================== 转换选项 ====================
    unfreeze_post_attention_layernorm: bool = True
    
    # ==================== 图像尺寸（文本预训练也需要，用于兼容） ====================
    image_size: int = 384  # 文本预训练也保留 image_size 以确保兼容性
    
    # ==================== 图像增强配置（文本预训练时禁用） ====================
    image_augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enable_pathology_augmentation": False,  # 文本预训练不需要图像增强
        "image_size": 384,  # 统一使用 image_size
    })

@dataclass
class VisionLanguagePretrainConfig(VLMoBaseConfig):
    """
    视觉-语言预训练任务的配置类
    """
    # ==================== 任务标识 ====================
    task_type: str = "vision_language_pretrain"
    exp_name: str = "vlmo_vision_language_pretrain"
    datamodule_name: str = "multitask"
    
    # ==================== 数据配置 ====================
    batch_size: int = 64
    datasets: List[str] = field(default_factory=lambda: ["coco", "sbu"])
    
    # ==================== 图像配置 ====================
    image_size: int = 384
    
    # ==================== 文本配置 ====================
    max_text_len: int = 196
    tokenizer: str = "/gz-fs/Tokenizer"
    
    # ==================== 权重加载配置 ====================
    weight_path: str = ""
    
    # ==================== 损失权重配置 ====================
    itc_loss_weight: float = 1.0
    mlm_loss_weight: float = 1.0
    use_siglip_loss: bool = True
    
    # ==================== 视觉-语言预训练特定的 Checkpoint 配置 ====================
    checkpoint_save_every_n_epochs: Optional[int] = 2          # 每2个epoch保存一次
    checkpoint_save_every_n_train_steps: Optional[int] = None  # 不按步数保存
    save_top_k: int = 5                                        # 保存最好的5个检查点
    checkpoint_monitor: str = "val/the_metric"                 # 监控综合指标
    checkpoint_mode: str = "max"                               # 最大化指标
    
    # 目录配置
    output_dir: str = r"E:\article_code\output\vlmo\vision_language_pretrain\checkpoints"
    log_dir: str = r"E:\article_code\output\vlmo\vision_language_pretrain\logs"
    
    # ==================== 解码器配置 ====================
    decoder_depth: int = 2
    decoder_drop_path_rate: float = 0.1
    decoder_num_experts: int = 4
    decoder_num_experts_per_tok: int = 2
    
    # ==================== 学习率配置 ====================
    learning_rate: float = 1e-4
    
    # ==================== 病理图像增强配置 ====================
    image_augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enable_pathology_augmentation": True,
        "image_size": 384,  # 统一使用 image_size
        "min_crop_scale": 0.08,
        "max_crop_scale": 1.0,
        "train_interpolation": "bicubic",
        "imagenet_default_mean_and_std": True,
        
        # 病理图像特定增强
        "color_jitter_brightness": 0.08,
        "color_jitter_contrast": 0.08,
        "color_jitter_saturation": 0.05,
        "color_jitter_hue": 0.02,
        "color_jitter_probability": 0.7,
        
        # RandStainNA 配置
        "randstainna_enabled": True,
        "randstainna_yaml_file": r"E:\article_code\Vision_Encoder\RandStainNA\CRC_LAB_randomTrue_n0.yaml",
        "randstainna_std_hyper": 0.1,
        "randstainna_probability": 0.8,
        "randstainna_distribution": "normal",
        
        # 几何变换
        "random_vertical_flip_prob": 0.5,
        "random_horizontal_flip_prob": 0.2,
        
        # 多尺度裁剪策略
        "multi_scale_cropping": True,
        "crop_scale_ranges": [
            (0.15, 0.4),   # 高倍镜视野
            (0.08, 0.25),  # 中倍镜视野
            (0.05, 0.15),  # 低倍镜视野
        ],
    })

@dataclass
class DownstreamTaskConfig(VLMoBaseConfig):
    """
    下游任务的基础配置类
    """
    # ==================== 任务标识 ====================
    task_type: str = "downstream"
    datamodule_name: str = "multitask"
    
    # ==================== 权重加载配置 ====================
    weight_path: str = ""
    freeze_encoder: bool = False
    
    # ==================== 下游任务通用配置 ====================
    batch_size: int = 32
    learning_rate: float = 3e-5
    lr_mult: float = 1.0
    
    # ==================== 下游任务特定的 Checkpoint 配置 ====================
    checkpoint_save_every_n_epochs: Optional[int] = None       # 不按固定频率保存
    checkpoint_save_every_n_train_steps: Optional[int] = None  # 不按步数保存
    save_top_k: int = 1                                        # 只保存最好的1个检查点
    checkpoint_monitor: str = "val/the_metric"                 # 监控任务特定指标
    checkpoint_mode: str = "max"                               # 通常最大化准确率等指标
    
    # 目录配置
    output_dir: str = r"E:\article_code\output\vlmo\downstream\checkpoints"
    log_dir: str = r"E:\article_code\output\vlmo\downstream\logs"
    
    # ==================== 评估配置 ====================
    get_recall_metric: bool = False
    get_recall_rerank_metric: bool = False
    k_test: int = 32
    
    # ==================== 下游任务图像增强（更保守） ====================
    image_augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "enable_pathology_augmentation": True,
        "image_size": 384,  # 统一使用 image_size
        "min_crop_scale": 0.2,  # 更保守的裁剪
        "max_crop_scale": 1.0,
        "train_interpolation": "bicubic",
        "imagenet_default_mean_and_std": True,
        
        # 较温和的增强
        "color_jitter_brightness": 0.02,
        "color_jitter_contrast": 0.02,
        "color_jitter_saturation": 0.01,
        "color_jitter_hue": 0.005,
        "color_jitter_probability": 0.3,
        
        # RandStainNA 配置（更保守）
        "randstainna_enabled": True,
        "randstainna_yaml_file": r"E:\article_code\Vision_Encoder\RandStainNA\CRC_LAB_randomTrue_n0.yaml",
        "randstainna_std_hyper": 0.02,
        "randstainna_probability": 0.4,
        "randstainna_distribution": "normal",
        
        # 几何变换
        "random_vertical_flip_prob": 0.3,
        "random_horizontal_flip_prob": 0.0,
        
        # 禁用多尺度裁剪，使用标准裁剪
        "multi_scale_cropping": False,
    })

# ==================== Sacred配置兼容性函数 ====================
@ex.config
def config():
    """
    默认配置函数，用于与现有Sacred系统兼容
    """
    task = "base"
    base_config = VLMoBaseConfig()
    
    # 初始化基础配置字典
    seed = base_config.seed
    exp_name = base_config.exp_name
    datamodule_name = "multitask"
    
    # 模型配置
    model_arch = base_config.model_arch
    embed_dim = base_config.model_arch["embed_dim"]
    depth = base_config.model_arch["depth"]
    num_heads = base_config.model_arch["num_heads"]
    num_kv_heads = base_config.model_arch["num_kv_heads"]
    drop_path_rate = base_config.model_arch["drop_path_rate"]
    
    # 图像增强配置
    image_augmentation = base_config.image_augmentation
    
    # 训练配置
    batch_size = 32
    learning_rate = base_config.learning_rate
    weight_decay = base_config.weight_decay
    max_epochs = base_config.max_epochs
    max_steps = base_config.max_steps
    warmup_steps = base_config.warmup_steps
    
    # 环境配置
    data_root = base_config.data_root
    output_dir = base_config.output_dir 
    log_dir = base_config.log_dir
    num_workers = base_config.num_workers
    precision = base_config.precision
    num_gpus = base_config.num_gpus
    num_nodes = base_config.num_nodes
    
    # MoE配置
    moe_balance_loss_weight = base_config.moe_balance_loss_weight
    moe_router_z_loss_weight = base_config.moe_router_z_loss_weight
    
    # 向后兼容的配置
    gradient_clip_val = base_config.gradient_clip_val
    fast_dev_run = base_config.fast_dev_run
    val_check_interval = base_config.val_check_interval
    resume_during_training = base_config.resume_during_training
    load_path = ""
    test_only = False
    
    # Checkpoint 控制配置
    checkpoint_save_every_n_train_steps = base_config.checkpoint_save_every_n_train_steps
    checkpoint_save_every_n_epochs = base_config.checkpoint_save_every_n_epochs
    save_top_k = base_config.save_top_k
    checkpoint_save_last = base_config.checkpoint_save_last
    checkpoint_monitor = base_config.checkpoint_monitor
    checkpoint_mode = base_config.checkpoint_mode
    checkpoint_save_weights_only = base_config.checkpoint_save_weights_only
    checkpoint_auto_insert_metric_name = base_config.checkpoint_auto_insert_metric_name
    checkpoint_filename_template = base_config.checkpoint_filename_template
    
    # 添加默认的 image_size
    image_size = 384
    
    # 根据任务类型设置特定配置
    if task == "text_pretrain":
        text_config = TextPretrainConfig()
        task_type = text_config.task_type
        exp_name = text_config.exp_name
        datamodule_name = text_config.datamodule_name
        batch_size = text_config.batch_size
        datasets = text_config.datasets
        max_text_len = text_config.max_text_len
        tokenizer = text_config.tokenizer
        mlm_prob = text_config.mlm_prob
        whole_word_masking = text_config.whole_word_masking
        learning_rate = text_config.learning_rate
        weight_path = text_config.weight_path
        unfreeze_post_attention_layernorm = text_config.unfreeze_post_attention_layernorm
        image_augmentation = text_config.image_augmentation
        image_size = text_config.image_size  
        output_dir = text_config.output_dir
        log_dir = text_config.log_dir
        checkpoint_save_every_n_train_steps = text_config.checkpoint_save_every_n_train_steps
        checkpoint_save_every_n_epochs = text_config.checkpoint_save_every_n_epochs
        save_top_k = text_config.save_top_k
        checkpoint_monitor = text_config.checkpoint_monitor
        checkpoint_mode = text_config.checkpoint_mode
        
    elif task == "vision_language_pretrain":
        vl_config = VisionLanguagePretrainConfig()
        task_type = vl_config.task_type
        exp_name = vl_config.exp_name
        datamodule_name = vl_config.datamodule_name
        batch_size = vl_config.batch_size
        datasets = vl_config.datasets
        image_size = vl_config.image_size
        max_text_len = vl_config.max_text_len
        tokenizer = vl_config.tokenizer
        itc_loss_weight = vl_config.itc_loss_weight
        mlm_loss_weight = vl_config.mlm_loss_weight
        use_siglip_loss = vl_config.use_siglip_loss
        learning_rate = vl_config.learning_rate
        weight_path = vl_config.weight_path
        decoder_depth = vl_config.decoder_depth
        decoder_drop_path_rate = vl_config.decoder_drop_path_rate
        decoder_num_experts = vl_config.decoder_num_experts
        decoder_num_experts_per_tok = vl_config.decoder_num_experts_per_tok
        image_augmentation = vl_config.image_augmentation
        output_dir = vl_config.output_dir
        log_dir = vl_config.log_dir
        checkpoint_save_every_n_train_steps = vl_config.checkpoint_save_every_n_train_steps
        checkpoint_save_every_n_epochs = vl_config.checkpoint_save_every_n_epochs
        save_top_k = vl_config.save_top_k
        checkpoint_monitor = vl_config.checkpoint_monitor
        checkpoint_mode = vl_config.checkpoint_mode

    elif task == "downstream":
        downstream_config = DownstreamTaskConfig()
        task_type = downstream_config.task_type
        datamodule_name = downstream_config.datamodule_name
        batch_size = downstream_config.batch_size
        learning_rate = downstream_config.learning_rate
        weight_path = downstream_config.weight_path
        freeze_encoder = downstream_config.freeze_encoder
        get_recall_metric = downstream_config.get_recall_metric
        get_recall_rerank_metric = downstream_config.get_recall_rerank_metric
        k_test = downstream_config.k_test
        image_augmentation = downstream_config.image_augmentation
        output_dir = downstream_config.output_dir
        log_dir = downstream_config.log_dir
        checkpoint_save_every_n_train_steps = downstream_config.checkpoint_save_every_n_train_steps
        checkpoint_save_every_n_epochs = downstream_config.checkpoint_save_every_n_epochs
        save_top_k = downstream_config.save_top_k
        checkpoint_monitor = downstream_config.checkpoint_monitor
        checkpoint_mode = downstream_config.checkpoint_mode
    
    else:
        # 默认配置
        task_type = "downstream"
        tokenizer = "/gz-fs/Tokenizer"  # 默认tokenizer
        max_text_len = 196
        image_size = 384
        datasets = ["coco"]
        output_dir = text_config.output_dir
        log_dir = text_config.log_dir
        checkpoint_save_every_n_train_steps = text_config.checkpoint_save_every_n_train_steps
        checkpoint_save_every_n_epochs = text_config.checkpoint_save_every_n_epochs
        save_top_k = text_config.save_top_k
        checkpoint_monitor = text_config.checkpoint_monitor
        checkpoint_mode = text_config.checkpoint_mode

# ==================== Named配置 ====================
@ex.named_config
def text_pretrain_base():
    """纯文本预训练基础配置"""
    task = "text_pretrain"
    
@ex.named_config  
def vision_language_pretrain_base():
    """视觉-语言预训练基础配置"""
    task = "vision_language_pretrain"

@ex.named_config
def text_pretrain_large():
    """纯文本预训练大模型配置"""
    task = "text_pretrain"
    datamodule_name = "text_pretrain_large"
    batch_size = 64
    max_text_len = 256
    datasets = ["wikibk", "pmc", "pubmed"]
    
    model_arch = {
        "img_size": 384,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "num_kv_heads": 8,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_eps": 1e-6,
        "layer_scale_init_values": 0.01,
        "init_std": 0.02,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "mlp_ratio": 4.0,
        "norm_topk_prob": True,
        "moe_hidden_act": "silu",
        "rope_base": 10000,
        "num_token_types": 2,
        "padding_idx": 0,
    }

@ex.named_config
def vision_language_pretrain_large():
    """视觉-语言预训练大模型配置"""
    task = "vision_language_pretrain"
    datamodule_name = "multitask_large"
    batch_size = 32
    image_size = 480
    max_text_len = 256
    datasets = ["coco", "sbu", "vg", "gcc"]
    decoder_depth = 4
    
    model_arch = {
        "img_size": 480,
        "patch_size": 16,
        "in_chans": 3,
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "num_kv_heads": 8,
        "qkv_bias": True,
        "qk_scale": None,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,
        "norm_eps": 1e-6,
        "layer_scale_init_values": 0.01,
        "init_std": 0.02,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "mlp_ratio": 4.0,
        "norm_topk_prob": True,
        "moe_hidden_act": "silu",
        "rope_base": 10000,
        "num_token_types": 2,
        "padding_idx": 0,
    }
    
    # 大模型的增强配置
    image_augmentation = {
        "enable_pathology_augmentation": True,
        "image_size": 480,  # 统一使用 image_size
        "min_crop_scale": 0.08,
        "max_crop_scale": 1.0,
        "train_interpolation": "bicubic",
        "imagenet_default_mean_and_std": True,
        "color_jitter_brightness": 0.08,
        "color_jitter_contrast": 0.08,
        "color_jitter_saturation": 0.05,
        "color_jitter_hue": 0.02,
        "color_jitter_probability": 0.8,
        "randstainna_enabled": True,
        "randstainna_yaml_file": r"E:\article_code\Vision_Encoder\RandStainNA\CRC_LAB_randomTrue_n0.yaml",
        "randstainna_std_hyper": 0.1,
        "randstainna_probability": 0.9,
        "randstainna_distribution": "normal",
        "random_vertical_flip_prob": 0.5,
        "random_horizontal_flip_prob": 0.2,
        "multi_scale_cropping": True,
        "crop_scale_ranges": [
            (0.15, 0.4),   # 高倍镜视野
            (0.08, 0.25),  # 中倍镜视野
            (0.05, 0.15),  # 低倍镜视野
        ],
    }