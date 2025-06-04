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
        "num_kv_heads": 8,  # 如果为None，则等于num_heads
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
        # 注意：移除了vocab_size，将动态确定
    })
    
    model_name: str = "vlmo_base_patch16"
    
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
    
    # ==================== 环境配置 ====================
    data_root: str = ""
    output_dir: str = ""
    log_dir: str = "/gz-fs/log/vlmo"
    resume_from: Optional[str] = None
    num_workers: int = 10
    precision: str = "16-mixed"
    
    # ==================== 训练控制 ====================
    fast_dev_run: bool = False
    val_check_interval: float = 1.0
    resume_during_training: bool = True
    
    # ==================== GPU配置 ====================
    num_gpus: int = 1
    num_nodes: int = 1

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
    datasets: List[str] = field(default_factory=lambda: ["wikibk", "pmc"])
    
    # ==================== 文本特定配置 ====================
    max_text_len: int = 512
    tokenizer: str = "/gz-fs/Tokenizer"  # 保留tokenizer路径
    # 移除 vocab_size: int，将动态确定
    mlm_prob: float = 0.30
    whole_word_masking: bool = True
    
    # ==================== 权重加载配置 ====================
    weight_path: str = ""
    
    # ==================== 学习率配置 ====================
    learning_rate: float = 2e-4
    lr_mult_head: float = 1.0
    
    # ==================== 转换选项 ====================
    convert_beit2_to_textpt: bool = True
    unfreeze_post_attention_layernorm: bool = True

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
    tokenizer: str = "/gz-fs/Tokenizer"  # 保留tokenizer路径
    # 移除 vocab_size: int，将动态确定
    
    # ==================== 权重加载配置 ====================
    weight_path: str = ""
    
    # ==================== 损失权重配置 ====================
    itc_loss_weight: float = 1.0
    mlm_loss_weight: float = 1.0
    use_siglip_loss: bool = True
    
    # ==================== 解码器配置 ====================
    decoder_depth: int = 2
    decoder_drop_path_rate: float = 0.1
    decoder_num_experts: int = 4
    decoder_num_experts_per_tok: int = 2
    
    # ==================== 学习率配置 ====================
    learning_rate: float = 1e-4

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
    
    # ==================== 评估配置 ====================
    get_recall_metric: bool = False
    get_recall_rerank_metric: bool = False
    k_test: int = 32

# ==================== Sacred配置兼容性函数 ====================
@ex.config
def config():
    """
    默认配置函数，用于与现有Sacred系统兼容
    """
    task = "base"
    base_config = VLMoBaseConfig()
    
    config_dict = {
        # 基础设置
        "seed": base_config.seed,
        "exp_name": base_config.exp_name,
        "datamodule_name": "multitask",
        
        # 模型配置（移除vocab_size）
        "model_arch": base_config.model_arch,
        "embed_dim": base_config.model_arch["embed_dim"],
        "depth": base_config.model_arch["depth"],
        "num_heads": base_config.model_arch["num_heads"],
        "num_kv_heads": base_config.model_arch["num_kv_heads"],
        "drop_path_rate": base_config.model_arch["drop_path_rate"],
        
        # 训练配置
        "batch_size": 32,
        "learning_rate": base_config.learning_rate,
        "weight_decay": base_config.weight_decay,
        "max_epochs": base_config.max_epochs,
        "max_steps": base_config.max_steps,
        "warmup_steps": base_config.warmup_steps,
        
        # 环境配置
        "data_root": base_config.data_root,
        "log_dir": base_config.log_dir,
        "num_workers": base_config.num_workers,
        "precision": base_config.precision,
        "num_gpus": base_config.num_gpus,
        "num_nodes": base_config.num_nodes,
        
        # MoE配置
        "moe_balance_loss_weight": base_config.moe_balance_loss_weight,
        "moe_router_z_loss_weight": base_config.moe_router_z_loss_weight,
        
        # 向后兼容的配置
        "gradient_clip_val": base_config.gradient_clip_val,
        "fast_dev_run": base_config.fast_dev_run,
        "val_check_interval": base_config.val_check_interval,
        "resume_during_training": base_config.resume_during_training,
        "load_path": "",
        "test_only": False,
    }
    
    # 根据任务类型更新配置
    if task == "text_pretrain":
        text_config = TextPretrainConfig()
        config_dict.update({
            "task_type": text_config.task_type,
            "exp_name": text_config.exp_name,
            "datamodule_name": text_config.datamodule_name,
            "batch_size": text_config.batch_size,
            "datasets": text_config.datasets,
            "max_text_len": text_config.max_text_len,
            "tokenizer": text_config.tokenizer,
            # 移除vocab_size
            "mlm_prob": text_config.mlm_prob,
            "whole_word_masking": text_config.whole_word_masking,
            "learning_rate": text_config.learning_rate,
            "weight_path": text_config.weight_path,
            "convert_beit2_to_textpt": text_config.convert_beit2_to_textpt,
            "unfreeze_post_attention_layernorm": text_config.unfreeze_post_attention_layernorm,
        })
    elif task == "vision_language_pretrain":
        vl_config = VisionLanguagePretrainConfig()
        config_dict.update({
            "task_type": vl_config.task_type,
            "exp_name": vl_config.exp_name,
            "datamodule_name": vl_config.datamodule_name,
            "batch_size": vl_config.batch_size,
            "datasets": vl_config.datasets,
            "image_size": vl_config.image_size,
            "max_text_len": vl_config.max_text_len,
            "tokenizer": vl_config.tokenizer,
            # 移除vocab_size
            "itc_loss_weight": vl_config.itc_loss_weight,
            "mlm_loss_weight": vl_config.mlm_loss_weight,
            "use_siglip_loss": vl_config.use_siglip_loss,
            "learning_rate": vl_config.learning_rate,
            "weight_path": vl_config.weight_path,
            "decoder_depth": vl_config.decoder_depth,
            "decoder_drop_path_rate": vl_config.decoder_drop_path_rate,
            "decoder_num_experts": vl_config.decoder_num_experts,
            "decoder_num_experts_per_tok": vl_config.decoder_num_experts_per_tok,
        })
    elif task == "downstream":
        downstream_config = DownstreamTaskConfig()
        config_dict.update({
            "task_type": downstream_config.task_type,
            "datamodule_name": downstream_config.datamodule_name,
            "batch_size": downstream_config.batch_size,
            "learning_rate": downstream_config.learning_rate,
            "lr_mult": downstream_config.lr_mult,
            "weight_path": downstream_config.weight_path,
            "freeze_encoder": downstream_config.freeze_encoder,
            "get_recall_metric": downstream_config.get_recall_metric,
            "get_recall_rerank_metric": downstream_config.get_recall_rerank_metric,
            "k_test": downstream_config.k_test,
        })
    
    return config_dict

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
        "embed_dim": 1024,
        "depth": 24,
        "num_heads": 16,
        "num_experts": 16,
        # 移除vocab_size
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
        "embed_dim": 1024,
        "depth": 24, 
        "num_heads": 16,
        "num_experts": 16,
        # 移除vocab_size
    }