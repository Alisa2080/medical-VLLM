from transformers import PretrainedConfig
from typing import Optional, Dict, Any, Union
import functools
import warnings

# 导入 RMSNorm 以便在 from_encoder_decoder_configs 中使用
try:
    from .RMSNorm import RMSNorm
except ImportError:
    # 提供一个备用方案
    RMSNorm = None
    warnings.warn("RMSNorm could not be imported. Config creation from dicts might fail if norm_layer is needed.")


class VLMoEncoderDecoderConfig(PretrainedConfig):
    """
    VLMoEncoderDecoder 模型的配置类。

    继承自 `transformers.PretrainedConfig`。定义了实例化 MultiWayTransformer Encoder
    和 TransformerDecoder 所需的参数。
    """
    model_type = "vlmo_encoder_decoder"
    is_composition = True

    def __init__(
        self,
        encoder=None,
        decoder=None,
        max_seq_len=256,
        rope_base=10000,
        moe_balance_loss_weight=0.01,
        moe_router_z_loss_weight=0.001,
        encoder_checkpoint_path=None,
        image_size=384,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if encoder is None:
            # Default encoder config if needed
            encoder = {}
        if decoder is None:
            # Default decoder config if needed
            decoder = {}

        # Use PretrainedConfig's nested handling
        self.use_cache = use_cache
        self.encoder = PretrainedConfig(**encoder)
        self.decoder = PretrainedConfig(**decoder)

        self.max_seq_len = max_seq_len
        self.rope_base = rope_base
        self.moe_balance_loss_weight = moe_balance_loss_weight
        self.moe_router_z_loss_weight = moe_router_z_loss_weight
        self.encoder_checkpoint_path = encoder_checkpoint_path
        # --- 确保 image_size 和 patch_size 可用 ---
        # 优先从 encoder config 获取，否则使用顶层参数
        self.image_size = getattr(self.encoder, "img_size", image_size)
        if not hasattr(self.encoder, "patch_size"):
             # Default patch size if not specified in encoder config
             self.encoder.patch_size = 16

        # Ensure vocab_size consistency if needed later
        if hasattr(self.decoder, "vocab_size") and hasattr(self.encoder, "vocab_size"):
             assert self.decoder.vocab_size == self.encoder.vocab_size, "Encoder and Decoder vocab sizes must match if both specified"

    @classmethod
    def from_encoder_decoder_configs(cls, encoder_config: Dict[str, Any], decoder_config: Dict[str, Any], **kwargs) -> "VLMoEncoderDecoderConfig":
        """
        从独立的 encoder 和 decoder 配置字典创建 VLMoEncoderDecoderConfig 的替代构造函数。
        """
        # 提取通用参数 (如果 kwargs 中提供，否则使用默认值)
        max_seq_len = kwargs.pop("max_seq_len", 512)
        rope_base = kwargs.pop("rope_base", 10000)

        # 提取编码器参数
        encoder_nested_config = encoder_config.get("config", {}) # 处理嵌套配置
        # 尝试从 norm_layer partial 获取 eps
        encoder_norm_eps = 1e-6 # Default
        if isinstance(encoder_config.get("norm_layer"), functools.partial):
            encoder_norm_eps = encoder_config["norm_layer"].keywords.get("eps", 1e-6)
        elif isinstance(encoder_config.get("norm_layer"), dict) and "eps" in encoder_config["norm_layer"]:
             encoder_norm_eps = encoder_config["norm_layer"]["eps"]

        encoder_params = {
            "encoder_model_name": encoder_config.get("model_name", "vlmo_base_patch16"),
            "encoder_img_size": encoder_config.get("img_size", 384),
            "encoder_patch_size": encoder_config.get("patch_size", 16),
            "encoder_in_chans": encoder_config.get("in_chans", 3),
            "encoder_embed_dim": encoder_config.get("embed_dim", 768),
            "encoder_depth": encoder_config.get("depth", 12),
            "encoder_num_heads": encoder_config.get("num_heads", 12),
            "encoder_mlp_ratio": encoder_config.get("mlp_ratio", 4.0),
            "encoder_qkv_bias": encoder_config.get("qkv_bias", True),
            "encoder_drop_rate": encoder_config.get("drop_rate", 0.0),
            "encoder_attn_drop_rate": encoder_config.get("attn_drop_rate", 0.0),
            "encoder_drop_path_rate": encoder_nested_config.get("drop_path_rate", 0.1), # 从嵌套配置获取
            "encoder_norm_layer_eps": encoder_norm_eps,
            "encoder_use_abs_pos_emb": encoder_config.get("use_abs_pos_emb", False),
            "encoder_layer_scale_init_values": encoder_config.get("layer_scale_init_values", 0.1),
            "encoder_max_text_len": encoder_nested_config.get("max_text_len", 196), 
        }

        # 提取解码器参数
        decoder_norm_eps = 1e-6 # Default
        if isinstance(decoder_config.get("norm_layer"), functools.partial):
            decoder_norm_eps = decoder_config["norm_layer"].keywords.get("eps", 1e-6)
        elif isinstance(decoder_config.get("norm_layer"), dict) and "eps" in decoder_config["norm_layer"]:
             decoder_norm_eps = decoder_config["norm_layer"]["eps"]

        decoder_params = {
            "decoder_vocab_size": decoder_config.get("vocab_size", 30522),
            "decoder_depth": decoder_config.get("depth", 6),
            "decoder_dim": decoder_config.get("dim", 768),
            "decoder_num_heads": decoder_config.get("num_heads", 12),
            "decoder_mlp_ratio": decoder_config.get("mlp_ratio", 4.0),
            "decoder_num_kv_heads": decoder_config.get("num_kv_heads", None),
            "decoder_norm_layer_eps": decoder_norm_eps,
            "decoder_qkv_bias": decoder_config.get("qkv_bias", True),
            "decoder_drop_rate": decoder_config.get("drop_rate", 0.0),
            "decoder_attn_drop_rate": decoder_config.get("attn_drop_rate", 0.0),
            "decoder_drop_path_rate": decoder_config.get("drop_path_rate", 0.0),
            "decoder_layer_scale_init_values": decoder_config.get("layer_scale_init_values", 0.1),
            "decoder_num_experts": decoder_config.get("num_experts", 2),
            "decoder_add_output_proj": decoder_config.get("add_output_proj", True),
            "decoder_padding_idx": decoder_config.get("padding_idx", 0),
            "moe_balance_loss_weight": decoder_config.get("moe_balance_loss_weight", 0.01), # 从 decoder config 获取 MoE 权重
            "moe_router_z_loss_weight": decoder_config.get("moe_router_z_loss_weight", 0.001),
        }

        # 合并所有参数
        combined_params = {
            "max_seq_len": max_seq_len,
            "rope_base": rope_base,
            **encoder_params,
            **decoder_params,
            **kwargs, # 包含任何其他覆盖
        }

        return cls(**combined_params)
