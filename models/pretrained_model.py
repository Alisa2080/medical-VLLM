import math
import torch
import torch.nn as nn
from functools import partial
from timm.models import register_model
from timm.layers import trunc_normal_ as __call_trunc_normal_
from modules.VITForMIM import apply_layer_wise_scaling
from modules.VITForMIM import VisionTransformerForMaskedImageModeling
from modules.Encoder import TransformerEncoder
from modules.RMSNorm import RMSNorm


@register_model
def VisionEncoder_base_patch16_384_8k(**kwargs): 
    model = VisionTransformerForMaskedImageModeling(
        img_size=384,patch_size=16, in_chans=3,dim=512, depth=6, num_heads=8, num_kv_heads=4, mlp_ratio=4, qkv_bias=False,
        norm_layer=RMSNorm, eps=1e-6, attn_drop_rate=0.01, drop_path_rate=0.0, hidden_act="silu", layer_scale_init_values=0.1, vocab_size=8192,
        rope_base=10000, init_std=0.02,embed_smooth_alpha = 0.9, **kwargs)
    return model

@register_model
def TransformerEncoder_base_patch16_384(**kwargs): 
    model = TransformerEncoder(
        patch_size=16, 
        dim=512, 
        depth=6, 
        num_heads=8, 
        num_kv_heads=4, 
        qkv_bias=False,
        norm_layer=RMSNorm, 
        eps=1e-6, 
        attn_drop_rate=0.01,
        drop_path_rate=0.0, 
        layer_scale_init_values=0.1, 
        num_experts= 4,
        num_experts_per_tok = 2,
        mlp_ratio=4.0, 
        norm_topk_prob=False,
        moe_hidden_act="silu",
        max_seq_len = 512,
        vocab_size=kwargs.get('vocab_size', 8192),
        rope_base=10000, 
        init_std=0.02, 
        num_token_types=2,
        **kwargs)
    return model

