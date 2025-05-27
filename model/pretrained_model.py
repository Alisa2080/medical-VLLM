import math
import torch
import torch.nn as nn
from functools import partial
from timm.models import register_model
from timm.layers import trunc_normal_ as __call_trunc_normal_
from modules.VITForMIM import apply_layer_wise_scaling
from modules.VITForMIM_CLS import VisionTransformerForMaskedImageModelingCLS
from modules.VITForMIM import VisionTransformerForMaskedImageModeling
from modules.RMSNorm import RMSNorm

def cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

@register_model
def beit_base_patch16_224_8k_vocab_cls_pt(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_384_8k_vocab_used(pretrained=False, **kwargs): 

    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, dim=512, depth=6, num_heads=8, num_kv_heads=4, mlp_ratio=4, qkv_bias=False,
        norm_layer=RMSNorm, eps=1e-6, attn_drop_rate=0.01, drop_path_rate=0.0, hidden_act="silu", layer_scale_init_values=0.1, vocab_size=kwargs.get('vocab_size', 8192),
        rope_base=10000, init_std=0.02, **kwargs)

    model.default_cfg = cfg()
    model.default_cfg['input_size'] = (3, 384, 384)
    if pretrained and kwargs["init_ckpt"] is not None:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_384_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)
    model.default_cfg = cfg()
    model.default_cfg['input_size'] = (3, 384, 384)
    if pretrained and kwargs["init_ckpt"] is not None:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_192_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=192, patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch16_256_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=256, patch_size=16, dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)  # 可以调整decay_factor
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_24x544_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        img_size=224, patch_size=16, dim=544, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_24x544_patch16_224_8k_vocab_cls_pt(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        img_size=224, patch_size=16, dim=544, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_large_patch16_224_8k_vocab(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=16, dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_large_patch16_224_8k_vocab_cls_pt(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModelingCLS(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_huge_patch14_224_8k_vocab(pretrained=False, **kwargs):
    # patch_size=14, embed_dim=1280, depth=32, num_heads=16
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMaskedImageModeling(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), vocab_size=8192, **kwargs)
    apply_layer_wise_scaling(model, decay_factor=0.9)
    model.default_cfg = cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model