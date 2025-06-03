import math
import torch
import torch.nn as nn
from functools import partial
from timm.models import register_model
from timm.layers import trunc_normal_ as __call_trunc_normal_
import timm.models as models
from modules.VisionTransformer import VisionTransformer
from modules.VITForMIM import VisionTransformerForMaskedImageModeling

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

@models.register_model
def beit_base_patch16_224_finetune(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@models.register_model
def beit_base_patch16_256_finetune(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@models.register_model
def beit_base_patch16_384_finetune(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, #qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@models.register_model
def beit_24x544_patch16_224_finetune(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=544, depth=24, num_heads=16, mlp_ratio=4, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@models.register_model
def beit_large_patch16_224_fine(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, #qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@models.register_model
def beit_large_patch16_384_fine(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, #qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@models.register_model
def beit_large_patch16_512_fine(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@models.register_model
def beit_huge_patch14_224_fine(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@models.register_model
def beit_giant_patch14_224_fine(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
