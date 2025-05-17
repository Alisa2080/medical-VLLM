from functools import partial, reduce
import torch
import torch.nn as nn
from timm.models import register_model
from modules.VQKD import VQKD

def get_model_default_params():
    return dict(img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12,  
                             mlp_ratio=4., qkv_bias=True,  qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., use_abs_pos_emb=True, 
                             use_rel_pos_bias=False, use_shared_rel_pos_bias=False, use_mean_pooling=True, init_scale=0.001)


def vqkd_encoder_base_decoder_1x768x12_clip(pretrained=False, pretrained_weight=None, as_tokenzer=False, img_size=224, 
                                            n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
    
    # encoder settings
    encoder_config['img_size'] = img_size
    encoder_config['num_classes'] = 0
    # decoder settings
    decoder_config['img_size'] = img_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 1
    # teacher settings
    _ = kwargs.pop("teacher_model_type", "clip")

    teacher_model_type = 'clip' if not as_tokenzer else 'None'
    decoder_out_dim = 512

    model = VQKD(encoder_config, decoder_config, n_code, code_dim, teacher_model_type=teacher_model_type, 
                 decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu')
            
        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())
        
        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]
        model.load_state_dict(weights)
    return model

@register_model
def vqkd_encoder_base_decoder_3x768x12_clip(pretrained=False, pretrained_weight=None, as_tokenzer=False, img_size=384, 
                                            n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
    
    # encoder settings
    encoder_config['img_size'] = img_size
    encoder_config['num_classes'] = 0
    # decoder settings
    decoder_config['img_size'] = img_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 3
    # teacher settings
    _ = kwargs.pop("teacher_model_type", "clip")
    
    teacher_model_type = 'clip' if not as_tokenzer else 'None'
    decoder_out_dim = 1024

    model = VQKD(encoder_config, decoder_config, n_code, code_dim, teacher_model_type=teacher_model_type, 
                 decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu',weights_only=False)
        
        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())
        
        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]
        model.load_state_dict(weights)
    return model


@register_model
def vqkd_encoder_base_decoder_1x768x12_dino(pretrained=False, pretrained_weight=None, as_tokenzer=False, img_size=224, 
                                            n_code=8192, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()
    
    # encoder settings
    encoder_config['img_size'] = img_size
    encoder_config['num_classes'] = 0
    # decoder settings
    decoder_config['img_size'] = img_size // decoder_config['patch_size']
    decoder_config['patch_size'] = 1
    decoder_config['in_chans'] = code_dim
    decoder_config['num_classes'] = 0
    decoder_config['depth'] = 1
    # teacher settings
    _ = kwargs.pop("teacher_model_type", "dino")

    teacher_model_type = 'dino' if not as_tokenzer else 'None'
    decoder_out_dim = 768

    model = VQKD(encoder_config, decoder_config, n_code, code_dim, teacher_model_type=teacher_model_type, 
                 decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None
        
        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu')
        
        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())
        
        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]
        model.load_state_dict(weights)
    return model


if __name__ == '__main__':
    pass