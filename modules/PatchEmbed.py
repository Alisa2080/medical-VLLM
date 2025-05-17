
from timm.layers import to_2tuple
import torch
import torch.nn as nn
from modules.AttentionSeries import P2TAttention
from modules.RMSNorm import RMSNorm
import torch.nn.functional as F

class Learnable2DInterpPosEmb(nn.Module):
    def __init__(
        self, pos_shape, embed_dim, interpolation_mode="bicubic",
    ) -> None:
        super().__init__()
        self.pos_shape = to_2tuple(pos_shape)
        self.embed_dim = embed_dim
        self.interpolation_mode = interpolation_mode
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.pos_shape[0], self.pos_shape[1]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x: torch.Tensor, patch_shape: tuple) -> torch.Tensor:
        B, num_patches, C = x.shape
        H_patches, W_patches = patch_shape
        if self.pos_shape[0] != H_patches or self.pos_shape[1] != W_patches:
            # (1, embed_dim, H_init, W_init) -> (1, embed_dim, H_curr, W_curr)
            pos_embed_resized = F.interpolate(
                self.pos_embed,
                size=(H_patches, W_patches),
                mode=self.interpolation_mode,
                align_corners=False,
            )
        else:
            pos_embed_resized = self.pos_embed
        
        pos_embed_flat = pos_embed_resized.flatten(2).transpose(1, 2)
        return x + pos_embed_flat

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
            # f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed_PT(nn.Module):
    """ (Overlapped) Image to Patch Embedding
    """
    def __init__(self, img_size=384, patch_size=16, kernel_size=16, in_chans=3, embed_dim=768, overlap=False,use_abs_pos=True):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.img_size_tuple = to_2tuple(img_size)
        self.patch_size_tuple = to_2tuple(patch_size)

        assert img_size % patch_size == 0 , \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.Height, self.Width = int(img_size // patch_size), int(img_size // patch_size)
        self.num_patches = self.Height * self.Width
        self.patch_shape = (self.Height, self.Width)
        self.grid_size = self.patch_shape
        if not overlap:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                                  padding=kernel_size // 2)

        self.norm = RMSNorm(embed_dim, eps=1e-6)
        self.attention = P2TAttention(dim=embed_dim, num_heads=8, qkv_bias=True, qk_scale=None,
                                attn_drop=0., proj_drop=0., pool_ratios=[1, 2, 3, 6])
        self.use_abs_pos = use_abs_pos
        if self.use_abs_pos:
            # 假设初始训练或常见 patch 网格大小为 self.patch_shape
            self.abs_pos_embed = Learnable2DInterpPosEmb(self.patch_shape, embed_dim) 


    def forward(self, hidden_state,**kwargs):
        batch_size,channel, height, width = hidden_state.shape
        grid_size = to_2tuple(height//self.patch_size)
        device = hidden_state.device
        # (B, C’, H', W') --> (B, C, H, W)    H', W'是图像的高度和宽度, 划分完patch之后,图像每列有H个patch，每行有W个patch
        hidden_state = self.proj(hidden_state)  #（b,768,24,24）
        B, C, H, W = hidden_state.shape
        # (B, C, H, W) --> (B,C,HW) --> (B,HW,C)   HW:patch的总数;  flatten(): 指定维度后的所有维度合并到指定维度上;
        hidden_state = hidden_state.flatten(2).transpose(1, 2) # (B, H*W, C)  # (B, 576, 768)
        hidden_state = self.norm(hidden_state)
        hidden_state = self.attention(hidden_state, H, W) ## (B, H*W, C) -> (B, H*W, C)
        if self.use_abs_pos:
            hidden_state = self.abs_pos_embed(hidden_state, grid_size)
        return hidden_state



class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
