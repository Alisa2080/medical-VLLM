import math
import torch
import torch.nn as nn
from modules.Block import Block,EncoderBlock
from modules.PatchEmbed import PatchEmbed,PatchEmbed_PT
from typing import Optional, Tuple,Callable
from modules.VITForMIM import VisionTransformerForMaskedImageModeling
from modules.RMSNorm import RMSNorm


class VisionTransformerForMaskedImageModelingCLS(VisionTransformerForMaskedImageModeling):
    def __init__(self, 
                 dim:int=512,
                 img_size:int = 384, 
                 patch_size:int=16,
                 in_chans:int=3,
                 vocab_size:int=8192,
                 depth:int=12,
                 num_heads:int=12,
                 num_kv_heads: int = None,
                 mlp_ratio:float=4.,
                 qkv_bias:bool=False,
                 qk_scale:Optional[float]=None,
                 attn_drop_rate:float=0.,
                 drop_path_rate:float=0.,
                 norm_layer:Callable=RMSNorm,
                 norm_eps: float = 1e-6,
                 hidden_act: str = "silu",
                 init_std: float = 0.02,
                 layer_scale_init_values: Optional[float] =  1e-5,
                 rope_base: int = 10000,
                 early_layers: int = 6,
                 head_layers: int = 2, 
                 shared_lm_head: bool = True,
                 ):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, vocab_size=vocab_size, dim=dim, depth=depth,
                 num_heads=num_heads, num_kv_heads=num_kv_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, norm_eps=norm_eps,  hidden_act=hidden_act, rope_base=rope_base,
                 init_std=init_std, layer_scale_init_values=layer_scale_init_values)

        self.early_layers = early_layers
        print(f'early layer {early_layers}, late layer {depth - early_layers}, condenser head layers {head_layers}, shared_lm_head {shared_lm_head}')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(depth, early_layers + head_layers))]  # stochastic depth decay rule
        self.cls_pt_layers = nn.ModuleList([
            EncoderBlock(
                dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads,mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,norm_eps=norm_eps, hidden_act=hidden_act,
            )
            for i in range(early_layers, early_layers + head_layers)])
        self.fix_init_cls_pt_weight()

        self.shared_lm_head = shared_lm_head
        if not shared_lm_head:
            self.cls_pt_norm = norm_layer(dim,eps=norm_eps if norm_eps else 1e-6)
            self.cls_pt_lm_head = nn.Linear(dim, vocab_size, bias=False)

            self.cls_pt_norm.apply(self._init_weights)
            self.cls_pt_lm_head.apply(self._init_weights)

    def fix_init_cls_pt_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.cls_pt_layers):
            rescale(layer.attn.proj.weight.data, self.early_layers + layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, self.early_layers + layer_id + 1)

    def forward_features(self, 
                         image_tensor: torch.Tensor, 
                         bool_masked_pos: torch.Tensor,
                         **kwargs
                         ):
        x = self.patch_embed(image_tensor, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i + 1 == self.early_layers:
                early_states = x[:, 1:]

        x_cls_pt = torch.cat([x[:, [0]], early_states], dim=1)
        for blk in self.cls_pt_layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=rel_pos_bias)

        return self.norm(x), self.norm(x_cls_pt) if self.shared_lm_head else self.cls_pt_norm(x_cls_pt)

    def forward(self, x, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        x, x_cls_pt = self.forward_features(x, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        if return_patch_tokens:
            return [x, x_cls_pt]
        if return_all_tokens:
            return [self.lm_head(x), self.lm_head(x_cls_pt) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt)]
        else:
            # return the masked tokens
            return [self.lm_head(x[bool_masked_pos]), self.lm_head(x_cls_pt[bool_masked_pos]) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt[bool_masked_pos])]