# -*- coding: utf-8 -*-
from functools import partial
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from typing import Type
from timm.layers import DropPath, to_2tuple, trunc_normal_
from timm.models import register_model
from pytorch_lightning.utilities import rank_zero_info
from modules.PatchEmbed import PatchEmbed_PT
from modules.RMSNorm import RMSNorm 
from modules.Block import EncoderBlock
from modules.rope import SimpleQwen3RotaryEmbedding,Rope2DPosEmb
from transformers.modeling_outputs import MoEModelOutput

class TransformerEncoder(nn.Module):

    def __init__(
        self,
        img_size: int = 384,
        patch_size: int = 16,
        in_chans: int = 3,
        num_kv_heads: int = None,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        qkv_bias: bool = True,
        qk_scale: float = None,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = RMSNorm,
        norm_eps: float = 1e-6,
        layer_scale_init_values: Optional[float] =  0.01,
        init_std: float = 0.02,
        # ---  MoE paraments ---,
        # --- MoE Params for Blocks ---
        num_experts: int = 2,
        num_experts_per_tok: int = 2,
        mlp_ratio: float = 4.0,
        norm_topk_prob: bool = True,
        moe_hidden_act: str = "silu",
         # --- RoPE Params ---
        max_seq_len: int = 512,
        rope_base=10000,
        # --- Token Type ---
        num_token_types=2, # Usually 2 (text, image)
        vocab_size: int = 30522,
        padding_idx: int = 0,
        **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.drop_path_rate = drop_path_rate 
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.head_dim = embed_dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.num_token_types = num_token_types # --- Token Type ---
        self.init_std = init_std
        self.norm_layer = norm_layer
        self.moe_hidden_act = moe_hidden_act
        self.attn_drop_rate = attn_drop_rate
        self.patch_embed = PatchEmbed_PT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.token_type_embeddings = nn.Embedding(num_token_types, embed_dim)
        self.rope1d = SimpleQwen3RotaryEmbedding(
            dim=self.head_dim, max_position_embeddings=max_seq_len, base=rope_base
        )
        self.rope2d = Rope2DPosEmb( # Need to ensure this can provide cos/sin or adapt logic
             dim=self.head_dim, max_height=self.grid_size[0]+ 1, max_width=self.grid_size[1]+ 1,theta_base=rope_base
        )

        # --- Blocks ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    qkv_bias=qkv_bias,
                    qk_scale=self.scale,
                    attn_drop=self.attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=self.norm_layer,
                    norm_eps=norm_eps,
                    layer_scale_init_values=layer_scale_init_values,
                    num_experts=num_experts,
                    mlp_ratio=mlp_ratio,
                    num_experts_per_tok=num_experts_per_tok,
                    norm_topk_prob=norm_topk_prob,
                    moe_hidden_act=moe_hidden_act,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim, eps=norm_eps)
        self.apply(self.init_weights)


    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            if m.weight is not None:
                m.weight.data.normal_(mean=0.0, std=self.init_std)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def visual_embed(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.patch_embed(image_tensor)
        input_size = hidden_states.shape[:1]

        cls_tokens = self.cls_token.expand(input_size[0], -1, -1)
        image_hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        num_img_tokens = image_hidden_states.shape[1]
        image_mask = torch.ones(input_size[0], num_img_tokens, dtype=torch.bool, device=image_hidden_states.device)
        image_pos_ids = torch.arange(num_img_tokens, dtype=torch.long, device=image_hidden_states.device).unsqueeze(0).expand(input_size[0], -1)
        return image_hidden_states, image_mask, image_pos_ids
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None, # Shape (B, N_text, C)
        text_mask: Optional[torch.Tensor] = None,    # Bool mask (B, N_text), True=VALID
        image_tensor: Optional[torch.Tensor] = None, # Raw image (B, C_in, H_img, W_img)
        attention_mask: Optional[torch.Tensor] = None, # Optional precomputed combined mask
        output_attentions: bool = False,
        output_hidden_states: bool = False, # Not implemented in detail here
        output_router_logits: bool = False, # Control returning logits
        **kwargs
    ):
        output_attentions = output_attentions if output_attentions is not None else False
        output_router_logits = output_router_logits if output_router_logits is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        co_embeds_list = []
        co_masks_list = []

        N_text_actual = 0
        cos_text, sin_text = None, None
        if input_ids is not None: 
            text_word_embeds = self.word_embeddings(input_ids)
            N_text_actual = input_ids.shape[1]
            text_pos_ids = torch.arange(N_text_actual, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        
            text_type_ids = torch.zeros_like(input_ids, dtype=torch.long)    
            text_token_type_embeds = self.token_type_embeddings(text_type_ids)
            text_embeds = text_word_embeds + text_token_type_embeds    
            
            cos_text, sin_text = self.rope1d(text_embeds, text_pos_ids)    
            
            co_embeds_list.append(text_embeds)

            if text_mask is None:
                text_mask = torch.ones_like(input_ids, dtype=torch.bool)
            co_masks_list.append(text_mask)
  
            N_text_actual = text_embeds.shape[1]
        
        
        image_embeds, image_mask, image_pos_ids = None, None, None
        if image_tensor is not None:
            image_embeds, image_mask, image_pos_ids = self.visual_embed(image_tensor)
            if self.token_type_embeddings is not None:
                image_type_idx = 1 if self.num_token_types > 1 and text_embeds is not None else 0
            
                image_embeds = image_embeds + self.token_type_embeddings(
                 torch.full_like(image_mask, image_type_idx, dtype=torch.long) # Type 1 for image
            )
            co_embeds_list.append(image_embeds)
            co_masks_list.append(image_mask)

        if not co_embeds_list:
            raise ValueError("Must provide at least text_embeds or image_embeds")

        co_embeds = torch.cat(co_embeds_list, dim=1) # (B, N, C) where N = N_text + N_img

        # 3. Prepare Combined Attention Mask
        attention_mask_for_block: Optional[torch.Tensor] = None
        if attention_mask is None:
            co_masks = torch.cat(co_masks_list, dim=1) # (B, N), assuming mask is 1 for valid, 0 for pad
            # Create 4D float mask (0/-inf) expected by attention layers
            # This basic version assumes full attention between all valid tokens
            attention_mask_for_block = torch.zeros_like(co_masks, dtype=co_embeds.dtype)
            attention_mask_for_block.masked_fill_(~co_masks, torch.finfo(co_embeds.dtype).min)
            
            if attention_mask_for_block.ndim == 2:
                attention_mask_for_block = attention_mask_for_block.unsqueeze(1).unsqueeze(2) # (B, 1, 1, N_total)
            elif attention_mask_for_block.ndim == 3: # Should not happen with co_masks
                attention_mask_for_block = attention_mask_for_block.unsqueeze(1) # (B, 1, N_q, N_k)   
        else:
            if attention_mask.dtype == torch.bool:
                attention_mask_for_block = torch.zeros_like(attention_mask, dtype=co_embeds.dtype)
                attention_mask_for_block.masked_fill_(~attention_mask, torch.finfo(co_embeds.dtype).min)
            else:
                attention_mask_for_block = attention_mask
        
        if attention_mask_for_block is not None:
            if attention_mask_for_block.ndim == 2: # (B, N_total)
                attention_mask_for_block = attention_mask_for_block.unsqueeze(1).unsqueeze(2)
            elif attention_mask_for_block.ndim == 3: # (B, N_q, N_k)
                attention_mask_for_block = attention_mask_for_block.unsqueeze(1)

        # 4. Pass through Encoder Blocks
        all_block_attentions = [] if output_attentions else None
        all_block_router_logits = [] if output_router_logits else None
        all_hidden_states = [] if output_hidden_states else None

        hidden_states = co_embeds
        for i, block in enumerate(self.blocks):
            
            if output_hidden_states: 
                all_hidden_states.append(hidden_states,)
            
            hidden_states = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask, 
                N_text=N_text_actual if text_embeds is not None else 0,
                cos=cos_text,
                sin=sin_text,
                rope_2d_instance=self.rope2d if image_tensor is not None else None,
                image_pos_ids=image_pos_ids if image_tensor is not None else None, # (B, N_img_with_cls)
                grid_hw=self.grid_size if image_tensor is not None else None, # (H_patches, W_patches)
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )

            hidden_states = hidden_states[0]
            current_output_idx = 1
            if output_attentions:
                all_block_attentions.append(hidden_states[current_output_idx])
                current_output_idx += 1
            if output_router_logits:
                all_block_router_logits.append(hidden_states[current_output_idx])

        # 5. Final Norm
        last_hidden_states = self.norm(hidden_states)
        if output_hidden_states: 
                all_hidden_states.append(hidden_states,)   
        # 6. Return results
        return MoEModelOutput( 
            last_hidden_state=last_hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None, 
            attentions=tuple(all_block_attentions) if output_attentions else None,
            router_probs=tuple(all_block_router_logits) if output_router_logits else None
        )


# VLMo base/p16
@register_model
def vlmo_base_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 384)
    model = TransformerEncoder(
        img_size=img_size, embed_dim=256, depth=6, num_heads=4, 
        mlp_ratio=4, qkv_bias=False,
        norm_layer=partial(RMSNorm, eps=1e-6), **kwargs)
    return model

# VLMo large/p16
@register_model
def vlmo_large_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 384)
    model = TransformerEncoder(
        img_size=img_size, patch_size=16, embed_dim=1024, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6), **kwargs)
    return model

# VLMo base+/p16
@register_model
def vlmo_base_plus_patch16(pretrained=False, **kwargs):
    img_size = kwargs.pop("img_size", 384)
    model = TransformerEncoder(
        img_size=img_size, patch_size=16, embed_dim=544, depth=24, num_heads=16, 
        mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, need_relative_position_embed=False, 
        layer_scale_init_values=None, norm_layer=partial(RMSNorm, eps=1e-6), **kwargs)
    return model

# 示例: 定义一个新的 Encoder 架构
@register_model
def vlmo_custom_patch16(pretrained=False, **kwargs):
    """我的自定义 VLMo Encoder 配置"""
    img_size = kwargs.pop("img_size", 384)
    model = TransformerEncoder(
        img_size=img_size,
        patch_size=16,
        embed_dim=640,      # <--- 自定义 embed_dim
        depth=10,           # <--- 自定义 depth
        num_heads=10,       # <--- 自定义 num_heads
        mlp_ratio=3.5,      # <--- 自定义 mlp_ratio
        qkv_bias=True,
        norm_layer=partial(RMSNorm, eps=1e-6),
        # 可以添加或修改其他 MultiWayTransformer 参数
        **kwargs)
    return model
