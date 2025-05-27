import torch
import torch.nn as nn
from typing import Optional, Tuple,Callable
from modules.AttentionSeries import Attention,VisionEncoderSelfAttention,EncoderSelfAttention,DecoderSelfAttention,DecoderCrossAttention
from modules.Mlp import Mlp,Mlp_original, MoeSparseMoeBlock
from modules.DropPath import DropPath
from modules.RMSNorm import RMSNorm
from .rope import Rope2DPosEmb
from transformers.cache_utils import Cache
from typing import Optional, Tuple, Callable, Type


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float=4., qkv_bias: bool=False, qk_scale: Optional[float]=None, drop: float=0., attn_drop: float=0.,
                 drop_path: float=0., init_values: Optional[float]=None, act_layer: Callable=nn.GELU, norm_layer: Callable=nn.LayerNorm,
                 window_size: Optional[Tuple[int, int]]=None, attn_head_dim: Optional[int]=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_original(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None, return_attention=False, return_qkv=False):
        if return_attention:
            return self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_attention=True)
        if return_qkv:
            y, qkv = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, return_qkv=return_qkv)
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
            return x, qkv
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VisionEncoderBlock(nn.Module):

    def __init__(self, 
                dim:int,
                num_heads:int,
                num_kv_heads: Optional[int] = None,
                mlp_ratio:float=4.,
                qkv_bias:bool=False,
                qk_scale:Optional[float]=None,
                attn_drop:float=0.,
                drop_path:float=0.,
                layer_scale_init_values: Optional[float] =  1e-5,
                hidden_act: str = "silu",
                norm_layer:Callable=RMSNorm,
                norm_eps: float = 1e-6,
                ):
        super().__init__()
        self.intermediate_size = int(dim * mlp_ratio)
        self.hidden_act = hidden_act
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.input_layernorm = norm_layer(dim, eps=norm_eps)
        
        # --- Self-Attention Path ---
        self.attn = VisionEncoderSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            norm_eps=norm_eps,
        )
        self.gamma_sa = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        
        self.drop_path_sa = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # ---  FFN Path ---
        self.post_attention_layernorm = norm_layer(dim, eps=norm_eps)
        self.mlp = Mlp(hidden_size=dim, intermediate_size=self.intermediate_size, hidden_act=self.hidden_act)

        self.gamma_ffn = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                rope_2d_instance: Optional[Rope2DPosEmb] = None,
                image_pos_ids: Optional[torch.Tensor] = None,
                grid_hw: Optional[Tuple[int, int]] = None,
                output_attentions: bool = False,
                **kwargs: Optional[dict] ,
                ):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self_attn_output, self_attn_weights= self.attn(
            hidden_states,
            mask=attention_mask,
            rope_2d_instance=rope_2d_instance,
            image_pos_ids=image_pos_ids,
            grid_hw=grid_hw
        )
        hidden_states = residual + self.drop_path_sa(self.gamma_sa * self_attn_output)

        residual = hidden_states
        hidden_states_for_ffn = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states_for_ffn)
        
        hidden_states = residual + hidden_states
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        
        return outputs
    
class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = RMSNorm,
        norm_eps: float = 1e-6,
        layer_scale_init_values: Optional[float] =  1e-5,
        # ---  MoE paraments ---
        num_experts: int = 4,
        mlp_ratio: float = 4.0,
        num_experts_per_tok: int = 2,
        norm_topk_prob: bool = False,
        moe_hidden_act: str = "silu",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.moe_intermediate_size = int(dim * mlp_ratio)
        self.hidden_act = moe_hidden_act
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        # --- Self-Attention Path ---
        self.input_layernorm = norm_layer(dim, eps=norm_eps)
        self.attn = EncoderSelfAttention( 
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads, 
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            norm_eps=norm_eps,
        )
        self.gamma_sa = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        
        self.drop_path_sa = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

         # --- MoE FFN Path ---
        self.post_attention_layernorm = norm_layer(dim, eps=norm_eps)
        # MoE Blcok
        if num_experts > 1:
            self.moe_block = MoeSparseMoeBlock(
                hidden_size=dim,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=self.moe_intermediate_size,
                norm_topk_prob=norm_topk_prob,
                moe_hidden_act=moe_hidden_act,
                )
        else: # Dense FFN
            self.moe_block = Mlp(hidden_size=dim,
                            intermediate_size=self.moe_intermediate_size,hidden_act =moe_hidden_act)

        self.gamma_ffn = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0

        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    
    def forward(
        self,
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        # --- RoPE 相关参数 ---
        N_text: Optional[int] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        rope_2d_instance: Optional[Rope2DPosEmb] = None,
        image_pos_ids: Optional[torch.Tensor] = None,
        grid_hw: Optional[Tuple[int, int]] = None,
        output_attentions: bool = False,
        output_router_logits: bool = True,
        **kwargs, 
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        self_attn_output, self_attn_weights= self.attn(
            hidden_states,
            mask=attention_mask,
            N_text=N_text,
            cos=cos,
            sin=sin,
            rope_2d_instance=rope_2d_instance,
            image_pos_ids=image_pos_ids,
            grid_hw=grid_hw
        )

        hidden_states = residual + self.drop_path_sa(self.gamma_sa * self_attn_output)

        residual = hidden_states
        hidden_states_for_ffn = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe_block(hidden_states_for_ffn)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
               
        if router_logits is None and output_router_logits:
            B, N, _ = hidden_states_for_ffn.shape
            router_logits = hidden_states_for_ffn.new_zeros(B * N, 1)  # (B*N,1)

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, )
        if output_router_logits:
            outputs += (router_logits,)

        return outputs

class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[nn.Module] = RMSNorm,
        layer_scale_init_values: Optional[float] =  1e-5, # Qwen3/Llama 通常用较小或不用 LayerScale
        context_dim: Optional[int] = None,  # Encoder output dim
        num_kv_heads: Optional[int] = None, # GQA KV heads
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-6,
        # ---  MoE paraments ---
        num_experts_per_tok: int = 2,
        norm_topk_prob: bool = False,
        num_experts: int = 4, # MoE Experts
        mlp_ratio: float = 4.0,
        moe_hidden_act: str = "silu", # Qwen3MoeMLP 使用
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_experts = num_experts
        self.moe_intermediate_size = int(dim * mlp_ratio)
        self.hidden_act = moe_hidden_act
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads

         # --- Self-Attention Path ---
        self.input_layernorm = norm_layer(dim, eps=norm_eps)
        self.self_attn = DecoderSelfAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            layer_idx=layer_idx,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
        )

        self.gamma_sa = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.drop_path_sa = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # --- Cross-Attention Path ---
        self.cross_attn_layernorm = norm_layer(dim, eps=norm_eps)
        self.cross_attn = DecoderCrossAttention(
            dim=dim,
            context_dim=context_dim,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            layer_idx=layer_idx,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            norm_eps=norm_eps,
        )
        self.gamma_ca = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.drop_path_ca = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()


        # --- MoE FFN Path ---
        self.post_attention_layernorm = norm_layer(dim, eps=norm_eps)
        # MoE Blcok
        self.moe_block = MoeSparseMoeBlock(
            hidden_size=dim,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=self.moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            moe_hidden_act=moe_hidden_act,
        )
        self.gamma_ffn = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
            if layer_scale_init_values is not None else 1.0
        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,               # (B, Nq, C)
        attention_mask: Optional[torch.Tensor] = None, # Self-Attention 4D Causal Mask (0/-inf)
        context: Optional[torch.Tensor] = None, # Encoder context (B, Nkv, C_kv)
        context_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None, # RoPE cos for query
        sin: Optional[torch.Tensor] = None, # RoPE cos for query
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None, # For KV Cache update
        past_key_value: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs, # 其他参数
    ):
        residual = hidden_states
        
        hidden_states = self.input_layernorm(hidden_states)
        # 自注意力层，现在支持KV缓存
        self_attn_output, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            cos=cos,
            sin=sin,
            output_attentions=output_attentions,
            past_key_value=past_key_value, # Pass Cache object
            cache_position=cache_position,
            use_cache=use_cache,
        )

        hidden_states = residual + self.drop_path_sa(self.gamma_sa * self_attn_output)

        # --- 2. Cross-Attention ---
        if context is not None:
            residual = hidden_states
            hidden_states_for_cross = self.cross_attn_layernorm(hidden_states)
            cross_attn_output, cross_attn_weights = self.cross_attn(
                hidden_states=hidden_states_for_cross,
                context=context,
                context_mask=context_mask,
                cos=cos,
                sin=sin,
                output_attentions=output_attentions,
            )
        
        # Apply DropPath and add residual
        hidden_states = residual + self.drop_path_sa(self.gamma_ca * cross_attn_output)


        # --- 3. MoE FFN ---
        residual = hidden_states
        hidden_states_for_ffn = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe_block(hidden_states_for_ffn)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None

        hidden_states = residual + hidden_states
        outputs = (hidden_states,)

        if output_attentions:
            outputs += ((self_attn_weights, cross_attn_weights), )
        if output_router_logits:
            outputs += (router_logits,)

        return outputs

