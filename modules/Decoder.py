import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,MoEModelOutputWithPastAndCrossAttentions
from modules.Block import DecoderBlock
from modules.RMSNorm import RMSNorm
import warnings
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from modules.rope import SimpleQwen3RotaryEmbedding
from typing import Optional, Tuple, Callable, Type

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder 模块，包含词嵌入、多个 DecoderBlock 和一个最终的归一化层。
    可以选择性地包含一个输出投影层 (LM Head)。
    """
    def __init__(
        self,
        vocab_size: int,
        depth: int = 6,
        dim: int = 256,
        num_heads: int = 4,
        num_kv_heads: Optional[int] = None,
        context_dim: Optional[int] = None,
        norm_layer=RMSNorm,
        qkv_bias: bool = False,
        norm_eps: float = 1e-6,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        layer_scale_init_values: float = 1e-5,
        # MoE specific parameters for DecoderBlock
        num_experts: int = 4,
        mlp_ratio: float = 4.0,
        num_experts_per_tok: int = 2,
        norm_topk_prob: bool = False,
        moe_hidden_act: str = "silu",
        # General parameters
        padding_idx: int = 0, # Padding index for embedding
        max_seq_len: int = 512,
        rope_base: int = 10000,
        rope_scaling: Optional[dict] = None,
        **kwargs, # 其他参数
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.depth = depth
        self.dim = dim
        self.num_layers = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.context_dim = context_dim if context_dim is not None else dim

        # 1. Token Embedding
        self.padding_idx = padding_idx
        self.tok_embeddings = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)

        
        self.head_dim = context_dim // num_heads
        self.rotary_emb = SimpleQwen3RotaryEmbedding(dim=self.head_dim, max_position_embeddings=max_seq_len,
            base=rope_base)
        
        
        # 2. Decoder Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.layers = nn.ModuleList([
            DecoderBlock(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_eps=norm_eps,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                layer_scale_init_values=layer_scale_init_values,
                context_dim=context_dim,
                num_kv_heads=num_kv_heads,
                layer_idx=i,
                # MoE parameters for each block
                mlp_ratio=mlp_ratio,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                norm_topk_prob=norm_topk_prob,
                moe_hidden_act=moe_hidden_act,
            )
            for i in range(depth)])

        # 3. Final Normalization
        self.norm = norm_layer(dim, eps=norm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """ Initialize weights """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: # 检查 bias 是否存在
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding): 
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if hasattr(m, 'padding_idx') and m.padding_idx is not None: 
                # 确保 padding_idx 在有效范围内
                if m.weight is not None and 0 <= m.padding_idx < m.num_embeddings:
                     with torch.no_grad(): # 确保在 no_grad 上下文中修改
                         m.weight[m.padding_idx].fill_(0) # 使用 fill_ 更安全
                else:
                     warnings.warn(f"Module {m} has padding_idx {m.padding_idx} but it's invalid or weight is None.")
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
             if hasattr(m, 'bias') and m.bias is not None: 
                 nn.init.constant_(m.bias, 0)
             if hasattr(m, 'weight') and m.weight is not None: 
                 nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # Decoder padding mask (B, Nq), **True is VALID**
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None, # Context mask (B, Nkv), **True is VALID**      # Encoder output, shape (B, N_kv, C_kv)
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        output_router_logits: Optional[bool] = None, # New parameter
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs, # 其他参数
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        should_output_router_probs = output_router_logits if output_router_logits is not None else False
        use_cache = use_cache if use_cache is not None else False

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You have to specify either input_ids or inputs_embeds")
            inputs_embeds = self.tok_embeddings(input_ids)

        hidden_states = inputs_embeds
        B, Nq, C = hidden_states.shape
        device = hidden_states.device

        past_length = 0
        if past_key_values is not None:
            # 尝试获取缓存长度，如果失败则为 0
            try:
                past_length = past_key_values.get_seq_length(self.layers[0].layer_idx)
            except (AttributeError, KeyError, IndexError): # 处理各种可能的缓存错误
                past_length = 0
                if use_cache: # 如果明确要用缓存但获取长度失败，可能需要重置或警告
                     warnings.warn("Could not get sequence length from past_key_values, resetting past_length to 0.")
                     past_key_values = None # 或者创建一个新的空缓存

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        
        #  计算 cache_position 和 RoPE position_ids
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        self.rotary_emb.to(device)
        cos, sin = self.rotary_emb(hidden_states, position_ids=position_ids)
        # cos/sin shape: (B, Nq, D_head)

        causal_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (B, Nq), inputs_embeds, past_length
        ) # Returns float mask (0/-inf) shape (B, 1, Nq, kv_seq_len)

        # --- Prepare Context Mask ---
        context_mask_4d = None
        if context is not None and context_mask is not None:
             if context_mask.dim() == 2: # (B, Nkv), True=VALID
                 _mask = context_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                 context_mask_4d = (_mask - 1.0) * torch.finfo(hidden_states.dtype).max # 1->0, 0->-inf
             elif context_mask.dim() == 4: # Assume already float 0/-inf
                 context_mask_4d = context_mask
             else: 
                 warnings.warn(f"Unexpected context_mask dim: {context_mask.dim()}")
        
        # --- Decoder Layer Loop ---
        all_hidden_states_tuple = () if output_hidden_states else None
        all_self_attns_tuple = () if output_attentions else None
        all_cross_attns_tuple = () if output_attentions else None
        all_router_logits_tuple = () if output_router_logits else None
        # 2. Pass through Decoder Layers
        for layer in self.layers:
            if output_hidden_states: 
                all_hidden_states_tuple += (hidden_states,)
            
            # 调用层的前向传播
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask_4d,
                context=context,
                context_mask=context_mask_4d,
                cos=cos,
                sin=sin,
                cache_position=cache_position,              
                past_key_value=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )

            hidden_states = layer_outputs[0]

            current_output_idx = 1
            if output_attentions:
                self_attn_weights, cross_attn_weights = layer_outputs[current_output_idx]
                all_self_attns_tuple += (self_attn_weights,)
                if cross_attn_weights is not None:
                    all_cross_attns_tuple += (cross_attn_weights,)
                else:
                    all_cross_attns_tuple += (None,)
                current_output_idx += 1
            
            if should_output_router_probs:
                router_logits = layer_outputs[current_output_idx]
                all_router_logits_tuple += (router_logits,)
                current_output_idx += 1
            

        # 3. Final Normalization
        hidden_states = self.norm(hidden_states)

        if output_hidden_states: 
            all_hidden_states_tuple += (hidden_states,)      
        
        if should_output_router_probs and all_router_logits_tuple is not None: # Check if the tuple is not empty
            all_router_probs_tuple = tuple(
                    [F.softmax(logits, dim=-1) for logits in all_router_logits_tuple if logits is not None]
                )
        else: # Handle case where tuple might be initialized but empty
                all_router_probs_tuple = ()


        return MoEModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states_tuple,
            attentions=all_self_attns_tuple,
            cross_attentions=all_cross_attns_tuple,
            router_probs=all_router_probs_tuple,
    )
        
        