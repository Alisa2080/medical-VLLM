import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from modules.AttentionSeries import DecoderCrossAttention
from modules.RMSNorm import RMSNorm
from modules.DropPath import DropPath
from modules.Mlp import MoeSparseMoeBlock

class CrossModalDecoderLayer(nn.Module):
    """
    交叉模态解码器层，包含交叉注意力和前馈网络
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        num_kv_heads: Optional[int] = 4,
        context_dim: Optional[int] = None,  # 编码器输出维度
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_eps: float = 1e-6,
        layer_scale_init_values: Optional[float] = 1e-5,
        # MoE参数
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        mlp_ratio: float = 4.0,
        norm_topk_prob: bool = True,
        moe_hidden_act: str = "silu",
    ):
        super().__init__()
        self.context_dim = context_dim if context_dim is not None else dim
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        # 交叉注意力层归一化
        self.cross_attn_layernorm = RMSNorm(dim, eps=norm_eps)
        
        # 交叉注意力模块
        self.cross_attn = DecoderCrossAttention(
            dim=dim,
            context_dim=self.context_dim,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            norm_layer=RMSNorm,
            norm_eps=norm_eps,
        )
        
        # Layer Scale for cross attention
        self.gamma_ca = nn.Parameter(
            layer_scale_init_values * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_values is not None else 1.0
        
        self.drop_path_ca = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
        # 前馈网络层归一化
        self.post_attention_layernorm = RMSNorm(dim, eps=norm_eps)
        
        # MoE前馈网络
        if num_experts > 1:
            self.moe_block = MoeSparseMoeBlock(
                hidden_size=dim,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=int(dim * mlp_ratio),
                norm_topk_prob=norm_topk_prob,
                moe_hidden_act=moe_hidden_act,
            )
        else:
            # 普通前馈网络
            from modules.Mlp import Mlp
            self.moe_block = Mlp(
                hidden_size=dim,
                intermediate_size=int(dim * mlp_ratio),
                hidden_act=moe_hidden_act
            )
        
        # Layer Scale for FFN
        self.gamma_ffn = nn.Parameter(
            layer_scale_init_values * torch.ones((dim)), requires_grad=True
        ) if layer_scale_init_values is not None else 1.0
        
        self.drop_path_ffn = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,  # 查询（语言嵌入）：(B, N_q, dim)
        context: torch.Tensor,        # 键值（图像嵌入）：(B, N_kv, context_dim)
        context_mask: Optional[torch.Tensor] = None,  # 图像掩码：(B, N_kv)
        output_attentions: bool = False,
        output_router_logits: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: 语言嵌入作为查询 (B, N_text, dim)
            context: 图像嵌入作为键值 (B, N_img, context_dim)
            context_mask: 图像掩码，True表示有效位置 (B, N_img)
            output_attentions: 是否输出注意力权重
            output_router_logits: 是否输出MoE路由logits
        
        Returns:
            tuple: (hidden_states, [attentions], [router_logits])
        """
        # 1. 交叉注意力
        residual = hidden_states
        hidden_states_for_cross = self.cross_attn_layernorm(hidden_states)
        
        cross_attn_output, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states_for_cross,
            context=context,
            context_mask=context_mask,
        )
        
        # 应用残差连接和LayerScale
        hidden_states = residual + self.drop_path_ca(self.gamma_ca * cross_attn_output)
        
        # 2. 前馈网络
        residual = hidden_states
        hidden_states_for_ffn = self.post_attention_layernorm(hidden_states)
        
        ffn_output = self.moe_block(hidden_states_for_ffn)
        
        # 处理MoE输出
        router_logits = None
        if isinstance(ffn_output, tuple):
            ffn_output, router_logits = ffn_output
        elif router_logits is None and output_router_logits:
            # 如果不是MoE但需要router_logits，创建dummy
            B, N, _ = hidden_states_for_ffn.shape
            router_logits = hidden_states_for_ffn.new_zeros(B * N, 1)
        
        # 应用残差连接和LayerScale
        hidden_states = residual + self.drop_path_ffn(self.gamma_ffn * ffn_output)
        
        # 准备输出
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (cross_attn_weights,)
        
        if output_router_logits:
            outputs += (router_logits,)
        
        return outputs
    
class CrossModalDecoder(nn.Module):
    """
    轻量级交叉模态解码器，用于交叉模态MLM任务
    图像嵌入作为键值，语言嵌入作为查询
    """
    def __init__(
        self,
        dim: int,
        context_dim: int,
        depth: int = 2,  # 轻量级，只用2层
        num_heads: int = 12,
        num_kv_heads: Optional[int] = 4,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_eps: float = 1e-6,
        layer_scale_init_values: Optional[float] = 1e-5,
        # MoE参数
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        mlp_ratio: float = 4.0,
        norm_topk_prob: bool = True,
        moe_hidden_act: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.context_dim = context_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        
        # 计算每层的drop_path率
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # 构建解码器层
        self.layers = nn.ModuleList([
            CrossModalDecoderLayer(
                dim=dim,
                context_dim=context_dim,
                num_heads=num_heads,
                num_kv_heads=self.num_kv_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                drop_path=dpr[i],
                norm_eps=norm_eps,
                layer_scale_init_values=layer_scale_init_values,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                mlp_ratio=mlp_ratio,
                norm_topk_prob=norm_topk_prob,
                moe_hidden_act=moe_hidden_act,
            )
            for i in range(depth)
        ])
        
        # 最终归一化层
        self.norm = RMSNorm(dim, eps=norm_eps)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化权重"""
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,  # 语言嵌入 (B, N_text, dim)
        context: torch.Tensor,        # 图像嵌入 (B, N_img, context_dim)
        context_mask: Optional[torch.Tensor] = None,  # 图像掩码 (B, N_img)
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        output_router_logits: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: 语言嵌入作为查询 (B, N_text, dim)
            context: 图像嵌入作为键值 (B, N_img, context_dim)  
            context_mask: 图像掩码，True表示有效位置 (B, N_img)
            output_attentions: 是否输出所有层的注意力权重
            output_hidden_states: 是否输出所有层的隐藏状态
            output_router_logits: 是否输出MoE路由logits
            
        Returns:
            tuple: 包含最终隐藏状态和可选的注意力权重、隐藏状态、路由logits
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        
        # 处理context_mask：将bool mask转换为attention mask格式
        attn_context_mask = None
        if context_mask is not None:
            if context_mask.dtype == torch.bool:
                # 创建attention mask：True->0, False->-inf
                attn_context_mask = torch.zeros_like(context_mask, dtype=hidden_states.dtype)
                attn_context_mask.masked_fill_(~context_mask, torch.finfo(hidden_states.dtype).min)
            else:
                attn_context_mask = context_mask
        
        current_hidden_states = hidden_states
        
        # 通过所有解码器层
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (current_hidden_states,)
            
            layer_outputs = layer(
                hidden_states=current_hidden_states,
                context=context,
                context_mask=attn_context_mask,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
            )
            
            current_hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            if output_router_logits:
                router_idx = 2 if output_attentions else 1
                if len(layer_outputs) > router_idx:
                    all_router_logits = all_router_logits + (layer_outputs[router_idx],)
        
        # 最终归一化
        last_hidden_state = self.norm(current_hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (last_hidden_state,)
        
        # 构建输出
        outputs = (last_hidden_state,)
        
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        
        if output_attentions:
            outputs = outputs + (all_attentions,)
        
        if output_router_logits:
            outputs = outputs + (all_router_logits,)
        
        return outputs

class CrossModalMLMHead(nn.Module):
    """
    用于交叉模态MLM的预测头
    """
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        hidden_act: str = "silu",
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = RMSNorm(hidden_size, eps=layer_norm_eps)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # 激活函数
        from transformers.activations import ACT2FN
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation function: {hidden_act}")
        self.act_fn = ACT2FN[hidden_act]
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: 解码器输出 (B, N_text, hidden_size)
        
        Returns:
            logits: MLM预测logits (B, N_text, vocab_size)
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits