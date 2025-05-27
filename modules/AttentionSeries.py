import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.layers import trunc_normal_
from typing import Optional, Tuple
import numpy as np
from einops import rearrange
from .rope import apply_rope, Rope2DPosEmb
from typing import Optional, Tuple, Callable
from .RMSNorm import RMSNorm
from transformers.activations import ACT2FN

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def eager_attention_forward(
    module: nn.Module, # 注意：需要传入 Attention 模块实例
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs, 
):
    # K, V 的形状预期是 (B, num_kv_heads, N, D)
    # Query 的形状预期是 (B, num_heads, N, D)
    key_states = repeat_kv(key, module.num_key_value_groups)   # (B, H, Nk, D)
    value_states = repeat_kv(value, module.num_key_value_groups) # (B, H, Nk, D)

    # 计算 Attention Scores
    # query: (B, H, Nq, D), key_states.transpose: (B, H, D, Nk)
    # attn_weights: (B, H, Nq, Nk)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    # 应用 Mask
    if attention_mask is not None:
        # 裁剪 mask 以匹配 K 的长度 (B, 1, Nq, Nk) or (B, H, Nq, Nk)
        # 注意 key_states 的形状是 (B, H, Nk, D)，所以长度在 dim=2
        current_mask = attention_mask[..., :key_states.shape[2]]
        if current_mask.dim() == 2:  # 例如调试信息中的情况: (B, Nk)
            # 扩展为 (B, 1, 1, Nk) 以便在 H (注意力头) 和 Nq (查询序列长度) 维度上广播
            expanded_mask = current_mask.unsqueeze(1).unsqueeze(1)
        elif current_mask.dim() == 3:  # 例如 (B, Nq, Nk)
            # 扩展为 (B, 1, Nq, Nk) 以便在 H 维度上广播
            expanded_mask = current_mask.unsqueeze(1)
        elif current_mask.dim() == 4:  # 例如 (B, H, Nq, Nk) 或 (B, 1, Nq, Nk)
            # 如果是 (B, H, Nq, Nk)，则无需扩展
            # 如果是 (B, 1, Nq, Nk)，它会自动在 H 维度上广播
            expanded_mask = current_mask
        else:
            raise ValueError(
                f"Unsupported attention_mask dimension after slicing: {current_mask.dim()}. "
                f"Original mask shape: {attention_mask.shape}, Sliced mask shape: {current_mask.shape}"
            )
        attn_weights = attn_weights + expanded_mask # Mask 通常包含 -inf

    # Softmax
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

    # Dropout
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Weighted Value
    # (B, H, Nq, Nk) @ (B, H, Nk, D) -> (B, H, Nq, D)
    attn_output = torch.matmul(attn_weights, value_states)

    # Transpose 输出以匹配常见格式 (B, Nq, H, D) 
    attn_output = attn_output.transpose(1, 2).contiguous() # Qwen3 Attention 外层做了这个 transpose

    # 返回未 transpose 的结果和 attention weights
    return attn_output, attn_weights


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w],indexing='ij'))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)

            # trunc_normal_(self.relative_position_bias_table, std=.0)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedAttention(nn.Module):
    """
    基于门控注意力的多实例学习聚合模块。
    """
    def __init__(self, 
                 input_dim: int,
                   num_classes: int, 
                   intermediate_dim: int = 512, 
                   hidden_dim_att: int = 256, 
                   hidden_act="silu",
                   init_std: float = 0.02):
        """
        初始化 GatedAttentionMIL 模块。

        Args:
            input_dim_vit (int): ViT输出的patch特征的维度。  512
            num_classes (int): 分类任务的类别数量。  3
            hidden_dim_att (int): 门控注意力网络中隐藏层的维度。默认为 384。
            mapped_dim_fc (int): 初始特征映射层输出的维度。默认为 512。
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.intermediate_dim = intermediate_dim
        self.hidden_dim_att = hidden_dim_att
        self.init_std = init_std

        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation function: {hidden_act}. Supported: {list(ACT2FN.keys())}")
        self.act_fn = ACT2FN[hidden_act]
        # 1. 全连接层和ReLU，将输入patch特征映射到 intermediate_dim 维
        self.feature_mapper = nn.Sequential(
            nn.Linear(self.input_dim, self.intermediate_dim),
            self.act_fn
        )

        # 2. 门控注意力网络
        self.attention_V = nn.Linear(self.intermediate_dim, self.hidden_dim_att)
        self.attention_U = nn.Linear(self.intermediate_dim, self.hidden_dim_att)
        self.attention_weights = nn.Linear(self.hidden_dim_att, 1)

        # 3. 全连接分类器头部
        self.classifier = nn.Linear(self.intermediate_dim, self.num_classes)
        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=self.init_std) # 使用和ViT类似的截断正态初始化
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, patch_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patch_features: (B_slides, N_patches, D_vit)
        
        # 1. 映射特征
        # mapped_features: (B_slides, N_patches, D_mapped)
        mapped_features = self.feature_mapper(patch_features)

        # 2. 计算门控注意力
        # att_v: (B_slides, N_patches, D_hidden_att)
        att_v = torch.tanh(self.attention_V(mapped_features))
        # att_u: (B_slides, N_patches, D_hidden_att)
        att_u = torch.sigmoid(self.attention_U(mapped_features))

        # A_raw: (B_slides, N_patches, 1) - 每个patch的原始注意力分数
        A_raw = self.attention_weights(att_v * att_u)
        # att_softmax: (B_slides, N_patches, 1) - 对每个slide内的patch应用softmax
        # 如果 B_slides > 1, 这里的 softmax 是在所有 N_patches 上进行的，
        # 如果希望在每个 slide 内部的 N_patches 上独立 softmax，需要调整。
        # 假设当前我们一次处理一个 slide (B_slides=1)，或者注意力在所有 patch 上计算。
        att_softmax = F.softmax(A_raw, dim=1) # dim=1 表示在 num_patches 维度上 softmax

        # 3. 聚合得到slide级别的嵌入
        # slide_embedding: (B_slides, D_mapped)
        slide_embedding = torch.sum(att_softmax * mapped_features, dim=1)

        # 4. 分类
        # logits: (B_slides, num_classes)
        logits = self.classifier(slide_embedding)

        return logits, att_softmax
    

class EncoderSelfAttention(nn.Module):

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            num_kv_heads: Optional[int] = 4,
            qkv_bias: bool = False,
            qk_scale: Optional[float] = None,
            attn_drop: float = 0.,
            norm_layer: Callable = RMSNorm,
            norm_eps: float = 1e-6,
            
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads # K/V 头数量
        assert self.num_heads % self.num_kv_heads == 0, \
            f"num_heads ({self.num_heads}) 必须能被 num_kv_heads ({self.num_kv_heads}) 整除"
        self.attention_dropout = attn_drop
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
 
        self.q_proj = nn.Linear(dim, self.num_heads * self.head_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim, eps=norm_eps)
        self.k_norm = norm_layer(self.head_dim, eps=norm_eps)


    def forward(self, hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                rope_2d_instance: Optional[Rope2DPosEmb] = None, # 用于 2D RoPE
                image_pos_ids: Optional[torch.Tensor] = None, # 可选的图像位置 ID 张量，形状为 (B, N_img)，其中 N_img 是图像 token 的数量。
                grid_hw: Optional[Tuple[int, int]] = None,
                **kwargs) :
        input_shape = hidden_states.shape[:-1] # B, sequence length
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        assert rope_2d_instance is not None and image_pos_ids is not None and grid_hw is not None, "2D RoPE args needed"
        q_cls, q_patches = query_states[:, :, :1, :], query_states[:, :, 1:, :]
        k_cls, k_patches = key_states[:, :, :1, :], key_states[:, :, 1:, :]
        
        patch_rel_pos_ids = image_pos_ids[:, 1:]
        h_indices = (patch_rel_pos_ids // grid_hw[1]) % grid_hw[0]
        w_indices = patch_rel_pos_ids % grid_hw[1]
        patch_2d_pos_idx = torch.stack([h_indices, w_indices], dim=-1) # (B, P, 2)
        patch_mask = torch.ones(patch_2d_pos_idx.shape[:-1], dtype=torch.bool, device=hidden_states.device)

        try:
            freqs_cis_patches = rope_2d_instance.get_freqs_cis_by_idx(patch_2d_pos_idx, patch_mask) # (B, P, head_dim/2)
            q_r, k_r = apply_rope(q_patches.transpose(1, 2), k_patches.transpose(1, 2), freqs_cis_patches)
                
            q_patches_rot = q_r.transpose(1, 2)
            k_patches_rot = k_r.transpose(1, 2)
        except IndexError as e:
            print(f"Warning: IndexError during 2D RoPE application. Check pos_idx and max H/W. Error: {e}")
            # 降级处理：使用原始patches
            q_patches_rot = q_patches
            k_patches_rot = k_patches
        
        query_states = torch.cat([q_cls, q_patches_rot], dim=2) # Concatenate along sequence dimension
        key_states = torch.cat([k_cls, k_patches_rot], dim=2)

        attn_mask_for_eager = None
        if attention_mask is not None:
           if attention_mask.dtype == torch.bool:
               attn_mask_for_eager = torch.zeros_like(attention_mask, dtype=query_states.dtype)
               attn_mask_for_eager.masked_fill_(~attention_mask, torch.finfo(query_states.dtype).min)
           else:
               attn_mask_for_eager = attention_mask

        attention_interface: Callable = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,        
            query=query_states,            
            key=key_states,              
            value=value_states,            
            attention_mask=attn_mask_for_eager, 
            scaling=self.scale,
            dropout=self.attention_dropout,
        )   
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class P2TAttention(nn.Module):
    def __init__(self, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pool_ratios=[1 ,2 ,3 ,6]):

        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([ t *t for t in pool_ratios]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool_ratios = pool_ratios  # 池化窗口大小
        self.pools = nn.ModuleList()

        self.norm = nn.LayerNorm(dim)
        self.d_convs1 = nn.ModuleList([nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in self.pool_ratios])

    def forward(self, x, H, W, d_convs=None):
        B, N, C = x.shape

        # 通过输入x生成q矩阵: (B,N,C) --q-> (B,N,C) --reshape-> (B,N,h,d) --permute-> (B,h,N,d);   C=h*d
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        pools = []

        # 为了便于在x上执行多尺度池化操作,我们将其reshape重塑为2D类型: (B,N,C) --permute-> (B,C,N) --reshape-> (B,C,H,W)
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # 遍历多个池化层, 假设池化窗口为: [1 ,2 ,3 ,6]
        for (pool_ratio, l) in zip(self.pool_ratios, self.d_convs1):
            # 分别计算当前池化窗口下的输出: input:(B,C,H,W);  1th_pool: (B,C,H/1,W/1); 2th_pool: (B,C,H/2,W/2); 3th_pool: (B,C,H/3,W/3); 4th_pool: (B,C,H/6,W/6)
            pool = F.adaptive_avg_pool2d(x_, (round( H /pool_ratio), round( W /pool_ratio)))
            # 将每一个尺度对应的池化层的输出, 再通过3*3的深度卷积进行相对位置编码, 然后与池化的输出相加
            pool = pool + l(pool)
            # 将每个尺度的输出重塑为与原始输入相同的shape: 1th_pool: (B,C,H/1,W/1) -->(B,C,(HW/1^2));  2th_pool: (B,C,H/2,W/2) --> (B,C,(HW/2^2));  3th_pool: (B,C,H/3,W/3) --> (B,C,(HW/3^2));   3th_pool: 
            pools.append(pool.view(B, C, -1))

        # 将多个尺度池化层的输出在token维度进行拼接,其具有多尺度的上下文信息: (B,C,(HW/1^2)+(HW/2^2)+(HW/3^2)+(HW/6^2))==(B,C,token_num) , 令token_num = (HW/1^2)+(HW/2^2)+(HW/3^2)+(HW/6^2)
        pools = torch.cat(pools, dim=2)
        # 将其进行维度转换, 以便于后续计算: (B,C,token_num)--permute->(B,token_num,C)
        pools = self.norm(pools.permute(0 ,2 ,1))

        # 多尺度的上下文信息生成kv: (B,token_num,C) --kv-> (B,token_num,2C) --reshape-> (B,token_num,2,h,d) --permute-> (2,B,h,token_num,d);   C=h*d
        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k:(B,h,token_num,d); v:(B,h,token_num,d)
        k, v = kv[0], kv[1]

        # 计算Token-to-Region化的注意力矩阵(region是指池化是在窗口上进行的,窗口可以看作region): (B,h,N,d) @ (B,h,d,token_num) = (B,h,N,token_num)  N:输入的token总数, token_num:池化后的Token总数量
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 通过注意力矩阵对value加权求和: (B,h,N,token_num) @ (B,h,token_num,d) = (B,h,N,d)
        x = (attn @ v)

        # 通过对输入进行重塑shape得到与原始输入相同的shape: (B,h,N,d) --transpose-> (B,N,h,d) --reshape-> (B,N,C)
        x = x.transpose(1 ,2).contiguous().reshape(B, N, C)
        # 最后通过一个线性层进行映射, 得到最终输出: (B,N,C)-->(B,N,C)
        x = self.proj(x)

        return x
    