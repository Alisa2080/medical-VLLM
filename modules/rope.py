import torch
import torch.nn as nn
from functools import cached_property # Python 3.8+
from typing import Optional, Tuple


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope_input_validation(x, freqs_cis):
    assert x.ndim == freqs_cis.ndim + 1, (x.shape, freqs_cis.shape)
    assert x.shape[:-2] == freqs_cis.shape[:-1], (x.shape, freqs_cis.shape)
    assert x.shape[-1] == 2 * freqs_cis.shape[-1], (x.shape, freqs_cis.shape)
    assert freqs_cis.dtype == torch.complex64, freqs_cis.dtype


def apply_rope(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2), dtype=torch.complex64. It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    _apply_rope_input_validation(xq, freqs_cis)
    _apply_rope_input_validation(xk, freqs_cis)
    if freqs_cis.dim() == xq.dim() - 1: # 检查是否缺少 head 维度
         freqs_cis = freqs_cis.unsqueeze(-2) # 添加 head 维度 -> (B, P, 1, D/2)
    # ..., num_heads, head_dim/2
    xq_ = torch.view_as_complex(xq.float().view(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().view(*xk.shape[:-1], -1, 2))

    xq_rotated = xq_ * freqs_cis
    xk_rotated = xk_ * freqs_cis

    xq_out = torch.view_as_real(xq_rotated).flatten(-2)  # ..., num_heads, head_dim
    xk_out = torch.view_as_real(xk_rotated).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Rope2DPosEmb(nn.Module):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each forward pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the forward pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
        device (str): the device to store the precomputed cis
    """

    def __init__(
        self, dim: int, max_height: int, max_width: int, theta_base=10000):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.register_buffer('_dummy', torch.tensor(0.0), persistent=False)

    @property
    def device(self):
        return self._dummy.device
    
    def extra_repr(self):
        return f"dim={self.dim}, max_height={self.max_height}, max_width={self.max_width}, theta_base={self.theta_base}"

    @cached_property
    def precomputed_freqs_cis(self) -> torch.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = torch.arange(0, N).float().to(self.device)
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = (
            torch.arange(0, self.dim, 4)[: (self.dim // 4)].float().to(self.device)
        )  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = torch.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = torch.outer(y_pos, freqs).float()  # N, C/4
        x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)  # N, C/4
        # N, C/4, 2
        freqs_cis = torch.cat(
            [x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1
        )
        # max_height, max_width, C/2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)
        return freqs_cis

    def get_freqs_cis_by_seqlens(self, grid_hws: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid_hws (torch.Tensor): containing list of (height, width) or (t, height, width) tuples.
        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2)
        """
        shapes = grid_hws.tolist()
        assert all(
            1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes
        ), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = torch.cat(
            [
                self.precomputed_freqs_cis[:h, :w].reshape(-1, self.dim // 2)
                for h, w in shapes
            ],
            dim=0,
        )
        return freqs_cis

    def get_freqs_cis_by_idx(
        self, pos_idx: torch.Tensor, pos_idx_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pos_idx: tensor of shape (..., 2), It contains the (h, w) position indices of each 2D token.
            pos_idx_mask: a mask of shape (...), the leading dimensions should be the same as pos_idx.
                Rope will only be applied to the tokens with True mask. `freqs_cis` for the tokens with False mask with be ones.
        Return:
            freqs_cis: tensor of shape (..., dim//2)
        """
        assert (
            pos_idx.shape[:-1] == pos_idx_mask.shape
            and pos_idx.shape[-1] == 2
            and pos_idx.ndim == pos_idx_mask.ndim + 1
        ), (pos_idx.shape, pos_idx_mask.shape)
        assert pos_idx_mask.dtype == torch.bool, pos_idx_mask.dtype

        shp = pos_idx_mask.shape + (self.dim // 2,)  # ..., head_dim/2
        freqs_cis = torch.ones(
            shp, dtype=torch.complex64, device=self.device
        )  # ..., head_dim/2
        freqs_cis[pos_idx_mask] = self.precomputed_freqs_cis[
            pos_idx[..., 0][pos_idx_mask], pos_idx[..., 1][pos_idx_mask]
        ]
        return freqs_cis
    

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class DeepseekV3RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
    
# apply_rotary_pos_emb_qwen3 函数要求 query 和 key 的序列长度必须相同，最后的维度也相同
def apply_rotary_pos_emb_qwen3(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies Rotary Position Embedding to the query and key tensors.
    This version expects `cos` and `sin` to be pre-selected for the correct positions.
    The `position_ids` argument is ignored here.
    """
        # cos/sin 已经是针对当前 token 的值，形状类似 (bs, 1, seq_len_q, head_dim/2) 或可广播
        # q/k 形状类似 (bs, heads, seq_len_q, head_dim)

    # 直接 unsqueeze 准备广播 
    cos = cos.unsqueeze(unsqueeze_dim) #(b,1,s,head_dim)
    sin = sin.unsqueeze(unsqueeze_dim) #(b,1,s,head_dim)

    # 应用 RoPE 公式
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SimpleQwen3RotaryEmbedding(nn.Module):
    """
    简化的 Qwen3 RoPE 实现，无 Scaling。
    根据 position_ids 动态计算 cos/sin。
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        # 计算逆频率 inv_freq
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # 无需缓存 cos/sin

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        Args:
            x: 仅用于获取 dtype 和 device。
            position_ids: 需要计算 RoPE 的位置索引 (B, Nq)。
        Returns:
            cos, sin: 对应位置的 cos/sin 值，形状 (B, Nq, dim)。
        """
        # inv_freq: (D/2)
        # position_ids: (B, Nq)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # (B, D/2, 1)
        position_ids_expanded = position_ids[:, None, :].float() # (B, 1, Nq)

        # 强制 float32 计算以保证精度
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: (B, D/2, 1) @ (B, 1, Nq) -> (B, D/2, Nq)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float())
            # 转置: (B, Nq, D/2)
            freqs = freqs.transpose(1, 2)
            # 拼接: (B, Nq, D)
            emb = torch.cat((freqs, freqs), dim=-1)
            # 计算 cos/sin: (B, Nq, D)
            cos = emb.cos()
            sin = emb.sin()
            # scaling = 1.0 # 简化版无 scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)