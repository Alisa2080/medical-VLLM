import torch
import torch.nn as nn
from functools import cached_property # Python 3.8+
# 或者 from torch.functional import cached_property (较新 PyTorch 版本)
# 或者自己实现一个简单的缓存机制
from typing import Optional, Tuple
# 可能还需要 math, numpy 等，根据实际代码补充


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
        self, dim: int, max_height: int, max_width: int, theta_base=10000, device="cuda"
    ):
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base
        self.device = device

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