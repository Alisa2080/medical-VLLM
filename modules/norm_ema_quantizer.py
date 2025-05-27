# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on VQGAN code bases
# https://github.com/CompVis/taming-transformers
# --------------------------------------------------------'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            dists = -(diffs ** 2).sum(dim = -1)

        buckets = dists.max(dim = -1).indices
        bins = torch.bincount(buckets, minlength = num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EmbeddingEMA(nn.Module):
    """
    于实现带有指数移动平均（Exponential Moving Average, EMA）的嵌入层。
    这个类主要用于在向量量化（Vector Quantization, VQ）过程中更新和维护码本（codebook）。
    """
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, kmeans_init=True, codebook_init_path=''):
        super().__init__()
        self.num_tokens = num_tokens # 词汇数
        self.codebook_dim = codebook_dim # 词维度
        self.decay = decay
        self.eps = eps 
        if codebook_init_path == '':   
            if not kmeans_init:
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                weight = torch.zeros(num_tokens, codebook_dim) # 8192, 32
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
            
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        # self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return
        print("Performing Kemans init for codebook")
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim = True)
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))
        
    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        # embed_normalized = l2norm(self.embed_avg / smoothed_cluster_size.unsqueeze(1))
        self.weight.data.copy_(embed_normalized)   

def norm_ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))
    moving_avg.data.copy_(l2norm(moving_avg.data))

class NormEMAVectorQuantizer(nn.Module):
    # 主要功能是将输入的连续向量 z 映射到离散的码本中，并计算量化损失。同时，它使用指数移动平均（EMA）来更新码本，以提高量化的稳定性和性能。
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        """
        初始化 NormEMAVectorQuantizer 类。

        参数:
        n_embed (int): 码本中嵌入向量的数量，即词汇数。
        embedding_dim (int): 每个嵌入向量的维度，即词维度。
        beta (float): 量化损失的权重系数。
        decay (float, 可选): 指数移动平均的衰减率，默认为 0.99。
        eps (float, 可选): 用于数值稳定性的小常数，默认为 1e-5。
        statistic_code_usage (bool, 可选): 是否统计码本的使用情况，默认为 True。
        kmeans_init (bool, 可选): 是否使用 K-means 算法初始化码本，默认为 False。
        codebook_init_path (str, 可选): 码本初始化权重的文件路径，默认为空字符串。
        """
        super().__init__()
        self.codebook_dim = embedding_dim # 32 # 词维度
        self.num_tokens = n_embed # 8192 词汇数
        self.beta = beta
        self.decay = decay
        
        # learnable = True if orthogonal_reg_weight > 0 else False
        # 初始化 EmbeddingEMA 类，用于管理码本
        self.embedding = EmbeddingEMA(self.num_tokens, self.codebook_dim, decay, eps, kmeans_init, codebook_init_path)
        
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            # 注册一个缓冲区来保存码本中每个嵌入向量的使用次数
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            # 如果使用分布式训练，使用 distributed.all_reduce 来同步统计信息
            self.all_reduce_fn = distributed.all_reduce
        else:
            # 如果不使用分布式训练，使用 nn.Identity 作为占位符
            self.all_reduce_fn = nn.Identity()
    
    def reset_cluster_size(self, device):
        """
        重置码本的聚类大小统计信息。

        参数:
        device (torch.device): 用于存放新的聚类大小统计信息的设备。
        """
        # 检查是否需要统计码本的使用情况
        if self.statistic_code_usage:
            # 注册一个新的缓冲区，用于保存码本中每个嵌入向量的使用次数，并初始化为零
            self.register_buffer('cluster_size', torch.zeros(self.num_tokens))
            # 将新的聚类大小统计信息移动到指定设备
            self.cluster_size = self.cluster_size.to(device)


    def forward(self, z):
        """
        前向传播方法，将输入的连续向量 z 映射到离散的码本中，并计算量化损失。

        参数:
        z (torch.Tensor): 输入的连续向量，形状为 (batch, channel, height, width)。

        返回:
        tuple: 包含量化后的向量 z_q、量化损失 loss 和编码索引 encoding_indices 的元组。
        """
        # reshape z -> (batch, height, width, channel) and flatten
        # 将输入的 z 从 (batch, channel, height, width) 形状转换为 (batch, height, width, channel)
        #z, 'b c h w -> b h w c'
        z = rearrange(z, 'b c h w -> b h w c')
        # 对 z 进行 L2 归一化
        z = l2norm(z)
        # 将 z 展平为二维张量，形状为 (-1, self.codebook_dim)
        z_flattened = z.reshape(-1, self.codebook_dim)
        
        # 如果码本未初始化，则使用输入数据进行 K-means 初始化
        self.embedding.init_embed_(z_flattened)
        
        # 计算输入向量与码本向量之间的距离
        d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
            self.embedding.weight.pow(2).sum(dim=1) - 2 * \
            torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight) # 'n d -> d n'
        
        # 找到每个输入向量对应的最近码本向量的索引
        encoding_indices = torch.argmin(d, dim=1)

        # 根据编码索引从码本中获取量化后的向量，并调整形状与输入 z 一致
        z_q = self.embedding(encoding_indices).view(z.shape)
        
        # 将编码索引转换为 one-hot 编码
        encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
        
        # 如果不在训练模式下，更新码本的聚类大小统计信息
        if not self.training:
            with torch.no_grad():
                # 计算每个码本向量的使用次数
                cluster_size = encodings.sum(0)
                # 如果使用分布式训练，同步聚类大小统计信息
                self.all_reduce_fn(cluster_size)
                # 使用指数移动平均更新聚类大小统计信息
                ema_inplace(self.cluster_size, cluster_size, self.decay)
        
        # 如果在训练模式下且允许更新码本
        if self.training and self.embedding.update:
            #EMA cluster size
            # 计算每个码本向量的使用次数
            bins = encodings.sum(0)
            # 如果使用分布式训练，同步聚类大小统计信息
            self.all_reduce_fn(bins)

            # self.embedding.cluster_size_ema_update(bins)
            # 使用指数移动平均更新聚类大小统计信息
            ema_inplace(self.cluster_size, bins, self.decay)

            # 找到未被使用的码本向量的掩码
            zero_mask = (bins == 0)
            # 将未被使用的码本向量的使用次数设置为 1，避免除零错误
            bins = bins.masked_fill(zero_mask, 1.)

            # 计算每个码本向量对应的输入向量的总和
            embed_sum = z_flattened.t() @ encodings
            # 如果使用分布式训练，同步嵌入向量总和信息
            self.all_reduce_fn(embed_sum)
                        
            # 计算每个码本向量的平均嵌入向量
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            # 对平均嵌入向量进行 L2 归一化
            embed_normalized = l2norm(embed_normalized)
            
            # 对于未被使用的码本向量，保持其原始值不变
            embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                           embed_normalized)
            # 使用归一化的指数移动平均更新码本向量
            norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

        # 计算量化损失
        loss = self.beta * F.mse_loss(z_q.detach(), z) 
        
        # 保留梯度信息，确保量化过程不会影响梯度传播
        z_q = z + (z_q - z).detach()

        # 将量化后的向量 z_q 形状转换回与输入 z 相同的形状
        #z_q, 'b h w c -> b c h w'
        z_q = rearrange(z_q, 'b h w c -> b c h w')
        
        # z_q是量化后的向量。在向量量化过程中，输入的连续向量 z 会被映射到离散的码本中，z_q 就是从码本中选取的与输入向量最接近的向量组合而成的结果。
        # loss 是量化损失。它用于衡量量化后的向量 z_q 与原始输入向量 z 之间的差异。在代码中，量化损失是通过均方误差（MSE）来计算的，并且乘以一个权重系数 beta
        # encoding_indices 是编码索引。它记录了每个输入向量对应的最近码本向量的索引。
        return z_q, loss, encoding_indices
    
