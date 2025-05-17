import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import partial, reduce
from einops import rearrange
from timm.layers import trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import register_model

from modules.VisionTransformer import VisionTransformer
from norm_ema_quantizer import NormEMAVectorQuantizer
from modules.teachermodel_image_preprocess import ScalingLayerForClip, ScalingLayerForIM
import utils


class VQKD(nn.Module):
   def __init__(self,
                encoder_config,
                decoder_config,
                n_embed=8192, 
                embed_dim=32,
                decay=0.99,
                process_type='default',
                quantize_kmeans_init=True,
                teacher_model_type='clip',
                decoder_out_dim=1024,
                rec_loss_type='cosine',
                **kwargs
                ):
       super().__init__()
       print(kwargs)
       if decoder_config['in_chans'] != embed_dim:
           print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
           decoder_config['in_chans'] = embed_dim
       
       # encoder & decode params
       print('Final encoder config', encoder_config)
       self.encoder = VisionTransformer(**encoder_config)
       print('Final decoder config', decoder_config)
       self.decoder = VisionTransformer(**decoder_config)
               
       self.quantize = NormEMAVectorQuantizer(
           n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
       )
       
       self.patch_size = encoder_config['patch_size']
       self.token_shape = (encoder_config['img_size'] // self.patch_size, encoder_config['img_size'] // self.patch_size)
       self.decoder_out_dim = 1024
    
       self.teacher_model = None
       # 冻结教师模型的参数，切换为评估模式
       if self.teacher_model is not None:
           for param in self.teacher_model.parameters():
               param.requires_grad = False # fix teacher_model model
           self.teacher_model.eval()
           self.teacher_input_size = kwargs.get('teacher_input_size', 224)
       # task layer
       self.encode_task_layer = nn.Sequential(
           nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
           nn.Tanh(),
           nn.Linear(encoder_config['embed_dim'], embed_dim) # for quantize
       )
       self.decode_task_layer = nn.Sequential(
           nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
           nn.Tanh(),
           nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
       )
       
       self.rec_loss_type = rec_loss_type # cosine
       print(f"process type for VQKD: {process_type}")
       self.process_type = process_type # in ['default', 'dall-e']
       self.logit_laplace_eps = 0.1
       self.kwargs = kwargs
       
       self.encode_task_layer.apply(self._init_weights)
       self.decode_task_layer.apply(self._init_weights)
   
   def _init_weights(self, m):
       if isinstance(m, nn.Linear):
           trunc_normal_(m.weight, std=.02)
           if isinstance(m, nn.Linear) and m.bias is not None:
               nn.init.constant_(m.bias, 0)
       elif isinstance(m, nn.LayerNorm):
           nn.init.constant_(m.bias, 0)
           nn.init.constant_(m.weight, 1.0)
           
   @torch.jit.ignore
   def no_weight_decay(self):
       return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 
               'encoder.cls_token', 'encoder.pos_embed'}
   @property
   def device(self):
       return self.decoder.cls_token.device
   def pre_process(self, data):
       """
       对输入数据进行预处理。
       参数:
       data (torch.Tensor): 输入的图像数据。
       返回:
       torch.Tensor: 预处理后的图像数据。
       """
       if self.process_type == 'default':
           # TODO: modify for adapt
           # 将数据移动到模型所在的设备上
           data = data.to(self.device)
           # 如果数据的最大值小于等于 1，则将数据乘以 255 以将像素值范围从 [0, 1] 扩展到 [0, 255]
           if data.max() <= 1.:
               data = data * 255.
           # 将数据的像素值范围从 [0, 255] 缩放到 [-1, 1]
           data = data / 127.5 - 1.0
       elif self.process_type == 'imagenet_norm':
           # 将 ImageNet 的均值转换为张量并移动到模型所在的设备上
           mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :, None, None]
           # 将 ImageNet 的标准差转换为张量并移动到模型所在的设备上
           std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :, None, None]
           # 对数据进行归一化处理，减去均值并除以标准差
           data = (data - mean) / std
       return data
       
   def get_number_of_tokens(self):
       return self.quantize.n_e
   def get_tokens(self, data, **kwargs):
       """
       该方法用于从输入数据中获取码本索引，并将其整理成适合输出的格式。
       参数:
       data (torch.Tensor): 输入的图像数据，形状通常为 [B, 3, H, W]。
       **kwargs: 其他可选参数。
       返回:
       dict: 包含码本索引和预处理后输入图像的字典。
       """
       # 对输入数据进行预处理，使其符合模型的输入要求
       data = self.pre_process(data)
       # 对预处理后的数据进行编码，得到量化后的特征、码本索引和量化损失
       quantize, embed_ind, loss = self.encode(data)
       # 初始化一个空字典，用于存储输出结果
       output = {}
       # 将码本索引重塑为 [B, -1] 的形状，并存储到输出字典中
       output['token'] = embed_ind.view(data.shape[0], -1)
       # 将预处理后的输入图像存储到输出字典中
       output['input_img'] = data
       return output
   def encode(self, x):
       """
       对输入数据进行编码操作，将输入数据转换为量化后的特征、对应的码本索引，并计算量化损失。
       参数:
       x (torch.Tensor): 输入的图像数据张量。
       返回:
       tuple: 包含量化后的特征、码本索引和量化损失的元组。
       """
       # 调用编码器对输入数据 x 进行编码，并返回所有块的特征标记
       encoder_features = self.encoder(x, return_patch_tokens=True)
       # 使用 torch.cuda.amp.autocast 上下文管理器暂时禁用自动混合精度
       # 确保在执行 encode_task_layer 时使用全精度计算，避免精度损失
       with torch.amp.autocast(enabled=False,device_type='cuda'):
           # 将编码器输出的特征转换为与 encode_task_layer 最后一层权重相同的数据类型
           # 然后通过 encode_task_layer 对特征进行进一步处理，使其适合量化器的输入
           to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))
       # 获取处理后特征的第二个维度的大小，即特征标记的数量
       N = to_quantizer_features.shape[1]
       # 假设特征标记可以排列成一个正方形的网格，计算网格的高度和宽度
       h, w = int(math.sqrt(N)), int(math.sqrt(N))
       # 使用 einops 库的 rearrange 函数将特征张量从 [b, (h w), c] 形状重塑为 [b, c, h, w] 形状
       # 以适应量化器的输入要求
       to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
       # 调用量化器对重塑后的特征进行量化操作
       # quantize 是量化后的特征张量
       # loss 是量化过程中产生的损失
       # embed_ind 是量化特征对应的码本索引
       quantize, loss, embed_ind = self.quantize(to_quantizer_features)
       return quantize, embed_ind, loss
   
   def decode(self, quantize, **kwargs):
       """
       对量化后的特征进行解码操作，将量化特征转换为重建数据。
       参数:
       quantize (torch.Tensor): 量化后的特征张量。
       **kwargs: 其他可选参数。
       返回:
       torch.Tensor: 解码后得到的重建数据。
       """
       # reshape tokens to feature maps for patch embed in decoder
       # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
       decoder_features = self.decoder(quantize, return_patch_tokens=True)
       rec = self.decode_task_layer(decoder_features)
       return rec
   
   def get_codebook_indices(self, x, **kwargs):
       # for beit pre-training
       return self.get_tokens(x, **kwargs)['token']
   @torch.no_grad()
   def get_regress_target(self, x, **kwargs):
       """
       获取回归目标，用于计算重建损失。
       参数:
       x (torch.Tensor): 输入的图像数据，经过预处理后的图像。
       **kwargs: 其他可选参数。
       返回:
       torch.Tensor: 回归目标，用于后续的重建损失计算。
       """
       norm_imgs = self.scaling_layer(x)
       if self.teacher_model_type == 'clip':
           #torch.Size([2, 196, 512])
           target = self.teacher_model.encode_image(norm_imgs, return_all_tokens=True) @ self.teacher_model.visual.proj
       elif self.teacher_model_type == 'dino':
           target = self.teacher_model.forward(norm_imgs, return_patch_tokens=True)
       else:
           raise NotImplementedError
       return target
   def calculate_rec_loss(self, rec, target):  
       """
       计算重建损失，根据 `self.rec_loss_type` 选择不同的损失计算方式。
       参数:
       rec (torch.Tensor): 模型解码后得到的重建数据。
       target (torch.Tensor): 回归目标数据，用于与重建数据对比计算损失。
       返回:
       torch.Tensor: 计算得到的重建损失。
       
       """
       if self.rec_loss_type == 'cosine':
           target = target / target.norm(dim=-1, keepdim=True)
           rec = rec / rec.norm(dim=-1, keepdim=True)
           rec_loss = (1 - (target * rec).sum(-1)).mean()
       else:
           raise NotImplementedError
       return rec_loss
   def forward(self, x, **kwargs):
       """
       前向传播方法，定义了模型的前向计算过程。
       参数:
       x (torch.Tensor): 输入的图像数据，形状为 [B, 3, H, W]，像素值范围在 [0, 1] 之间。
       **kwargs: 其他可选参数。
       返回:
       tuple: 包含总损失和日志信息的元组。
       """
       # 对输入数据进行预处理，将像素值范围从 [0, 1] 重新缩放到 [-1, 1]
       x = self.pre_process(x) # rescale to [-1, 1]
       # 获取回归目标，用于计算重建损失
       target = self.get_regress_target(x, **kwargs)
       # 对预处理后的输入数据进行编码，得到量化后的特征、码本索引和量化损失
       # torch.Size([2, 32, 14, 14])
       quantize, embed_ind, emb_loss = self.encode(x)
       # 对量化后的特征进行解码，得到重建的图像数据
       xrec = self.decode(quantize)
       # 计算重建损失，衡量重建图像与目标图像之间的差异
       rec_loss = self.calculate_rec_loss(xrec, target)
       # 总损失为量化损失和重建损失之和
       loss = emb_loss + rec_loss
       # 初始化一个空字典，用于存储日志信息
       log = {}
       # 根据模型的训练状态确定当前的阶段（训练或验证）
       split="train" if self.training else "val"
       # 将量化损失的均值添加到日志中，使用 detach() 方法避免梯度传播
       log[f'{split}/quant_loss'] = emb_loss.detach().mean()
       # 将重建损失的均值添加到日志中，使用 detach() 方法避免梯度传播
       log[f'{split}/rec_loss'] = rec_loss.detach().mean()
       # 将总损失的均值添加到日志中，使用 detach() 方法避免梯度传播
       log[f'{split}/total_loss'] = loss.detach().mean()
       return loss, log
