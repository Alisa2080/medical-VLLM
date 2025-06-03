import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
import models.pretrained_model
import models.vqkd_model
from pathlib import Path
from timm.models import create_model
from PIL import PngImagePlugin
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datasets.vision_datasets.datasets import build_beit_pretraining_dataset
from modules.VITForMIM import BEiTLightningModule
from utils.utils import bool_flag
from datasets.vision_datasets.datasets import DataAugmentationForBEiT

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

def get_args():
    parser = argparse.ArgumentParser('BEiT pre-training script', add_help=False)
    # 默认的命令行参数
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save_ckpt_freq', default=10, type=int)
    parser.add_argument('--tokenizer_weight', type=str,default= r"E:\article_code\weight\image_tokenizer.pth")
    parser.add_argument('--tokenizer_model', type=str, default="vqkd_encoder_base_decoder_3x768x12_clip")
    parser.add_argument('--model', default='VisionEncoder_base_patch16_384_8k', type=str)

    #掩码参数
    parser.add_argument('--num_mask_patches', default=230, type=int,
                        help='maximum number of patches to be masked (threshold)')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=75)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    
    # 更新预训练脚本参数
    parser.add_argument('--masking_strategy', type=str, default='pathology_aware',
                        choices=['random', 'block', 'pathology_aware', 'entropy_based', 'gradient_based'],
                        help='Masking strategy for pathology images')
    parser.add_argument('--foreground_bias', type=float, default=0.8,
                        help='Bias towards foreground regions in masking [0, 1]')
    parser.add_argument('--complexity_adaptive', action='store_true', default=True,
                        help='Adapt masking ratio based on image complexity')
    parser.add_argument('--curriculum_masking', action='store_true', default= True,
                        help='Use curriculum learning for masking difficulty')
    
    # 输入尺寸参数
    parser.add_argument('--input_size', default=384, type=int,
                        help='images input size for backbone')
    parser.add_argument('--possible_input_sizes', type=int, nargs='+', default=None, # 或者 default=[224]
                        help='List of possible input sizes to randomly sample from during training. e.g., 192 224 256')
    parser.add_argument('--second_input_size', default=384, type=int,
                        help='images input size for discrete vae')
    
    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='number of codebook')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=[0.9, 0.98], type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=3, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--layer_decay_rate', type=float, default= 0.8, 
                        help='Layer-wise learning rate decay rate (e.g., 0.75). If None, no layer decay.')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    # 学习率参数
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=2, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    
    # 梯度累积
    parser.add_argument('--grad_accum_steps', type=int, default=1, 
                        help='Number of steps to accumulate gradients (default: 1)')
    
    # 数据增强参数
    parser.add_argument('--decoupling_aug', default=False, type=bool_flag, help="use decoupling aug for tokenizer and vit")
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--min_crop_scale', type=float, default=0.08, metavar='PCT',
                        help='min_crop_scale (default: 0.08)')
    
    # 数据集参数
    parser.add_argument('--data_path', default=r'E:\datasets\medical_images\patches', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='', type=str, help='dataset path')
    parser.add_argument('--data_set', default='image_folder',  type=str, help='dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    # 输出和设备参数
    parser.add_argument('--output_dir', default=r'E:\datasets\medical_images\output\beit2', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=r'E:\datasets\medical_images\output\beit2', type=str,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', type=str,
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')    
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
 
    return parser.parse_args()


class GradientNormLogger(pl.Callback):
    def __init__(self, group_names):
        """
        group_names: list of module name 前缀, 比如 ['patch_embed','blocks']
        """
        super().__init__()
        self.group_names = group_names

    def on_before_optimizer_step(self, trainer, pl_module,optimizer):
        for g in self.group_names:
            norms = [
                p.grad.detach().norm().item()
                for n, p in pl_module.model.named_parameters()
                if n.startswith(g) and p.grad is not None
            ]
            if len(norms) == 0:
                continue
            avg_norm = sum(norms) / len(norms)
            pl_module.log(f"grad_norm/{g}", avg_norm,
                          on_step=True, on_epoch=True, prog_bar=False)

class PathologyMaskingCallback(pl.Callback):
    """病理掩码回调，用于更新掩码生成器的epoch信息"""
    
    def __init__(self, data_augmentation):
        super().__init__()
        self.data_augmentation = data_augmentation
    
    def on_train_epoch_start(self, trainer, pl_module):
        """在每个训练epoch开始时更新掩码生成器"""
        current_epoch = trainer.current_epoch
        
        # 更新数据增强器中的掩码生成器
        if hasattr(self.data_augmentation, 'update_epoch'):
            self.data_augmentation.update_epoch(current_epoch)
        
        # 更新掩码生成器的epoch信息
        if hasattr(self.data_augmentation, 'masking_generator'):
            if hasattr(self.data_augmentation.masking_generator, 'update_epoch'):
                self.data_augmentation.masking_generator.update_epoch(current_epoch)
        
        # 日志记录掩码策略信息
        strategy = getattr(self.data_augmentation.masking_generator, 'strategy', 'unknown')
        curriculum_masking = getattr(self.data_augmentation.masking_generator, 'curriculum_masking', False)
        curriculum_progress = getattr(self.data_augmentation.masking_generator, 'curriculum_progress', 0.0)
        max_masks = getattr(self.data_augmentation.masking_generator, 'max_masking_patches', 0)
        
        pl_module.log("masking/strategy", 
                      hash(strategy) % 1000,  # 简单的策略编码
                      on_epoch=True, prog_bar=False)
        pl_module.log("masking/curriculum_progress", 
                      curriculum_progress, 
                      on_epoch=True, prog_bar=False)
        pl_module.log("masking/max_allowed_masks", 
                      max_masks, 
                      on_epoch=True, prog_bar=False)
        
        # 记录当前epoch的预期掩码数量
        if curriculum_masking and hasattr(self.data_augmentation.masking_generator, '_get_curriculum_base_masks'):
            expected_masks = self.data_augmentation.masking_generator._get_curriculum_base_masks()
            pl_module.log("masking/expected_base_masks", expected_masks, on_epoch=True, prog_bar=False)
        else:
            # 非课程学习模式，记录最大掩码数量作为期望值
            pl_module.log("masking/expected_base_masks", max_masks, on_epoch=True, prog_bar=False)
        
        # 如果是课程学习，记录当前的掩码参数
        if curriculum_masking:
            adaptive_min, adaptive_max = self.data_augmentation.masking_generator._compute_adaptive_block_size()
            pl_module.log("masking/adaptive_min_patches", adaptive_min, on_epoch=True, prog_bar=False)
            pl_module.log("masking/adaptive_max_patches", adaptive_max, on_epoch=True, prog_bar=False)
        
        # 记录当前epoch的掩码策略
        if current_epoch == 0:
            print(f"\n=== Pathology Masking Configuration ===")
            print(f"Strategy: {strategy}")
            print(f"Curriculum masking: {curriculum_masking}")
            print(f"Max masking patches: {max_masks}")
            print(f"Foreground bias: {getattr(self.data_augmentation.masking_generator, 'foreground_bias', 'N/A')}")
            print(f"Complexity adaptive: {getattr(self.data_augmentation.masking_generator, 'complexity_adaptive', 'N/A')}")
            
            if curriculum_masking and hasattr(self.data_augmentation.masking_generator, 'curriculum_min_masks'):
                print(f"Curriculum min patches: {self.data_augmentation.masking_generator.curriculum_min_masks}")
                print(f"Masking will start low and gradually increase")
            else:
                variation = int(max_masks * 0.05)
                print(f"No curriculum learning - will use close to {max_masks} masks")
                print(f"Allowed range: {max_masks - variation} - {max_masks}")
            print("=====================================\n")
        
        # 每10个epoch记录一次进度
        if current_epoch % 10 == 0:
            if curriculum_masking and hasattr(self.data_augmentation.masking_generator, '_get_curriculum_base_masks'):
                current_base = self.data_augmentation.masking_generator._get_curriculum_base_masks()
                print(f"Epoch {current_epoch}: Curriculum progress = {curriculum_progress:.3f}, "
                      f"Expected base masks = {current_base}")
            else:
                print(f"Epoch {current_epoch}: Using close to max masks = {max_masks}")

class MaskingMetricsCallback(pl.Callback):
    """掩码质量监控回调 - 增强版"""
    
    def __init__(self):
        super().__init__()
        self.mask_stats = []
        self.complexity_stats = []
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录掩码统计信息"""
        if batch_idx % 50 == 0:  # 增加记录频率
            try:
                # 从batch中获取掩码信息
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    packed_data = batch[0]
                    if isinstance(packed_data, (list, tuple)) and len(packed_data) >= 3:
                        _, _, bool_masked_pos = packed_data
                        
                        if isinstance(bool_masked_pos, torch.Tensor):
                            # 处理不同的张量形状
                            original_shape = bool_masked_pos.shape
                            
                            if bool_masked_pos.dim() == 1:
                                # 计算每个样本的patch数
                                total_elements = bool_masked_pos.numel()
                                # 假设是24x24=576个patches per sample
                                patches_per_sample = 576
                                batch_size = total_elements // patches_per_sample
                                
                                if batch_size * patches_per_sample == total_elements:
                                    bool_masked_pos = bool_masked_pos.view(batch_size, patches_per_sample)
                                else:
                                    print(f"[WARNING] Cannot reshape mask tensor: total={total_elements}")
                                    return
                            
                            elif bool_masked_pos.dim() == 2:
                                batch_size, patches_per_sample = bool_masked_pos.shape
                            else:
                                print(f"[WARNING] Unexpected mask tensor dimensions: {bool_masked_pos.shape}")
                                return
                            
                            # 计算详细的掩码统计
                            mask_counts = bool_masked_pos.sum(dim=-1).cpu().numpy()  # 每个样本的掩码数量
                            mask_ratios = mask_counts.astype(float) / patches_per_sample
                            
                            # 统计指标
                            avg_mask_ratio = float(np.mean(mask_ratios))
                            std_mask_ratio = float(np.std(mask_ratios))
                            min_masks = int(np.min(mask_counts))
                            max_masks = int(np.max(mask_counts))
                            avg_masks = float(np.mean(mask_counts))
                            
                            # 计算掩码数量的分布
                            unique_counts, count_frequencies = np.unique(mask_counts, return_counts=True)
                            mask_diversity = len(unique_counts)  # 不同掩码数量的种类数
                            
                            # 记录到TensorBoard
                            pl_module.log("masking/avg_mask_ratio", avg_mask_ratio, 
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("masking/std_mask_ratio", std_mask_ratio, 
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("masking/min_mask_count", min_masks, 
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("masking/max_mask_count", max_masks, 
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("masking/avg_mask_count", avg_masks, 
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("masking/mask_diversity", mask_diversity, 
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("masking/mask_range", max_masks - min_masks, 
                                        on_step=True, on_epoch=True, prog_bar=False)
                            
                            # 详细调试信息
                            if batch_idx % 500 == 0:
                                print(f"\n[MASKING METRICS] Batch {batch_idx}:")
                                print(f"  Batch size: {batch_size}")
                                print(f"  Patches per sample: {patches_per_sample}")
                                print(f"  Mask counts: min={min_masks}, max={max_masks}, avg={avg_masks:.1f}")
                                print(f"  Mask ratios: min={np.min(mask_ratios):.3f}, max={np.max(mask_ratios):.3f}, avg={avg_mask_ratio:.3f}")
                                print(f"  Std deviation: {std_mask_ratio:.3f}")
                                print(f"  Mask diversity: {mask_diversity} different counts")
                                print(f"  Unique counts: {unique_counts[:10]}...")  # 显示前10个
                                
                                # 检查是否存在复杂度自适应
                                if std_mask_ratio < 0.001:
                                    print(f"  [ALERT] Very low mask variance detected!")
                                    print(f"  This suggests complexity adaptation may not be working properly.")
                                else:
                                    print(f"  [OK] Good mask variance detected, complexity adaptation appears active.")
                            
                            # 异常检测
                            if std_mask_ratio == 0.0 and batch_idx > 100:
                                print(f"[CRITICAL] Zero mask variance at batch {batch_idx}!")
                                print(f"This indicates all samples have identical mask counts: {mask_counts}")
                                print(f"Complexity adaptive masking may be disabled or malfunctioning.")
                        
            except Exception as e:
                print(f"[ERROR] Failed to log mask statistics: {e}")
                import traceback
                traceback.print_exc()

class ComplexityDistributionCallback(pl.Callback):
    """监控图像复杂度分布的回调"""
    
    def __init__(self):
        super().__init__()
        self.complexity_samples = []
        self.mask_count_samples = []
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录复杂度和掩码数量的关系"""
        if batch_idx % 200 == 0:  # 每200个batch检查一次
            try:
                # 模拟获取一些复杂度样本（实际中需要从数据增强器获取）
                # 这里我们记录当前batch的掩码统计作为代理
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    packed_data = batch[0]
                    if isinstance(packed_data, (list, tuple)) and len(packed_data) >= 3:
                        _, _, bool_masked_pos = packed_data
                        
                        if isinstance(bool_masked_pos, torch.Tensor):
                            if bool_masked_pos.dim() == 1:
                                patches_per_sample = 576
                                batch_size = bool_masked_pos.numel() // patches_per_sample
                                if batch_size * patches_per_sample == bool_masked_pos.numel():
                                    bool_masked_pos = bool_masked_pos.view(batch_size, patches_per_sample)
                                else:
                                    return
                            
                            mask_counts = bool_masked_pos.sum(dim=-1).cpu().numpy()
                            
                            # 计算复杂度分布统计
                            complexity_range = np.max(mask_counts) - np.min(mask_counts)
                            complexity_cv = np.std(mask_counts) / (np.mean(mask_counts) + 1e-8)  # 变异系数
                            
                            pl_module.log("complexity/mask_count_range", complexity_range,
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("complexity/mask_count_cv", complexity_cv,
                                        on_step=True, on_epoch=True, prog_bar=False)
                            
                            # 记录分位数
                            percentiles = np.percentile(mask_counts, [10, 25, 50, 75, 90])
                            pl_module.log("complexity/mask_p10", percentiles[0],
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("complexity/mask_p25", percentiles[1],
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("complexity/mask_p50", percentiles[2],
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("complexity/mask_p75", percentiles[3],
                                        on_step=True, on_epoch=True, prog_bar=False)
                            pl_module.log("complexity/mask_p90", percentiles[4],
                                        on_step=True, on_epoch=True, prog_bar=False)
                            
                            if batch_idx % 1000 == 0:
                                print(f"\n[COMPLEXITY DISTRIBUTION] Batch {batch_idx}:")
                                print(f"  Mask count range: {complexity_range}")
                                print(f"  Coefficient of variation: {complexity_cv:.3f}")
                                print(f"  Percentiles: {percentiles}")
                        
            except Exception as e:
                print(f"[ERROR] Complexity distribution logging failed: {e}")


def get_visual_tokenizer(args):

    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    # 创建视觉分词器模型
    model = create_model(
        args.tokenizer_model,
        pretrained=True,
        pretrained_weight=args.tokenizer_weight,
        as_tokenzer=True,
        n_code=args.codebook_size, 
        code_dim=args.codebook_dim,
    # 将模型设置为评估模式
    ).eval()
    return model

def print_masking_strategy_info(args):
    """打印掩码策略配置信息"""
    print("\n" + "="*60)
    print("PATHOLOGY MASKING STRATEGY CONFIGURATION")
    print("="*60)
    print(f"Strategy: {args.masking_strategy}")
    print(f"Max masking patches (threshold): {args.num_mask_patches}")
    print(f"Min patches per block: {args.min_mask_patches_per_block}")
    print(f"Max patches per block: {args.max_mask_patches_per_block}")
    print(f"Foreground bias: {args.foreground_bias}")
    print(f"Complexity adaptive: {args.complexity_adaptive}")
    print(f"Curriculum masking: {args.curriculum_masking}")
    
    if args.curriculum_masking:
        curriculum_min = max(args.min_mask_patches_per_block, int(args.num_mask_patches * 0.3))
        print(f"\nCurriculum learning configuration:")
        print(f"  - Initial masks: ~{curriculum_min} patches")
        print(f"  - Final masks: ~{args.num_mask_patches} patches")
        print(f"  - Progress over {int(args.epochs * 0.8)} epochs")
        print(f"  - All strategies will follow curriculum progression")
    else:
        print(f"\nNo curriculum learning:")
        print(f"  - All strategies will use close to {args.num_mask_patches} masks")
        print(f"  - Allowed variation: ±{int(args.num_mask_patches * 0.05)} patches")
        print(f"  - Range: {args.num_mask_patches - int(args.num_mask_patches * 0.05)} - {args.num_mask_patches}")
    
    if args.complexity_adaptive:
        print(f"\nComplexity adaptation enabled:")
        print(f"  - Complex images: reduce masks by up to 20%")
        print(f"  - Simple images: increase masks by up to 20%")
        print(f"  - Applied on top of base mask count")
    
    if args.masking_strategy == 'pathology_aware':
        print("\nPathology-aware masking features:")
        print("  - Foreground/background detection")
        print("  - Information entropy analysis")
        print("  - Gradient intensity analysis")
        print("  - Weighted block generation (70% blocks + 30% weighted random)")
    elif args.masking_strategy == 'entropy_based':
        print("\nEntropy-based masking:")
        print("  - Prioritizes high-entropy regions")
        print("  - Focuses on texturally complex areas")
    elif args.masking_strategy == 'gradient_based':
        print("\nGradient-based masking:")
        print("  - Prioritizes high-gradient regions")
        print("  - Focuses on edge-rich areas")
    elif args.masking_strategy == 'block':
        print("\nBlock masking:")
        print("  - Optimized block-wise masking")
        print("  - Maintains spatial coherence")
    else:
        print("\nRandom masking:")
        print("  - Pure random patch selection")
        print("  - No spatial or content bias")
    
    print("="*60 + "\n")

def main(args):

    print(args)
    cudnn.benchmark = True
    
    # 打印掩码策略信息
    print_masking_strategy_info(args)
    
    # 根据命令行参数创建模型
    model = create_model(args.model)
    patch_size = model.patch_embed.patch_size
    args.window_size = (args.input_size // patch_size, args.input_size // patch_size)
    args.patch_size = patch_size
    
        # 添加详细的参数验证和调试信息
    total_patches = args.window_size[0] * args.window_size[1]
    print(f"\n[DEBUG] Detailed Masking Configuration:")
    print(f"  Input size: {args.input_size}")
    print(f"  Patch size: {patch_size}")
    print(f"  Window size: {args.window_size}")
    print(f"  Total patches: {total_patches}")
    print(f"  Max mask patches: {args.num_mask_patches}")
    print(f"  Max mask ratio: {args.num_mask_patches / total_patches:.3f}")
    
    if args.curriculum_masking:
        curriculum_min = max(args.min_mask_patches_per_block, int(args.num_mask_patches * 0.3))
        print(f"  Curriculum min masks: {curriculum_min}")
        print(f"  Curriculum min ratio: {curriculum_min / total_patches:.3f}")
    
    if args.num_mask_patches >= total_patches:
        print(f"[ERROR] Max mask patches ({args.num_mask_patches}) >= total patches ({total_patches})")
        args.num_mask_patches = total_patches - 10  # 留一些余量
        print(f"[FIXED] Adjusted max mask patches to {args.num_mask_patches}")
   
    # 创建数据集和数据增强器
    print("\nBuilding dataset with pathology-specific augmentations...")
    dataset_train = build_beit_pretraining_dataset(args)

    # 获取数据增强器的引用，用于epoch更新
    data_augmentation = DataAugmentationForBEiT(args)
    print(f"Data augmentation initialized with masking strategy: {args.masking_strategy}")
    
    # 获取视觉分词器
    vqkd = get_visual_tokenizer(args)

    if args.device == "cuda" and torch.cuda.is_available():
        num_gpus_estimate = torch.cuda.device_count()
        print(f"Detected {num_gpus_estimate} GPUs")
    else:
        num_gpus_estimate = 1
        print("Using CPU or single device")

        # 创建数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
        shuffle=True,  # 确保数据被打乱
    )
    total_batch_size = args.batch_size * num_gpus_estimate * args.grad_accum_steps
    num_training_steps_per_epoch = len(data_loader_train) // args.grad_accum_steps
    
    if num_training_steps_per_epoch == 0: # Handle small datasets
        num_training_steps_per_epoch = 1
        print("Warning: Very small dataset detected, setting num_training_steps_per_epoch=1")
    
    
    
    print(f"\nTraining Configuration:")
    print(f"  Dataset size: {len(dataset_train)}")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Total batch size: {total_batch_size}")
    print(f"  Number of training steps per epoch: {num_training_steps_per_epoch}")
    print(f"  Total training steps: {num_training_steps_per_epoch * args.epochs}")

    # 创建Lightning模块
    lightning_module = BEiTLightningModule(
        model=model,
        vqkd=vqkd,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        optimizer_name=args.opt,
        opt_eps=args.opt_eps,
        opt_betas=list(args.opt_betas) if args.opt_betas else [0.9, 0.95], 
        num_training_steps_per_epoch=num_training_steps_per_epoch,
        weight_decay_end=args.weight_decay_end,
        momentum=args.momentum if 'sgd' in args.opt or 'momentum' in args.opt else None,
        layer_decay_rate=args.layer_decay_rate,
        embed_smooth_alpha=0.1,
    )
    # 设置日志和回调
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="beit2_pathology",      # 日志根目录名
        version=None       # 使用 Lightning 默认的自动增量 version_0, version_1...
    )

    # 创建检查点目录
    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"version_{tb_logger.version}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # 设置回调
    callbacks = []

    # 检查点回调
    checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='beitv2-pathology-{epoch:02d}-{step:06d}',
            save_last=True, 
            every_n_epochs=args.save_ckpt_freq if args.save_ckpt_freq > 0 else 1,
            save_top_k=-1, 
        )
    
    callbacks.append(checkpoint_callback)

    # 学习率监控
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # 梯度范数监控
    callbacks.append(GradientNormLogger(group_names=['patch_embed','blocks']))
    
    # 病理掩码回调（用于epoch更新）
    pathology_masking_callback = PathologyMaskingCallback(data_augmentation)
    callbacks.append(pathology_masking_callback)

    # 掩码质量监控回调
    masking_metrics_callback = MaskingMetricsCallback()
    callbacks.append(masking_metrics_callback)
    
    # 复杂度分布监控回调
    complexity_distribution_callback = ComplexityDistributionCallback()
    callbacks.append(complexity_distribution_callback)
    
    # 检查点恢复逻辑
    ckpt_path = None
    if args.resume: # Explicitly provided checkpoint
        ckpt_path = args.resume
    elif args.auto_resume:
        # 在当前输出目录中查找最新的检查点
        if args.output_dir:
            ckpt_dir = os.path.join(args.output_dir, "checkpoints")
            if os.path.exists(ckpt_dir):
                # 查找所有version目录
                version_dirs = [d for d in os.listdir(ckpt_dir) if d.startswith("version_")]
                if version_dirs:
                    # 按版本号排序，获取最新的
                    latest_version = sorted(version_dirs, key=lambda x: int(x.split("_")[1]))[-1]
                    last_ckpt = os.path.join(ckpt_dir, latest_version, "last.ckpt")
                    if os.path.exists(last_ckpt):
                        ckpt_path = last_ckpt
                        print(f"Auto-resuming from: {ckpt_path}")
                    else:
                        print(f"No last.ckpt found in {os.path.join(ckpt_dir, latest_version)}")
                else:
                    print("No version directories found for auto-resume")
            else:
                print(f"Checkpoint directory {ckpt_dir} does not exist")
    
    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint {ckpt_path} does not exist, starting from scratch")
        ckpt_path = None

    trainer = pl.Trainer(
        accelerator="gpu" if args.device == "cuda" and torch.cuda.is_available() else args.device,
        devices="auto" if args.device == "cuda" and torch.cuda.is_available() else 1,
        max_epochs=args.epochs,
        precision='16-mixed' if args.device == "cuda" and torch.cuda.is_available() else '32-true',
        accumulate_grad_batches=args.grad_accum_steps,
        callbacks=callbacks, 
        logger=tb_logger,
        gradient_clip_val=args.clip_grad if args.clip_grad > 0 else None,
        log_every_n_steps=min(50, num_training_steps_per_epoch) if num_training_steps_per_epoch > 0 else 1,
        gradient_clip_algorithm="norm",
        val_check_interval=None,  # 禁用验证
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=False,  # 在生产环境中关闭异常检测以提高性能
    )

    # 开始训练
    print(f"\n{'='*60}")
    print(f"STARTING PATHOLOGY-SPECIFIC BEIT PRETRAINING")
    print(f"{'='*60}")
    print(f"Training epochs: {args.epochs}")
    print(f"Masking strategy: {args.masking_strategy}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log directory: {args.log_dir}")

    if ckpt_path:
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("Starting training from scratch.")
    
    print(f"{'='*60}\n")

    # 保存训练配置
    config_path = os.path.join(ckpt_dir, "training_config.json")
    config_dict = vars(args).copy()
    # 转换不可序列化的对象
    for key, value in config_dict.items():
        if not isinstance(value, (str, int, float, bool, list, type(None))):
            config_dict[key] = str(value)
    
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Training configuration saved to: {config_path}")

    # 开始训练
    trainer.fit(
        model=lightning_module,
        train_dataloaders=data_loader_train,
        ckpt_path=ckpt_path # Handles resuming from a checkpoint
    )

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Final model saved to: {ckpt_dir}")
    print(f"Logs available at: {tb_logger.log_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    main(args)
