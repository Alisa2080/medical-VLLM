import argparse
import os
import time
import datetime
import json
import random
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import PngImagePlugin 
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# Your custom modules

from dataset.WSIDataModule import WSIDataModule
from model.VIT_CLassifer import ViTClassifier, MILFineTuningModule 

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
torch.set_float32_matmul_precision('high')

def get_args_parser():
    parser = argparse.ArgumentParser('MIL ViT Classifier Training with PyTorch Lightning', add_help=False)
    
    # Data parameters
    parser.add_argument('--data_csv', type=str, default=r"F:\dataset\CLAM-WSI\tumor_segment\process_list_autogen_train.csv", help='Path to the main CSV file with slide_ids and labels')
    parser.add_argument('--train_csv', type=str, default=None, help='Path to the training CSV file (overrides data_csv for training)')
    parser.add_argument('--val_csv', type=str, default=r"F:\dataset\CLAM-WSI\tumor_segment\process_list_autogen_val.csv", help='Path to the validation CSV file')
    parser.add_argument('--test_csv', type=str, default=None, help='Path to the test CSV file')
    parser.add_argument('--patches_root_dir', type=str, default=r"F:\dataset\CLAM-WSI\tumor_segment\patches", help='Root directory for WSI patch folders')
    parser.add_argument('--model_input_size', default=384, type=int, help='Input patch size for ViT')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of workers for the main WSI DataLoader (WSIDataModule)')
    parser.add_argument('--patch_loader_num_workers', default=2, type=int, help='Number of workers for the internal patch DataLoader (in MILFineTuningModule)')
    
    # Model parameters
    parser.add_argument('--model_name', default='beit_base_patch16_384_8k_vocab_used', type=str, metavar='MODEL', help='Name of model to create')
    parser.add_argument('--pretrained_path', default=r"E:\BaiduNetdiskDownload\checkpoints\version_0\last.ckpt", type=str, help='Path to pretrained ViT weights')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for MIL task')
    parser.add_argument('--embed_dim', default=512, type=int, help='ViT output embedding dimension (auto-detected if None)')
    parser.add_argument('--mil_hidden_dim_att', default=256, type=int, help='Hidden dimension for GatedAttention')
    parser.add_argument('--mil_intermediate_dim', default=384, type=int, help='Intermediate dimension for GatedAttention')
    parser.add_argument('--freeze', type=bool, default=False, help='Freeze ViT encoder weights')

    # LDAM + DRW parameters
    parser.add_argument('--ldam_C_factor', type=float, default=1.0, help='LDAM loss margin scaling factor')
    parser.add_argument('--drw_start_epoch', type=int, default=None, help='Epoch to start DRW (switch to CB loss). Default: 80% of total epochs')
    parser.add_argument('--cb_beta', type=float, default=0.999, help='Beta parameter for class-balanced loss effective number calculation')
    
    
    # Training parameters (PyTorch Lightning Trainer)
    parser.add_argument('--epochs', default=30, type=int, help="Max number of epochs for PyTorch Lightning Trainer")
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Gradient accumulation steps (handled by Trainer)')
    parser.add_argument('--patch_batch_size', default=4, type=int, help='Batch size for processing patches within a WSI (in MILFineTuningModule)')
    parser.add_argument('--output_dir', default=r"E:\article_code\output\beit2\finetuning_pl", type=str, help='Base path to save logs and checkpoints')
    parser.add_argument('--log_dir_name', default="lightning_logs", type=str, help="Name of the subdirectory for TensorBoard logs within output_dir")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--precision', default='16-mixed', type=str, help='Precision for training (e.g., 32-true, 16-mixed, bf16-mixed)')
    parser.add_argument('--accelerator', default='gpu', type=str, help='Accelerator to use (e.g., cpu, gpu, tpu)')
    parser.add_argument('--devices', default='auto', help='Devices to use (e.g., 1, [0, 1], "auto")')
    parser.add_argument('--save_ckpt_freq', default=5, type=int, help='Save checkpoint every N epochs. Set to 0 or -1 to disable epoch-based saving (only last.ckpt).')
    parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='Path to a specific checkpoint to resume from.')
    parser.add_argument('--auto_resume', action='store_true', help='Automatically resume from the last checkpoint in the log directory.')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    # Optimizer parameters (passed to MILFineTuningModule)
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw")')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon')
    parser.add_argument('--opt_betas', default=[0.9, 0.98], type=float, nargs='+', metavar='BETA', help='Optimizer Betas')
    parser.add_argument('--clip_grad', type=float, default=3.0, metavar='NORM', help='Clip gradient norm (0 for no clipping, handled by Trainer)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay for ViT encoder')
    parser.add_argument('--weight_decay_mil', type=float, default=1e-4, help='Weight decay for MIL aggregator part')
    parser.add_argument('--layer_decay_rate', type=float, default=0.8, help='Layer-wise LR decay for ViT. If 1.0, no layer decay.')

    # Learning rate scheduler parameters (passed to MILFineTuningModule)
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='Learning rate (base_lr)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR', help='Warmup learning rate')
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR', help='Lower LR bound for scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N', help='Epochs to warmup LR')
    # 新增智能采样相关参数
    parser.add_argument('--use_smart_sampling', action='store_true', 
                        help='Use smart patch sampling based on quality scores')
    parser.add_argument('--quality_scores_dir', type=str, default=r"F:\dataset\CLAM-WSI\tumor_segment\patch_score",
                        help='Directory containing patch quality assessment results')
    parser.add_argument('--sampling_strategy', type=str, default='hybrid',
                        choices=['quality_based', 'stratified', 'diversity_aware', 'hybrid'],
                        help='Patch sampling strategy for smart sampling')
    parser.add_argument('--max_patches_per_wsi', type=int, default=1000,
                        help='Maximum number of patches to sample per WSI')
    parser.add_argument('--min_patches_per_wsi', type=int, default=300,
                        help='Minimum number of patches to sample per WSI')
    parser.add_argument('--quality_threshold', type=float, default=0.3,
                        help='Quality threshold for patch filtering')
    return parser

class GradientNormLogger(pl.Callback):
    def __init__(self, group_names, log_individual_layers=False):
        """
        group_names: list of module name prefixes, e.g., ['encoder.patch_embed', 'encoder.blocks', 'mil_aggregator']
        log_individual_layers: bool, if True, logs norm for each layer within a block group.
        """
        super().__init__()
        self.group_names = group_names
        self.log_individual_layers = log_individual_layers

    def on_before_optimizer_step(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", optimizer):
        if trainer.global_step % trainer.log_every_n_steps == 0: # Log at the same frequency as other logs
            for group_prefix in self.group_names:
                norms = []
                # For logging individual layers within a block group
                layer_norms = {}

                for n, p in pl_module.model.named_parameters():
                    if p.grad is not None and n.startswith(group_prefix):
                        norm = p.grad.detach().norm().item()
                        norms.append(norm)

                        if self.log_individual_layers and group_prefix.endswith("blocks"): # Specific to 'blocks'
                            # Try to extract layer index, e.g., blocks.0.attn -> blocks.0
                            layer_name_parts = n.split('.')
                            if len(layer_name_parts) > 2 and layer_name_parts[0] + '.' + layer_name_parts[1] == group_prefix: # e.g., encoder.blocks
                                block_layer_idx_str = layer_name_parts[2] # Should be the block index
                                if block_layer_idx_str.isdigit():
                                    layer_specific_name = f"{group_prefix}.{block_layer_idx_str}"
                                    if layer_specific_name not in layer_norms:
                                        layer_norms[layer_specific_name] = []
                                    layer_norms[layer_specific_name].append(norm)
                
                if norms:
                    avg_norm = sum(norms) / len(norms)
                    pl_module.log(f"grad_norm_avg/{group_prefix}", avg_norm,
                                  on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

                if self.log_individual_layers:
                    for ln, l_norms in layer_norms.items():
                        if l_norms:
                            avg_ln_norm = sum(l_norms) / len(l_norms)
                            pl_module.log(f"grad_norm_avg/{ln}", avg_ln_norm,
                                          on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
def main(args):
    print("Starting PyTorch Lightning MIL Fine-tuning script...")
    print(args)

    pl.seed_everything(args.seed, workers=True)

    
     # 1. 实例化数据模块
    if args.use_smart_sampling:
        if not args.quality_scores_dir:
            raise ValueError("--quality_scores_dir must be provided when using smart sampling")
        
        if not os.path.exists(args.quality_scores_dir):
            raise ValueError(f"Quality scores directory not found: {args.quality_scores_dir}")
        
        print(f"Using Smart WSI Data Module with sampling strategy: {args.sampling_strategy}")
        
        # 导入SmartWSIDataModule
        from dataset.WSIDataModule_Smart import SmartWSIDataModule

        data_module = SmartWSIDataModule(
            data_csv=args.data_csv,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            patches_root_dir=args.patches_root_dir,
            model_input_size=args.model_input_size,
            train_batch_size=1,
            val_batch_size=1,
            num_workers=args.num_workers,
            # Smart sampling specific parameters
            quality_scores_dir=args.quality_scores_dir,
            sampling_strategy=args.sampling_strategy,
            max_patches_per_wsi=args.max_patches_per_wsi,
            min_patches_per_wsi=args.min_patches_per_wsi,
            quality_threshold=args.quality_threshold
        )
    else:
    # 2. Instantiate WSIDataModule
        data_module = WSIDataModule(
            data_csv=args.data_csv,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            test_csv=args.test_csv,
            patches_root_dir=args.patches_root_dir,
            model_input_size=args.model_input_size,
            train_batch_size=1, # Outer DataLoader batch_size is always 1 for WSI
            val_batch_size=1,   # Outer DataLoader batch_size is always 1 for WSI
            num_workers=args.num_workers
        )
    
    # 2. Setup数据模块并获取类别信息
    data_module.setup('fit') 
    class_info = data_module.get_class_info()
    samples_per_cls = class_info['samples_per_cls']
    num_classes = class_info['num_classes']
    print(f"Class information: {class_info}")

    # 3. 设置DRW切换点
    if args.drw_start_epoch is None:
        args.drw_start_epoch = int(args.epochs * 0.8)  # 默认在80%的epoch处切换
    print(f"DRW strategy: LDAM Loss for epochs 0-{args.drw_start_epoch-1}, CB Loss for epochs {args.drw_start_epoch}-{args.epochs-1}")
    

    # 4. 实例化ViT模型
    vit_model_instance = ViTClassifier(
        model_name=args.model_name,
        pretrained_path=args.pretrained_path,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        hidden_dim_att=args.mil_hidden_dim_att,
        intermediate_dim=args.mil_intermediate_dim,
        freeze=args.freeze
    )

    # 5. 计算优化器步数
    try:
        num_optimizer_steps_per_epoch = len(data_module.train_dataloader()) // args.grad_accum_steps
        if num_optimizer_steps_per_epoch == 0 and len(data_module.train_dataloader()) > 0:
            num_optimizer_steps_per_epoch = 1
    except Exception as e:
        print(f"Could not determine num_optimizer_steps_per_epoch from dataloader: {e}. Defaulting to 1. Ensure train_dataset is not empty.")
        num_optimizer_steps_per_epoch = 1 # Fallback, though this might cause issues if dataloader is truly empty

    if num_optimizer_steps_per_epoch == 0:
         print("Warning: num_optimizer_steps_per_epoch is 0. This might happen if the training dataset is empty or too small. Training might not proceed correctly.")

    # 6. 实例化Lightning模块
    lightning_module = MILFineTuningModule(
        model_instance=vit_model_instance,
        # LDAM + DRW parameters
        samples_per_cls=samples_per_cls,
        ldam_C_factor=args.ldam_C_factor,
        drw_start_epoch=args.drw_start_epoch,
        cb_beta=args.cb_beta,
        # Optimizer hparams
        opt=args.opt,
        opt_eps=args.opt_eps,
        opt_betas=args.opt_betas,
        momentum=args.momentum,
        # LR and WD hparams
        base_lr=args.lr,
        weight_decay=args.weight_decay, # For ViT
        weight_decay_mil=args.weight_decay_mil,
        layer_decay_rate=args.layer_decay_rate,
        # Scheduler hparams
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        # Data/Model specific hparams
        patch_batch_size=args.patch_batch_size,
        patch_loader_num_workers=args.patch_loader_num_workers,
        model_input_size=args.model_input_size,
        num_optimizer_steps_per_epoch=num_optimizer_steps_per_epoch,
        # Misc model hparams
        freeze=args.freeze, # Passed to LightningModule's hparams
        num_classes=args.num_classes
    )

     # 7. 配置Logger和Callbacks
    tb_logger = None
    callbacks_list = []

    log_dir_base = Path(args.output_dir) / args.log_dir_name
    tb_logger = TensorBoardLogger(
        save_dir=str(log_dir_base.parent), # save_dir is the root for 'lightning_logs' or custom name
        name=log_dir_base.name,       
        version=None # Auto-increment version
    )
    
    ckpt_base_dir = Path(args.output_dir) / "mil_checkpoints"
    run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}" if tb_logger.version is None else f"version_{tb_logger.version}"
    checkpoint_dir = ckpt_base_dir / run_name
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename='mil-model-{epoch:02d}-{step:06d}',
        save_last=True,
        every_n_epochs=args.save_ckpt_freq if args.save_ckpt_freq > 0 else None, # None means only save_last if >0
        save_top_k=-1, # Save all checkpoints if every_n_epochs is used, or just last if every_n_epochs is None/0
        verbose=True
    )

    callbacks_list.append(checkpoint_callback)
    callbacks_list.append(LearningRateMonitor(logging_interval='step'))
    grad_norm_logger_groups = ['encoder.patch_embed', 'encoder.blocks', 'mil_aggregator']
    callbacks_list.append(GradientNormLogger(group_names=grad_norm_logger_groups, log_individual_layers=True))
    
     # 8. 处理checkpoint恢复
    ckpt_path_to_resume = None
    if args.resume_from_checkpoint:
        ckpt_path_to_resume = args.resume_from_checkpoint
        if not os.path.exists(ckpt_path_to_resume):
            print(f"Warning: Explicit resume checkpoint {ckpt_path_to_resume} not found. Starting fresh.")
            ckpt_path_to_resume = None
    elif args.auto_resume:
        potential_last_ckpt = checkpoint_dir / "last.ckpt"
        if potential_last_ckpt.exists():
            ckpt_path_to_resume = str(potential_last_ckpt)
            print(f"Auto-resuming from: {ckpt_path_to_resume}")
        else:
            # More advanced: search all version folders in log_dir_base for the true last.ckpt
            # For now, this basic auto-resume is fine.
            print(f"Auto-resume: No 'last.ckpt' found in the current run's checkpoint directory: {checkpoint_dir}. Starting fresh.")
    run_validation = data_module.val_dataloader() is not None
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices if args.devices != "auto" else "auto",
        max_epochs=args.epochs,
        precision=args.precision,
        accumulate_grad_batches=args.grad_accum_steps,
        callbacks=callbacks_list,
        logger=tb_logger,
        gradient_clip_val=args.clip_grad if args.clip_grad > 0 else None,
        gradient_clip_algorithm="norm",
        log_every_n_steps=min(50, num_optimizer_steps_per_epoch) if num_optimizer_steps_per_epoch > 0 else 1,
        num_sanity_val_steps=2 if run_validation else 0,
        deterministic=False,
    )
        # 7. Start training
    print(f"Starting training with PyTorch Lightning for {args.epochs} epochs...")
    if ckpt_path_to_resume:
        print(f"Resuming training from checkpoint: {ckpt_path_to_resume}")
    else:
        print("Starting training from scratch.")
    
    trainer.fit(
        model=lightning_module,
        datamodule=data_module,
        ckpt_path=ckpt_path_to_resume
    )


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    main(args)