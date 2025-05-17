import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
from pathlib import Path
from timm.models import create_model
from PIL import PngImagePlugin
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from dataset.datasets import build_beit_pretraining_dataset
from modules.VITForMIM import BEiTLightningModule
import utils
import model.pretrained_model
import model.vqkd_model

PngImagePlugin.MAX_TEXT_CHUNK = 100 * (1024 ** 2)
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)


def get_args():
    parser = argparse.ArgumentParser('BEiT pre-training script', add_help=False)
    # 默认的命令行参数
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)
    parser.add_argument('--tokenizer_weight', type=str,default= r"/gz-data/beit2/image_tokenizer.pth")
    parser.add_argument('--tokenizer_model', type=str, default="vqkd_encoder_base_decoder_3x768x12_clip")
    parser.add_argument('--model', default='beit_base_patch16_384_8k_vocab_used', type=str)

    parser.add_argument('--num_mask_patches', default=230, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--mask_schedule_epochs',type=int,default=None,help='用于 mask 难度调度的 epoch 数，默认等于 warmup_epochs')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=75)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)
    parser.add_argument('--input_size', default=384, type=int,
                        help='images input size for backbone')
    
    parser.add_argument('--possible_input_sizes', type=int, nargs='+', default=None, # 或者 default=[224]
                        help='List of possible input sizes to randomly sample from during training. e.g., 192 224 256')
    parser.add_argument('--second_input_size', default=384, type=int,
                        help='images input size for discrete vae')
    
    # cls-pretraining settings
    parser.add_argument('--early_layers', default=9, type=int, help='early_layers, default 9 for base and 21 for large')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--shared_lm_head', default=True, type=utils.bool_flag, help='head_layers')

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

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    # 梯度累积
    parser.add_argument('--grad_accum_steps', type=int, default=1, 
                        help='Number of steps to accumulate gradients (default: 1)')
    # Augmentation parameters
    parser.add_argument('--decoupling_aug', default=False, type=utils.bool_flag, help="use decoupling aug for tokenizer and vit")
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--min_crop_scale', type=float, default=0.08, metavar='PCT',
                        help='min_crop_scale (default: 0.08)')
    
    # Dataset parameters
    parser.add_argument('--data_path', default=r'/gz-data/patchset', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='', type=str, help='dataset path')
    parser.add_argument('--data_set', default='image_folder',  type=str, help='dataset path')

    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')

    parser.add_argument('--output_dir', default=r'/gz-data/output/beit2', type=str,
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=r'/gz-data/log/beit2', type=str,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')    
    parser.add_argument('--num_workers', default=10, type=int)
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

def get_model(args):
    print(f"Creating model: {args.model}")
    if 'cls_pt' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            early_layers=args.early_layers,
            # 头部层的数量
            head_layers=args.head_layers,
            # 是否使用共享语言模型头部
            shared_lm_head=args.shared_lm_head,
            # 初始化参数
        )
    else:

        model = create_model(
            args.model,
            pretrained=False,
        )
    return model

def get_visual_tokenizer(args):
    # 打印正在创建的视觉分词器的名称
    print(f"Creating visual tokenizer: {args.tokenizer_model}")
    # 创建视觉分词器模型
    model = create_model(
        # 视觉分词器的模型名称
        args.tokenizer_model,
        # 是否使用预训练权重
        pretrained=True,
        # 预训练权重文件的路径
        pretrained_weight=args.tokenizer_weight,
        # 是否将模型作为分词器使用
        as_tokenzer=True,
        # 码本的大小
        n_code=args.codebook_size, 
        # 码本的维度
        code_dim=args.codebook_dim,
    # 将模型设置为评估模式
    ).eval()
    return model

def main(args):

    print(args)

    cudnn.benchmark = True

    # 根据命令行参数创建模型
    model = get_model(args)
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size, args.input_size // patch_size)
    args.patch_size = patch_size
    
    dataset_train = build_beit_pretraining_dataset(args)

    vqkd = get_visual_tokenizer(args)

    num_gpus_estimate = torch.cuda.device_count() if args.device == "cuda" and torch.cuda.is_available() else 1
    num_training_steps_per_epoch = len(dataset_train) // (args.batch_size * num_gpus_estimate * args.grad_accum_steps)
    if num_training_steps_per_epoch == 0 and len(dataset_train) > 0: # Handle small datasets
        num_training_steps_per_epoch = 1

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    num_training_steps_per_epoch = len(data_loader_train) // args.grad_accum_steps
    if num_training_steps_per_epoch == 0 and len(data_loader_train) > 0 :
        num_training_steps_per_epoch = 1

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
        opt_betas=tuple(args.opt_betas) if args.opt_betas else (0.9, 0.95), 
        num_training_steps_per_epoch=num_training_steps_per_epoch,
        weight_decay_end=args.weight_decay_end,
        momentum=args.momentum if 'sgd' in args.opt or 'momentum' in args.opt else None,
        layer_decay_rate=args.layer_decay_rate,
        embed_smooth_alpha=0.1,
    )

    tb_logger = None
    callbacks = []
    tb_logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name="beit2",      # 日志根目录名
        version=None       # 使用 Lightning 默认的自动增量 version_0, version_1...
    )

    ckpt_dir = os.path.join(args.output_dir, "checkpoints", f"version_{tb_logger.version}")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='beitv2-{epoch:02d}-{step:06d}',
            save_last=True, 
            every_n_epochs=args.save_ckpt_freq if args.save_ckpt_freq > 0 else 1,
            save_top_k=-1, 
        )
    
    callbacks.append(checkpoint_callback)

    # Add LearningRateMonitor if you're using it
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(GradientNormLogger(group_names=['patch_embed','blocks']))
    
    # Resume logic using Trainer's ckpt_path
    ckpt_path = None
    if args.resume: # Explicitly provided checkpoint
        ckpt_path = args.resume
    elif args.auto_resume:
        # Try to find 'last.ckpt' in the *latest* version directory if output_dir and log_dir are set
        if args.output_dir and args.log_dir:
            log_root = Path(args.log_dir) / "lightning_logs"
            if log_root.exists():
                versions = sorted([d for d in log_root.iterdir() if d.is_dir() and d.name.startswith("version_")],
                                  key=lambda p: int(p.name.split("_")[-1]), reverse=True)
                if versions:
                    latest_version_dir = versions[0]
                    last_ckpt_path = latest_version_dir / "checkpoints" / "last.ckpt"
                    if last_ckpt_path.exists():
                        ckpt_path = str(last_ckpt_path)
                        print(f"Auto-resuming from latest version's last.ckpt: {ckpt_path}")
                    else:
                        print(f"Auto-resume: Found latest version dir {latest_version_dir}, but no last.ckpt found there.")
                else:
                    print("Auto-resume: No versioned log directories found to resume from.")
            else:
                print(f"Auto-resume: Log directory {log_root} does not exist.")
        else:
            print("Auto-resume: --output_dir and --log_dir must be set to find the latest checkpoint.")

    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint path {ckpt_path} provided for resume does not exist. Starting from scratch.")
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
    )

    print(f"Starting training with PyTorch Lightning for {args.epochs} epochs...")
    if ckpt_path:
        print(f"Resuming training from checkpoint: {ckpt_path}")
    else:
        print("Starting training from scratch.")

    trainer.fit(
        model=lightning_module,
        train_dataloaders=data_loader_train,
        ckpt_path=ckpt_path # Handles resuming from a checkpoint
    )



if __name__ == '__main__':
    args = get_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    main(args)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Total script execution time: {total_time_str}')
