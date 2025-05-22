import torch
import argparse
import torch.nn as nn
from modules.VITForMIM import VisionTransformerForMaskedImageModeling 
from modules.AttentionSeries import GatedAttention 
from timm.models import create_model
from typing import Optional, Tuple,Callable, Iterable
from model import pretrained_model
from PIL import Image 
import pytorch_lightning as pl
from dataset.WSIBagDatasetMTL import WSIBagDatasetMIL, SinglePatchDataset
from torch.utils.data import DataLoader
from modules.optim_factory import create_optimizer, get_parameter_groups
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class ViTClassifier(nn.Module):
    def __init__(self,
                 model_name: str, # 用于从timm或本地加载ViT模型的名称或配置
                 pretrained_path: Optional[str] = None, # 预训练权重的路径
                 num_classes: int = 2, # MIL分类任务的类别数
                 embed_dim: Optional[int] = None, # ViT输出的特征维度
                 hidden_dim_att: int = 384,
                 intermediate_dim: int = 512,
                 freeze: bool = False, # 是否冻结ViT的权重
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_vit_flag = freeze
        self.encoder = create_model(model_name, pretrained=False,)

        # 获取ViT的输出维度
        # 如果vit_params中没有显式给出dim，并且模型有dim属性
        if embed_dim is None:
            if hasattr(self.encoder, 'dim'):
                self.embed_dim = self.encoder.dim
            else:
                raise ValueError("Cannot automatically determine embed_dim. Please provide it.")
        else:
            self.embed_dim = embed_dim
        
        print(f"ViT Initialized. Expected feature dimension (embed_dim): {self.embed_dim}")

        if pretrained_path:
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                # 根据您的权重文件结构加载权重
                # 可能是 'model', 'state_dict', 或者直接是权重字典
                # PyTorch Lightning saves model weights under 'state_dict'
                if 'state_dict' in checkpoint:
                    raw_state_dict = checkpoint['state_dict']
                else:
                    # Fallback for other checkpoint types
                    raw_state_dict = checkpoint.get('model', checkpoint)
                
                # 处理可能的 'module.' 前缀 (来自DataParallel/DDP训练)
                processed_state_dict = {}
                for k, v in raw_state_dict.items():
                    name = k
                    if name.startswith('model.'): # Strip 'model.' prefix from BEiTLightningModule
                        name = name[len('model.'):]
                    if name.startswith('module.'): # Strip 'module.' prefix if trained with DDP/DP
                        name = name[len('module.'):]
                    processed_state_dict[name] = v

                encoder_model_dict = self.encoder.state_dict()

                final_state_dict_to_load = {
                    k: v for k, v in processed_state_dict.items() 
                    if k in encoder_model_dict and encoder_model_dict[k].shape == v.shape
                }
                missing_keys, unexpected_keys = self.encoder.load_state_dict(final_state_dict_to_load, strict=False)
                print(f"Successfully loaded ViT weights from {pretrained_path}")
                if missing_keys:
                    print(f"Missing keys in ViT: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in ViT: {unexpected_keys}")

            except Exception as e:
                print(f"Error loading ViT pretrained weights from {pretrained_path}: {e}")
        else:
            print("No ViT pretrained path provided, ViT will be initialized randomly (or by its own init).")

        # 冻结ViT层 (根据您的要求，先不冻结)
        if self.freeze_vit_flag:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("ViT encoder weights are frozen.")
        else:
            print("ViT encoder weights are NOT frozen (all layers will be fine-tuned).")

        # 2. MIL聚合模块
        self.mil_aggregator = GatedAttention(
            input_dim=self.embed_dim,
            num_classes=self.num_classes,
            hidden_dim_att=hidden_dim_att,
            intermediate_dim=intermediate_dim
        )

    def forward(self, 
                patch_batch_iterable: Iterable[torch.Tensor], # Expects an iterable of patch batches
                device: torch.device
                ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
        """
        Processes all patches for a single WSI (provided as an iterable of batches), 
        extracts features, and performs MIL aggregation.

        Args:
            patch_batch_iterable (Iterable[torch.Tensor]): An iterable (e.g., a DataLoader) 
                                                           that yields batches of transformed patch tensors.
            device (torch.device): The device to perform computations on for the MIL aggregator.
                                   Patch tensors from iterable should already be on this device.

        Returns:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], bool]:
                - logits (torch.Tensor or None): Classification logits.
                - attention_scores (torch.Tensor or None): Attention weights.
                - success (bool): True if processing was successful.
        """
        all_cls_tokens_for_wsi = []
        processed_any_batch = False
        for patch_tensor_batch in patch_batch_iterable: # Iterate through batches from the patch_loader
            if patch_tensor_batch is None or patch_tensor_batch.numel() == 0: # Skip if a batch is empty or None (e.g. error handling in SinglePatchDataset)
                print("ViTClassifier.forward: Encountered an empty or None patch batch. Skipping.")
                continue
            
            patch_tensor_batch = patch_tensor_batch.to(device) # Ensure batch is on the correct device
            processed_any_batch = True
 
            with torch.set_grad_enabled(not self.freeze_vit_flag):
                # Assuming self.encoder is your ViT model (e.g., VisionTransformerForMaskedImageModeling)
                # and its forward method can take 'return_cls_feature=True'
                cls_tokens_batch = self.encoder(patch_tensor_batch, return_cls_feature=True) 
            all_cls_tokens_for_wsi.append(cls_tokens_batch.cpu())

        if not processed_any_batch or not all_cls_tokens_for_wsi:
            return None, None, False 
        
        slide_features_cpu = torch.cat(all_cls_tokens_for_wsi, dim=0)
        slide_features_for_mil = slide_features_cpu.unsqueeze(0).to(device) 

        logits, attention_scores = self.mil_aggregator(slide_features_for_mil)
        
        return logits, attention_scores, True
 
class MILFineTuningModule(pl.LightningModule):
    def __init__(self,
                 model_instance: ViTClassifier,
                 # Optimizer hparams
                 opt: str, opt_eps: float, 
                 opt_betas: list, 
                 momentum: float,
                 # LR and WD hparams
                 base_lr: float, 
                 weight_decay: float, # for ViT
                 weight_decay_mil: float, 
                 layer_decay_rate: float,
                 # Scheduler hparams
                 warmup_lr: float, 
                 min_lr: float, 
                 warmup_epochs: int, 
                 max_epochs: int,
                 # Data/Model specific hparams for inner loader & transforms
                 patch_batch_size: int, 
                 patch_loader_num_workers: int, 
                 model_input_size: int,
                 num_optimizer_steps_per_epoch: int,
                 # Misc model hparams
                 freeze: bool = False,
                 num_classes: int = 2, # Passed for criterion or metrics if needed
                 **kwargs): # Catches other args
        super().__init__()
        # `ignore` should include non-scalar/non-simple types that shouldn't be saved in hparams.yaml
        # or are objects that will be part of the model.
        self.save_hyperparameters(ignore=['model_instance'])

        self.model = model_instance
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize transforms here as they are static per instance
        self.train_patch_transforms = WSIBagDatasetMIL.patch_transforms(
            model_input_size=self.hparams.model_input_size, is_train=True
        )
        self.val_patch_transforms = WSIBagDatasetMIL.patch_transforms(
            model_input_size=self.hparams.model_input_size, is_train=False
        )

    def forward(self, patch_batch_iterable, device):
        return self.model(patch_batch_iterable=patch_batch_iterable, device=device)

    def training_step(self, batch, batch_idx):
        # batch is the output of WSIBagDatasetMIL for a single WSI
        slide_id = batch['slide_id'][0] 
        patch_paths_list = batch['patch_paths'][0]
        wsi_label = batch['label'] # Already on self.device due to Lightning

        if not patch_paths_list:
            self.log(f"train_skipped_wsi/{slide_id}", 1.0, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            return torch.tensor(0.0, device=self.device, requires_grad=True)


        current_wsi_patch_dataset = SinglePatchDataset(
            patch_paths=patch_paths_list,
            transform=self.train_patch_transforms
        )
        
        patch_loader = DataLoader(
            current_wsi_patch_dataset,
            batch_size=self.hparams.patch_batch_size,
            shuffle=False, 
            num_workers=self.hparams.patch_loader_num_workers,
            pin_memory=True if self.hparams.patch_loader_num_workers > 0 and self.device.type == 'cuda' else False,
            drop_last=False
        )

        # The self.model (ViTClassifier) handles moving patches to its processing device internally
        logits, _, success = self.model(
            patch_batch_iterable=patch_loader,
            device=self.device 
        )

        if not success or logits is None:
            self.log(f"train_failed_wsi/{slide_id}", 1.0, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            return torch.tensor(0.0, device=self.device, requires_grad=True)


        loss = self.criterion(logits, wsi_label)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == wsi_label).float().mean()

        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1,logger=True, sync_dist=True)
        self.log("train_acc_step", acc, on_step=True, on_epoch=False, prog_bar=True, batch_size=1,logger=True, sync_dist=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1,logger=True, sync_dist=True, reduce_fx="mean")
        self.log("train_acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=1,logger=True, sync_dist=True, reduce_fx="mean")

        return loss
    
 
    def validation_step(self, batch, batch_idx):
        slide_id = batch['slide_id'][0]
        patch_paths_list = batch['patch_paths'][0]
        wsi_label = batch['label']

        if not patch_paths_list:
            self.log(f"val_skipped_wsi/{slide_id}", 1.0, on_epoch=True, batch_size=1, sync_dist=True)
            return None # Or a dict with val_loss: tensor(0.0) if needed by metrics

        current_wsi_patch_dataset = SinglePatchDataset(
            patch_paths=patch_paths_list,
            transform=self.val_patch_transforms # Use validation transforms
        )
        patch_loader = DataLoader(
            current_wsi_patch_dataset,
            batch_size=self.hparams.patch_batch_size,
            shuffle=False,
            num_workers=self.hparams.patch_loader_num_workers,
            pin_memory=True if self.hparams.patch_loader_num_workers > 0 and self.device.type == 'cuda' else False,
            drop_last=False
        )
        logits, _, success = self.model(
            patch_batch_iterable=patch_loader,
            device=self.device
        )
        if not success or logits is None:
            self.log(f"val_failed_wsi/{slide_id}", 1.0, on_epoch=True, batch_size=1, sync_dist=True)
            return None

        val_loss = self.criterion(logits, wsi_label)
        preds = torch.argmax(logits, dim=1)
        val_acc = (preds == wsi_label).float().mean()

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, batch_size=1,sync_dist=True)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, batch_size=1,sync_dist=True)
        return val_loss
    
    def configure_optimizers(self):
        param_groups_vit = []
        if not self.hparams.freeze:
            skip_list_vit = self.model.encoder.no_weight_decay() if hasattr(self.model.encoder, 'no_weight_decay') else set()
            num_layers_for_decay_vit = None
            if self.hparams.layer_decay_rate is not None and self.hparams.layer_decay_rate < 1.0:
                if hasattr(self.model.encoder, 'get_num_layers'):
                    num_layers_for_decay_vit = self.model.encoder.get_num_layers()
                elif hasattr(self.model.encoder, 'depth'): 
                    num_layers_for_decay_vit = self.model.encoder.depth
                else:
                    print("Warning: Layer decay for ViT enabled, but num_layers not found. LLRD might not work as expected.")
            
            if num_layers_for_decay_vit is not None and self.hparams.layer_decay_rate < 1.0 :
                param_groups_vit = get_parameter_groups(
                    self.model.encoder,
                    weight_decay=self.hparams.weight_decay,
                    skip_list=skip_list_vit,
                    layer_decay_rate=self.hparams.layer_decay_rate,
                    num_layers=num_layers_for_decay_vit
                )
            else: 
                # Filter out parameters that don't require gradients (e.g. if some parts are frozen differently)
                vit_params_to_optimize = [p for p in self.model.encoder.parameters() if p.requires_grad]
                if vit_params_to_optimize: # Only add group if there are params to optimize
                    param_groups_vit = [{
                        'params': vit_params_to_optimize,
                        'weight_decay': self.hparams.weight_decay,
                        'lr_scale': 1.0 
                    }]
        
        # Ensure mil_aggregator params are added only if they require grads
        mil_params_to_optimize = [p for p in self.model.mil_aggregator.parameters() if p.requires_grad]
        param_groups_mil = []
        if mil_params_to_optimize:
            param_groups_mil = [{
                'params': mil_params_to_optimize,
                'weight_decay': self.hparams.weight_decay_mil,
                'lr_scale': 1.0 
            }]

        all_param_groups = param_groups_vit + param_groups_mil
        if not all_param_groups:
             raise ValueError("No parameters to optimize. Check model freezing logic.")

        
        opt_config_ns = argparse.Namespace()
        opt_config_ns.opt = self.hparams.opt
        opt_config_ns.opt_eps = self.hparams.opt_eps
        opt_config_ns.opt_betas = tuple(self.hparams.opt_betas) if self.hparams.opt_betas else None
        opt_config_ns.momentum = self.hparams.momentum if 'sgd' in self.hparams.opt or 'momentum' in self.hparams.opt else None

        optimizer = create_optimizer(
            opt_config_ns,
            all_param_groups,
            base_lr=self.hparams.base_lr 
        )

        # Scheduler
        total_optimizer_steps = self.hparams.num_optimizer_steps_per_epoch * self.hparams.max_epochs
        warmup_optimizer_steps = self.hparams.num_optimizer_steps_per_epoch * self.hparams.warmup_epochs

        schedulers_list = []
        if warmup_optimizer_steps > 0:
            warmup_start_factor = self.hparams.warmup_lr / self.hparams.base_lr if self.hparams.base_lr > 0 else 0
            scheduler_warmup = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_optimizer_steps)
            schedulers_list.append(scheduler_warmup)

        cosine_t_max = total_optimizer_steps - warmup_optimizer_steps if warmup_optimizer_steps > 0 else total_optimizer_steps
        if cosine_t_max <= 0 and total_optimizer_steps > 0 : 
            cosine_t_max = 1 # Ensure T_max is at least 1
        
        if total_optimizer_steps > 0 : 
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_t_max if cosine_t_max > 0 else 1, eta_min=self.hparams.min_lr)
            schedulers_list.append(scheduler_cosine)

        if not schedulers_list:
            return optimizer # No scheduler

        if len(schedulers_list) > 1:
            # Ensure milestones are correctly set for SequentialLR
            scheduler = SequentialLR(optimizer, schedulers=schedulers_list, milestones=[warmup_optimizer_steps])
        elif schedulers_list:
            scheduler = schedulers_list[0]
        else: # Should not happen if total_optimizer_steps > 0
            return optimizer
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1,
            }
        }