import torch
import argparse
import numpy as np
import torch.nn as nn
from modules.VITForMIM import VisionTransformerForMaskedImageModeling 
from modules.AttentionSeries import GatedAttention 
from timm.models import create_model
from typing import Optional, Tuple,Callable, Iterable,List
from model import pretrained_model
from modules.Losses import LDAMLoss, ClassBalancedLoss,CBCrossEntropyLoss
from PIL import Image 
import pytorch_lightning as pl
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
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
                 opt: str, 
                 opt_eps: float, 
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
                 # Class imbalance handling parameters
                 samples_per_cls: List[int] = None, 
                 ldam_C_factor: float = 1.0,
                 drw_start_epoch: int = 20,
                 cb_beta: float = 0.999,
                 **kwargs): # Catches other args
        super().__init__()
        # `ignore` should include non-scalar/non-simple types that shouldn't be saved in hparams.yaml
        # or are objects that will be part of the model.
        self.save_hyperparameters(ignore=['model_instance'])

        self.model = model_instance

        # 初始化两个损失函数
        print(f"Initializing LDAM Loss and CB Loss with samples_per_cls: {samples_per_cls}")

        # LDAM Loss for stage 1
        self.criterion_ldam = LDAMLoss(
            samples_per_cls=samples_per_cls,
            num_classes=num_classes,
            C_factor=ldam_C_factor
        )
        # Class-Balanced Cross-Entropy Loss for stage 2
        self.criterion_cb_ce = CBCrossEntropyLoss(
            samples_per_cls=samples_per_cls,
            num_classes=num_classes,
            beta=cb_beta
        )
        # 用于验证的标准交叉熵
        self.criterion_standard = nn.CrossEntropyLoss()

        # Initialize transforms here as they are static per instance
        self.train_patch_transforms = WSIBagDatasetMIL.patch_transforms(
            model_input_size=self.hparams.model_input_size, is_train=True
        )
        self.val_patch_transforms = WSIBagDatasetMIL.patch_transforms(
            model_input_size=self.hparams.model_input_size, is_train=False
        )
        self.val_predictions = []
        self.val_targets = []
        print(f"MILFineTuningModule initialized with:")
        print(f"  - num_classes: {num_classes}")
        print(f"  - drw_start_epoch: {drw_start_epoch}")
        print(f"  - samples_per_cls: {samples_per_cls}")
        print(f"  - Validation cache initialized: predictions={len(self.val_predictions)}, targets={len(self.val_targets)}")
    
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


        
        current_epoch = self.current_epoch
        if current_epoch < self.hparams.drw_start_epoch:
            loss = self.criterion_ldam(logits, wsi_label)
            loss_stage = "LDAM"
        else:
            loss = self.criterion_cb_ce(logits, wsi_label)
            loss_stage = "CB_CE"

        preds = torch.argmax(logits, dim=1)
        acc = (preds == wsi_label).float().mean()

        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=1,logger=True, sync_dist=True)
        self.log("train_acc_step", acc, on_step=True, on_epoch=False, prog_bar=True, batch_size=1,logger=True, sync_dist=True)
        self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1,logger=True, sync_dist=True, reduce_fx="mean")
        self.log("train_acc_epoch", acc, on_step=False, on_epoch=True, prog_bar=False, batch_size=1,logger=True, sync_dist=True, reduce_fx="mean")
        # 记录当前使用的损失函数类型
        self.log(f"train_loss_stage/{loss_stage}", loss, on_step=False, on_epoch=True, prog_bar=False, batch_size=1, logger=True, sync_dist=True)
        
        # 每个epoch记录一次当前阶段
        if batch_idx == 0:
            print(f"Epoch {current_epoch}: Using {loss_stage} loss (DRW switch at epoch {self.hparams.drw_start_epoch})")
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

        val_loss = self.criterion_standard(logits, wsi_label)
        preds = torch.argmax(logits, dim=1)
        val_acc = (preds == wsi_label).float().mean()
        
        self.val_predictions.append(preds.cpu())
        self.val_targets.append(wsi_label.cpu())

        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, batch_size=1,sync_dist=True)
        self.log("val_acc", val_acc, on_epoch=True, prog_bar=True, batch_size=1,sync_dist=True)
        return val_loss
    
    def on_validation_epoch_end(self):
        """在验证epoch结束时计算类别感知指标"""
        if len(self.val_predictions) == 0 or len(self.val_targets) == 0:
            print("Warning: No validation predictions or targets collected. Skipping metrics computation.")
            self._clear_validation_cache()
            return
        
        try:
            # 合并所有预测和标签
            all_preds = torch.cat(self.val_predictions, dim=0).numpy()
            all_targets = torch.cat(self.val_targets, dim=0).numpy()
                
            # 检查是否有有效的预测和标签
            if len(all_preds) == 0 or len(all_targets) == 0:
                print("Warning: No valid predictions or targets found for validation metrics computation.")
                self._clear_validation_cache()
                return
            
            # 检查并修复预测值超出范围的问题
            valid_classes = set(range(self.hparams.num_classes))
            target_classes = set(all_targets)
            pred_classes = set(all_preds)

            # 将超出范围的预测值裁剪到有效范围内
            all_preds = np.clip(all_preds, 0, self.hparams.num_classes - 1)

            print(f"Validation metrics calculation: {len(all_preds)} predictions, {len(all_targets)} targets")
            print(f"Valid classes range: {valid_classes}")
            print(f"Target classes: {target_classes}")
            print(f"Prediction classes (after clipping): {set(all_preds)}")
            print(f"Predictions shape: {all_preds.shape}, Targets shape: {all_targets.shape}") 

            # 如果目标类别超出了模型类别数，也需要处理
            if max(all_targets) >= self.hparams.num_classes:
                print(f"Warning: Target contains classes >= num_classes ({self.hparams.num_classes})")
                all_targets = np.clip(all_targets, 0, self.hparams.num_classes - 1)

                # 计算平衡准确率
            balanced_acc = balanced_accuracy_score(all_targets, all_preds)
            self.log("val_balanced_acc", balanced_acc, on_epoch=True, prog_bar=True, sync_dist=True)
                
            # 计算每个类别的指标
            try:
                labels = list(range(self.hparams.num_classes))
                # 生成分类报告
                report = classification_report(all_targets, all_preds, labels=labels, output_dict=True, zero_division=0)

                # 记录每个类别的F1分数
                for class_idx in range(self.hparams.num_classes):
                    class_key = str(class_idx)
                    if class_key in report:
                        f1_score = report[class_key]['f1-score']
                        precision = report[class_key]['precision']
                        recall = report[class_key]['recall']
                            
                        self.log(f"val_f1_class_{class_idx}", f1_score, on_epoch=True, sync_dist=True)
                        self.log(f"val_precision_class_{class_idx}", precision, on_epoch=True, sync_dist=True)
                        self.log(f"val_recall_class_{class_idx}", recall, on_epoch=True, sync_dist=True)
                    
                    # 记录宏平均指标
                macro_f1 = report['macro avg']['f1-score']
                macro_precision = report['macro avg']['precision']
                macro_recall = report['macro avg']['recall']
                    
                self.log("val_macro_f1", macro_f1, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log("val_macro_precision", macro_precision, on_epoch=True, sync_dist=True)
                self.log("val_macro_recall", macro_recall, on_epoch=True, sync_dist=True)
                    
                # 打印详细的验证结果
                print(f"\nValidation Results - Epoch {self.current_epoch}:")
                print(f"Balanced Accuracy: {balanced_acc:.4f}")
                print(f"Macro F1-Score: {macro_f1:.4f}")
                print("\nPer-class metrics:")
                for class_idx in range(self.hparams.num_classes):
                    class_key = str(class_idx)
                    if class_key in report:
                        print(f"  Class {class_idx}: F1={report[class_key]['f1-score']:.4f}, "
                                  f"Precision={report[class_key]['precision']:.4f}, "
                                  f"Recall={report[class_key]['recall']:.4f}")
                    
                # 打印混淆矩阵
                cm = confusion_matrix(all_targets, all_preds, labels=labels)
                print(f"Confusion Matrix:\n{cm}")
                    
            except Exception as e:
                print(f"Error computing detailed metrics: {e}")
                    
        except Exception as e:
            print(f"Error computing validation metrics: {e}")
            print(f"val_predictions length: {len(self.val_predictions)}")
            print(f"val_targets length: {len(self.val_targets)}")
            if len(self.val_predictions) > 0:
                print(f"First prediction shape: {self.val_predictions[0].shape if len(self.val_predictions) > 0 else 'N/A'}")
            if len(self.val_targets) > 0:
                print(f"First target shape: {self.val_targets[0].shape if len(self.val_targets) > 0 else 'N/A'}")
        # 清空存储的预测和标签
        self._clear_validation_cache()

    def _clear_validation_cache(self):
        """清空验证缓存"""
        self.val_predictions = []
        self.val_targets = []
    
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