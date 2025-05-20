
import torch
import argparse
import torch.nn as nn
from modules.Block import EncoderBlock
from modules.PatchEmbed import PatchEmbed, HybridEmbed,PatchEmbed_PT
from timm.layers import trunc_normal_ as __call_trunc_normal_
from modules.RMSNorm import RMSNorm
from.rope import Rope2DPosEmb
from typing import Optional, Tuple,Callable
import pytorch_lightning as pl
from transformers.modeling_outputs import BaseModelOutput
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from modules.optim_factory import create_optimizer
from modules.optim_factory import get_parameter_groups, create_optimizer # Ensure create_optimizer is imported

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

def apply_layer_wise_scaling(model, decay_factor=0.9):
    blocks = model.blocks
    n_layers = len(blocks)
    if n_layers < 2:
        return

    with torch.no_grad():
        for i, block in enumerate(blocks):
            scale = decay_factor ** (i / (n_layers - 1))

            if hasattr(block, 'attn'):
                if hasattr(block.attn, 'proj'):
                    block.attn.proj.weight.mul_(scale)
                elif hasattr(block.attn, 'o_proj'):
                   block.attn.o_proj.weight.mul_(scale)
            
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'fc2'):
                block.mlp.fc2.weight.mul_(scale)


class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, 
                 dim:int=512,
                 img_size:int = 384, 
                 patch_size:int=16,
                 in_chans:int=3,
                 vocab_size:int=8192,
                 depth:int=12,
                 num_heads:int=12,
                 num_kv_heads: int = None,
                 mlp_ratio:float=4.,
                 qkv_bias:bool=False,
                 qk_scale:Optional[float]=None,
                 attn_drop_rate:float=0.,
                 drop_path_rate:float=0.,
                 norm_layer:Callable=RMSNorm,
                 norm_eps: float = 1e-6,
                 hidden_act: str = "silu",
                 layer_scale_init_values: Optional[float] =  1e-5,
                 rope_base: int = 10000,
                 init_std: float = 0.02, 
                 embed_smooth_alpha: float = 1.0,
                 **kwargs):
        super().__init__()
        self.dim = dim
        self.img_size = img_size
        self.in_chans = in_chans
        self.drop_path_rate = drop_path_rate 
        self.embed_smooth_alpha = embed_smooth_alpha
        self.depth = depth
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.head_dim = self.dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.norm_layer = norm_layer
        self.hidden_act = hidden_act
        self.attn_drop_rate = attn_drop_rate
        self.patch_embed = PatchEmbed_PT(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.dim,
        )
        self.grid_size = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.dim))

        self.mask_token = nn.Parameter(torch.empty(1, 1, self.dim))


        self.rope2d = Rope2DPosEmb( # Need to ensure this can provide cos/sin or adapt logic
             dim=self.head_dim, max_height=self.grid_size[0]+ 1, max_width=self.grid_size[1]+ 1,theta_base=rope_base
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        # 初始化 Transformer 块
        self.blocks = nn.ModuleList([
            EncoderBlock( 
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                norm_eps=norm_eps,
                layer_scale_init_values=layer_scale_init_values,
                hidden_act=hidden_act,
            )
            for i in range(depth)])
        self.norm = norm_layer(dim, eps=norm_eps)

        self.init_std = init_std

        self.lm_head = nn.Linear(dim, vocab_size)
        nn.init.normal_(self.cls_token, std=self.init_std)
        nn.init.normal_(self.mask_token, std=self.init_std)

        self.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, RMSNorm)):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            if m.weight is not None:
                m.weight.data.normal_(mean=0.0, std=self.init_std)
            if m.padding_idx is not None:
                m.weight.data[m.padding_idx].zero_()
    
    @torch.jit.ignore
    def no_weight_decay(self):
        no_decay = {'mask_token', 'cls_token'}
        return no_decay

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self,
                            image_tensor: torch.Tensor,
                            bool_masked_pos,
                            output_attentions: bool = False,
                            output_hidden_states: bool = False,
                            **kwargs
                            ):
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        hidden_states = self.patch_embed(hidden_state=image_tensor) # (B, N_patches, C)
        if self.embed_smooth_alpha < 1.0:
            hidden_states = (
                hidden_states * self.embed_smooth_alpha
                + hidden_states.detach() * (1.0 - self.embed_smooth_alpha)
            )
        # 获取输入张量的批次大小和序列长度
        input_size = hidden_states.shape[:2]

        # 扩展分类令牌以匹配批次大小
        cls_tokens = self.cls_token.expand(input_size[0], -1, -1)  # (B, 1, C)
        # 扩展掩码令牌以匹配批次大小和序列长度
        mask_token = self.mask_token.expand(input_size[0], input_size[1], -1)  # (B, N_patches, C)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token) # (B, N_patches, 1)
        hidden_states = hidden_states * (1 - w) + mask_token * w # (B, N_patches, C)

        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1) # (B, N_patches + 1, C)
        num_img_tokens = hidden_states.shape[1]
        image_pos_ids = torch.arange(num_img_tokens, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(input_size[0], -1)
        
        all_block_attentions = [] if output_attentions else None
        all_hidden_states = [] if output_hidden_states else None
        
        for i, block in enumerate(self.blocks):

            if output_hidden_states: 
                all_hidden_states.append(hidden_states,)

            hidden_states = block(hidden_states = hidden_states, 
                                  image_pos_ids=image_pos_ids, 
                                  grid_hw=self.grid_size, 
                                  rope_2d_instance=self.rope2d,
                                  output_attentions=output_attentions,
                                  )
            hidden_states = hidden_states[0]
            
            if output_attentions:
                all_block_attentions.append(hidden_states[1])
        
        last_hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(last_hidden_states)
        
        return BaseModelOutput(
            last_hidden_state=last_hidden_states,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_block_attentions) if output_attentions else None,
        )

    def forward(self, 
                image_tensor: torch.Tensor,
                bool_masked_pos=None,
                output_attentions: bool = False,
                output_hidden_states: bool = False,
                return_patch_tokens=False, 
                return_all_tokens=False,
                return_cls_feature: bool = False,
                **kwargs):
        output_attentions = output_attentions if output_attentions is not None else False
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((image_tensor.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(image_tensor.device)
        
        hidden_states = self.forward_features(image_tensor, bool_masked_pos=bool_masked_pos,
                                             output_attentions=output_attentions,
                                             output_hidden_states=output_hidden_states) # (B, N, C) N = num_patches + 1
        last_hidden_states = hidden_states.last_hidden_state 
        patch_tokens = last_hidden_states[:, 1:]  # 取除 cls_token 外的patch tokens, shape: [B, 576, C]

        if output_hidden_states:
            return hidden_states.hidden_states # tuple(all_hidden_states)
            
        if output_attentions:
            return hidden_states.attentions # tuple(all_block_attentions)
            
        if return_patch_tokens:
            return patch_tokens

        if return_cls_feature:
            # Return the [CLS] token embedding after all blocks and final normalization
            return last_hidden_states[:, 0, :] # Shape: (B, C)
        
        if return_all_tokens:
            return last_hidden_states
        else:
            masked_output = self.lm_head(patch_tokens[bool_masked_pos])
            return masked_output


    def forward_intermediate(self, 
                             image_tensor: torch.Tensor,
                             bool_masked_pos=None, 
                             layer_id:int=12):

        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((image_tensor.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(image_tensor.device)
        hidden_states = self.patch_embed(image_tensor)
        input_size = hidden_states.shape[:1]

        cls_tokens = self.cls_token.expand(input_size[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(input_size[0], input_size[1], -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        hidden_states = hidden_states * (1 - w) + mask_token * w

        hidden_states = torch.cat((cls_tokens, hidden_states), dim=1)
        num_img_tokens = hidden_states.shape[1]
        image_pos_ids = torch.arange(num_img_tokens, dtype=torch.long, device=hidden_states.device).unsqueeze(0).expand(input_size[0], -1)
        if isinstance(layer_id, list):
            output_list = []
            for l, block in enumerate(self.blocks):
                hidden_states = block(
                            hidden_states = hidden_states, 
                            image_pos_ids=image_pos_ids, 
                            grid_hw=self.grid_size, 
                            rope_2d_instance=self.rope2d,
                            output_attentions=False,
                          )
                if l in layer_id:
                    output_list.append(hidden_states[0][:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for i, block in enumerate(self.blocks):
                hidden_states = block(hidden_states=hidden_states, 
                                      image_pos_ids=image_pos_ids, 
                                      grid_hw=self.grid_size, 
                                      rope_2d_instance=self.rope2d, 
                                      output_attentions=False)
                if i == layer_id:
                    break
            return hidden_states[0][:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")


    def get_last_selfattention(self, 
                               image_tensor: torch.Tensor,
                               ):
        all_attention_tuple = self.forward(image_tensor=image_tensor,
                                           output_attentions=True,
                                           bool_masked_pos=None,
                                           output_hidden_states=False)
        return all_attention_tuple[-1:]  
    


    
class BEiTLightningModule(pl.LightningModule):
    def __init__(self,
                model: nn.Module,
                vqkd: nn.Module,
                lr: float = 5e-5, # This will be the base_lr for the optimizer
                weight_decay: float = 0.05,
                warmup_epochs: int = 5,
                max_epochs: int = 100,
                min_lr: float = 1e-5,
                warmup_lr: float = 1e-6,
                optimizer_name: str = 'adamw',
                opt_eps: Optional[float] = 1e-8,
                opt_betas: Optional[list[float, float]] = [0.9, 0.98], # Ensure this is a list for hparams
                num_training_steps_per_epoch: Optional[int] = None,
                momentum: Optional[float] = None, # Added momentum for SGD 
                **kwargs):

        super().__init__()
        self.save_hyperparameters(ignore=['model', 'vqkd'])
        self.model = model
        self.vqkd = vqkd.eval()
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        if num_training_steps_per_epoch is None:
            raise ValueError("num_training_steps_per_epoch must be provided for LR scheduler.")
            

    def forward(self, x, bool_masked_pos=None, **kwargs):
            
        return self.model(x, bool_masked_pos=bool_masked_pos, **kwargs)
    
    def training_step(self, batch, batch_idx):
        
        packed_data, _ = batch
        samples, images, bool_masked_pos = packed_data
            
        with torch.no_grad():
            autocast_enabled = self.trainer.precision.endswith("mixed")
            with torch.amp.autocast(device_type=self.device.type,dtype=self.dtype, enabled=autocast_enabled):
                token_ids = self.vqkd.get_codebook_indices(images)

        bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
        labels = token_ids[bool_masked_pos]
    
        outputs = self.model(samples, bool_masked_pos=bool_masked_pos)
            
        loss = 0
        mlm_acc = 0.0
    
        if not isinstance(outputs, torch.Tensor):
            raise ValueError(
                f"Expected model output to be a torch.Tensor for MLM, but got {type(outputs)}. "
                "If your model is designed to return a list for MLM, "
                "you'll need to adjust this training_step logic."
            )
        
        if outputs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Shape mismatch between outputs ({outputs.shape}) and labels ({labels.shape}). "
                f"Number of predictions ({outputs.shape[0]}) must match number of labels ({labels.shape[0]})."
            )
        
        if outputs.ndim != 2 or labels.ndim != 1:
             raise ValueError(
                f"Dimension mismatch. Outputs should be 2D (N, C) and labels 1D (N). "
                f"Got outputs.ndim={outputs.ndim}, labels.ndim={labels.ndim}."
            )
        
        loss = self.loss_fn(outputs, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN or Inf detected in loss at step {self.global_step}")

        mlm_acc = (outputs.max(-1)[1] == labels).float().mean()

        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("mlm_acc", mlm_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        pass



    def configure_optimizers(self):
        skip_list = self.model.no_weight_decay() if hasattr(self.model, 'no_weight_decay') else set()
        
        num_layers_for_decay = None
        if self.hparams.layer_decay_rate is not None and self.hparams.layer_decay_rate < 1.0:
            if hasattr(self.model, 'get_num_layers'): # Standard way to get depth for ViTs
                num_layers_for_decay = self.model.get_num_layers()
            elif hasattr(self.model, 'depth'): # Fallback
                 num_layers_for_decay = self.model.depth
            else:
                print("Warning: Layer decay rate is set, but cannot determine number of layers from model. LLRD might not work as expected.")
        
        parameter_groups_with_scale = get_parameter_groups(
            self.model,
            weight_decay=self.hparams.weight_decay,
            skip_list=skip_list,
            layer_decay_rate=self.hparams.layer_decay_rate,
            num_layers=num_layers_for_decay # Pass the number of transformer blocks
        )

    # Create a mock args object for create_optimizer
    # Ensure all necessary attributes for create_optimizer are present in self.hparams
        opt_config = argparse.Namespace()
        opt_config.opt = self.hparams.optimizer_name
        opt_config.opt_eps = self.hparams.opt_eps
        opt_config.opt_betas = tuple(self.hparams.opt_betas) if self.hparams.opt_betas is not None else None
        # Add other args if create_optimizer needs them, e.g., momentum for SGD
        if 'sgd' in self.hparams.optimizer_name or 'momentum' in self.hparams.optimizer_name:
            opt_config.momentum = self.hparams.get('momentum', 0.9)
        else:
            opt_config.momentum = None # Ensure momentum is None if not SGD-like

        # 3. Create the optimizer using the base learning rate
        # The LLRD scales are applied inside create_optimizer now
        optimizer = create_optimizer(
                opt_config,
                parameter_groups_with_scale,
                base_lr=self.hparams.lr,
            )

        total_steps = self.hparams.num_training_steps_per_epoch * self.hparams.max_epochs
        warmup_steps = self.hparams.num_training_steps_per_epoch * self.hparams.warmup_epochs

        if warmup_steps > 0:
            if self.hparams.lr == 0: # Avoid division by zero if peak LR is 0
                 warmup_start_factor = 0 
            else:
                 warmup_start_factor = self.hparams.warmup_lr / self.hparams.lr

            scheduler_warmup = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_steps)
            # 余弦退火调度器：从 lr 到 min_lr
            cosine_t_max = total_steps - warmup_steps
            # This case means warmup is longer than or equal to total steps, which is unusual.
            if cosine_t_max <= 0:
                scheduler_cosine = CosineAnnealingLR(optimizer, T_max=1, eta_min=self.hparams.min_lr) # Effectively no cosine if T_max is too small
            else:
                scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_t_max, eta_min=self.hparams.min_lr)

            scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
        else:
            # No warmup, just cosine annealing from peak_lr to min_lr
            scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=self.hparams.min_lr)
            scheduler = scheduler_cosine

            
        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step", 
                    "frequency": 1,
                }
            }
