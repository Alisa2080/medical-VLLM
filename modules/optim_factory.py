# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------'
import torch
from torch import optim as optim

from timm.optim.adafactor import Adafactor
from timm.optim.adahessian import Adahessian
from timm.optim.adamp import AdamP
from timm.optim.lookahead import Lookahead
from timm.optim.nadam import NAdamLegacy
from timm.optim.nvnovograd import NvNovoGrad
from timm.optim.radam import RAdamLegacy
from timm.optim.rmsprop_tf import RMSpropTF
from timm.optim.sgdp import SGDP

import json

try:
    from apex.optimizers import FusedNovoGrad, FusedAdam, FusedLAMB, FusedSGD
    has_apex = Trueset
except ImportError:
    has_apex = False


def get_num_layer_for_vit(var_name, num_max_layer):
    if var_name in ("cls_token", "mask_token", "pos_embed"):
        return 0
    elif var_name.startswith("patch_embed"):
        return 0
    elif var_name.startswith("rel_pos_bias"): # For BEiT v1 style, BEiT v2 might not have this or handle differently
        return num_max_layer - 1
    elif var_name.startswith("blocks"):
        try:
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        except (IndexError, ValueError): # Handle cases where block name might not be as expected
            return num_max_layer - 1
    else:
        return num_max_layer - 1


class LayerDecayValueAssigner(object):
    def __init__(self, values):
        self.values = values

    def get_scale(self, layer_id):
        clamped_layer_id = max(0, min(layer_id, len(self.values) - 1))
        return self.values[clamped_layer_id]

    def get_layer_id(self, var_name):
        return get_num_layer_for_vit(var_name, len(self.values))


def get_parameter_groups(model, weight_decay=1e-5, skip_list=(), layer_decay_rate=None, num_layers=None, **kwargs):
    parameter_group_names = {}
    parameter_group_vars = {}

    if num_layers is None: # num_layers should be the max_layer_id for which decay applies
        raise ValueError("num_layers must be provided for layer_wise_lr_decay.")
    
    num_effective_layers_for_decay = num_layers + 1 # embeds + blocks
    lr_scales = [layer_decay_rate ** (num_effective_layers_for_decay - 1 - i) for i in range(num_effective_layers_for_decay)]
    assigner = LayerDecayValueAssigner(lr_scales)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(kwargs.get('filter_name', [])) > 0:
            flag = False
            for filter_n in kwargs.get('filter_name', []):
                if filter_n in name:
                    print(f"filter {name} because of the pattern {filter_n}")
                    flag = True
            if flag:
                continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list: # param.ndim <= 1 len(param.shape) == 1
            group_name = "no_decay"
            this_weight_decay = 0.
        else:
            group_name = "decay"
            this_weight_decay = weight_decay

        layer_id = -1 # Default if no assigner
        current_lr_scale = 1.0

        if assigner is not None:
            layer_id = assigner.get_layer_id(name) # This calls get_num_layer_for_vit
            current_lr_scale = assigner.get_scale(layer_id)
            group_name = f"layer_{layer_id}_{group_name}"

 
        if group_name not in parameter_group_vars:
            parameter_group_vars[group_name] = {
                "params": [],
                "weight_decay": this_weight_decay,
                "lr_scale": current_lr_scale # Store the scale
            }
            # For logging/debugging parameter group names
            parameter_group_names[group_name] = {
                "params_name": [],
                "weight_decay": this_weight_decay,
                "lr_scale": current_lr_scale
            }
        
        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params_name"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    
    final_parameter_groups = []
    for group_name, group_vars in parameter_group_vars.items():
        group = {
            "params": group_vars["params"],
            "weight_decay": group_vars["weight_decay"],
            "lr_scale": group_vars["lr_scale"]
        }
        final_parameter_groups.append(group)

    return final_parameter_groups


def create_optimizer(opt_config, parameter_groups_with_scale, base_lr):
    """
    Creates an optimizer.
    opt_config: A namespace or dict containing optimizer settings (opt, opt_eps, opt_betas, momentum).
    parameter_groups_with_scale: List of dicts from get_parameter_groups, each containing 'params', 'weight_decay', 'lr_scale'.
    base_lr: The base learning rate.
    """

    opt_lower = opt_config.opt.lower()

     # Apply lr_scale to each group's learning rate
    processed_parameter_groups = []
    for group in parameter_groups_with_scale:
        processed_group = {
            "params": group["params"],
            "weight_decay": group["weight_decay"],
            "lr": base_lr * group.get("lr_scale", 1.0) # Apply lr_scale here
        }
        # Optimizer-specific params like betas, eps are usually global, not per-group
        processed_parameter_groups.append(processed_group)

    
    # Common optimizer arguments (global for the optimizer)
    optimizer_kwargs = {}
    if hasattr(opt_config, 'opt_eps') and opt_config.opt_eps is not None:
        optimizer_kwargs['eps'] = opt_config.opt_eps
    if hasattr(opt_config, 'opt_betas') and opt_config.opt_betas is not None:
        # Ensure opt_betas is a tuple
        optimizer_kwargs['betas'] = tuple(opt_config.opt_betas) if isinstance(opt_config.opt_betas, list) else opt_config.opt_betas

    opt_split = opt_lower.split('_')
    opt_type = opt_split[-1]

    optimizer = None
    if opt_type == 'sgd' or opt_type == 'nesterov':
        optimizer_kwargs.pop('eps', None) # SGD doesn't use eps
        optimizer_kwargs.pop('betas', None) # SGD doesn't use betas
        optimizer = optim.SGD(processed_parameter_groups, lr=base_lr, momentum=opt_config.momentum, nesterov=True, **optimizer_kwargs)
    elif opt_type == 'momentum':
        optimizer_kwargs.pop('eps', None)
        optimizer_kwargs.pop('betas', None)
        optimizer = optim.SGD(processed_parameter_groups, lr=base_lr, momentum=opt_config.momentum, nesterov=False, **optimizer_kwargs)
    elif opt_type == 'adam':
        optimizer = optim.Adam(processed_parameter_groups, lr=base_lr, **optimizer_kwargs)
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(processed_parameter_groups, lr=base_lr, **optimizer_kwargs)
    else:
        print(f"Warning: Optimizer type '{opt_type}' not explicitly handled, defaulting to AdamW or check config.")
        optimizer_kwargs.setdefault('eps', 1e-8)
        optimizer_kwargs.setdefault('betas', (0.9, 0.999))
        optimizer = optim.AdamW(processed_parameter_groups, lr=base_lr, **optimizer_kwargs)

    if len(opt_split) > 1 and opt_split[0] == 'lookahead':
        optimizer = Lookahead(optimizer)

    return optimizer
