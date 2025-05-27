import torch
import random
import json

from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from vlmo.modules.objectives import compute_irtr_recall, compute_irtr_recall_with_rerank
from vlmo.gadgets.my_metrics import Accuracy, VQAScore, Scalar
from pytorch_lightning.utilities import rank_zero_info
TEXT_PRETRAIN_METRIC_NAME = "textmlm_accuracy"
TEXT_PRETRAIN_TASK_KEY = "textmlm"

def set_metrics_for_text_pretraining(pl_module):
    for split in ["train", "val"]:
        metric_attr_name = f"{split}_{TEXT_PRETRAIN_METRIC_NAME}"
        rank_zero_info(f"Setting up '{TEXT_PRETRAIN_METRIC_NAME}' metric as '{metric_attr_name}' for {split} phase.")
        setattr(pl_module, metric_attr_name, Accuracy())


def epoch_wrapup_for_text_pretraining(pl_module):
    phase = "val" 
    computed_metrics = pl_module.trainer.callback_metrics
    the_metric = 0.0
    accuracy_log_key = f"{phase}/{TEXT_PRETRAIN_METRIC_NAME}"
    cur_epoch = getattr(pl_module.trainer, "current_epoch", "N/A")
    accuracy_val_tensor = computed_metrics.get(accuracy_log_key)
    if accuracy_val_tensor is not None:
        the_metric = accuracy_val_tensor.item()
        rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} {TEXT_PRETRAIN_TASK_KEY.upper()} Accuracy: {the_metric:.4f}")
    else:
        rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} {TEXT_PRETRAIN_TASK_KEY.upper()} Accuracy not found in callback_metrics with key '{accuracy_log_key}'. 'the_metric' will be 0.")

    pl_module.log(f"{phase}/the_metric", the_metric, rank_zero_only=True)
    rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} 'the_metric' (Text Pretraining): {the_metric:.4f}")


def set_metrics(pl_module):
    """
    为训练和验证阶段设置不同任务的评估指标和损失记录器。

    参数:
        pl_module (pl.LightningModule): PyTorch Lightning 模块实例，包含配置信息。
    """
    # 遍历训练和验证阶段
    for split in ["train", "val"]:
        # 遍历配置文件中定义的所有损失名称
        for k, v in pl_module.hparams.config["loss_names"].items():
            # 如果损失权重小于 1，则跳过该任务
            if v < 1:
                continue
            # 针对不同的任务类型设置不同的评估指标和损失记录器
            if k == "vqa":
                # 设置 VQA 任务的得分评估指标
                setattr(pl_module, f"{split}_vqa_score", VQAScore())
                # 设置 VQA 任务的损失记录器
                # setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "nlvr2":
                if split == "train":
                    # 训练阶段设置 NLVR2 任务的准确率评估指标
                    setattr(pl_module, f"train_{k}_accuracy", Accuracy())
                    # 训练阶段设置 NLVR2 任务的损失记录器
                    # setattr(pl_module, f"train_{k}_loss", Scalar())
                else:
                    # 验证阶段设置 NLVR2 任务在开发集上的准确率评估指标
                    setattr(pl_module, f"dev_{k}_accuracy", Accuracy())
                    # 验证阶段设置 NLVR2 任务在开发集上的损失记录器
                    # setattr(pl_module, f"dev_{k}_loss", Scalar())
                    # 验证阶段设置 NLVR2 任务在测试集上的准确率评估指标
                    setattr(pl_module, f"test_{k}_accuracy", Accuracy())
                    # 验证阶段设置 NLVR2 任务在测试集上的损失记录器
                    # setattr(pl_module, f"test_{k}_loss", Scalar())
                    setattr(pl_module, f"val_{k}_accuracy", Accuracy())
            elif k == "irtr":
                # 设置 IRTR 任务图像到文本的准确率评估指标
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                # 设置 IRTR 任务文本到图像的准确率评估指标
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                # 设置 IRTR 任务的损失记录器
                # setattr(pl_module, f"{split}_{k}_loss", Scalar())
                # 设置 IRTR 任务的 logit 缩放因子记录器
                # setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())
            elif k == "itm":
                # 设置 ITM 任务的准确率评估指标
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                # 设置 ITM 任务的损失记录器
                # setattr(pl_module, f"{split}_{k}_loss", Scalar())
            elif k == "itc":
                # 设置 ITC 任务图像到文本的准确率评估指标
                setattr(pl_module, f"{split}_{k}_i2t_accuracy", Accuracy())
                # 设置 ITC 任务文本到图像的准确率评估指标
                setattr(pl_module, f"{split}_{k}_t2i_accuracy", Accuracy())
                # 设置 ITC 任务的损失记录器
                # setattr(pl_module, f"{split}_{k}_loss", Scalar())
                # 设置 ITC 任务的 logit 缩放因子记录器
                # setattr(pl_module, f"{split}_{k}_logit_scale", Scalar())

                # 设置 ITC 任务视觉语言图像到文本的准确率评估指标
                setattr(pl_module, f"{split}_{k}_vl_i2t_accuracy", Accuracy())
                # 设置 ITC 任务视觉语言文本到图像的准确率评估指标
                setattr(pl_module, f"{split}_{k}_vl_t2i_accuracy", Accuracy())
                # 设置 ITC 任务视觉语言的 logit 缩放因子记录器
                # setattr(pl_module, f"{split}_{k}_vl_logit_scale", Scalar())
            else:
                # 其他任务设置准确率评估指标
                setattr(pl_module, f"{split}_{k}_accuracy", Accuracy())
                # 其他任务设置损失记录器
                # setattr(pl_module, f"{split}_{k}_loss", Scalar())


def epoch_wrapup(pl_module):
    """
    在每个训练或验证周期结束时执行的操作，包括计算和记录召回率、各种任务的评估指标和损失，并计算总指标。

    参数:
        pl_module (pl.LightningModule): PyTorch Lightning 模块实例，包含配置信息和评估指标。
    """
    phase = "train" if pl_module.training else "val"


    # 如果配置中要求计算召回率指标且当前处于验证阶段
    if pl_module.hparams.config["get_recall_metric"] and not pl_module.training:
        # 计算验证集的图像到文本和文本到图像的召回率（R@1, R@5, R@10）
        (val_ir_r1, val_ir_r5, val_ir_r10, val_tr_r1, val_tr_r5, val_tr_r10) = compute_irtr_recall(pl_module, split="val")
        val_avg = (val_ir_r1.item() + val_ir_r5.item() + val_ir_r10.item() + val_tr_r1.item() + val_tr_r5.item() + val_tr_r10.item()) / 6.0
        # 使用 LightningModule 的 log 方法记录召回率，而不是直接访问 logger
        pl_module.log("recalls/val_avg", val_avg, rank_zero_only=True)

        # 计算测试集的图像到文本和文本到图像的召回率（R@1, R@5, R@10）
        (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10) = compute_irtr_recall(pl_module, split="test")
        test_avg = (ir_r1.item() + ir_r5.item() + ir_r10.item() + tr_r1.item() + tr_r5.item() + tr_r10.item()) / 6.0
        pl_module.log("recalls/test_avg", test_avg, rank_zero_only=True)

        rank_zero_info("val_avg:{}, test_avg:{}".format(val_avg, test_avg))
        rank_zero_info("test ir_r1:{}, ir_r5:{}, ir_r10:{}, tr_r1:{}, tr_r5:{}, tr_r10:{}".format(ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10))

        pl_module.log("recalls/ir_r1", ir_r1, rank_zero_only=True)
        pl_module.log("recalls/ir_r5", ir_r5, rank_zero_only=True)
        pl_module.log("recalls/ir_r10", ir_r10, rank_zero_only=True)
        pl_module.log("recalls/tr_r1", tr_r1, rank_zero_only=True)
        pl_module.log("recalls/tr_r5", tr_r5, rank_zero_only=True)
        pl_module.log("recalls/tr_r10", tr_r10, rank_zero_only=True)


def set_schedule_for_MLM(pl_module):
    if not hasattr(pl_module, 'hparams') or not hasattr(pl_module.hparams, 'config'):
        rank_zero_info("Error: pl_module.hparams.config not found. Ensure hparams are saved correctly.")
        if hasattr(pl_module.hparams, 'learning_rate'):
            config = pl_module.hparams
        else:
            rank_zero_info("Cannot determine config structure in hparams.")
            config = {}
    else:
        config = pl_module.hparams.config

    lr = config.get("learning_rate", 1e-4)
    wd = config.get("weight_decay", 0.01)

    # 定义不需要进行权重衰减的参数名称模式
    no_decay = [
        "bias",
        ".Encoder.norm.weight",
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        ".patch_embed.norm.weight",
        ".Encoder.cls_token",
        ".Encoder.token_type_embeddings.weight",
        ".gamma_sa", 
        ".gamma_ffn",
    ]
    head_names = ["mlm_score."]
    lr_mult_head = config.get("lr_mult_head", config.get("lr_mult", 1.0))

    params_main_decay = []
    params_main_no_decay = []
    params_head_decay = []
    params_head_no_decay = []

    for name, param in pl_module.named_parameters():
        if not param.requires_grad:
            continue

        is_head_param = any(name.startswith(hn) for hn in head_names)
        apply_no_decay = any(nd_pattern in name for nd_pattern in no_decay)

        if is_head_param:
            if apply_no_decay:
                params_head_no_decay.append(param)
            else:
                params_head_decay.append(param)
        else: # Main model parameters (e.g., Encoder)
            if apply_no_decay:
                params_main_no_decay.append(param)
            else:
                params_main_decay.append(param)

    optimizer_grouped_parameters = []
    if params_main_decay:
        optimizer_grouped_parameters.append({
            "params": params_main_decay, "weight_decay": wd, "lr": lr, "name": "main_decay"
        })
    if params_main_no_decay:
        optimizer_grouped_parameters.append({
            "params": params_main_no_decay, "weight_decay": 0.0, "lr": lr, "name": "main_no_decay"
        })
    if params_head_decay:
        optimizer_grouped_parameters.append({
            "params": params_head_decay, "weight_decay": wd, "lr": lr * lr_mult_head, "name": "head_decay"
        })
    if params_head_no_decay:
        optimizer_grouped_parameters.append({
            "params": params_head_no_decay, "weight_decay": 0.0, "lr": lr * lr_mult_head, "name": "head_no_decay"
        })
 

    optimizer_grouped_parameters = [pg for pg in optimizer_grouped_parameters if len(pg["params"]) > 0]
    if not optimizer_grouped_parameters:
        rank_zero_info("No parameters requiring gradients found for the optimizer.")
        return [], []
    
    optim_type = config.get("optim_type", "adamw")
    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=config.get("adam_epsilon", 1e-8), betas=config.get("adam_betas", (0.9, 0.98))
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr, eps=config.get("adam_epsilon", 1e-8), betas=config.get("adam_betas", (0.9, 0.999)))
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=config.get("sgd_momentum", 0.9))

    if pl_module.trainer is None:
        rank_zero_info("Trainer not attached to pl_module. Cannot determine max_steps for scheduler. Scheduler will not be configured.")
        return [optimizer], []
    
    # 计算最大训练步数
    max_steps = -1
    if hasattr(pl_module.trainer, 'max_steps') and pl_module.trainer.max_steps is not None and pl_module.trainer.max_steps != -1 :
        max_steps = pl_module.trainer.max_steps
    elif hasattr(pl_module.trainer, 'datamodule') and hasattr(pl_module.trainer.datamodule, 'train_dataloader') and \
        hasattr(pl_module.trainer, 'max_epochs') and hasattr(pl_module.trainer, 'accumulate_grad_batches'):
        try:
            train_dataloader = pl_module.trainer.datamodule.train_dataloader()
            if train_dataloader is not None and pl_module.trainer.max_epochs is not None and pl_module.trainer.accumulate_grad_batches is not None:
                len_train_loader = len(train_dataloader)
                if len_train_loader > 0: # 确保 dataloader 不是空的或无限的
                    max_steps = (
                        len_train_loader
                        * pl_module.trainer.max_epochs
                        // pl_module.trainer.accumulate_grad_batches
                    )
                else:
                    rank_zero_info("Warning: Train dataloader length is 0 or not determined. Cannot calculate max_steps for scheduler.")
            else:
                rank_zero_info("Warning: Could not determine train_dataloader length, max_epochs, or accumulate_grad_batches. Cannot calculate max_steps.")
        except Exception as e:
            rank_zero_info(f"Warning: Error obtaining train_dataloader length: {e}. Cannot calculate max_steps.")

        if max_steps <= 0:
            rank_zero_info(f"Max_steps ({max_steps}) is not valid for scheduler. Scheduler will not be configured.")
            return [optimizer], []
    
    warmup_steps_config = config.get("warmup_steps", 0)
    if isinstance(warmup_steps_config, float) and 0 < warmup_steps_config < 1: # 如果是比例
        warmup_steps = int(max_steps * warmup_steps_config)
    else:
        warmup_steps = int(warmup_steps_config)

    rank_zero_info("Warmup_steps:{} \t Max_steps:{}".format(warmup_steps, max_steps))

    decay_power_config = config.get("decay_power", "cosine")
    end_lr_config = config.get("end_lr", 0.0)

    scheduler = None
    if decay_power_config == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        try:
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr_config,
                power=float(decay_power_config) if isinstance(decay_power_config, (str,int,float)) and decay_power_config != "cosine" else 1.0,
            )
        except ValueError:
            rank_zero_info(f"Warning: Invalid decay_power '{decay_power_config}'. Defaulting to linear decay (power=1.0).")
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
                lr_end=end_lr_config,
                power=1.0,
            )

    # 定义学习率调度器的配置
    sched = {"scheduler": scheduler, "interval": "step","frequency": 1}

    return (
        [optimizer],
        [sched],
    )
