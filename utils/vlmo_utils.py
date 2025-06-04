import torch
import random
import json

from torch.optim import AdamW
from transformers import (
    get_polynomial_decay_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from modules.objectives import compute_irtr_recall, compute_irtr_recall_with_rerank
from metrics.my_metrics import Accuracy, VQAScore, Scalar
from pytorch_lightning.utilities import rank_zero_info

# ==================== Text Pretraining Constants ====================
TEXT_PRETRAIN_METRIC_NAME = "textmlm_accuracy"
TEXT_PRETRAIN_TASK_KEY = "textmlm"

# ==================== Vision-Language Pretraining Constants ====================
VL_PRETRAIN_ITC_I2T_METRIC_NAME = "itc_i2t_accuracy"
VL_PRETRAIN_ITC_T2I_METRIC_NAME = "itc_t2i_accuracy"
VL_PRETRAIN_MLM_METRIC_NAME = "mlm_accuracy"
VL_PRETRAIN_CROSS_MODAL_MLM_METRIC_NAME = "cross_modal_mlm_accuracy"

# ==================== Text Pretraining Functions ====================
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

# ==================== Vision-Language Pretraining Functions ====================
def set_metrics_for_vision_language_pretraining(pl_module):
    """为视觉-语言预训练任务设置评估指标"""
    for split in ["train", "val"]:
        # ITC (Image-Text Contrastive) 指标
        setattr(pl_module, f"{split}_{VL_PRETRAIN_ITC_I2T_METRIC_NAME}", Accuracy())
        setattr(pl_module, f"{split}_{VL_PRETRAIN_ITC_T2I_METRIC_NAME}", Accuracy())
        
        # MLM 指标
        setattr(pl_module, f"{split}_{VL_PRETRAIN_MLM_METRIC_NAME}", Accuracy())
        
        # Cross-Modal MLM 指标（第三阶段）
        setattr(pl_module, f"{split}_{VL_PRETRAIN_CROSS_MODAL_MLM_METRIC_NAME}", Accuracy())
        
        rank_zero_info(f"Setting up vision-language pretraining metrics for {split} phase:")
        rank_zero_info(f"  - {split}_{VL_PRETRAIN_ITC_I2T_METRIC_NAME}")
        rank_zero_info(f"  - {split}_{VL_PRETRAIN_ITC_T2I_METRIC_NAME}")
        rank_zero_info(f"  - {split}_{VL_PRETRAIN_MLM_METRIC_NAME}")
        rank_zero_info(f"  - {split}_{VL_PRETRAIN_CROSS_MODAL_MLM_METRIC_NAME}")

def epoch_wrapup_for_vision_language_pretraining(pl_module):
    """视觉-语言预训练任务的周期结束处理"""
    phase = "val"
    computed_metrics = pl_module.trainer.callback_metrics
    cur_epoch = getattr(pl_module.trainer, "current_epoch", "N/A")
    
    # 获取当前训练阶段
    current_stage = pl_module.hparams.config.get("current_stage", "stage2")
    
    # 收集所有指标
    itc_i2t_accuracy = 0.0
    itc_t2i_accuracy = 0.0
    mlm_accuracy = 0.0
    cross_modal_mlm_accuracy = 0.0
    
    # ITC 指标
    itc_i2t_key = f"{phase}/{VL_PRETRAIN_ITC_I2T_METRIC_NAME}"
    itc_t2i_key = f"{phase}/{VL_PRETRAIN_ITC_T2I_METRIC_NAME}"
    
    itc_i2t_tensor = computed_metrics.get(itc_i2t_key)
    itc_t2i_tensor = computed_metrics.get(itc_t2i_key)
    
    if itc_i2t_tensor is not None:
        itc_i2t_accuracy = itc_i2t_tensor.item()
        rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} ITC I2T Accuracy: {itc_i2t_accuracy:.4f}")
    
    if itc_t2i_tensor is not None:
        itc_t2i_accuracy = itc_t2i_tensor.item()
        rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} ITC T2I Accuracy: {itc_t2i_accuracy:.4f}")
    
    # MLM 指标
    if current_stage == "stage2":
        mlm_key = f"{phase}/{VL_PRETRAIN_MLM_METRIC_NAME}"
        mlm_tensor = computed_metrics.get(mlm_key)
        if mlm_tensor is not None:
            mlm_accuracy = mlm_tensor.item()
            rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} MLM Accuracy: {mlm_accuracy:.4f}")
    elif current_stage == "stage3":
        cross_modal_mlm_key = f"{phase}/{VL_PRETRAIN_CROSS_MODAL_MLM_METRIC_NAME}"
        cross_modal_mlm_tensor = computed_metrics.get(cross_modal_mlm_key)
        if cross_modal_mlm_tensor is not None:
            cross_modal_mlm_accuracy = cross_modal_mlm_tensor.item()
            rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} Cross-Modal MLM Accuracy: {cross_modal_mlm_accuracy:.4f}")
    
    # 计算总指标
    if current_stage == "stage2":
        # Stage 2: ITC + MLM 的平均值
        the_metric = (itc_i2t_accuracy + itc_t2i_accuracy + mlm_accuracy) / 3.0
        rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} Stage2 Combined Metric: {the_metric:.4f}")
    elif current_stage == "stage3":
        # Stage 3: ITC + Cross-Modal MLM 的加权平均
        itc_weight = 0.5
        cross_modal_mlm_weight = 0.5
        the_metric = (itc_i2t_accuracy + itc_t2i_accuracy) * itc_weight / 2.0 + cross_modal_mlm_accuracy * cross_modal_mlm_weight
        rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} Stage3 Combined Metric: {the_metric:.4f}")
    else:
        # 默认使用ITC指标的平均值
        the_metric = (itc_i2t_accuracy + itc_t2i_accuracy) / 2.0
        rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} Default Combined Metric: {the_metric:.4f}")
    
    # 记录总指标
    pl_module.log(f"{phase}/the_metric", the_metric, rank_zero_only=True)
    rank_zero_info(f"Epoch {cur_epoch} {phase.capitalize()} 'the_metric' (Vision-Language Pretraining): {the_metric:.4f}")


def set_schedule_for_vision_language_pretraining(pl_module):
    """为视觉-语言预训练任务设置优化器和学习率调度器"""
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
        ".cross_attn_layernorm.weight",
        ".patch_embed.norm.weight",
        ".Encoder.cls_token",
        ".Encoder.token_type_embeddings.weight",
        ".gamma_sa", 
        ".gamma_ffn",
        ".gamma_ca",  # 交叉注意力的layer scale
        ".decoder_norm.weight",  # 解码器归一化层
        "logit_scale",  # 对比学习的logit scale
    ]
    
    # 定义头部参数（高学习率）
    head_names = [
        "mlm_score.",
        "cross_modal_mlm_score.",
        "itc_text_proj.",
        "itc_image_proj.",
        "cross_modal_decoder.",  # 交叉模态解码器
    ]
    
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
        else: # 主模型参数（编码器等）
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
            optimizer_grouped_parameters, lr=lr, eps=config.get("adam_epsilon", 1e-8), 
            betas=config.get("adam_betas", (0.9, 0.98))
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters, lr=lr, eps=config.get("adam_epsilon", 1e-8), 
            betas=config.get("adam_betas", (0.9, 0.999))
        )
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(
            optimizer_grouped_parameters, lr=lr, momentum=config.get("sgd_momentum", 0.9)
        )

    if pl_module.trainer is None:
        rank_zero_info("Trainer not attached to pl_module. Cannot determine max_steps for scheduler. Scheduler will not be configured.")
        return [optimizer], []
    
    # 计算最大训练步数
    max_steps = -1
    if hasattr(pl_module.trainer, 'max_steps') and pl_module.trainer.max_steps is not None and pl_module.trainer.max_steps != -1:
        max_steps = pl_module.trainer.max_steps
    elif (hasattr(pl_module.trainer, 'datamodule') and hasattr(pl_module.trainer.datamodule, 'train_dataloader') and 
          hasattr(pl_module.trainer, 'max_epochs') and hasattr(pl_module.trainer, 'accumulate_grad_batches')):
        try:
            train_dataloader = pl_module.trainer.datamodule.train_dataloader()
            if (train_dataloader is not None and pl_module.trainer.max_epochs is not None and 
                pl_module.trainer.accumulate_grad_batches is not None):
                len_train_loader = len(train_dataloader)
                if len_train_loader > 0:
                    max_steps = (
                        len_train_loader * pl_module.trainer.max_epochs 
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
    if isinstance(warmup_steps_config, float) and 0 < warmup_steps_config < 1:
        warmup_steps = int(max_steps * warmup_steps_config)
    else:
        warmup_steps = int(warmup_steps_config)

    rank_zero_info("Vision-Language Pretraining - Warmup_steps:{} \t Max_steps:{}".format(warmup_steps, max_steps))

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
                power=float(decay_power_config) if isinstance(decay_power_config, (str, int, float)) and decay_power_config != "cosine" else 1.0,
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

    sched = {"scheduler": scheduler, "interval": "step", "frequency": 1}

    return [optimizer], [sched]

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

def check_non_acc_grad(pl_module):
    if pl_module.token_type_embeddings.weight.grad is None:
        return True
    else:
        grad = pl_module.token_type_embeddings.weight.grad
        return (grad.sum() == 0).item()


def set_task(pl_module):
    pl_module.current_tasks = [
        k for k, v in pl_module.hparams.config["loss_names"].items() if v >= 1
    ]
    return


def set_schedule(pl_module):
    lr = pl_module.hparams.config["learning_rate"]
    wd = pl_module.hparams.config["weight_decay"]

    no_decay = [
        "bias",
        "LayerNorm.bias",
        "LayerNorm.weight",
        "norm.bias",
        "norm.weight",
        "norm1.bias",
        "norm1.weight",
        "norm2.bias",
        "norm2.weight",
    ]
    head_names = ["vqa_classifier", "nlvr2_classifier"]
    lr_mult = pl_module.hparams.config["lr_mult"]
    end_lr = pl_module.hparams.config["end_lr"]
    decay_power = pl_module.hparams.config["decay_power"]
    optim_type = pl_module.hparams.config["optim_type"]

    names = [n for n, p in pl_module.named_parameters()]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay)
                and not any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if not any(nd in n for nd in no_decay)
                and any(bb in n for bb in head_names)
            ],
            "weight_decay": wd,
            "lr": lr * lr_mult,
        },
        {
            "params": [
                p
                for n, p in pl_module.named_parameters()
                if any(nd in n for nd in no_decay) and any(bb in n for bb in head_names)
            ],
            "weight_decay": 0.0,
            "lr": lr * lr_mult,
        },
    ]

    if optim_type == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98)
        )
    elif optim_type == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    elif optim_type == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)

    if pl_module.trainer.max_steps is None or pl_module.trainer.max_steps==-1:
        max_steps = (
            len(pl_module.trainer.datamodule.train_dataloader())
            * pl_module.trainer.max_epochs
            // pl_module.trainer.accumulate_grad_batches
        )
    else:
        max_steps = pl_module.trainer.max_steps

    warmup_steps = pl_module.hparams.config["warmup_steps"]
    if isinstance(pl_module.hparams.config["warmup_steps"], float):
        warmup_steps = int(max_steps * warmup_steps)
    rank_zero_info("Warmup_steps:{} \t Max_steps:{}".format(warmup_steps, max_steps))

    if decay_power == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
            lr_end=end_lr,
            power=decay_power,
        )

    sched = {"scheduler": scheduler, "interval": "step"}

    return (
        [optimizer],
        [sched],
    )