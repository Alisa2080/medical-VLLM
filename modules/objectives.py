import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from pytorch_lightning.utilities import rank_zero_info
import torchmetrics
from metrics.my_metrics import Accuracy as MyAccuracy, Scalar as MyScalar, VQAScore as MyVQAScore


def _get_mlm_acc_metric(pl_module):
    """
    取得（或懒加载创建）一个自定义的 Accuracy metric。
    挂到 pl_module 上，避免重复创建和跨 GPU 不一致。
    """
    if not hasattr(pl_module, "_mlm_acc_metric"):
        # Use the custom Accuracy metric from my_metrics.py
        rank_zero_info("Initializing custom MLM Accuracy metric.")
        pl_module._mlm_acc_metric = MyAccuracy().to(pl_module.device)
    return pl_module._mlm_acc_metric


def compute_mlm(pl_module, batch):
    # 对输入数据进行推理，启用文本掩码，禁用图像掩码
    infer = pl_module.infer(batch, mask_text=True, mask_image=False)
    # 通过 MLM 评分函数计算掩码语言模型的 logits
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    # 获取真实的文本标签
    mlm_labels = infer["text_labels"]

    # 计算交叉熵损失，忽略标签为 -100 的 token
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )
    
    # 获取精度评估器
    acc_metric = _get_mlm_acc_metric(pl_module)
    preds = mlm_logits.argmax(-1)
    mask = mlm_labels.ne(-100)
    
    # 修改：仅当有掩码标记时更新指标，避免不必要的计算
    if mask.sum() > 0:
        acc_metric.update(preds[mask], mlm_labels[mask])
        
    # if hasattr(pl_module, "_mlm_acc_ever_updated") and pl_module._mlm_acc_ever_updated:
        # acc = acc_metric.compute()
    # else:
        # acc = torch.tensor(0.0).to(pl_module.device)
# 
    # 根据当前阶段（训练或验证）获取相应的损失和准确率计算函数
    phase = "train" if pl_module.training else "val"
    # 记录当前阶段的损失和准确率
    pl_module.log(f"mlm/{phase}/loss", mlm_loss * 0.25, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    pl_module.log(f"mlm/{phase}/accuracy", acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    return {
        "mlm_loss": mlm_loss * 0.25,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }

def compute_textonly_mlm(pl_module, batch):
    # 对纯文本数据进行推理，启用文本掩码
    infer = pl_module.infer_text_mlm(batch, mask_text=True)
    # 通过 MLM 评分函数计算掩码语言模型的 logits
    mlm_logits = pl_module.mlm_score(infer["text_feats"])
    # 获取真实的文本标签
    mlm_labels = infer["text_labels"]

    # 计算交叉熵损失，忽略标签为 -100 的 token
    mlm_loss = F.cross_entropy(
        mlm_logits.view(-1, pl_module.hparams.config["vocab_size"]),
        mlm_labels.view(-1),
        ignore_index=-100,
    )

    # 构建返回结果字典，包含损失、logits、标签和文本 ID
    acc_metric = _get_mlm_acc_metric(pl_module)
    preds = mlm_logits.argmax(-1)
    mask = mlm_labels.ne(-100)
    
    # 修改：仅当有掩码标记时更新指标，避免不必要的计算
    if mask.sum() > 0:
        acc_metric.update(preds[mask], mlm_labels[mask])
    # if hasattr(pl_module, "_mlm_acc_ever_updated") and pl_module._mlm_acc_ever_updated:
        # acc = acc_metric.compute()
    # 获取当前累积的准确率，但不改变内部状态
    # else:
        # acc = torch.tensor(0.0).to(pl_module.device)
# 
    # 根据当前阶段（训练或验证）获取相应的损失和准确率计算函数
    phase = "train" if pl_module.training else "val"
    # 记录当前阶段的损失和准确率
    pl_module.log(f"textmlm/{phase}/loss", mlm_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    pl_module.log(f"textmlm/{phase}/accuracy", acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    return {
        "mlm_loss": mlm_loss,
        "mlm_logits": mlm_logits,
        "mlm_labels": mlm_labels,
        "mlm_ids": infer["text_ids"],
    }


def compute_itm_hardneg(pl_module, batch, sim_i2t, sim_t2i):
    pos_len = batch["text_ids"].size(0)
    neg_len = batch["text_ids"].size(0)
    bsz = batch["text_ids"].size(0)
    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len), torch.zeros(neg_len)]).to(
        pl_module.device
    )

    batch = {k: v for k, v in batch.items()}
    infer_pos = pl_module.infer(batch, mask_text=False, mask_image=False)

    batch_text_ids = infer_pos["text_ids"]
    batch_text_masks = infer_pos["text_masks"]
    batch_image = infer_pos["image"]

    with torch.no_grad():
        all_text_ids = batch_text_ids
        all_text_masks = batch_text_masks
        all_image = batch_image

    with torch.no_grad():       
        weights_i2t = F.softmax(sim_i2t[:bsz, :].float(), dim=1)
        weights_t2i = F.softmax(sim_t2i[:bsz, :].float(), dim=1)

        weights_i2t.fill_diagonal_(0)
        weights_t2i.fill_diagonal_(0)
    
    images_neg = []    
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        images_neg.append(all_image[neg_idx])
    images_neg = torch.stack(images_neg, dim=0)   

    # select a negative text for each image
    text_ids_neg = []
    text_masks_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(all_text_ids[neg_idx])
        text_masks_neg.append(all_text_masks[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)     
    text_masks_neg = torch.stack(text_masks_neg, dim=0)      

    # text_labels is not used in ITM loss
    batch_imgs_neg = {"image":[images_neg], "text_ids":batch["text_ids"], "text_labels":batch["text_labels"], "text_masks":batch["text_masks"]}
    infer_imags_neg = pl_module.infer(batch_imgs_neg, mask_text=False, mask_image=False)
    
    batch_text_neg = {"image":batch["image"], "text_ids":text_ids_neg, "text_labels":batch["text_labels"], "text_masks":text_masks_neg}
    infer_text_neg = pl_module.infer(batch_text_neg, mask_text=False, mask_image=False)

    all_cls_feats = torch.cat([infer_pos["cls_feats"], infer_imags_neg["cls_feats"], infer_text_neg["cls_feats"]], dim=0)

    itm_logits = pl_module.itm_score(all_cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }

    phase = "train" if pl_module.training else "val"
    pl_module.log(f"itm/{phase}/loss", ret["itm_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
    acc_metric = getattr(pl_module, f"{phase}_itm_accuracy")
    preds = ret["itm_logits"].argmax(dim=-1)
    acc_metric.update(preds, ret["itm_labels"])
    pl_module.log(f"itm/{phase}/accuracy", acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    return ret


def compute_itc(pl_module, batch, aggregate=False):
    """
    计算图像-文本对比（ITC）损失。
    
    参数:
        pl_module (LightningModule): 模型的Lightning模块实例。
        batch (dict): 包含输入数据的字典，如文本和图像。
        aggregate (bool): 在多GPU环境下是否聚合特征，默认为False。
    
    返回:
        dict: 包含ITC损失、logits、标签和logit缩放因子的字典。
    """
    # 对输入的图像数据进行推理，不进行图像掩码处理
    infer_imag = pl_module.infer_image(batch, mask_image=False)
    # 对输入的文本数据进行推理，不进行文本掩码处理
    infer_text = pl_module.infer_text(batch, mask_text=False)

    # 从图像和文本推理结果中获取CLS特征
    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    
    # 计算logit缩放因子
    logit_scale = pl_module.logit_scale.exp().mean()

    # 计算图像和文本特征之间的相似度得分
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logits_per_image.t()

    # 创建一个表示真实标签的张量，即图像和文本的匹配索引
    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

    # 计算图像-文本对比损失，通过交叉熵损失计算图像到文本和文本到图像的损失并取平均
    itc_loss = (
        F.cross_entropy(logits_per_image.float(), ground_truth)
        + F.cross_entropy(logits_per_text.float(), ground_truth)
    ) / 2

    # 构建包含损失、logits、标签和缩放因子的返回字典
    ret = {
        "itc_loss": itc_loss,
        "itc_i2t_logits": logits_per_image,
        "itc_t2i_logits": logits_per_text,
        "itc_labels": ground_truth,
        "itc_logit_scale": logit_scale,
    }

    # 判断当前是训练阶段还是验证阶段
    phase = "train" if pl_module.training else "val"
    
    # 计算并记录当前阶段的ITC损失
    pl_module.log(f"itc/{phase}/loss", ret["itc_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
    pl_module.log(f"itc/{phase}/logit_scale", ret["itc_logit_scale"], prog_bar=False, logger=True, on_step=True, on_epoch=True) # 直接记录缩放因子
    # 获取准确率指标，更新它们，并记录指标对象
    i2t_acc_metric = getattr(pl_module, f"{phase}_itc_i2t_accuracy")
    i2t_preds = ret["itc_i2t_logits"].argmax(dim=-1)
    i2t_acc_metric.update(i2t_preds, ret["itc_labels"])
    pl_module.log(f"itc/{phase}/i2t_accuracy", i2t_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    t2i_acc_metric = getattr(pl_module, f"{phase}_itc_t2i_accuracy")
    t2i_preds = ret["itc_t2i_logits"].argmax(dim=-1) # 获取预测类别
    t2i_acc_metric.update(t2i_preds, ret["itc_labels"])
    pl_module.log(f"itc/{phase}/t2i_accuracy", t2i_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    return ret

def compute_itc_siglip(pl_module, batch):
    """
    使用SigLIP损失计算图像-文本对比损失，简化版本无需cls_vlffn_feats。
    
    参数:
        pl_module (LightningModule): 模型的Lightning模块实例
        batch (dict): 包含输入数据的字典，如文本和图像
        
    返回:
        dict: 包含ITC损失、logits、标签和logit缩放因子的字典
    """
    # 对输入的图像数据进行推理
    infer_imag = pl_module.infer_image(batch, mask_image=False)
    # 对输入的文本数据进行推理
    infer_text = pl_module.infer_text(batch, mask_text=False)

    # 获取特征表示
    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()
    
    # 计算原始logits矩阵
    raw_logits = image_features @ text_features.t()
    
    # 应用温度缩放和可选偏置
    if hasattr(pl_module, "siglip_bias"):
        logits = raw_logits * logit_scale + pl_module.siglip_bias
    else:
        logits = raw_logits * logit_scale
    
    # 创建一个表示真实标签的张量
    batch_size = logits.size(0)
    ground_truth = torch.arange(batch_size, device=logits.device)
    
    # SigLIP损失的核心实现:
    # 创建对角矩阵标识正样本对
    eye = torch.eye(batch_size, device=logits.device)
    # 对角线=+1(正样本)，非对角线=-1(负样本)
    m1_diag1 = -torch.ones_like(logits) + 2 * eye
    # 应用log_sigmoid，正样本求log_sigmoid(logit)，负样本求log_sigmoid(-logit)
    loglik = F.logsigmoid(m1_diag1 * logits)
    # 每个样本的负对数似然
    nll = -torch.sum(loglik, dim=-1)
    # 最终损失是NLL的平均值
    itc_loss = torch.mean(nll)

    # 构建返回结果字典
    ret = {
        "itc_loss": itc_loss,
        "itc_i2t_logits": logits,  # 使用相同的logits用于评估准确率
        "itc_t2i_logits": logits.t(),
        "itc_labels": ground_truth,
        "itc_logit_scale": logit_scale,
    }

    # 记录度量
    phase = "train" if pl_module.training else "val"
    pl_module.log(f"itc/{phase}/loss", ret["itc_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
    pl_module.log(f"itc/{phase}/logit_scale", ret["itc_logit_scale"], prog_bar=False, logger=True, on_step=True, on_epoch=True)

    i2t_acc_metric = getattr(pl_module, f"{phase}_itc_i2t_accuracy")
    i2t_preds = ret["itc_i2t_logits"].argmax(dim=-1) # 获取预测类别
    i2t_acc_metric.update(i2t_preds, ret["itc_labels"])
    pl_module.log(f"itc/{phase}/i2t_accuracy", i2t_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    t2i_acc_metric = getattr(pl_module, f"{phase}_itc_t2i_accuracy")
    t2i_preds = ret["itc_t2i_logits"].argmax(dim=-1)
    t2i_acc_metric.update(t2i_preds, ret["itc_labels"])
    pl_module.log(f"itc/{phase}/t2i_accuracy", t2i_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    return ret

# The implementation of image-text contrastive refers to open_clip (https://github.com/mlfoundations/open_clip))失并取平均
def compute_irtr(pl_module, batch, aggregate=False):
    # pl_module.logit_scale.data = torch.clamp(pl_module.logit_scale.data, 0, 4.6052)

    infer_imag = pl_module.infer_image_ft(batch, mask_image=False)
    infer_text = pl_module.infer_text_ft(batch, mask_text=False)

    image_features = infer_imag["cls_feats"]
    text_features = infer_text["cls_feats"]
    logit_scale = pl_module.logit_scale.exp().mean()

    if aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_image_features = [
            torch.zeros_like(image_features) for _ in range(world_size)
        ]
        gathered_text_features = [
            torch.zeros_like(text_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_image_features, image_features)
        dist.all_gather(gathered_text_features, text_features)

        all_image_features = torch.cat(
            [image_features]
            + gathered_image_features[:rank]
            + gathered_image_features[rank + 1 :]
        )
        all_text_features = torch.cat(
            [text_features]
            + gathered_text_features[:rank]
            + gathered_text_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_image = logit_scale * all_image_features @ all_text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

    ground_truth = torch.arange(len(logits_per_image)).long().to(device=logits_per_image.get_device())

    irtr_loss = (
        F.cross_entropy(logits_per_image.float(), ground_truth)
        + F.cross_entropy(logits_per_text.float(), ground_truth)
    ) / 2

    ret = {
        "irtr_loss": irtr_loss,
        "irtr_i2t_logits": logits_per_image,
        "irtr_t2i_logits": logits_per_text,
        "irtr_labels": ground_truth,
        "irtr_logit_scale": logit_scale,
    }

    phase = "train" if pl_module.training else "val"
    pl_module.log(f"irtr/{phase}/loss", ret["irtr_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
    pl_module.log(f"irtr/{phase}/logit_scale", ret["irtr_logit_scale"], prog_bar=False, logger=True, on_step=True, on_epoch=True)

    i2t_acc_metric = getattr(pl_module, f"{phase}_irtr_i2t_accuracy")
    i2t_preds = ret["irtr_i2t_logits"].argmax(dim=-1) # 获取预测类别
    i2t_acc_metric.update(i2t_preds, ret["irtr_labels"])
    pl_module.log(f"irtr/{phase}/i2t_accuracy", i2t_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    t2i_acc_metric = getattr(pl_module, f"{phase}_irtr_t2i_accuracy")
    t2i_preds = ret["irtr_t2i_logits"].argmax(dim=-1) # 获取预测类别
    t2i_acc_metric.update(t2i_preds, ret["irtr_labels"])
    pl_module.log(f"irtr/{phase}/t2i_accuracy", t2i_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    return ret


def compute_vqa(pl_module, batch):
    infer = pl_module.infer(batch, mask_text=False, mask_image=False)
    vqa_logits = pl_module.vqa_classifier(infer["cls_feats"])
    vqa_targets = torch.zeros(
        len(vqa_logits), pl_module.hparams.config["vqav2_label_size"]
    ).to(pl_module.device)

    vqa_labels = batch["vqa_labels"]
    vqa_scores = batch["vqa_scores"]

    for i, (_label, _score) in enumerate(zip(vqa_labels, vqa_scores)):
        for l, s in zip(_label, _score):
            vqa_targets[i, l] = s

    vqa_loss = (
        F.binary_cross_entropy_with_logits(vqa_logits, vqa_targets)
        * vqa_targets.shape[1]
    )  # https://github.com/jnhwkim/ban-vqa/blob/master/train.py#L19

    ret = {
        "vqa_loss": vqa_loss,
        "vqa_logits": vqa_logits,
        "vqa_targets": vqa_targets,
        "vqa_labels": vqa_labels,
        "vqa_scores": vqa_scores,
    }

    phase = "train" if pl_module.training else "val"
    pl_module.log(f"vqa/{phase}/loss", ret["vqa_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)

    # VQA score 通常是自定义的，检查指标对象如何工作
    # 假设它是一个需要 update 的 torchmetrics 对象
    score_metric = getattr(pl_module, f"{phase}_vqa_score")
    score_metric.update(ret["vqa_logits"], ret["vqa_targets"])
    pl_module.log(f"vqa/{phase}/score", score_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    return ret


def compute_nlvr2(pl_module, batch):
    infer1 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=1
    )
    infer2 = pl_module.infer(
        batch, mask_text=False, mask_image=False, image_token_type_idx=2
    )

    cls_feats = torch.cat([infer1["cls_feats"], infer2["cls_feats"]], dim=-1)
    nlvr2_logits = pl_module.nlvr2_classifier(cls_feats)

    nlvr2_labels = batch["answers"]
    nlvr2_labels = torch.tensor(nlvr2_labels).to(pl_module.device).long()
    nlvr2_loss = F.cross_entropy(nlvr2_logits, nlvr2_labels)

    ret = {
        "nlvr2_loss": nlvr2_loss,
        "nlvr2_logits": nlvr2_logits,
        "nlvr2_labels": nlvr2_labels,
    }

    phase = "train" if pl_module.training else "val"

    if phase == "train":
        pl_module.log(f"nlvr2/{phase}/loss", ret["nlvr2_loss"], prog_bar=False, logger=True, on_step=True, on_epoch=True)
        acc_metric = getattr(pl_module, f"{phase}_nlvr2_accuracy")
        preds = ret["nlvr2_logits"].argmax(dim=-1) # 获取预测类别
        acc_metric.update(preds, ret["nlvr2_labels"])
        pl_module.log(f"nlvr2/{phase}/accuracy", acc_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True)
 
    else:
        dev_batches = [i for i, n in enumerate(batch["table_name"]) if "dev" in n]
        test_batches = [i for i, n in enumerate(batch["table_name"]) if "test" in n]

        pl_module.log(f"nlvr2/{phase}/loss", ret["nlvr2_loss"], prog_bar=False, logger=True, on_step=False, on_epoch=True) # 记录 epoch 损失
        acc_metric = getattr(pl_module, f"{phase}_nlvr2_accuracy") # 这应该是主要的验证指标
        preds = ret["nlvr2_logits"].argmax(dim=-1) # 获取预测类别
        acc_metric.update(preds, ret["nlvr2_labels"])
        pl_module.log(f"nlvr2/{phase}/accuracy", acc_metric, prog_bar=True, logger=True, on_step=False, on_epoch=True)

        # 如果需要，使用单独的指标处理 dev/test 特定的日志记录
        if dev_batches:
            # 注意：为 dev/test 创建单独的指标对象（在 __init__ 或 set_metrics 中）
            dev_acc_metric = getattr(pl_module, f"dev_nlvr2_accuracy")
            dev_preds = ret["nlvr2_logits"][dev_batches].argmax(dim=-1) # 获取预测类别
            dev_labels = ret["nlvr2_labels"][dev_batches]
            dev_loss = F.cross_entropy(ret["nlvr2_logits"][dev_batches], ret["nlvr2_labels"][dev_batches])
            pl_module.log(f"nlvr2/dev/loss", dev_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            dev_acc_metric.update(dev_preds, dev_labels)
            pl_module.log(f"nlvr2/dev/accuracy", dev_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True, add_dataloader_idx=False) # 单独记录 dev 准确率
            
        if test_batches:
            # 注意：为 dev/test 创建单独的指标对象（在 __init__ 或 set_metrics 中）
            test_loss = F.cross_entropy(ret["nlvr2_logits"][test_batches], ret["nlvr2_labels"][test_batches])
            pl_module.log(f"nlvr2/test/loss", test_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            test_acc_metric = getattr(pl_module, f"test_nlvr2_accuracy")
            test_preds = ret["nlvr2_logits"][test_batches].argmax(dim=-1)
            test_labels = ret["nlvr2_labels"][test_batches]
            test_acc_metric.update(test_preds, test_labels)
            pl_module.log(f"nlvr2/test/accuracy", test_acc_metric, prog_bar=False, logger=True, on_step=False, on_epoch=True, add_dataloader_idx=False) # 单独记录 test 准确率

    return ret


@torch.no_grad()
def compute_irtr_recall(pl_module, split="test"):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if split == "val":
        rank_zero_info("Use val set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    else:
        rank_zero_info("Use test set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()

    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    if split == "val":
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
            image_only=True
        )
    else:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(
            image_only=True
        )

    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    for pre in text_preload:
        tiids += pre["img_index"]
    tiids = torch.tensor(tiids)

    rank_zero_info("len(tiids): {}".format(len(tiids)))

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append(
            {
                "image": [_b["image"][0].to(pl_module.device)],
                "img_index": _b["img_index"],
            }
        )
    iids = list()
    for pre in image_preload:
        iids += pre["img_index"]
    iids = torch.tensor(iids)

    rank_zero_info("len(iids): {}".format(len(iids)))

    txt_cls_feats = list()
    for txt_batch in text_preload:
        with torch.amp.autocast(device_type="cuda"):
            cls_feats = pl_module.infer_text_ft(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    }
                )["cls_feats"]
        txt_cls_feats.append(cls_feats)

    img_cls_feats = list()
    for img_batch in image_preload:
        with torch.amp.autocast(device_type="cuda"):
            cls_feats = pl_module.infer_image_ft(
                    {
                        "image": img_batch["image"],
                    }
                )["cls_feats"]
        img_cls_feats.append(cls_feats)

    txt_cls_feats = torch.cat(txt_cls_feats)
    img_cls_feats = torch.cat(img_cls_feats)

    rank_zero_info("txt_cls_feats.size(): {}\t{}".format(txt_cls_feats.size(), split))
    rank_zero_info("img_cls_feats.size(): {}\t{}".format(img_cls_feats.size(), split))

    scores = img_cls_feats @ txt_cls_feats.t()

    rank_zero_info("scores.size(): {}".format(scores.size(), split))

    topk10 = scores.topk(10, dim=1)
    topk5 = scores.topk(5, dim=1)
    topk1 = scores.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices.to(tiids.device)]
    topk5_iids = tiids[topk5.indices.to(tiids.device)]
    topk1_iids = tiids[topk1.indices.to(tiids.device)]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores.topk(10, dim=0)
    topk5 = scores.topk(5, dim=0)
    topk1 = scores.topk(1, dim=0)
    topk10_iids = iids[topk10.indices.to(iids.device)]
    topk5_iids = iids[topk5.indices.to(iids.device)]
    topk1_iids = iids[topk1.indices.to(iids.device)]

    ir_r10 = (tiids.unsqueeze(0) == topk10_iids).float().max(dim=0)[0].mean()
    ir_r5 = (tiids.unsqueeze(0) == topk5_iids).float().max(dim=0)[0].mean()
    ir_r1 = (tiids.unsqueeze(0) == topk1_iids).float().max(dim=0)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


@torch.no_grad()
def compute_irtr_recall_with_rerank(pl_module, split="test"):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    if split == "val":
        rank_zero_info("Use val set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset()
    else:
        rank_zero_info("Use test set...")
        text_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset()

    text_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    text_loader = torch.utils.data.DataLoader(
        text_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            text_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    if split == "val":
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_val_dset(
            image_only=True
        )
    else:
        image_dset = pl_module.trainer.datamodule.dms[0].make_no_false_test_dset(
            image_only=True
        )

    image_dset.tokenizer = pl_module.trainer.datamodule.dms[0].tokenizer
    image_loader = torch.utils.data.DataLoader(
        image_dset,
        batch_size=32,
        num_workers=2, #pl_module.hparams.config["num_workers"],
        pin_memory=True,
        collate_fn=functools.partial(
            image_dset.collate,
            mlm_collator=pl_module.trainer.datamodule.dms[0].mlm_collator,
        ),
    )

    text_preload = list()
    for _b in tqdm.tqdm(text_loader, desc="text prefetch loop"):
        text_preload.append(
            {
                "text_ids": _b["text_ids"].to(pl_module.device),
                "text_masks": _b["text_masks"].to(pl_module.device),
                "text_labels": _b["text_labels"].to(pl_module.device),
                "img_index": _b["img_index"],
            }
        )

    tiids = list()
    text_ids_list = list()
    text_masks_list = list()
    text_labels_list = list()
    for pre in text_preload:
        tiids += pre["img_index"]
        text_ids_list.append(pre["text_ids"])
        text_masks_list.append(pre["text_masks"])
        text_labels_list.append(pre["text_labels"])
    tiids = torch.tensor(tiids)
    all_text_ids = torch.cat(text_ids_list)
    all_text_masks = torch.cat(text_masks_list)
    all_text_labels = torch.cat(text_labels_list)

    image_preload = list()
    for _b in tqdm.tqdm(image_loader, desc="image prefetch loop"):
        image_preload.append(
            {
                "image": [_b["image"][0].to(pl_module.device)],
                "img_index": _b["img_index"],
            }
        )
    iids = list()
    image_list = list()
    for pre in image_preload:
        iids += pre["img_index"]
        image_list.append(pre["image"][0])
    iids = torch.tensor(iids)
    all_image = torch.cat(image_list)

    txt_cls_feats = list()
    for txt_batch in text_preload:
        with torch.amp.autocast(device_type="cuda"):
            cls_feats = pl_module.infer_text_ft(
                    {
                        "text_ids": txt_batch["text_ids"],
                        "text_masks": txt_batch["text_masks"],
                        "text_labels": txt_batch["text_labels"],
                    }
                )["cls_feats"]
        txt_cls_feats.append(cls_feats)

    img_cls_feats = list()
    for img_batch in image_preload:
        with torch.amp.autocast(device_type="cuda"):
            cls_feats = pl_module.infer_image_ft(
                    {
                        "image": img_batch["image"],
                    }
                )["cls_feats"]
        img_cls_feats.append(cls_feats)

    txt_cls_feats = torch.cat(txt_cls_feats)
    img_cls_feats = torch.cat(img_cls_feats)

    scores = img_cls_feats @ txt_cls_feats.t()

    rank_zero_info("scores.size(): {}".format(scores.size(), split))


    scores_i2t = torch.full((len(iids), len(tiids)), -100.0).to(pl_module.device)
    
    k_test = pl_module.hparams.config["k_test"]
    num_tasks = world_size
    step = scores.size(0) // num_tasks + 1
    start = rank * step
    end = min(scores.size(0), start+step)

    for i, sims in enumerate(scores[start:end]): 
        if i%100 == 0:
            rank_zero_info("TR Rerank: {}".format(i))
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        cur_images = all_image[start+i].repeat(k_test, 1, 1, 1)
        cur_text_ids = all_text_ids[topk_idx]
        cur_text_masks = all_text_masks[topk_idx]
        cur_text_labels = all_text_labels[topk_idx]
        
        cur_rerank_batch = {"image":[cur_images], "text_ids":cur_text_ids, "text_labels":cur_text_labels, "text_masks":cur_text_masks}
        infer_rerank = pl_module.infer(cur_rerank_batch, mask_text=False, mask_image=False)
        itm_logits = pl_module.itm_score(infer_rerank["cls_feats"])
        itm_scores = itm_logits[:,1]
        scores_i2t[start+i,topk_idx] = itm_scores

    scores = scores.t()
    scores_t2i = torch.full((len(tiids), len(iids)), -100.0).to(pl_module.device)
    
    step = scores.size(0) // num_tasks + 1
    start = rank * step
    end = min(scores.size(0), start+step)    
    
    for i,sims in enumerate(scores[start:end]): 
        topk_sim, topk_idx = sims.topk(k=k_test, dim=0)
        cur_images = all_image[topk_idx]
        cur_text_ids = all_text_ids[start+i].repeat(k_test, 1) 
        cur_text_masks = all_text_masks[start+i].repeat(k_test, 1) 
        cur_text_labels = all_text_labels[start+i].repeat(k_test, 1)
        cur_rerank_batch = {"image":[cur_images], "text_ids":cur_text_ids, "text_labels":cur_text_labels, "text_masks":cur_text_masks}
        infer_rerank = pl_module.infer(cur_rerank_batch, mask_text=False, mask_image=False)
        itm_logits = pl_module.itm_score(infer_rerank["cls_feats"])
        itm_scores = itm_logits[:,1]
        scores_t2i[start+i, topk_idx] = itm_scores

    dist.barrier()   
    torch.distributed.all_reduce(scores_i2t, op=torch.distributed.ReduceOp.SUM) 
    torch.distributed.all_reduce(scores_t2i, op=torch.distributed.ReduceOp.SUM)    


    scores_t2i = scores_t2i + scores
    scores_i2t = scores_i2t + scores.t()

    topk10 = scores_i2t.topk(10, dim=1)
    topk5 = scores_i2t.topk(5, dim=1)
    topk1 = scores_i2t.topk(1, dim=1)
    topk10_iids = tiids[topk10.indices]
    topk5_iids = tiids[topk5.indices]
    topk1_iids = tiids[topk1.indices]

    tr_r10 = (iids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    tr_r5 = (iids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    tr_r1 = (iids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    topk10 = scores_t2i.topk(10, dim=1)
    topk5 = scores_t2i.topk(5, dim=1)
    topk1 = scores_t2i.topk(1, dim=1)
    topk10_iids = iids[topk10.indices]
    topk5_iids = iids[topk5.indices]
    topk1_iids = iids[topk1.indices]

    ir_r10 = (tiids.unsqueeze(1) == topk10_iids).float().max(dim=1)[0].mean()
    ir_r5 = (tiids.unsqueeze(1) == topk5_iids).float().max(dim=1)[0].mean()
    ir_r1 = (tiids.unsqueeze(1) == topk1_iids).float().max(dim=1)[0].mean()

    return (ir_r1, ir_r5, ir_r10, tr_r1, tr_r5, tr_r10)


def init_weights(module):
    # 如果是线性层或嵌入层，使用正态分布初始化权重
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    # 如果是LayerNorm层，初始化偏置为0，权重为1
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

    # 如果是线性层且有偏置项，将偏置初始化为0
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def vqa_test_step(pl_module, batch, output):
    id2answer = (
        pl_module.trainer.datamodule.dm_dicts["vqa_trainval"].id2answer
        if "vqa_trainval" in pl_module.trainer.datamodule.dm_dicts
        else pl_module.trainer.datamodule.dm_dicts["vqa"].id2answer
    )
    vqa_logits = output["vqa_logits"]
    vqa_preds = vqa_logits.argmax(dim=-1)
    vqa_preds = [id2answer[pred.item()] for pred in vqa_preds]
    questions = batch["text"]
    qids = batch["qid"]
    return {"qids": qids, "preds": vqa_preds}


def vqa_test_wrapup(outs, model_name, output_dir):
    rank = torch.distributed.get_rank()
    qids, preds = list(), list()
    for out in outs:
        qids += out["qids"]
        preds += out["preds"]

    rets = list()
    for qid, pred in zip(qids, preds):
        rets.append({"question_id": qid, "answer": pred})
    with open(f"vqa_submit_{rank}.json", "w") as fp:
        json.dump(rets, fp, indent=4)

    torch.distributed.barrier()

    if rank == 0:
        jsons = list()
        paths = list(glob.glob("vqa_submit_*.json"))
        for path in paths:
            with open(path, "r") as fp:
                jsons += json.load(fp)
        # os.makedirs("result", exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/vqa_submit_{model_name}.json", "w") as fp:
            json.dump(jsons, fp, indent=4)

    torch.distributed.barrier()
    os.remove(f"vqa_submit_{rank}.json")
