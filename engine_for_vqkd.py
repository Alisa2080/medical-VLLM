# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import math
import sys
from typing import Iterable

import torch
import torch.nn as nn

import utils

def train_one_epoch(model: torch.nn.Module, 
                            data_loader: Iterable, 
                            optimizer: torch.optim.Optimizer,
                            device: torch.device, 
                            epoch: int, 
                            loss_scaler, 
                            clip_grad: float = 0,
                            log_writer=None, 
                            lr_scheduler=None, 
                            start_steps=None,
                            lr_schedule_values=None,
                            args=None,
                            ):
    """
    训练模型一个epoch的函数。

    参数:
    model (torch.nn.Module): 要训练的模型。
    data_loader (Iterable): 数据加载器，用于提供训练数据。
    optimizer (torch.optim.Optimizer): 优化器，用于更新模型参数。
    device (torch.device): 训练设备，如CPU或GPU。
    epoch (int): 当前训练的epoch数。
    loss_scaler: 损失缩放器，用于混合精度训练。
    clip_grad (float): 梯度裁剪的阈值，默认为0。
    log_writer: 日志记录器，用于记录训练信息。
    lr_scheduler: 学习率调度器，用于调整学习率。
    start_steps (int): 全局训练迭代的起始步数。
    lr_schedule_values (list): 学习率调度值列表。
    args: 包含其他参数的对象。

    返回:
    dict: 包含训练统计信息的字典。
    """
    model.train()  # 将模型设置为训练模式
    metric_logger = utils.MetricLogger(delimiter="  ")  # 创建一个指标记录器
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 添加学习率指标
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 添加最小学习率指标
    header = 'Epoch: [{}]'.format(epoch)  # 日志输出的头部信息
    print_freq = 10  # 日志打印频率

    if hasattr(model, 'quantize'):
        # 如果模型有量化模块，重置量化器中的码本统计信息
        try:
            model.module.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before each epoch")
        except:
            pass
        
    for step, (batch, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 为每个步骤分配学习率和权重衰减
        it = start_steps + step  # 全局训练迭代步数
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    # 根据调度值更新学习率
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
        images = batch.to(device, non_blocking=True)  # 将图像数据移动到指定设备

        with torch.amp.autocast(enabled=True, device_type='cuda'):
            # 前向传播计算损失
            loss, log_loss = model(images)

        loss_value = loss.item()  # 获取损失值

        if not math.isfinite(loss_value):
            # 如果损失值为非有限值，停止训练并保存模型
            print("Loss is {}, stopping training".format(loss_value), force=True)
            utils.save_nan_model(args, model)
            sys.exit(1)

        optimizer.zero_grad()  # 清零优化器的梯度
        # 检查是否为二阶优化器
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        # 反向传播并更新参数
        # grad_norm 代表梯度的范数（norm），也就是梯度向量的长度。
        grad_norm = loss_scaler(loss, optimizer, clip_grad=clip_grad,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]  # 获取损失缩放值
        
        torch.cuda.synchronize()  # 同步CUDA设备

        metric_logger.update(loss=loss_value)  # 更新损失指标
        
        # 处理日志中的损失信息
        new_log_loss = {k.split('/')[-1]:v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)  # 更新其他损失指标

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            # 找出最小和最大学习率
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)  # 更新最大学习率指标
        metric_logger.update(min_lr=min_lr)  # 更新最小学习率指标
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                # 获取权重衰减值
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)  # 更新权重衰减指标
        metric_logger.update(grad_norm=grad_norm)  # 更新梯度范数指标

        if log_writer is not None:
            # 记录损失信息到日志
            log_writer.update(**new_log_loss, head="train/loss")
            # 记录学习率信息到日志
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            # 记录权重衰减信息到日志
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            # 记录梯度范数信息到日志
            log_writer.update(grad_norm=grad_norm, head="opt")
            # 记录损失缩放值信息到日志
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            # 设置日志记录的步数
            log_writer.set_step()

        if lr_scheduler is not None:
            # 更新学习率调度器
            lr_scheduler.step_update(start_steps + step)
    # 同步所有进程的统计信息
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # 统计码本使用信息
    if hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()  # 统计未使用的码本数量
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        train_stat['Unused_code'] = zero_cnt  # 将未使用的码本数量添加到统计信息中
        print(f"Unused code in codebook: {zero_cnt}")
        return train_stat
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, log_writer=None, epoch=None, args=None):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Validation:'

    # switch to evaluation mode
    model.eval()

    if hasattr(model.module, 'quantize'):
        try:
            model.module.quantize.reset_cluster_size(device)
            print("Reset the codebook statistic info in quantizer before testing")
        except:
            pass

    for step, (batch, extra_info) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        images = batch.to(device, non_blocking=True)
        loss, log_loss = model(images)

        metric_logger.update(loss=loss.item())

        new_log_loss = {k.split('/')[-1]:v for k, v in log_loss.items() if k not in ['total_loss']}
        metric_logger.update(**new_log_loss)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # stat the codebook usage information
    if hasattr(model, 'module') and hasattr(model.module, 'quantize'):
        try:
            codebook_cluster_size = model.module.quantize._codebook.cluster_size
        except:
            codebook_cluster_size = model.module.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code'] = zero_cnt
        print(f"Unused code in codebook: {zero_cnt}")
        return test_stat

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def calculate_codebook_usage(data_loader, model, device, log_writer=None, epoch=None, args=None):
    """
    计算码本中每个码本条目的使用情况。

    参数:
    data_loader (Iterable): 数据加载器，用于提供评估数据。
    model (torch.nn.Module): 要评估的模型。
    device (torch.device): 评估设备，如CPU或GPU。
    log_writer: 日志记录器，用于记录评估信息。
    epoch (int): 当前评估的epoch数。
    args: 包含其他参数的对象。

    返回:
    无
    """
    # 创建一个指标记录器，用于记录评估过程中的指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 日志输出的头部信息
    header = 'Calculating codebook usage:'

    # 将模型设置为评估模式，以确保在评估过程中不进行训练相关的操作，如Dropout等
    model.eval()
    
    # 从参数中获取码本的条目数量
    codebook_num = args.codebook_n_emd
    # 初始化一个长度为码本条目数量的张量，用于记录每个码本条目的使用次数，初始值都为0
    codebook_cnt = torch.zeros(codebook_num, dtype=torch.float64).to(device)

    # 遍历数据加载器中的每个批次
    for step, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # 将图像数据移动到指定设备
        images = images.to(device, non_blocking=True)

        # 通过模型获取图像对应的码本索引，并将其展平为一维张量
        outputs = utils.get_model(model).get_tokens(images)['token'].view(-1)
        
        # 初始化一个列表，用于收集所有进程的输出结果
        outputs_gather_list = [torch.zeros_like(outputs) for _ in range(utils.get_world_size())]
        # 使用分布式通信机制，将当前进程的输出结果收集到所有进程的outputs_gather_list中
        
        # 统计每个码本条目在当前批次中的使用次数，并累加到codebook_cnt中
        codebook_cnt += torch.bincount(outputs, minlength=codebook_num)

    # 统计码本中未使用的条目数量
    zero_cnt = (codebook_cnt == 0).sum() # 0
    # 打印统计信息，包括未使用的条目数量和占比
    print(f"STAT:  {zero_cnt} tokens ({(zero_cnt / codebook_num) * 100}%) never are used in this codebook.")