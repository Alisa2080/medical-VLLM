import os
import copy
import pytorch_lightning as pl

from config import ex
from models.vlmo_module import VLMo
from datamodules.multitask_datamodule import MTDataModule
import torch
from pytorch_lightning.plugins import environments as pl_env
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import Callback
torch.set_float32_matmul_precision('high')

def get_cluster_plugin(num_gpus=1, num_nodes=1):
    # 如果不满足上述条件，返回 None
    return None


@ex.automain
def main(config):
    config = copy.deepcopy(config)
    # 打印模型配置参数
    print("Model configuration parameters:")
    for key, value in config.items():
        print(f"{key}: {value}")
    # 设置随机种子，确保结果可复现
    pl.seed_everything(config["seed"])

    if "textmlm" in config["exp_name"] or config.get("model_type") == "VLMoForTextPretraining":
        from models.vlmo_module import VLMoForTextPretraining # 确保导入
        model_class = VLMoForTextPretraining
        rank_zero_info("Using model: VLMoForTextPretraining")
    else:
        # 默认使用旧的 VLMo 或其他多任务模型
        from models.vlmo_module import VLMo # 确保导入
        model_class = VLMo
        rank_zero_info("Using model: VLMo (multitask)")


    # 初始化多任务数据模块，dist=True 表示使用分布式训练
    dm = MTDataModule(config, dist=False)

    # 初始化 VLMo 模型
    model = VLMo(config)
    # 提取实验名称
    exp_name = f'{config["exp_name"]}'

    # 创建日志目录，如果目录已存在则不会报错
    os.makedirs(config["log_dir"], exist_ok=True)
    # 配置检查点回调，保存所有检查点，监控验证集指标，保存最后一个检查点
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=-1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
    )
    # 配置 TensorBoard 日志记录器
    # logger = pl.loggers.TensorBoardLogger(
        # _config["log_dir"],
        # name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',flush_secs=10,
    # )
    load_path_base = os.path.basename(config["load_path"])  # 获取文件名，不带路径
    load_path_name = os.path.splitext(load_path_base)[0]  # 去除扩展名
    logger = pl.loggers.TensorBoardLogger(
        config["log_dir"],
        name=f'{exp_name}_seed{config["seed"]}_from_{load_path_name}',flush_secs=10,
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    grad_steps = 1
    # 记录梯度累积步数
    rank_zero_info("grad_steps: {}".format(grad_steps))

    # 确定最大训练步数
    max_steps = config["max_steps"] if config["max_steps"] is not None else -1

    # 初始化恢复检查点路径
    resume_ckpt = None
    
    if config["resume_during_training"]:
        base_log_path = os.path.join(config["log_dir"], logger.name)
        if os.path.isdir(base_log_path):
            versions = sorted([d for d in os.listdir(base_log_path) if d.startswith("version_") and os.path.isdir(os.path.join(base_log_path, d))], key=lambda x: int(x.split('_')[1]))
            if versions:
                latest_version_path = os.path.join(base_log_path, versions[-1], "checkpoints/last.ckpt")
                if os.path.exists(latest_version_path):
                    resume_ckpt = latest_version_path
                    rank_zero_info(f"Found resume_ckpt: {resume_ckpt}")
    
    gradient_clip_val = config.get("gradient_clip_val", 1.0)

    trainer = pl.Trainer(
        devices="auto",
        num_nodes=config["num_nodes"],
        precision=config["precision"],
        accelerator="gpu",
        strategy="auto",
        max_epochs=config["max_epochs"],
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        enable_model_summary=True,
        fast_dev_run=config["fast_dev_run"],
        val_check_interval=config["val_check_interval"],
        enable_progress_bar=True,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
    )


    if not config["test_only"]:
        # 执行训练
        trainer.fit(model, datamodule=dm,ckpt_path=resume_ckpt)
    else:
        # 执行测试
        trainer.test(model, datamodule=dm)

