import os
import sys
import argparse
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict,Any
import torch
import json
import logging
from torch import nn
import numpy as np
from tqdm import tqdm
from transformers.trainer_callback import TrainerCallback
from transformers import (
    HfArgumentParser, 
    TrainingArguments, 
    AutoTokenizer,
    set_seed,
    Trainer,
    BertTokenizer,
)
from modules.RMSNorm import RMSNorm
from transformers.trainer_utils import get_last_checkpoint
# filepath: /gz-fs/vlmo/vlmo/train.py

try:
    from torch.utils.tensorboard import SummaryWriter  # 官方实现
    _tb_available = True
except ModuleNotFoundError:
    try:
        from tensorboardX import SummaryWriter        # 退回 tensorboardX
        _tb_available = True
    except ModuleNotFoundError:
        SummaryWriter = None                          # 占位
        _tb_available = False
# -------------------------------------------------------
# 导入自定义模块
from vlmo.modules.vlmo_config import VLMoEncoderDecoderConfig
from vlmo.modules.vlmo_modeling import VLMoEncoderDecoderForConditionalGeneration
from vlmo.data.transforms import create_image_transform
from vlmo.datamodules.text_qa_datamodule import TextQADataModule
from vlmo.datamodules.vqa_gen_datamodule import VQAGenDataModule
from vlmo.metrics import compute_caption_metrics
   
# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    用于模型初始化的参数
    """
    model_path: Optional[str] = field(
        default=None,
        metadata={"help": "预训练模型的路径或HF模型ID"},
    )
    encoder_path: Optional[str] = field(
        default=None,
        metadata={"help": "预训练编码器的路径"},
    )
    encoder_config: Dict = field(
        default_factory=lambda: {
            "model_name": "vlmo_base_patch16",
            "img_size": 384,
            "patch_size": 16,
            "config": {"max_text_len": 256, "drop_path_rate": 0.1},
        },
        metadata={"help": "编码器配置"},
    )
    decoder_config: Dict = field(
        default_factory=lambda: {
            "vocab_size": 30522,
            "depth": 6,
            "dim": 256,
            "num_heads": 4,
            "num_kv_heads": 4,
            "mlp_ratio": 4.0,
            "drop_path_rate": 0.1,
        },
        metadata={"help": "解码器配置"},
    )
    freeze_encoder: bool = field(
        default=True,
        metadata={"help": "是否冻结编码器参数"},
    )
    max_seq_len: int = field(
        default=256,
        metadata={"help": "最大序列长度"},
    )
    rope_base: int = field(
        default=10000,
        metadata={"help": "RoPE base值"},
    )
    moe_balance_loss_weight: float = field(
        default=0.01,
        metadata={"help": "MoE负载均衡损失的权重"},
    )
    moe_router_z_loss_weight: float = field(
        default=0.001,
        metadata={"help": "MoE路由器Z损失的权重"},
    )

@dataclass
class DataArguments:
    """
    用于数据加载的参数
    """
    dataset: str = field(
        default="caption",
        metadata={"help": "数据集名称，如'caption', 'vqa'等"},
    )
    data_root: str = field(
        default=None,
        metadata={"help": "数据集根目录"},
    )
    tokenizer_path: str = field(
        default=None,
        metadata={"help": "分词器路径"},
    )
    image_size: int = field(
        default=384,
        metadata={"help": "图像大小"},
    )
    max_text_len: int = field(
        default=256,
        metadata={"help": "最大文本长度"},
    )
    max_answer_len: int = field(
        default=256, metadata={"help": "The maximum length for the generated answer sequence."}
    )
    num_workers: int = field(
        default=10,
        metadata={"help": "数据加载的工作线程数"},
    )
    vqa_label_size: int = field(
        default=3129, # Default for VQAv2
        metadata={"help": "Number of answer labels for VQA"},
    )

@dataclass
class VLMoTrainingArguments(TrainingArguments):
    """
    扩展 TrainingArguments 以添加特定于 VLMo 的训练参数
    """
    use_moe: bool = field(
        default=True,
        metadata={"help": "是否使用MoE层(混合专家模型)"},
    )
    evaluate_during_training: bool = field(
        default=True,
        metadata={"help": "是否在训练期间进行评估"},
    )
    
    save_every_n_steps: int = field(
        default=0,
        metadata={"help": "每N个训练步骤保存一次检查点，0表示禁用"},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "是否启用梯度检查点以减少显存占用"},
    )



class GradClipMonitor(TrainerCallback):
    """
    记录梯度 L2 范数。执行时机：on_backward_end（optimizer.step 之前，梯度仍在）。
    不额外裁剪梯度。
    """
    def __init__(self):
        self.tb_writer: Optional[Any] = None

    def _ensure_writer(self, log_dir: str):
        if self.tb_writer is None and _tb_available:
            self.tb_writer = SummaryWriter(log_dir=log_dir)

    def on_backward_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return

        model = kwargs["model"]
        # max_norm=inf → 只计算不修改梯度
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf")).item()
        logger.info(f"[grad_norm] = {grad_norm:.4f}")

        # 写 TensorBoard
        self._ensure_writer(args.logging_dir)
        if self.tb_writer is not None:
            self.tb_writer.add_scalar("train/grad_norm", grad_norm, state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()


class MoELossLogger(TrainerCallback):
    """把模型的 balance_loss / router_z_loss 写入 Trainer 日志（TensorBoard 可见）"""
    def __init__(self):
        self.tb_writer: Optional[Any] = None
    
    def _ensure_writer(self, log_dir: str):
        if self.tb_writer is None and _tb_available:
            self.tb_writer = SummaryWriter(log_dir=log_dir)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps != 0:
            return
        self._ensure_writer(args.logging_dir)
        if self.tb_writer is None:
            return
        model = kwargs["model"]                    # VLMoEncoderDecoderForConditionalGeneration
        moe = model.model.get_moe_losses()        # (balance, router) or None
        if moe is None:
            return
        balance, router = moe
        step = state.global_step
        self.tb_writer.add_scalar("moe/balance_loss", balance.detach().float().item(), step)
        self.tb_writer.add_scalar("moe/router_z_loss", router.detach().float().item(), step)
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

def main():
    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataArguments, VLMoTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    print("\n===== Model Arguments =====")
    for arg_name, arg_value in sorted(vars(model_args).items()):
        print(f"{arg_name}: {arg_value}")
    
    print("\n===== Data Arguments =====")
    for arg_name, arg_value in sorted(vars(data_args).items()):
        print(f"{arg_name}: {arg_value}")
    print("\n===== Training Arguments =====")
    for arg_name, arg_value in sorted(vars(training_args).items()):
        print(f"{arg_name}: {arg_value}")
    print("===========================\n")
    set_seed(training_args.seed)
    
    loss_weights = {
        "moe_balance": model_args.moe_balance_loss_weight,
        "moe_router_z": model_args.moe_router_z_loss_weight
    }
    print(f"data_args.tokenizer_path:{data_args.tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_path, local_files_only=True)
        logger.info(f"加载tokenizer: {data_args.tokenizer_path}")
    except Exception as e:
        logger.info(f"Error loading tokenizer: {e}. Falling back to bert-base-uncased.")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info(f"从bert-base-uncased加载tokenizer")
    
    model = None
    dataset_name = data_args.dataset.lower() # text_qa

    shared_config_kwargs = {
        "freeze_encoder": model_args.freeze_encoder,
    }
    encoder_config_overrides = {
        "model_name": model_args.encoder_config["model_name"],
        "img_size": model_args.encoder_config["img_size"],
        "patch_size": model_args.encoder_config["patch_size"],
        "config": model_args.encoder_config["config"],
    }
    decoder_config = {
        "vocab_size": model_args.decoder_config["vocab_size"],
        "depth": model_args.decoder_config["depth"],
        "dim": model_args.decoder_config["dim"],
        "num_heads": model_args.decoder_config["num_heads"],
        "mlp_ratio": model_args.decoder_config["mlp_ratio"],
        "num_kv_heads": model_args.decoder_config["num_kv_heads"],
        "drop_path_rate": model_args.decoder_config["drop_path_rate"],
    }
    model_config = VLMoEncoderDecoderConfig(
        encoder=encoder_config_overrides,
        decoder=decoder_config,
        max_seq_len=model_args.max_seq_len,
        rope_base=model_args.rope_base,
        moe_balance_loss_weight=model_args.moe_balance_loss_weight,
        moe_router_z_loss_weight=model_args.moe_router_z_loss_weight,
        encoder_checkpoint_path=model_args.encoder_path
    )
    print(f"model_args.model_path:{model_args.model_path}")

    if model_args.model_path:
        logger.info(f"Attempting to load model for task '{dataset_name}' from {model_args.model_path}")
        model = VLMoEncoderDecoderForConditionalGeneration.from_pretrained(model_args.model_path,
                                             local_files_only=True,**shared_config_kwargs)
        logger.info(f"Loaded {model.__class__.__name__} from {model_args.model_path}")

    else:
        logger.info(f"Creating new model for task '{dataset_name}' from config")
        model_args.decoder_config["vocab_size"] = tokenizer.vocab_size # 30522

        model = VLMoEncoderDecoderForConditionalGeneration(config=model_config, **shared_config_kwargs)
        logger.info(f"Created new {model.__class__.__name__}")

    logger.info(f"加载{data_args.dataset}数据集")
    
    datamodule_name = data_args.dataset.lower()
    logger.info(f"Loading DataModule: {datamodule_name}")
    if datamodule_name == "text_qa":
        dm = TextQADataModule(data_args,batch_size=training_args.per_device_train_batch_size)
        dm.setup("fit")
    elif datamodule_name == "vqa_gen":
        dm = VQAGenDataModule(data_args,batch_size=training_args.per_device_train_batch_size)
        dm.setup("fit")
    else:
        raise ValueError(f"Unsupported datamodule_name: {datamodule_name}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dm.train_dataset,
        eval_dataset=dm.val_dataset,
        tokenizer=tokenizer,
        data_collator=dm.collate_fn,
        callbacks=[GradClipMonitor(),MoELossLogger()],
        compute_metrics=compute_caption_metrics,
    )
    
    logger.info("Starting training...")
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change the `--output_dir` or add `--overwrite_output_dir`.")

    checkpoint = last_checkpoint if last_checkpoint is not None else training_args.resume_from_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict and hasattr(dm, "test_dataset"):
        logger.info("*** Test ***")
        predict_results = trainer.predict(dm.test_dataset)
        logger.info("*** Test completed ***")
    
    logger.info("任务完成!")

if __name__ == "__main__":
    main()
