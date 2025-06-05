import os
import copy
import inspect
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from config import ex

# 设置高精度矩阵运算
torch.set_float32_matmul_precision('high')

# ==================== DataModule 类注册表 ====================
def get_datamodule_registry():
    """
    获取DataModule类注册表
    
    Returns:
        dict: 映射datamodule_name到实际DataModule类的字典
    """
    # 延迟导入避免循环依赖
    from datamodules.multitask_datamodule import MTDataModule
    from datamodules.TextPretrainDataModule import TextPretrainDataModule, TextPretrainLargeDataModule
    
    # 可以在这里添加更多自定义DataModule
    registry = {
        "multitask": MTDataModule,
        "text_pretrain": TextPretrainDataModule,
        "text_pretrain_large": TextPretrainLargeDataModule,
        "multitask_large": MTDataModule,  # 大规模多任务，目前复用MTDataModule
    }
    
    return registry

DATAMODULE_CLASS_REGISTRY = get_datamodule_registry()

def get_model_class_and_datamodule_config(task_type: str, config: dict):
    """
    根据任务类型返回对应的模型类和数据模块配置
    
    Args:
        task_type: 任务类型 ("text_pretrain", "vision_language_pretrain", "downstream")
        config: 配置字典
    
    Returns:
        tuple: (model_class, datamodule_specific_config, model_name)
    """
    if task_type == "text_pretrain":
        from models.vlmo_module import VLMoForTextPretraining
        model_class = VLMoForTextPretraining
        model_name = "VLMoForTextPretraining"
        
        # 确保 image_size 存在于配置中
        image_size = config.get("image_size", 384)
        
        # 文本预训练数据模块特定配置
        datamodule_specific_config = {
            **config,
            "datasets": config.get("datasets", ["wikibk", "pmc"]),
            "batch_size": config.get("batch_size", 128),
            "max_text_len": config.get("max_text_len", 512),
            "mlm_prob": config.get("mlm_prob", 0.15),
            "whole_word_masking": config.get("whole_word_masking", True),
            "text_only": True,
            "image_only": False,
            "draw_false_image": 0,
            "draw_false_text": 0,
            "tokenizer": config["tokenizer"],
            "image_size": image_size,  # 确保顶级配置中有 image_size
            "image_augmentation": {
                "enable_pathology_augmentation": False,  # 文本预训练不需要图像增强
                "image_size": image_size,  # 图像增强配置中也统一使用 image_size
                **config.get("image_augmentation", {})
            },
        }
        
    elif task_type == "vision_language_pretrain":
        from models.vlmo_module import VLMoForVisionLanguagePretraining
        model_class = VLMoForVisionLanguagePretraining
        model_name = "VLMoForVisionLanguagePretraining"
        
        # 确保 image_size 存在于配置中
        image_size = config.get("image_size", 384)
        
        # 视觉-语言预训练数据模块特定配置
        datamodule_specific_config = {
            **config,
            "datasets": config.get("datasets", ["coco", "sbu"]),
            "batch_size": config.get("batch_size", 64),
            "max_text_len": config.get("max_text_len", 196),
            "image_size": image_size,  # 确保顶级配置中有 image_size
            "text_only": False,
            "image_only": False,
            "draw_false_image": config.get("draw_false_image", 1),
            "draw_false_text": config.get("draw_false_text", 1),
            "tokenizer": config["tokenizer"],
            "image_augmentation": {
                "enable_pathology_augmentation": True,
                "image_size": image_size,  # 统一使用 image_size
                "randstainna_enabled": True,
                "randstainna_yaml_file": r"E:\article_code\Vision_Encoder\RandStainNA\CRC_LAB_randomTrue_n0.yaml",
                "randstainna_probability": 0.8,
                "multi_scale_cropping": True,
                **config.get("image_augmentation", {})
            },
        }
        
    elif task_type == "downstream":
        from models.vlmo_module import VLMo
        model_class = VLMo
        model_name = "VLMo (Downstream)"
        
        # 确保 image_size 存在于配置中
        image_size = config.get("image_size", 384)
        
        # 下游任务数据模块特定配置
        datamodule_specific_config = {
            **config,
            "batch_size": config.get("batch_size", 32),
            "get_recall_metric": config.get("get_recall_metric", False),
            "draw_false_image": 0,
            "draw_false_text": 0,
            "tokenizer": config.get("tokenizer", "/gz-fs/Tokenizer"),
            "image_size": image_size,  # 确保顶级配置中有 image_size
            "image_augmentation": {
                "enable_pathology_augmentation": True,
                "image_size": image_size,  # 统一使用 image_size
                "randstainna_enabled": True,
                "randstainna_probability": 0.4,
                "multi_scale_cropping": False,
                "color_jitter_probability": 0.3,
                **config.get("image_augmentation", {})
            },
        }
        
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    
    rank_zero_info(f"Selected model: {model_name}")
    rank_zero_info(f"Task type: {task_type}")
    rank_zero_info(f"Tokenizer path: {datamodule_specific_config['tokenizer']}")
    rank_zero_info(f"Image size: {datamodule_specific_config['image_size']}")
    rank_zero_info(f"Image augmentation enabled: {datamodule_specific_config['image_augmentation'].get('enable_pathology_augmentation', False)}")
    
    return model_class, datamodule_specific_config, model_name

def create_datamodule(config: dict) -> pl.LightningDataModule:
    """
    根据配置动态创建DataModule实例
    
    Args:
        config: 配置字典，必须包含 datamodule_name 和 tokenizer 字段
        
    Returns:
        pl.LightningDataModule: 实例化的DataModule
        
    Raises:
        ValueError: 如果datamodule_name不在注册表中
        KeyError: 如果配置中缺少必要字段
    """
    datamodule_name = config.get("datamodule_name")
    if not datamodule_name:
        raise KeyError("Configuration must contain 'datamodule_name' field")
    
    # 验证tokenizer配置存在（用于动态vocab_size获取）
    if not config.get("tokenizer"):
        raise KeyError("Configuration must contain 'tokenizer' field for dynamic vocab_size determination")
    
    # 验证image_size配置存在
    if not config.get("image_size"):
        rank_zero_info("Warning: 'image_size' not found in config, using default 384")
        config["image_size"] = 384
    
    if datamodule_name not in DATAMODULE_CLASS_REGISTRY:
        available_names = list(DATAMODULE_CLASS_REGISTRY.keys())
        raise ValueError(
            f"Unknown datamodule_name: '{datamodule_name}'. "
            f"Available options: {available_names}"
        )
    
    DataModuleClass = DATAMODULE_CLASS_REGISTRY[datamodule_name]
    
    # 根据DataModule类型决定是否使用分布式
    use_dist = config.get("num_nodes", 1) > 1 or config.get("num_gpus", 1) > 1
    
    rank_zero_info(f"Creating DataModule: {DataModuleClass.__name__}")
    rank_zero_info(f"  - Datasets: {config.get('datasets', 'N/A')}")
    rank_zero_info(f"  - Batch size: {config.get('batch_size', 'N/A')}")
    rank_zero_info(f"  - Tokenizer: {config.get('tokenizer', 'N/A')}")
    rank_zero_info(f"  - Image size: {config.get('image_size', 'N/A')}")
    rank_zero_info(f"  - Distributed: {use_dist}")
    rank_zero_info(f"  - Dynamic vocab_size will be determined from tokenizer")
    
    try:
        # 检查DataModule构造函数的参数
        sig = inspect.signature(DataModuleClass.__init__)
        params = list(sig.parameters.keys())
        
        if 'dist' in params:
            # 支持dist参数的DataModule（如MTDataModule）
            datamodule = DataModuleClass(config, dist=use_dist)
        else:
            # 不支持dist参数的DataModule
            datamodule = DataModuleClass(config)
            
        rank_zero_info(f"Successfully created {DataModuleClass.__name__}")
        rank_zero_info(f"  - Dynamic vocab_size determined: {getattr(datamodule, 'vocab_size', 'Unknown')}")
        return datamodule
        
    except Exception as e:
        rank_zero_info(f"Error creating DataModule {DataModuleClass.__name__}: {e}")
        raise

def validate_config_for_task(config: dict, task_type: str):
    """
    验证配置是否适合指定的任务类型
    """
    # 更新required_fields：移除vocab_size、transform_keys，保留tokenizer和image_size
    required_fields = {
        "text_pretrain": ["tokenizer", "max_text_len", "datamodule_name", "image_size"],
        "vision_language_pretrain": ["tokenizer", "max_text_len", "image_size", "datamodule_name"],
        "downstream": ["datamodule_name", "image_size"],  # 下游任务的tokenizer可能有默认值
    }
    
    missing_fields = []
    for field in required_fields.get(task_type, []):
        if field not in config:
            missing_fields.append(field)
    
    if missing_fields:
        raise ValueError(f"Missing required fields for {task_type}: {missing_fields}")
    
    # 验证datamodule_name是否有效
    datamodule_name = config.get("datamodule_name")
    if datamodule_name not in DATAMODULE_CLASS_REGISTRY:
        available_names = list(DATAMODULE_CLASS_REGISTRY.keys())
        raise ValueError(
            f"Invalid datamodule_name '{datamodule_name}' for {task_type}. "
            f"Available options: {available_names}"
        )
    
    # 验证tokenizer路径
    tokenizer_path = config.get("tokenizer")
    if tokenizer_path and not isinstance(tokenizer_path, str):
        raise ValueError(f"Tokenizer path must be a string, got {type(tokenizer_path)}")
    
    # 验证image_size
    image_size = config.get("image_size")
    if not isinstance(image_size, (int, float)) or image_size <= 0:
        raise ValueError(f"image_size must be a positive number, got {image_size}")
    
    # 验证图像增强配置（如果存在）
    image_aug_config = config.get("image_augmentation", {})
    if image_aug_config.get("randstainna_enabled", False):
        yaml_file = image_aug_config.get("randstainna_yaml_file")
        if yaml_file and not os.path.exists(yaml_file):
            rank_zero_info(f"Warning: RandStainNA YAML file not found: {yaml_file}")
    
    # 任务特定验证
    if task_type == "text_pretrain":
        if not config.get("datasets"):
            raise ValueError("Text pretraining requires 'datasets' configuration")
        if config.get("max_text_len", 0) <= 0:
            raise ValueError("Text pretraining requires valid 'max_text_len'")
        # 验证文本预训练是否使用了合适的DataModule
        if datamodule_name not in ["text_pretrain", "text_pretrain_large", "multitask"]:
            rank_zero_info(f"Warning: Using '{datamodule_name}' for text pretraining. Consider using 'text_pretrain' for better performance.")
            
    elif task_type == "vision_language_pretrain":
        if not config.get("datasets"):
            raise ValueError("Vision-language pretraining requires 'datasets' configuration")
        if config.get("image_size", 0) <= 0:
            raise ValueError("Vision-language pretraining requires valid 'image_size'")
        # 验证图像增强配置的关键参数
        if image_aug_config.get("enable_pathology_augmentation", True):
            rank_zero_info("Vision-language pretraining: Pathology augmentation is enabled")
    
    rank_zero_info(f"Configuration validation passed for task: {task_type}")
    rank_zero_info(f"  - Using DataModule: {datamodule_name}")
    rank_zero_info(f"  - Tokenizer: {tokenizer_path}")
    rank_zero_info(f"  - Image size: {image_size}")
    rank_zero_info(f"  - Image augmentation: {image_aug_config.get('enable_pathology_augmentation', False)}")
    rank_zero_info(f"  - vocab_size will be determined dynamically from tokenizer")

def setup_callbacks(config: dict, logger: pl.loggers.TensorBoardLogger) -> list:
    """
    设置训练回调函数
    
    Args:
        config: 配置字典
        logger: 日志器实例

    Returns:
        list: 回调函数列表
    """
    callbacks = []
    
    version_str = f"version_{logger.version}" if logger.version is not None else "version_0" # Fallback for safety
    checkpoint_dir = os.path.join(config["output_dir"], logger.name, version_str, "checkpoints")
    rank_zero_info(f"Checkpoints will be saved to: {checkpoint_dir}")
    os.makedirs(checkpoint_dir, exist_ok=True) # 确保目录存在
    
    # 获取 checkpoint 保存频率配置
    save_every_n_train_steps = config.get("checkpoint_save_every_n_train_steps")
    save_every_n_epochs = config.get("checkpoint_save_every_n_epochs")
    checkpoint_save_last = config.get("checkpoint_save_last", True)  # 新增：是否保存最后一个检查点
    checkpoint_monitor = config.get("checkpoint_monitor", "val/the_metric")  # 新增：监控的指标
    checkpoint_mode = config.get("checkpoint_mode", "max")  # 新增：监控模式
    
    # 检查点回调
    if save_every_n_epochs is not None and save_every_n_epochs > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            every_n_epochs=save_every_n_epochs,
            save_last=checkpoint_save_last,
            save_top_k=-1, # 保存所有基于频率的检查点
            filename="{epoch:02d}-freqE",
            verbose=True,
        )
        rank_zero_info(f"Checkpointing every {save_every_n_epochs} epoch(s).")
    elif save_every_n_train_steps is not None and save_every_n_train_steps > 0:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            every_n_train_steps=save_every_n_train_steps,
            save_last=checkpoint_save_last,
            save_top_k=-1, # 保存所有基于频率的检查点
            filename="{epoch:02d}-{step:06d}-freqS",
            verbose=True,
        )
        rank_zero_info(f"Checkpointing every {save_every_n_train_steps} training step(s).")
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir, 
            save_top_k=config.get("save_top_k", 1), 
            verbose=True,
            monitor="val/the_metric",
            mode="max",
            save_last=checkpoint_save_last,
            filename="{epoch:02d}-{step:06d}-{val_the_metric:.4f}",
        )
        rank_zero_info(f"Checkpointing based on '{checkpoint_monitor}' (mode: {checkpoint_mode}). Save top k: {config.get('save_top_k', 1)}.")
        
    callbacks.append(checkpoint_callback)
    
    # 学习率监控
    lr_callback = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_callback)
    
    # 任务特定回调
    task_type = config.get("task_type", config.get("task", "downstream"))
    datamodule_name = config.get("datamodule_name", "multitask")
    
    if task_type == "text_pretrain":
        rank_zero_info(f"Added text pretraining specific callbacks for {datamodule_name}")
        
        
    elif task_type == "vision_language_pretrain":
        rank_zero_info(f"Added vision-language pretraining specific callbacks for {datamodule_name}")
        
    
    rank_zero_info(f"Setup {len(callbacks)} callbacks for {task_type} using {datamodule_name}")
    rank_zero_info(f"Directory structure:")
    rank_zero_info(f"  - Checkpoints: {checkpoint_dir}")
    rank_zero_info(f"  - Logs: {logger.log_dir}")
    
    return callbacks

def setup_logger(config: dict) -> pl.loggers.TensorBoardLogger:
    """
    设置TensorBoard日志器
    
    Args:
        config: 配置字典
    
    Returns:
        TensorBoard日志器
    """
    exp_name = config["exp_name"]
    task_type = config.get("task_type", config.get("task", "downstream"))
    datamodule_name = config.get("datamodule_name", "multitask")
    
    # 构建日志器名称，包含DataModule信息
    load_path_base = ""
    if config.get("load_path") or config.get("weight_path"):
        load_path = config.get("load_path") or config.get("weight_path")
        if load_path:
            load_path_base = os.path.basename(load_path)
            load_path_name = os.path.splitext(load_path_base)[0]
        else:
            load_path_name = "scratch"
    else:
        load_path_name = "scratch"
    
    logger_name = f'{exp_name}_{task_type}_{datamodule_name}_seed{config["seed"]}_from_{load_path_name}'
    
    logger = pl.loggers.TensorBoardLogger(
        save_dir=config["log_dir"],
        name=logger_name,
        flush_secs=10,
    )
    
    rank_zero_info(f"Setup logger: {logger_name}")
    return logger

def find_resume_checkpoint(config: dict, logger: pl.loggers.TensorBoardLogger) -> str:
    """
    查找恢复训练的检查点
    
    Args:
        config: 配置字典
        logger_name: 日志器名称
    
    Returns:
        检查点路径或None
    """
    if not config.get("resume_during_training", False):
        return None
    
    experiment_base_path = os.path.join(config["output_dir"], logger.name)
    
    if not os.path.isdir(experiment_base_path):
        rank_zero_info(f"Experiment output directory not found for resuming: {experiment_base_path}")
        return None

    versions = sorted(
        [d for d in os.listdir(experiment_base_path) if d.startswith("version_") and os.path.isdir(os.path.join(experiment_base_path, d))],
        key=lambda x: int(x.split('_')[1])
    )

    if not versions:
        rank_zero_info(f"No versions found in {experiment_base_path} for resuming.")
        return None
    
    latest_version_dir_name = versions[-1]
    resume_ckpt_path = os.path.join(experiment_base_path, latest_version_dir_name, "checkpoints", "last.ckpt")
    
    if os.path.exists(resume_ckpt_path):
        rank_zero_info(f"Found resume checkpoint: {resume_ckpt_path}")
        return resume_ckpt_path
    else:
        rank_zero_info(f"Resume checkpoint 'last.ckpt' not found in: {os.path.join(experiment_base_path, latest_version_dir_name, 'checkpoints')}")
        
    return None

def setup_trainer(config: dict, callbacks: list, logger: pl.loggers.TensorBoardLogger) -> pl.Trainer:
    """
    设置Trainer
    
    Args:
        config: 配置字典
        callbacks: 回调函数列表
        logger: 日志器
    
    Returns:
        Trainer实例
    """
    # 计算梯度累积步数
    grad_steps = config.get("grad_accum_steps", 1)
    rank_zero_info(f"Gradient accumulation steps: {grad_steps}")
    
    # 确定最大训练步数
    max_steps = config.get("max_steps")
    if max_steps is not None and max_steps > 0:
        max_steps = max_steps
    else:
        max_steps = -1
    
    # 梯度裁剪
    gradient_clip_val = config.get("gradient_clip_val", 1.0)
    
    trainer = pl.Trainer(
        devices="auto",
        num_nodes=config.get("num_nodes", 1),
        precision=config.get("precision", "16-mixed"),
        accelerator="gpu",
        strategy="auto",
        max_epochs=config.get("max_epochs", 10),
        max_steps=max_steps,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        enable_model_summary=True,
        fast_dev_run=config.get("fast_dev_run", False),
        val_check_interval=config.get("val_check_interval", 1.0),
        enable_progress_bar=True,
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm="norm",
    )
    
    rank_zero_info("Trainer setup completed")
    return trainer

# ==================== Sacred 主函数 ====================
@ex.automain
def main(_config):  
    config = dict(_config)
    config = copy.deepcopy(config)
    
    # 设置随机种子
    pl.seed_everything(config["seed"])
    
    # 确定任务类型
    task_type = config.get("task_type", config.get("task", "downstream"))
    datamodule_name = config.get("datamodule_name", "multitask")
    
    # 打印配置信息
    rank_zero_info("="*60)
    rank_zero_info(f"Task Type: {task_type}")
    rank_zero_info(f"DataModule: {datamodule_name}")
    rank_zero_info(f"Experiment Name: {config['exp_name']}")
    rank_zero_info(f"Seed: {config['seed']}")
    
    rank_zero_info("="*60)
    rank_zero_info(f"Checkpoint Directory: {config.get('output_dir', 'N/A')}")
    rank_zero_info(f"Log Directory: {config.get('log_dir', 'N/A')}")
    rank_zero_info(f"Data Root: {config.get('data_root', 'N/A')}")
    rank_zero_info(f"  - Save every N train steps: {config.get('checkpoint_save_every_n_train_steps', 'Disabled')}")
    rank_zero_info(f"  - Save every N epochs: {config.get('checkpoint_save_every_n_epochs', 'Disabled')}")
    rank_zero_info(f"  - Save top K (metric-based): {config.get('save_top_k', 1)}")
    rank_zero_info(f"  - Save last checkpoint: {config.get('checkpoint_save_last', True)}")
    rank_zero_info(f"  - Monitor metric: {config.get('checkpoint_monitor', 'val/the_metric')}")
    rank_zero_info(f"  - Monitor mode: {config.get('checkpoint_mode', 'max')}")
    
    rank_zero_info("="*60)
    rank_zero_info(f"Max Epochs: {config.get('max_epochs', 10)}")
    rank_zero_info(f"Batch Size: {config.get('batch_size', 32)}")
    rank_zero_info(f"Learning Rate: {config.get('learning_rate', 1e-4)}")
    rank_zero_info(f"Weight Path: {config.get('weight_path', config.get('load_path', 'None'))}")
    rank_zero_info(f"Datasets: {config.get('datasets', 'N/A')}")
    rank_zero_info(f"Tokenizer: {config.get('tokenizer', 'N/A')}")
    rank_zero_info(f"Image Size: {config.get('image_size', 'N/A')}")
    rank_zero_info(f"Output Directory (for checkpoints): {config.get('output_dir', 'N/A')}") # 新增
    rank_zero_info(f"Checkpoint every N train steps: {config.get('checkpoint_save_every_n_train_steps', 'Metric-based')}") # 新增
    rank_zero_info(f"Checkpoint every N epochs: {config.get('checkpoint_save_every_n_epochs', 'Metric-based')}") # 新增
    rank_zero_info(f"Save top K (metric-based): {config.get('save_top_k', 1)}") # 新增
    rank_zero_info(f"vocab_size: Will be determined dynamically from tokenizer")
    rank_zero_info("="*60)
    
    # 验证配置
    validate_config_for_task(config, task_type)
    
    # 获取模型类和数据模块特定配置
    model_class, datamodule_specific_config, model_name = get_model_class_and_datamodule_config(task_type, config)
    
    # 动态创建数据模块（将通过tokenizer获取vocab_size）
    rank_zero_info(f"Creating DataModule: {datamodule_name}")
    dm = create_datamodule(datamodule_specific_config)
    
    # 创建模型（将通过tokenizer获取vocab_size）
    rank_zero_info(f"Initializing model: {model_name}")
    model = model_class(config)
    
    # 验证vocab_size一致性
    model_vocab_size = getattr(model.hparams, 'actual_vocab_size', getattr(model.hparams, 'vocab_size', None))
    dm_vocab_size = getattr(dm, 'vocab_size', None)
    
    if model_vocab_size and dm_vocab_size and model_vocab_size != dm_vocab_size:
        raise ValueError(
            f"Vocab size mismatch: Model has vocab_size={model_vocab_size}, "
            f"but DataModule has vocab_size={dm_vocab_size}. "
            f"This indicates inconsistent tokenizer configurations."
        )
    
    rank_zero_info(f"Vocab size consistency verified: {model_vocab_size}")
    
    # 设置日志器
    logger = setup_logger(config)
    
    # 设置回调
    callbacks = setup_callbacks(config, logger)
    
    # 设置Trainer
    trainer = setup_trainer(config, callbacks, logger)
    
    # 查找恢复检查点
    resume_ckpt = find_resume_checkpoint(config, logger.name)
    
    # 训练或测试
    if not config.get("test_only", False):
        rank_zero_info("="*60)
        rank_zero_info(f"STARTING TRAINING - {task_type.upper()}")
        rank_zero_info("="*60)
        rank_zero_info(f"Model: {model_name}")
        rank_zero_info(f"DataModule: {datamodule_name}")
        rank_zero_info(f"Data: {datamodule_specific_config.get('datasets', 'N/A')}")
        rank_zero_info(f"Dynamic vocab_size: {model_vocab_size}")
        rank_zero_info(f"Image size: {datamodule_specific_config.get('image_size', 'N/A')}")
        rank_zero_info(f"Resume from: {resume_ckpt if resume_ckpt else 'None'}")
        rank_zero_info("="*60)
        
        # 执行训练
        trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
        
        rank_zero_info("="*60)
        rank_zero_info("TRAINING COMPLETED SUCCESSFULLY!")
        rank_zero_info("="*60)
        rank_zero_info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
        rank_zero_info(f"Last checkpoint: {trainer.checkpoint_callback.last_model_path}")
        rank_zero_info(f"Logs available at: {logger.log_dir}")
        rank_zero_info(f"Final vocab_size used: {model_vocab_size}")
        rank_zero_info("="*60)
        
    else:
        rank_zero_info("="*60)
        rank_zero_info(f"STARTING TESTING - {task_type.upper()}")
        rank_zero_info("="*60)
        rank_zero_info(f"DataModule: {datamodule_name}")
        rank_zero_info(f"Dynamic vocab_size: {model_vocab_size}")
        
        # 执行测试
        trainer.test(model, datamodule=dm)
        
        rank_zero_info("="*60)
        rank_zero_info("TESTING COMPLETED!")
        rank_zero_info("="*60)

# 为了兼容性，保留一个备用的主函数调用方式
if __name__ == "__main__":
    # Sacred 会自动调用被 @ex.automain 装饰的函数
    pass