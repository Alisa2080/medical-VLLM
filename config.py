# from sacred import Experiment

# ex = Experiment("VLMo")




# @ex.config
# def config():
#     """
#     此函数用于配置VLMo模型训练和评估的参数。

#     返回:
#         无。函数定义了一系列的参数，这些参数将被用于配置实验。
#     """
#     # 实验名称
#     exp_name = "vlmo"
#     # 随机种子，用于保证实验的可重复性
#     seed = 1
#     # 训练和验证使用的数据集列表
#     # datasets = ["coco", "vg", "sbu", "gcc"]
#     datasets = ["coco"]
#     # 期望的批量大小，PL训练器在每步批量较小时会累积梯度
#     batch_size = 2
#     # MoE相关参数
#     moe_balance_loss_weight = 0.01  # 负载均衡损失权重
#     moe_router_z_loss_weight = 0.001  # 路由器Z-损失权重
#     # Image setting
#     # 训练时使用的图像变换方法
#     train_transform_keys = ["square_transform_randaug"]
#     # 验证时使用的图像变换方法
#     val_transform_keys = ["square_transform"]
#     # 图像的大小
#     image_size = 384
#     # 绘制错误图像的数量
#     draw_false_image = 0
#     # 是否仅使用图像数据
#     image_only = False
#     # 是否仅使用文本数据
#     text_only = False

#     # Text Setting
#     # 文本的最大长度
#     max_text_len = 196
#     # 初始检查点的文本最大长度
#     max_text_len_of_initckpt = 196
#     # 使用的分词器
#     # tokenizer = r"E:\BaiduNetdiskDownload\Bert_tokenizer"
#     tokenizer = r"/gz-fs/Tokenizer"

#     # tokenizer = r"/autodl-fs/data/tokenizer"
    
#     use_siglip_loss = True
#     # 词汇表的大小
#     vocab_size = 30522
#     # 是否使用全词掩码
#     whole_word_masking = False
#     # 掩码语言模型的掩码概率
#     mlm_prob = 0.15
#     # 绘制错误文本的数量
#     draw_false_text = 0

#     # Transformer Setting
#     # 模型的架构
#     model_arch = "vlmo_base_patch16"
#     # 随机丢弃路径的概率
#     drop_path_rate = 0.1

#     # Optimizer Setting
#     # 优化器的类型
#     optim_type = "adamw"
#     # 学习率
#     learning_rate = 1e-4
#     # 权重衰减系数
#     weight_decay = 0.01
#     # 学习率衰减的幂次
#     decay_power = 1
#     # 最大训练轮数
#     max_epochs = 10
#     # 最大训练步数
#     max_steps = None
#     # 热身步数
#     warmup_steps = 500
#     # 最终学习率
#     end_lr = 0
#     # 下游任务头的学习率乘数
#     lr_mult = 1  

#     # Downstream Setting
#     # 是否获取召回率指标
#     get_recall_metric = False
#     # 是否获取重排序后的召回率指标
#     get_recall_rerank_metric = False
#     # 测试时的K值
#     k_test = 32

#     # PL Trainer Setting
#     # 从指定路径恢复训练
#     resume_from = None
#     # 是否进行快速开发运行
#     fast_dev_run = False
#     # 验证的间隔
#     val_check_interval = 1.0
#     # 是否仅进行测试
#     test_only = False
#     # 是否使用分片训练
#     use_sharded_training = False
#     # 在训练过程中是否恢复训练
#     resume_during_training = True

#     # below params varies with the environment
#     # 数据的根目录
#     data_root = ""
#     gradient_clip_val = 3.0
#     # 日志保存的目录
#     # log_dir = r"E:\article_code\unilm-master\vlmo\log"
#     log_dir = r"/gz-fs/log/vlmo"

#     # 使用的GPU数量
#     num_gpus = 1
#     # 使用的节点数量
#     num_nodes = 1
#     # 加载预训练模型的路径
#     load_path = "" # Path to load full model weights
#     # --- 添加 freeze_encoder 标志 ---
#     freeze_encoder = False # 默认不冻结，在特定任务中覆盖
#     # --- 添加答案最大长度 ---
#     max_answer_len = 256 # Default max length for generated answers
#     # 数据加载的工作线程数量
#     num_workers = 10
#     # 训练的精度
#     precision = "16-mixed"


# @ex.named_config
# def task_vlmo_vl_pretrain_base():
#     exp_name = "vlmo_vl_pretrain_base"
#     task_identifier = "vl_pretrain"
#     model_class_name = "VLMoForVisionLanguage" # 假设的类名

#     model_name = "beit_base_patch16_384"
#     datasets = ["coco", "sbu"] # 使用 CocoCaptionKarpathyDataset 等
#     datamodule_name = "VisionLanguagePretrainDataModule"

#     load_path = "path/to/your/text_pretrained_model.ckpt" # 上一阶段的输出

#     # 定义此阶段要计算的损失及其权重
#     active_tasks = {"mlm": 1.0, "itm": 1.0, "itc": 1.0} # 示例
#     use_siglip_loss = True # 如果ITC使用SigLIP

#     learning_rate = 1e-4
#     batch_size = 64
#     # ... 其他超参数 ...


# @ex.named_config
# def task_textmlm_base():
#     """
#     此函数定义了一个名为 'textmlm_base' 的配置，用于仅文本的掩码语言模型 (Text-only Masked Language Modeling) 任务。

#     此配置主要用于预训练阶段，仅使用文本数据进行训练。

#     返回:
#         无。函数定义了一系列的参数，这些参数将被用于配置实验。
#     """
#     # 实验名称
#     exp_name = "textmlm_base"
#     # 训练和验证使用的数据集列表，这里仅使用 'wikibk' 数据集
#     datasets = ["wikibk"]
#     # 损失函数的配置，使用 _loss_names 函数生成损失名称和初始值的字典，仅启用 'textmlm' 损失
#     loss_names = _loss_names({"textmlm": 1})
#     # 期望的批量大小，PL训练器在每步批量较小时会累积梯度
#     batch_size = 128
#     # 文本的最大长度
#     max_text_len = 196
#     # 学习率
#     learning_rate = 2e-3
#     # 是否使用全词掩码
#     whole_word_masking = True
#     # 训练时使用的图像变换方法
#     train_transform_keys = ["square_transform_randaug"]
#     # 验证时使用的图像变换方法
#     val_transform_keys = ["square_transform"]
#     # 模型的架构
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_textmlm_base_plus():
#     """
#     此函数定义了一个名为 'textmlm_base_plus' 的配置，用于仅文本的掩码语言模型 (Text-only Masked Language Modeling) 任务。
#     此配置在 'textmlm_base' 的基础上进行了增强，可能使用了更大或更复杂的模型架构。

#     返回:
#         无。函数定义了一系列的参数，这些参数将被用于配置实验。
#     """
#     # 实验名称
#     exp_name = "textmlm_base_plus"
#     # 训练和验证使用的数据集列表，这里仅使用 'wikibk' 数据集
#     datasets = ["wikibk"]
#     # 损失函数的配置，使用 _loss_names 函数生成损失名称和初始值的字典，仅启用 'textmlm' 损失
#     loss_names = _loss_names({"textmlm": 1})
#     # 期望的批量大小，PL训练器在每步批量较小时会累积梯度
#     batch_size = 1024
#     # 文本的最大长度
#     max_text_len = 196
#     # 学习率
#     learning_rate = 2e-4
#     # 是否使用全词掩码
#     whole_word_masking = True
#     # 训练时使用的图像变换方法
#     train_transform_keys = ["square_transform_randaug"]
#     # 验证时使用的图像变换方法
#     val_transform_keys = ["square_transform"]
#     # 模型的架构，使用 'vlmo_base_plus_patch16' 架构
#     model_arch = "vlmo_base_plus_patch16"


# # ----------------------- vision-language pretraining config -----------------------


# # Named configs for "task" which define datasets, loss_names and desired batch_size, warmup_steps, epochs, and exp_name
# @ex.named_config
# def task_mlm_itm_itc_base():
#     """
#     此函数定义了一个名为 'mlm_itm_itc_base' 的配置，用于视觉 - 语言预训练任务。
#     该任务结合了掩码语言模型 (MLM)、图像 - 文本匹配 (ITM) 和图像 - 文本对比 (ITC) 三种损失。

#     返回:
#         无。函数定义了一系列的参数，这些参数将被用于配置实验。
#     """
#     # 实验名称
#     exp_name = "mlm_itm_itc_base"
#     # 训练和验证使用的数据集列表
#     # datasets = ["coco", "vg", "sbu", "gcc"]
#     datasets = ["coco"]
#     # 损失函数的配置，使用 _loss_names 函数生成损失名称和初始值的字典，启用 'itm'、'mlm' 和 'itc' 损失
#     loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
#     # 期望的批量大小，PL训练器在每步批量较小时会累积梯度
#     batch_size = 20
#     # 是否使用全词掩码
#     whole_word_masking = True
#     # 学习率
#     learning_rate = 1e-4
#     # 训练时使用的图像变换方法
#     train_transform_keys = ["square_transform_randaug"]
#     # 验证时使用的图像变换方法
#     val_transform_keys = ["square_transform"]
#     # 模型的架构
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_mlm_itm_itc_base_plus():
#     """
#     此函数定义了一个名为 'mlm_itm_itc_base_plus' 的配置，用于视觉 - 语言预训练任务。
#     该配置在 'mlm_itm_itc_base' 的基础上进行增强，可能使用更大或更复杂的模型架构。
#     任务结合了掩码语言模型 (MLM)、图像 - 文本匹配 (ITM) 和图像 - 文本对比 (ITC) 三种损失。

#     返回:
#         无。函数定义了一系列的参数，这些参数将被用于配置实验。
#     """
#     # 实验名称
#     exp_name = "mlm_itm_itc_base_plus"
#     # 训练和验证使用的数据集列表
#     datasets = ["coco", "vg", "sbu", "gcc"]
#     # 损失函数的配置，使用 _loss_names 函数生成损失名称和初始值的字典，启用 'itm'、'mlm' 和 'itc' 损失
#     loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
#     # 期望的批量大小，PL训练器在每步批量较小时会累积梯度
#     batch_size = 1024
#     # 是否使用全词掩码
#     whole_word_masking = True
#     # 学习率
#     learning_rate = 1e-4
#     # 训练时使用的图像变换方法
#     train_transform_keys = ["square_transform_randaug"]
#     # 验证时使用的图像变换方法
#     val_transform_keys = ["square_transform"]
#     # 模型的架构，使用 'vlmo_base_plus_patch16' 架构
#     model_arch = "vlmo_base_plus_patch16"


# @ex.named_config
# def task_mlm_itm_itc_large():
#     """
#     此函数定义了一个名为 'mlm_itm_itc_large' 的配置，用于视觉 - 语言预训练任务。
#     该任务结合了掩码语言模型 (MLM)、图像 - 文本匹配 (ITM) 和图像 - 文本对比 (ITC) 三种损失。
#     此配置使用了较大的模型架构（vit_large_patch16_224）。

#     返回:
#         无。函数定义了一系列的参数，这些参数将被用于配置实验。
#     """
#     # 定义实验名称
#     exp_name = "mlm_itm_itc_large"
#     # 定义训练和验证使用的数据集列表
#     datasets = ["coco", "vg", "sbu", "gcc"]
#     # 定义损失函数的配置，使用 _loss_names 函数生成损失名称和初始值的字典，启用 'itm'、'mlm' 和 'itc' 损失
#     loss_names = _loss_names({"itm": 1, "mlm": 1, "itc": 1})
#     # 定义期望的批量大小，PL训练器在每步批量较小时会累积梯度
#     batch_size = 1024
#     # 定义是否使用全词掩码
#     whole_word_masking = True
#     # 定义学习率
#     learning_rate = 5e-5
#     # 定义训练时使用的图像变换方法
#     train_transform_keys = ["square_transform_randaug"]
#     # 定义验证时使用的图像变换方法
#     val_transform_keys = ["square_transform"]
#     # 定义模型的架构，使用 'vit_large_patch16_224' 架构
#     model_arch = "vit_large_patch16_224"


# # ----------------------- NLVR2 fine-tuning configs -----------------------


# @ex.named_config
# def task_finetune_nlvr2_base():
#     exp_name = "finetune_nlvr2_base"
#     datasets = ["nlvr2"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"nlvr2": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 5e-5
#     val_transform_keys = ["square_transform"]
#     use_sharded_training=False
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_finetune_nlvr2_base_plus():
#     exp_name = "finetune_nlvr2_base_plus"
#     datasets = ["nlvr2"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"nlvr2": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 3e-5
#     drop_path_rate = 0.2
#     val_transform_keys = ["square_transform"]
#     use_sharded_training=False
#     model_arch = "vlmo_base_plus_patch16"


# @ex.named_config
# def task_finetune_nlvr2_base_image384():
#     exp_name = "finetune_nlvr2_base_image384"
#     datasets = ["nlvr2"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"nlvr2": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 5e-5
#     val_transform_keys = ["square_transform"]
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_finetune_nlvr2_base_plus_image384():
#     exp_name = "finetune_nlvr2_base_plus_image384"
#     datasets = ["nlvr2"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"nlvr2": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 3e-5
#     drop_path_rate = 0.2
#     val_transform_keys = ["square_transform"]
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_base_plus_patch16"


# @ex.named_config
# def task_finetune_nlvr2_large():
#     exp_name = "finetune_nlvr2_large"
#     datasets = ["nlvr2"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"nlvr2": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 3e-5
#     drop_path_rate = 0.15
#     val_transform_keys = ["square_transform"]
#     use_sharded_training=False
#     model_arch = "vlmo_large_patch16"


# @ex.named_config
# def task_finetune_nlvr2_large_image384():
#     exp_name = "finetune_nlvr2_large_image384"
#     datasets = ["nlvr2"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"nlvr2": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 3e-5
#     drop_path_rate = 0.15
#     val_transform_keys = ["square_transform"]
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_large_patch16"


# # ----------------------- VQAv2 Fine-tuning configs -----------------------


# @ex.named_config
# def task_finetune_vqa_base_image480():
#     exp_name = "finetune_vqa_base_image480"
#     datasets = ["vqa"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"vqa": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 3e-5
#     drop_path_rate = 0.15
#     val_transform_keys = ["square_transform"]
#     lr_mult = 20
#     image_size = 480
#     use_sharded_training=False
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_finetune_vqa_base_plus_image480():
#     exp_name = "finetune_vqa_base_plus_image480"
#     datasets = ["vqa"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"vqa": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 3e-5
#     drop_path_rate = 0.15
#     val_transform_keys = ["square_transform"]
#     lr_mult = 20
#     image_size = 480
#     use_sharded_training=False
#     model_arch = "vlmo_base_plus_patch16"


# @ex.named_config
# def task_finetune_vqa_large_image480():
#     exp_name = "finetune_vqa_large_image480"
#     datasets = ["vqa"]
#     train_transform_keys = ["square_transform_randaug"]
#     loss_names = _loss_names({"vqa": 1})
#     batch_size = 128
#     max_epoch = 10
#     max_steps = None
#     warmup_steps = 0.1
#     learning_rate = 1.5e-5
#     drop_path_rate = 0.15
#     val_transform_keys = ["square_transform"]
#     lr_mult = 20
#     image_size = 480
#     use_sharded_training=False
#     model_arch = "vlmo_large_patch16"


# # ----------------------- F30K IR/TR Fine-tuning configs -----------------------


# @ex.named_config
# def task_finetune_irtr_f30k_base():
#     exp_name = "finetune_irtr_f30k_base"
#     datasets = ["f30k"]
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     loss_names = _loss_names({"irtr": 1.0})
#     batch_size = 3072
#     max_epoch = 50
#     max_steps = 1500
#     warmup_steps = 150
#     get_recall_metric = True
#     learning_rate = 3e-5
#     drop_path_rate = 0.15
#     use_sharded_training=False
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_finetune_irtr_f30k_base_image384():
#     exp_name = "finetune_irtr_f30k_base_image384"
#     datasets = ["f30k"]
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     loss_names = _loss_names({"irtr": 1.0})
#     batch_size = 3072
#     max_epoch = 50
#     max_steps = 1500
#     warmup_steps = 150
#     get_recall_metric = True
#     learning_rate = 3e-5
#     drop_path_rate = 0.15
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_finetune_irtr_f30k_base_plus_image384():
#     exp_name = "finetune_irtr_f30k_base_plus_image384"
#     datasets = ["f30k"]
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     loss_names = _loss_names({"irtr": 1.0})
#     batch_size = 3072
#     max_epoch = 50
#     max_steps = 1500
#     warmup_steps = 150
#     get_recall_metric = True
#     learning_rate = 3e-5
#     drop_path_rate = 0.2
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_base_plus_patch16"


# @ex.named_config
# def task_finetune_irtr_f30k_large_image384():
#     exp_name = "finetune_irtr_f30k_large_image384"
#     datasets = ["f30k"]
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     loss_names = _loss_names({"irtr": 1.0})
#     batch_size = 3072
#     max_epoch = 50
#     max_steps = 1500
#     warmup_steps = 150
#     get_recall_metric = True
#     learning_rate = 2e-5
#     drop_path_rate = 0.2
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_large_patch16"


# # ----------------------- COCO IR/TR Fine-tuning configs -----------------------


# @ex.named_config
# def task_finetune_irtr_coco_base_image384():
#     exp_name = "finetune_irtr_coco_base_image384"
#     datasets = ["coco"]
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     loss_names = _loss_names({"irtr": 1.0})
#     batch_size = 3072
#     max_epoch = 50
#     max_steps = 3000
#     warmup_steps = 300
#     get_recall_metric = True
#     learning_rate = 3e-5
#     drop_path_rate = 0.2
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_base_patch16"


# @ex.named_config
# def task_finetune_irtr_coco_base_plus_image384():
#     exp_name = "finetune_irtr_coco_base_plus_image384"
#     datasets = ["coco"]
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     loss_names = _loss_names({"irtr": 1.0})
#     batch_size = 3072
#     max_epoch = 50
#     max_steps = 3000
#     warmup_steps = 300
#     get_recall_metric = True
#     learning_rate = 3e-5
#     drop_path_rate = 0.2
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_base_plus_patch16"


# @ex.named_config
# def task_finetune_irtr_coco_large_image384():
#     exp_name = "finetune_irtr_coco_large_image384"
#     datasets = ["coco"]
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     loss_names = _loss_names({"irtr": 1.0})
#     batch_size = 3072
#     max_epoch = 50
#     max_steps = 3000
#     warmup_steps = 300
#     get_recall_metric = True
#     learning_rate = 2e-5
#     drop_path_rate = 0.2
#     image_size = 384
#     use_sharded_training=False
#     model_arch = "vlmo_large_patch16"


# # ----------------------- Other configs -----------------------


# # Named configs for "etc" which are orthogonal to "env" and "task", need to be added at the end
# @ex.named_config
# def step1_5k():
#     max_epoch = 100
#     warmup_steps = 150
#     max_steps = 1500


# @ex.named_config
# def step3k():
#     max_epoch = 100
#     warmup_steps = 300
#     max_steps = 3000


# @ex.named_config
# def step200k():
#     max_epoch = 200
#     warmup_steps = 2500
#     max_steps = 200000


# @ex.named_config
# def step500k():
#     max_epoch = 500
#     warmup_steps = 2500
#     max_steps = 500000


# # ----------------------- Text QA Generation Task Config -----------------------
# @ex.named_config
# def task_text_qa_gen_base():
#     """配置用于纯文本问答 (生成式) 任务 (阶段 1)"""
#     exp_name = "text_qa_gen_base"
#     datasets = ["my_text_qa"] # Example dataset name
#     # --- 定义损失 (仅生成损失) ---
#     loss_names = _loss_names({"gen": 1}) # Ensure gen loss is set
#     batch_size = 64
#     max_epoch = 5
#     warmup_steps = 0.1
#     learning_rate = 5e-5
#     # --- 指定使用 Text QA DataModule ---
#     datamodule_name = "text_qa" # <--- 必须设置
#     # --- 冻结编码器 ---
#     freeze_encoder = True # <--- 阶段 1 冻结编码器
#     # --- 加载预训练 VLMo Encoder 权重 ---
#     load_path = "path/to/pretrained/vlmo_encoder.ckpt" # <--- 设置预训练 VLMo 编码器权重路径
#     # --- 模型架构 (Encoder-Decoder) ---
#     model_arch = "vlmo_base_patch16"
#     # --- 文本长度 ---
#     max_text_len = 128 # 问题最大长度
#     max_answer_len = 64 # 答案最大长度
#     # --- 其他参数 ---
#     image_size = 384
#     train_transform_keys = ["square_transform"]
#     val_transform_keys = ["square_transform"]
#     moe_balance_loss_weight = 0.01
#     moe_router_z_loss_weight = 0.001


# # ----------------------- Medical VQA Generation Fine-tuning Config -----------------------
# @ex.named_config
# def task_finetune_medical_vqa_gen_base():
#     """配置用于医疗 VQA (生成式) 微调任务 (阶段 2)"""
#     exp_name = "finetune_medical_vqa_gen_base"
#     datasets = ["medical_vqa"] # Example dataset name
#     # --- 定义损失 (仅生成损失) ---
#     loss_names = _loss_names({"gen": 1}) # <--- 使用生成损失
#     batch_size = 64
#     max_epoch = 10
#     warmup_steps = 0.1
#     learning_rate = 3e-5
#     # --- 指定使用 VQA Gen DataModule ---
#     datamodule_name = "vqa_gen" # <--- 必须设置
#     # --- 指定从 Stage 1 加载完整模型权重 ---
#     load_path = "path/to/stage1/text_qa_gen_base/checkpoint.ckpt" # <--- 设置阶段 1 的检查点路径
#     # --- 不冻结编码器 ---
#     freeze_encoder = False # <--- 阶段 2 不冻结编码器 (全参微调)
#     # --- 模型架构 (Encoder-Decoder) ---
#     model_arch = "vlmo_base_patch16" # Should match Stage 1 arch
#     # --- 文本长度 ---
#     max_text_len = 40 # VQA 问题最大长度
#     max_answer_len = 64 # VQA 答案最大生成长度
#     # --- 其他参数 ---
#     image_size = 480
#     train_transform_keys = ["square_transform_randaug"]
#     val_transform_keys = ["square_transform"]
#     drop_path_rate = 0.1
#     moe_balance_loss_weight = 0.01
#     moe_router_z_loss_weight = 0.001

from sacred import Experiment

ex = Experiment("VLMo")

@ex.config
def config():
    """
    此函数用于配置VLMo模型训练和评估的通用参数。
    特定任务的参数将在 named_config 中覆盖或添加。
    """
    # 实验名称
    exp_name = "vlmo_base"
    # 随机种子
    seed = 1
    # 期望的批量大小 (可被任务覆盖)
    batch_size = 64 # Default, can be overridden by specific tasks
    # MoE相关参数 (如果模型使用MoE)
    moe_balance_loss_weight = 0.01
    moe_router_z_loss_weight = 0.001

    # Image Setting (通用部分)
    train_transform_keys = ["square_transform_randaug"]
    val_transform_keys = ["square_transform"]
    image_size = 384
    draw_false_image = 0 # 通常用于调试

    # Text Setting (通用部分)
    max_text_len = 196
    tokenizer = r"/gz-fs/Tokenizer" # 或者您的实际路径
    vocab_size = 30522 # 根据您的tokenizer调整
    mlm_prob = 0.15 # 如果任务涉及MLM
    draw_false_text = 0 # 通常用于调试

    # Encoder Model Setting (由timm.create_model使用)
    # 将在任务配置中指定具体的 encoder_model_name
    # 例如: encoder_model_name = "beit_base_patch16_384"
    drop_path_rate = 0.1

    # Optimizer Setting (通用部分)
    optim_type = "adamw"
    learning_rate = 1e-4 # Default, overridden by tasks
    weight_decay = 0.01
    decay_power = 1
    max_epochs = 10 # Default, overridden by tasks
    max_steps = None
    warmup_steps = 500 # Default, overridden by tasks
    end_lr = 0
    lr_mult = 1 # For downstream head learning rate multiplier

    # PL Trainer Setting (通用部分)
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    use_sharded_training = False # 通常用于大规模训练
    resume_during_training = True
    gradient_clip_val = 3.0
    num_gpus = 1
    num_nodes = 1
    num_workers = 10
    precision = "16-mixed"

    # Paths (通用部分)
    data_root = "" # 例如: "/path/to/your/datasets"
    log_dir = r"/gz-fs/log/vlmo" # 或者您的实际路径
    load_path = "" # 用于加载预训练权重的路径 (由任务配置具体指定)

    # Task identification (将在named_config中设置)
    task_identifier = None
    model_class_name = None
    datamodule_name = None

    # Specific model/task flags (can be overridden)
    convert_beit2_to_textpt = False # 用于 VLMoForTextPretraining 从视觉权重加载
    freeze_encoder = False # 用于某些微调任务
    max_answer_len = 256 # 用于生成任务

    # Vision-Language Pretraining specific (if applicable, can be in base or task config)
    use_siglip_loss = True # For ITC if using SigLIP variant
    active_tasks = {} # For VL-Pretrain, e.g., {"mlm": 1.0, "itm": 1.0, "itc": 1.0}

    # Downstream specific (can be in base or task config)
    get_recall_metric = False
    k_test = 32 # For retrieval tasks


# --- STAGE 1: Text-only Pretraining (MLM) ---
@ex.named_config
def task_vlmo_text_pretrain_base():
    exp_name = "vlmo_text_pretrain_base"
    task_identifier = "text_pretrain"
    model_class_name = "VLMoForTextPretraining"
    datamodule_name = "TextPretrainDataModule" # 假设的数据模块名

    encoder_model_name = "beit_base_patch16_384" # Timm模型名, 确保它在Encoder_version.py中定义且FFN是MoE
    datasets = ["pmc"] # 例如使用PMCDataset

    # 权重加载：从视觉预训练的Encoder加载
    load_path = "path/to/your/visual_pretrained_encoder.ckpt" # <--- 修改此路径
    convert_beit2_to_textpt = True # 因为视觉Encoder的MLP与文本MoE-MLP不同

    # 训练超参数
    learning_rate = 5e-5
    batch_size = 128
    max_epochs = 5
    warmup_steps = 500
    max_text_len = 196 # 确保与模型内部的max_seq_len一致
    # vocab_size, drop_path_rate 等从基础配置继承


# --- STAGE 2: Vision-Language Pretraining (MLM, ITM, ITC) ---
@ex.named_config
def task_vlmo_vl_pretrain_base():
    exp_name = "vlmo_vl_pretrain_base"
    task_identifier = "vl_pretrain"
    model_class_name = "VLMoForVisionLanguage" # 假设的pl.LightningModule类名
    datamodule_name = "VisionLanguagePretrainDataModule" # 假设的数据模块名

    encoder_model_name = "beit_base_patch16_384" # Timm模型名
    datasets = ["coco", "sbu", "vg", "gcc"] # 使用的图文对数据集

    # 权重加载：从文本预训练阶段的模型加载
    load_path = "path/to/your/text_pretrained_vlmo_model.ckpt" # <--- 修改此路径
    convert_beit2_to_textpt = False # 通常在此阶段不需要转换

    # 此阶段激活的任务及其权重 (由 VLMoForVisionLanguage 内部使用)
    active_tasks = {"mlm": 1.0, "itm": 1.0, "itc": 1.0}
    use_siglip_loss = True # 如果ITC损失使用SigLIP变体

    # 训练超参数
    learning_rate = 1e-4
    batch_size = 64 # 根据GPU内存调整
    max_epochs = 10
    warmup_steps = 1000
    image_size = 384
    max_text_len = 196


# ----------------------- Downstream Task: NLVR2 Fine-tuning -----------------------
@ex.named_config
def task_finetune_nlvr2_base():
    exp_name = "finetune_nlvr2_base"
    task_identifier = "nlvr2_finetune"
    model_class_name = "NLVR2FinetuneModule" # 假设的pl.LightningModule类名
    datamodule_name = "NLVR2DataModule"     # 假设的数据模块名

    encoder_model_name = "beit_base_patch16_384" # Encoder架构应与预训练一致
    datasets = ["nlvr2"]

    # 权重加载：从图文预训练阶段的模型加载
    load_path = "path/to/your/vl_pretrained_model.ckpt" # <--- 修改此路径

    learning_rate = 5e-5
    batch_size = 128
    max_epochs = 10
    warmup_steps = 200 # 可以是比例或绝对步数
    image_size = 384 # NLVR2通常使用特定尺寸
    # drop_path_rate 可以在微调时调整，例如增加一点


# ----------------------- Downstream Task: VQAv2 Fine-tuning -----------------------
@ex.named_config
def task_finetune_vqa_base(): # 移除了image480以保持简洁，可以在具体实验中添加
    exp_name = "finetune_vqa_base"
    task_identifier = "vqa_finetune"
    model_class_name = "VQAFinetuneModule" # 假设的pl.LightningModule类名
    datamodule_name = "VQADataModule"      # 假设的数据模块名

    encoder_model_name = "beit_base_patch16_384"
    datasets = ["vqa"] # 指向 VQAv2KarpathyDataset

    load_path = "path/to/your/vl_pretrained_model.ckpt" # <--- 修改此路径

    learning_rate = 3e-5
    batch_size = 128
    max_epochs = 10
    warmup_steps = 200
    lr_mult = 10 # VQA头部的学习率乘数
    image_size = 480 # VQA通常使用较大图像尺寸
    max_text_len = 128 # VQA问题长度
    # vqav2_label_size 需要在基础配置或这里定义，例如 3129


# ----------------------- Downstream Task: IR/TR Fine-tuning (COCO example) -----------------------
@ex.named_config
def task_finetune_irtr_coco_base():
    exp_name = "finetune_irtr_coco_base"
    task_identifier = "irtr_finetune_coco"
    model_class_name = "IRTRFinetuneModule" # 假设的pl.LightningModule类名
    datamodule_name = "COCODataModule"     # 或更通用的 IRTRDataModule

    encoder_model_name = "beit_base_patch16_384"
    datasets = ["coco"] # 使用 CocoCaptionKarpathyDataset

    load_path = "path/to/your/vl_pretrained_model.ckpt" # <--- 修改此路径

    learning_rate = 3e-5
    batch_size = 256 # IR/TR 通常需要较大 batch size (global batch)
    max_epochs = 10
    warmup_steps = 300
    get_recall_metric = True
    image_size = 384


# ----------------------- Example for a new model architecture -----------------------
@ex.named_config
def model_vlmo_custom_config(): # 这是一个修改模型的配置，可以与任务配置组合
    # encoder_model_name 应该在 Encoder_version.py 中注册
    encoder_model_name = "vlmo_custom_patch16"
    # 如果这个自定义模型有不同的 embed_dim, vocab_size 等，也需要在这里覆盖
    # 例如:
    # embed_dim = 640 # 假设 vlmo_custom_patch16 使用这个维度
    # vocab_size = 32000 # 如果tokenizer也变了


# ----------------------- Other Utility Configs (e.g., for training steps) -----------------------
@ex.named_config
def training_steps_5k():
    max_steps = 5000
    # 可能还需要调整 warmup_steps 和 max_epochs
    # max_epochs = 100 # 确保max_steps先生效

@ex.named_config
def training_steps_100k():
    max_steps = 100000
    # max_epochs = 200

