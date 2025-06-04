import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pretrained_model
from typing import Optional, Tuple, Callable, List, Dict
import numpy as np
from modules.Encoder import TransformerEncoder
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from modules import heads, objectives
from utils import vlmo_utils
from pytorch_lightning.utilities import rank_zero_info
from scipy import interpolate
from timm.models import create_model
from modules.CrossmodalMLMHead import CrossModalDecoder, CrossModalMLMHead
from modules.RMSNorm import RMSNorm
from transformers.activations import ACT2FN
import torch
from transformers import AutoTokenizer, BertTokenizer
from pytorch_lightning.utilities import rank_zero_info
from typing import Union, Tuple

def load_tokenizer_with_vocab_size(tokenizer_path: str, use_distributed_loading: bool = False) -> Tuple[Union[AutoTokenizer, BertTokenizer], int]:
    """
    加载tokenizer并返回tokenizer对象和vocab_size
    
    Args:
        tokenizer_path: tokenizer的路径或Hugging Face模型名称
        use_distributed_loading: 是否使用分布式加载策略
        
    Returns:
        tuple: (tokenizer对象, vocab_size)
    """
    rank_zero_info(f"Loading tokenizer from: {tokenizer_path}")
    
    if use_distributed_loading and torch.distributed.is_initialized():
        # 分布式加载策略：只有rank 0加载，然后同步
        if torch.distributed.get_rank() == 0:
            rank_zero_info("Rank 0: Loading tokenizer...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                rank_zero_info(f"AutoTokenizer failed, trying BertTokenizer: {e}")
                tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        
        # 同步所有进程
        torch.distributed.barrier()
        
        # 其他进程也加载tokenizer
        if torch.distributed.get_rank() != 0:
            try:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            except Exception as e:
                tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    else:
        # 非分布式加载
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            rank_zero_info(f"AutoTokenizer failed, trying BertTokenizer: {e}")
            tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    
    vocab_size = tokenizer.vocab_size
    rank_zero_info(f"Tokenizer loaded successfully. Vocab size: {vocab_size}")
    
    return tokenizer, vocab_size

def update_model_arch_with_vocab_size(model_arch: dict, vocab_size: int) -> dict:
    """
    更新model_arch字典，添加动态确定的vocab_size
    
    Args:
        model_arch: 原始的模型架构字典
        vocab_size: 动态确定的词汇表大小
        
    Returns:
        dict: 更新后的模型架构字典
    """
    updated_arch = model_arch.copy()
    updated_arch["vocab_size"] = vocab_size
    rank_zero_info(f"Updated model_arch with vocab_size: {vocab_size}")
    return updated_arch

class VLMoForTextPretraining(pl.LightningModule):
    def __init__(self, config_dict:dict):
        super().__init__()
        tokenizer_path = config_dict.get("tokenizer")
        if not tokenizer_path:
            raise ValueError("TextPretraining requires 'tokenizer' configuration")
        
        rank_zero_info("VLMoForTextPretraining: Loading tokenizer to determine vocab_size...")
        self.tokenizer, actual_vocab_size = load_tokenizer_with_vocab_size(
            tokenizer_path, 
            use_distributed_loading=False  # 在模型初始化时通常不需要分布式加载
        )
        
        # 第二步：更新config_dict和model_arch，添加动态确定的vocab_size
        config_dict = config_dict.copy()  # 避免修改原始配置
        config_dict["vocab_size"] = actual_vocab_size  # 添加到config中以便后续使用
        
        model_arch = config_dict.get("model_arch", {})
        model_arch = update_model_arch_with_vocab_size(model_arch, actual_vocab_size)
        config_dict["model_arch"] = model_arch
        
        # 第三步：保存超参数
        self.save_hyperparameters({
            "config": config_dict,
            "actual_vocab_size": actual_vocab_size,
            "tokenizer_path": tokenizer_path
        })
        self.config = config_dict
        
        # 第四步：创建编码器模型
        model_arch = config_dict.get("model_arch", {})
        self.Encoder = TransformerEncoder(
            img_size=model_arch.get("img_size", 384),
            patch_size=model_arch.get("patch_size", 16),
            in_chans=model_arch.get("in_chans", 3),
            embed_dim=model_arch.get("embed_dim", 512),
            depth=model_arch.get("depth", 6),
            num_heads=model_arch.get("num_heads", 8),
            num_kv_heads=model_arch.get("num_kv_heads", 4),
            qkv_bias=model_arch.get("qkv_bias", False),
            qk_scale=model_arch.get("qk_scale", None),
            attn_drop_rate=model_arch.get("attn_drop_rate", 0.0),
            drop_path_rate=model_arch.get("drop_path_rate", 0.1),
            norm_eps=model_arch.get("norm_eps", 1e-6),
            layer_scale_init_values=model_arch.get("layer_scale_init_values", 0.01),
            init_std=model_arch.get("init_std", 0.02),
            # MoE参数
            num_experts=model_arch.get("num_experts", 4),
            num_experts_per_tok=model_arch.get("num_experts_per_tok", 2),
            mlp_ratio=model_arch.get("mlp_ratio", 4.0),
            norm_topk_prob=model_arch.get("norm_topk_prob", True),
            moe_hidden_act=model_arch.get("moe_hidden_act", "silu"),
            # RoPE参数
            max_seq_len=config_dict.get("max_text_len", 196),
            rope_base=model_arch.get("rope_base", 10000),
            # Token Type参数
            num_token_types=model_arch.get("num_token_types", 2),
            vocab_size=actual_vocab_size,
            padding_idx=model_arch.get("padding_idx", 0),
        )

        # 获取模型属性
        self.activation = self.Encoder.moe_hidden_act
        self.img_size = self.Encoder.img_size
        self.patch_size = self.Encoder.patch_size
        self.num_layers = self.Encoder.depth
        self.embed_dim = self.Encoder.embed_dim
        self.head_dim = self.Encoder.head_dim
        
        # 创建BERT配置用于MLM头
        bert_config = BertConfig(
            vocab_size=config_dict["vocab_size"],
            hidden_size=self.embed_dim,
            hidden_dropout_prob=config_dict["drop_path_rate"],
        )
        
        # 创建MLM评分头
        self.mlm_score = heads.MLMHead(bert_config)
        self.mlm_score.apply(objectives.init_weights)

        # 设置文本预训练的评估指标
        vlmo_utils.set_metrics_for_text_pretraining(self)

        # 加载预训练权重并进行参数冻结设置
        self.setup_stage_weights_and_freezing()
    
    def _internal_convert_vision_to_textpt(self, source_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        将视觉预训练权重转换为文本预训练MoE架构的权重。
        将单个MLP的权重复制到所有MoE专家中，同时保留patch_embed（包含注意力）。
        
        Args:
            source_state_dict: 视觉预训练模型的状态字典
            
        Returns:
            转换后的状态字典，可直接加载到当前模型
        """
        new_state_dict = {}
        actual_vocab_size = self.hparams.actual_vocab_size
        rank_zero_info("VLMoForTextPretraining: Converting Vision Pretrained (single MLP) checkpoint for Text Pretraining (MoE)...")

        # 统计转换信息
        skipped_keys_count = 0
        mapped_mlp_keys_count = 0
        mapped_attention_keys_count = 0
        mapped_norm_keys_count = 0
        mapped_embedding_keys_count = 0
        other_mapped_keys_count = 0

        if not hasattr(self, 'Encoder') or not hasattr(self.Encoder, 'blocks') or not self.Encoder.blocks:
            rank_zero_info("Error: Encoder or blocks not found in current model!")

        # 获取当前模型的MoE配置信息
        try:
            first_block = self.Encoder.blocks[0]
            if hasattr(first_block, 'mlp') and hasattr(first_block.mlp, 'experts'):
                num_experts = first_block.mlp.experts
                rank_zero_info(f"Target model has {num_experts} experts per layer")
            else:
                num_experts = self.config["model_arch"].get("num_experts", 4)
                rank_zero_info(f"Using config MoE: {num_experts} experts per layer")
        except Exception as e:
            num_experts = self.config["model_arch"].get("num_experts", 4)
            rank_zero_info(f"Error determining MoE structure: {e}")

        for key, value in source_state_dict.items():
            # 跳过vocab相关的权重，因为vocab_size可能不匹配
            if "word_embeddings" in key:
                source_vocab_size = value.shape[0] if len(value.shape) > 0 else 0
                if source_vocab_size != actual_vocab_size:
                    rank_zero_info(f"Skipping {key}: source vocab_size={source_vocab_size}, target vocab_size={actual_vocab_size}")
                    skipped_keys_count += 1
                    continue
                else:
                    rank_zero_info(f"Mapping {key}: vocab sizes match ({actual_vocab_size})")
                    new_state_dict[key] = value
                    mapped_embedding_keys_count += 1
                    continue
            
            # 跳过其他可能依赖vocab_size的组件
            if any(vocab_key in key for vocab_key in ["mlm_score", "lm_head", "predictions", "classifier"]):
                # 检查是否是vocab相关的权重
                if len(value.shape) > 0 and value.shape[0] == actual_vocab_size:
                    rank_zero_info(f"Mapping vocab-related {key}: shapes compatible")
                    new_state_dict[key] = value
                    mapped_embedding_keys_count += 1
                elif len(value.shape) > 0 and value.shape[-1] == actual_vocab_size:
                    rank_zero_info(f"Mapping vocab-related {key}: output dim compatible")
                    new_state_dict[key] = value
                    mapped_embedding_keys_count += 1
                else:
                    rank_zero_info(f"Skipping {key}: shape {value.shape} not compatible with dynamic vocab_size={actual_vocab_size}")
                    skipped_keys_count += 1
                continue
            
            # 处理MLP -> MoE转换
            if ".mlp." in key and "experts" not in key:
                # 这是单个MLP的权重，需要复制到所有专家
                layer_match = None
                for i in range(len(self.Encoder.blocks)):
                    if f".blocks.{i}.mlp." in key:
                        layer_match = i
                        break
                
                if layer_match is not None:
                    # 提取MLP权重类型（如fc1.weight, fc2.weight等）
                    mlp_weight_name = key.split(f".blocks.{layer_match}.mlp.")[-1]
                    
                    # 复制到所有专家
                    for expert_idx in range(num_experts):
                        new_key = key.replace(f".blocks.{layer_match}.mlp.", f".blocks.{layer_match}.mlp.experts.{expert_idx}.")
                        new_state_dict[new_key] = value.clone()
                        mapped_mlp_keys_count += 1
                    
                    rank_zero_info(f"Converted MLP {key} to {num_experts} experts")
                    continue
            
            # 处理注意力层权重
            if any(attn_key in key for attn_key in [".attn.", ".attention.", ".self_attn."]):
                new_state_dict[key] = value
                mapped_attention_keys_count += 1
                continue
            
            # 处理归一化层权重
            if any(norm_key in key for norm_key in [".norm.", ".layernorm.", ".layer_norm."]):
                new_state_dict[key] = value
                mapped_norm_keys_count += 1
                continue
            
            # 处理token type embeddings
            if "token_type_embeddings" in key:
                new_state_dict[key] = value
                mapped_embedding_keys_count += 1
                continue
            
            # 处理其他权重（patch_embed, cls_token, position embeddings等）
            if any(other_key in key for other_key in ["patch_embed", "cls_token", "pos_embed", "rope"]):
                new_state_dict[key] = value
                other_mapped_keys_count += 1
                continue
            
            # 跳过MLM相关的权重（这些是新任务特定的）
            if any(skip_key in key for skip_key in ["mlm_score", "classifier", "pooler"]):
                rank_zero_info(f"Skipping task-specific weight: {key}")
                skipped_keys_count += 1
                continue
            
            # 其他未匹配的权重
            rank_zero_info(f"Unhandled key: {key}")
            skipped_keys_count += 1

        rank_zero_info(f"Conversion summary:")
        rank_zero_info(f"  - MLP layers converted to MoE: {mapped_mlp_keys_count}")
        rank_zero_info(f"  - Attention layers mapped: {mapped_attention_keys_count}")
        rank_zero_info(f"  - Norm layers mapped: {mapped_norm_keys_count}")
        rank_zero_info(f"  - Embedding layers mapped: {mapped_embedding_keys_count}")
        rank_zero_info(f"  - Other keys mapped: {other_mapped_keys_count}")
        rank_zero_info(f"  - Keys skipped: {skipped_keys_count}")
        
        return new_state_dict
        
    
    def setup_stage_weights_and_freezing(self):
        """
        设置阶段权重加载和参数冻结策略
        """
        load_path = self.hparams.config.get("weight_path", "")
        if not load_path:
            rank_zero_info("VLMoForTextPretraining: No weight path provided, using random initialization")
            self._apply_parameter_freezing()
            return

        rank_zero_info(f"VLMoForTextPretraining: Loading checkpoint from {load_path}")
        try:
            ckpt = torch.load(load_path, map_location="cpu")
        except Exception as e:
            rank_zero_info(f"Error loading checkpoint: {e}")
            self._apply_parameter_freezing()
            return

        # 获取state_dict
        raw_state_dict = ckpt.get("state_dict", ckpt.get("module", ckpt.get("model", ckpt)))
        if raw_state_dict is None:
            rank_zero_info("Error: No state_dict found in checkpoint")
            self._apply_parameter_freezing()
            return

        # 执行权重转换（从视觉预训练到文本预训练MoE）
        rank_zero_info("Converting vision pretrained weights for text pretraining with MoE...")
        processed_sd = self._internal_convert_vision_to_textpt(raw_state_dict)
         # 加载到Encoder
        encoder_model_dict = self.Encoder.state_dict()
        final_encoder_sd_to_load = {
            k: v for k, v in processed_sd.items()
            if k in encoder_model_dict and encoder_model_dict[k].shape == v.shape
        }
        
        if final_encoder_sd_to_load:
            missing_keys, unexpected_keys = self.Encoder.load_state_dict(final_encoder_sd_to_load, strict=False)
            rank_zero_info(f"Encoder loaded {len(final_encoder_sd_to_load)} weights successfully")
            if missing_keys:
                rank_zero_info(f"Missing Encoder keys: {len(missing_keys)} (some expected for new components)")
            if unexpected_keys:
                rank_zero_info(f"Unexpected Encoder keys: {len(unexpected_keys)}")
        else:
            rank_zero_info("Warning: No compatible weights found for Encoder")

        # MLM头保持随机初始化（因为是新任务）
        rank_zero_info("MLM head keeps random initialization for new text pretraining task")
        
        # 应用参数冻结策略
        self._apply_parameter_freezing()
        

    def _apply_parameter_freezing(self):
        """
        应用参数冻结策略：冻结注意力层，解冻MoE层
        """
        rank_zero_info("VLMoForTextPretraining: Applying parameter freezing strategy")
        
        # 首先冻结所有参数
        for param in self.parameters():
            param.requires_grad = False

        # 需要解冻的组件关键词
        unfreeze_keywords = [
            # 文本嵌入相关（新增组件）
            "Encoder.word_embeddings.",
            "Encoder.token_type_embeddings.",
            
            # MoE相关参数（主要训练目标）
            "moe_block.experts.",
            "moe_block.gate.",
            
            # MLM头（新任务头）
            "mlm_score.",
            
            # 最终归一化层（可能需要适应新任务）
            "Encoder.norm.",
            
            # Layer Scale参数（FFN部分，与MoE相关）
            ".gamma_ffn",
        ]
        
        # 可选解冻的组件（根据配置决定）
        if self.hparams.config.get("unfreeze_post_attention_layernorm", True):
            unfreeze_keywords.append("post_attention_layernorm.")
        
        rank_zero_info(f"Attempting to unfreeze parameters containing keywords: {unfreeze_keywords}")
        unfrozen_count = 0
        frozen_count = 0
        
        for name, param in self.named_parameters():
            should_unfreeze = any(keyword in name for keyword in unfreeze_keywords)
            
            if should_unfreeze:
                param.requires_grad = True
                unfrozen_count += 1
                # 可选：打印解冻的重要参数
                if any(important in name for important in ["moe_block", "mlm_score", "word_embeddings"]):
                    rank_zero_info(f"  Unfrozen: {name}")
            else:
                frozen_count += 1
                # 可选：打印冻结的注意力参数示例
                if "attn." in name and frozen_count <= 5:  # 只打印前几个示例
                    rank_zero_info(f"  Frozen (attention): {name}")

        rank_zero_info(f"Parameter freezing summary:")
        rank_zero_info(f"  - Total parameters unfrozen: {unfrozen_count}")
        rank_zero_info(f"  - Total parameters frozen: {frozen_count}")
        rank_zero_info(f"  - Strategy: Focus on MoE training while keeping attention frozen")

    def forward(self, batch):
        text_ids = batch["text_ids_mlm"]
        text_labels_mlm = batch["text_labels_mlm"]
        text_masks = batch["text_masks"]

        encoder_outputs = self.Encoder(
            input_ids=text_ids,
            text_mask=text_masks, 
            image_tensor=None, # 纯文本预训练阶段不使用图像
            output_router_logits=True,
        )
        text_feats = encoder_outputs.last_hidden_state
        mlm_logits = self.mlm_score(text_feats)
        
        return {"logits": mlm_logits, "labels": text_labels_mlm, "text_ids": text_ids, "encoder_outputs_obj": encoder_outputs}
    
    def training_step(self, batch, batch_idx):
        """
        训练步骤
        """
        self.current_batch_size = batch["text_ids_mlm"].size(0)
        outputs = self.forward(batch)
        mlm_logits = outputs["logits"]
        mlm_labels = outputs["labels"]

        mlm_loss = F.cross_entropy(
            mlm_logits.reshape(-1, self.hparams.config["vocab_size"]),
            mlm_labels.reshape(-1),
            ignore_index=-100,
        )

        moe_aux_loss = torch.tensor(0.0, device=mlm_loss.device)
        encoder_outputs_obj = outputs.get("encoder_outputs_obj")
        if encoder_outputs_obj is not None:
            router_logits_tuple = getattr(encoder_outputs_obj, "router_probs", None) 
            if router_logits_tuple is not None and len(router_logits_tuple) > 0:
                moe_aux_loss = self.compute_moe_aux_loss_from_router_logits(router_logits_tuple)
        
        total_loss = mlm_loss + moe_aux_loss

        self.log("train/mlm_loss", mlm_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/moe_aux_loss", moe_aux_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)

        acc_metric = getattr(self, f"train_textmlm_accuracy", None) # from set_metrics
        if acc_metric:
            preds = mlm_logits.argmax(dim=-1)
            mask = mlm_labels != -100
            if mask.sum() > 0:
                 acc_metric.update(preds[mask], mlm_labels[mask])

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        self.current_batch_size = batch["text_ids_mlm"].size(0)
        outputs = self.forward(batch)
        mlm_logits = outputs["logits"]
        mlm_labels = outputs["labels"]

        mlm_loss = F.cross_entropy(
            mlm_logits.reshape(-1, self.hparams.config["vocab_size"]),
            mlm_labels.reshape(-1),
            ignore_index=-100,
        )
    
        moe_aux_loss = torch.tensor(0.0, device=mlm_loss.device)
        encoder_outputs_obj = outputs.get("encoder_outputs_obj")
        if encoder_outputs_obj is not None:
            router_logits_tuple = getattr(encoder_outputs_obj, "router_probs", None)
            if router_logits_tuple is not None and len(router_logits_tuple) > 0:
                moe_aux_loss = self.compute_moe_aux_loss_from_router_logits(router_logits_tuple)

        total_loss = mlm_loss + moe_aux_loss

        self.log("val/mlm_loss", mlm_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/moe_aux_loss", moe_aux_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log("val/total_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)

        acc_metric = getattr(self, f"val_{vlmo_utils.TEXT_PRETRAIN_METRIC_NAME}", None)
        
        if acc_metric:
            preds = mlm_logits.argmax(dim=-1)
            mask = mlm_labels != -100
            if mask.sum() > 0:
                 acc_metric.update(preds[mask], mlm_labels[mask])
        
        return total_loss
    
    def on_validation_epoch_end(self):
        """
        验证周期结束时的处理
        """
        vlmo_utils.epoch_wrapup_for_text_pretraining(self)
        train_acc_metric = getattr(self, f"train_{vlmo_utils.TEXT_PRETRAIN_METRIC_NAME}", None)
        val_acc_metric = getattr(self, f"val_{vlmo_utils.TEXT_PRETRAIN_METRIC_NAME}", None)

        if train_acc_metric:
            train_acc_metric.reset()
        if val_acc_metric:
            val_acc_metric.reset()


    @staticmethod
    def _layer_moe_losses(router_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算单层的MoE损失（平衡损失和路由器Z损失）
        
        Args:
            router_logits: 路由器的logits，形状为 (batch_size*sequence_length, num_experts)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (balance_loss, router_z_loss)
        """

        num_experts = router_logits.size(-1)   
        
        if num_experts == 1:
            return torch.tensor(0.0, device=router_logits.device), torch.tensor(0.0, device=router_logits.device)                             # (E,)
        
        gates = F.softmax(router_logits, dim=-1)

        # 负载均衡损失
        density = gates.mean(dim=0)
        balance_loss = (density * num_experts).pow(2).sum()

        # 路由器z损失（稀疏性损失）
        router_z_loss = router_logits.pow(2).mean()
        
        return balance_loss, router_z_loss
    
    def compute_moe_aux_loss_from_router_logits(
        self, router_logits_tuple: Tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """
        从路由器logits元组计算MoE辅助损失
        
        Args:
            router_logits_tuple: 所有层的路由器logits元组
            
        Returns:
            torch.Tensor: 总的MoE辅助损失
        """
        if not router_logits_tuple:
            return torch.tensor(0.0, device=self.device)
        
        balance_loss_total = 0.0
        router_z_loss_total = 0.0

        for layer_logits in router_logits_tuple:                # 遍历所有层
            balance_loss, router_z_loss = self._layer_moe_losses(layer_logits)
            balance_loss_total += balance_loss
            router_z_loss_total += router_z_loss

        n_layer = len(router_logits_tuple)
        balance_loss_avg = balance_loss_total / n_layer
        router_z_loss_avg = router_z_loss_total / n_layer

        coef_b = self.hparams.config.get("moe_balance_loss_weight", 0.01)
        coef_z = self.hparams.config.get("moe_router_z_loss_weight", 0.001)
        
        total_aux_loss = coef_b * balance_loss_avg + coef_z * router_z_loss_avg

        # 记录详细的MoE损失信息
        self.log("train/moe_balance_loss", balance_loss_avg, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/moe_router_z_loss", router_z_loss_avg, prog_bar=False, logger=True, sync_dist=True)
        
        return total_aux_loss


    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        return vlmo_utils.set_schedule_for_MLM(self) # Assuming set_schedule can work with self.hparams


class VLMoForVisionLanguagePretraining(pl.LightningModule):
    def __init__(self, config_dict: dict):
        super().__init__()
        
        # 第一步：动态加载tokenizer并获取vocab_size
        tokenizer_path = config_dict.get("tokenizer")
        if not tokenizer_path:
            raise ValueError("VisionLanguagePretraining requires 'tokenizer' configuration")
        
        rank_zero_info("VLMoForVisionLanguagePretraining: Loading tokenizer to determine vocab_size...")
        self.tokenizer, actual_vocab_size = load_tokenizer_with_vocab_size(
            tokenizer_path, 
            use_distributed_loading=False
        )
        
        # 第二步：更新config_dict和model_arch
        config_dict = config_dict.copy()
        config_dict["vocab_size"] = actual_vocab_size
        
        model_arch = config_dict.get("model_arch", {})
        model_arch = update_model_arch_with_vocab_size(model_arch, actual_vocab_size)
        config_dict["model_arch"] = model_arch
        
        # 第三步：保存超参数
        self.save_hyperparameters({
            "config": config_dict,
            "actual_vocab_size": actual_vocab_size,
            "tokenizer_path": tokenizer_path
        })
        self.config = config_dict


        # 第四步：创建编码器
        model_arch = config_dict.get("model_arch", {})
        self.Encoder = TransformerEncoder(
            img_size=model_arch.get("img_size", 384),
            patch_size=model_arch.get("patch_size", 16),
            in_chans=model_arch.get("in_chans", 3),
            embed_dim=model_arch.get("embed_dim", 512),
            depth=model_arch.get("depth", 6),
            num_heads=model_arch.get("num_heads", 8),
            num_kv_heads=model_arch.get("num_kv_heads", 8),
            qkv_bias=model_arch.get("qkv_bias", False),
            qk_scale=model_arch.get("qk_scale", None),
            attn_drop_rate=model_arch.get("attn_drop_rate", 0.0),
            drop_path_rate=model_arch.get("drop_path_rate", 0.1),
            norm_eps=model_arch.get("norm_eps", 1e-6),
            layer_scale_init_values=model_arch.get("layer_scale_init_values", 0.01),
            init_std=model_arch.get("init_std", 0.02),
            # MoE参数
            num_experts=model_arch.get("num_experts", 4),
            num_experts_per_tok=model_arch.get("num_experts_per_tok", 2),
            mlp_ratio=model_arch.get("mlp_ratio", 4.0),
            norm_topk_prob=model_arch.get("norm_topk_prob", True),
            moe_hidden_act=model_arch.get("moe_hidden_act", "silu"),
            # RoPE参数
            max_seq_len=config_dict.get("max_text_len", 196),
            rope_base=model_arch.get("rope_base", 10000),
            # Token Type参数
            num_token_types=model_arch.get("num_token_types", 2),
            vocab_size=actual_vocab_size,
            padding_idx=model_arch.get("padding_idx", 0),
        )
        
        # 获取模型属性
        self.activation = self.Encoder.moe_hidden_act
        self.img_size = self.Encoder.img_size
        self.patch_size = self.Encoder.patch_size
        self.num_layers = self.Encoder.depth
        self.embed_dim = self.Encoder.embed_dim
        self.head_dim = self.Encoder.head_dim
        
        # 第五步：创建各种头部，使用动态vocab_size
        # 池化层
        self.pooler = heads.Pooler(self.embed_dim)
        
        # 创建ITC头
        self.itc_text_proj = heads.ITCHead(self.embed_dim)
        self.itc_image_proj = heads.ITCHead(self.embed_dim)

        # 可学习的温度参数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
        
        # MLM头部（使用动态vocab_size）
        bert_config = BertConfig(
            vocab_size=actual_vocab_size,
            hidden_size=self.embed_dim,
            hidden_dropout_prob=config_dict.get("drop_path_rate", 0.1),
        )
        self.mlm_score = heads.MLMHead(bert_config)
        
        # 交叉模态解码器和MLM头部
        decoder_config = config_dict
        self.cross_modal_decoder = CrossModalDecoder(
            dim=self.embed_dim,
            context_dim=self.embed_dim,  # 图像和文本embed_dim相同
            depth=decoder_config.get("decoder_depth", 2),
            num_heads=decoder_config.get("num_heads", 8),
            num_kv_heads=decoder_config.get("num_kv_heads", 8),
            qkv_bias=decoder_config.get("qkv_bias", False),
            qk_scale=decoder_config.get("qk_scale", None),
            attn_drop=decoder_config.get("attn_drop_rate", 0.0),
            drop_path_rate=decoder_config.get("drop_path_rate", 0.1),
            norm_eps=decoder_config.get("norm_eps", 1e-6),
            layer_scale_init_values=decoder_config.get("layer_scale_init_values", 0.01),
            # MoE参数
            num_experts=model_arch.get("decoder_num_experts", 4),
            num_experts_per_tok=model_arch.get("decoder_num_experts_per_tok", 2),
            mlp_ratio=model_arch.get("mlp_ratio", 4.0),
            norm_topk_prob=model_arch.get("norm_topk_prob", True),
            moe_hidden_act=model_arch.get("moe_hidden_act", "silu"),
        )
        
        # 交叉模态MLM预测头
        self.cross_modal_mlm_score = CrossModalMLMHead(
            hidden_size=self.embed_dim,
            vocab_size=actual_vocab_size,
            hidden_act=model_arch.get("moe_hidden_act", "silu"),
            layer_norm_eps=model_arch.get("norm_eps", 1e-6),
        )
        
        # 初始化权重
        self.mlm_score.apply(objectives.init_weights)
        
        # 设置评估指标
        vlmo_utils.set_metrics_for_vision_language_pretraining(self)
        
        # 加载预训练权重
        self.setup_stage_weights_and_freezing()
    
    def setup_stage_weights_and_freezing(self):
        """设置权重加载策略"""
        load_path = self.hparams.config.get("weight_path", "")
        if not load_path:
            rank_zero_info("VLMoForVisionLanguagePretraining: No weight path provided, using random initialization")
            return

        rank_zero_info(f"VLMoForVisionLanguagePretraining: Loading checkpoint from {load_path}")
        try:
            ckpt = torch.load(load_path, map_location="cpu")
 
            # 获取state_dict
            raw_state_dict = ckpt.get("state_dict", ckpt.get("module", ckpt.get("model", ckpt)))
            if raw_state_dict is None:
                rank_zero_info("Error: No state_dict found in checkpoint")
                return

            # 过滤兼容的权重（检查vocab_size一致性）
            model_dict = self.state_dict()
            compatible_state_dict = {}
            for k, v in raw_state_dict.items():
                if k in model_dict:
                    if model_dict[k].shape == v.shape:
                        compatible_state_dict[k] = v
                    else:
                        # 特别处理vocab相关的权重
                        if "word_embeddings" in k or "mlm_score" in k:
                            source_vocab_size = v.shape[0] if len(v.shape) > 0 else 0
                            target_vocab_size = self.hparams.actual_vocab_size
                            rank_zero_info(f"Skipping {k}: shape mismatch - source: {v.shape}, target: {model_dict[k].shape}")
                            rank_zero_info(f"  vocab_size difference: source={source_vocab_size}, target={target_vocab_size}")
                        else:
                            rank_zero_info(f"Skipping {k}: shape mismatch - {v.shape} vs {model_dict[k].shape}")
            
            if compatible_state_dict:
                missing_keys, unexpected_keys = self.load_state_dict(compatible_state_dict, strict=False)
                rank_zero_info(f"Loaded {len(compatible_state_dict)} compatible weights")
                if missing_keys:
                    rank_zero_info(f"Missing keys: {len(missing_keys)} (expected for new components)")
                if unexpected_keys:
                    rank_zero_info(f"Unexpected keys: {len(unexpected_keys)}")
            else:
                rank_zero_info("No compatible weights found")
                
        except Exception as e:
            rank_zero_info(f"Failed to load checkpoint: {e}")
        
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):

        text_ids = batch["text_ids"]
        text_masks = batch["text_masks"]
        images = batch["image"][0] if isinstance(batch["image"], list) else batch["image"]

        encoder_outputs = self.Encoder(
            input_ids=text_ids,
            text_mask=text_masks,
            image_tensor=images,
            output_router_logits=True,
        )
        
        co_embeds = encoder_outputs.last_hidden_state
        
        # 分离图像和文本特征
        text_len = text_ids.shape[1]
        text_feats = co_embeds[:, :text_len]
        image_feats = co_embeds[:, text_len:]
        
        # CLS特征用于对比学习
        cls_feats = self.pooler(co_embeds)
        
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "encoder_outputs": encoder_outputs,
        }
        
        if mask_text:
            ret["text_labels"] = batch["text_labels_mlm"]
        
        return ret
    
    def infer_image(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        images = batch["image"][0] if isinstance(batch["image"], list) else batch["image"]
        
        encoder_outputs = self.Encoder(
            input_ids=None,
            text_mask=None,
            image_tensor=images,
            output_router_logits=True,
        )
        
        image_feats = encoder_outputs.last_hidden_state
        cls_feats = self.pooler(image_feats)
        
        return {
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "encoder_outputs": encoder_outputs,
        }

    def infer_text(
        self,
        batch,
        mask_text=False,
    ):
        """推理纯文本"""
        text_ids = batch["text_ids"]
        text_masks = batch["text_masks"]
        
        encoder_outputs = self.Encoder(
            input_ids=text_ids,
            text_mask=text_masks,
            image_tensor=None,
            output_router_logits=True,
        )
        
        text_feats = encoder_outputs.last_hidden_state
        cls_feats = self.pooler(text_feats)
        
        ret = {
            "text_feats": text_feats,
            "cls_feats": cls_feats,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "encoder_outputs": encoder_outputs,
        }
        
        if mask_text:
            ret["text_labels"] = batch["text_labels_mlm"]
        
        return ret

    def infer_cross_modal_mlm(self, batch):
        """
        执行交叉模态MLM推理
        图像嵌入作为键值，被掩码的语言嵌入作为查询
        """
        # 首先获取图像和文本的独立表示
        image_outputs = self.infer_image(batch)
        text_outputs = self.infer_text(batch, mask_text=True)
        
        # 使用交叉模态解码器
        decoder_outputs = self.cross_modal_decoder(
            hidden_states=text_outputs["text_feats"],  # 查询
            context=image_outputs["image_feats"],      # 键值
            output_router_logits=True,
        )
        
        enhanced_text_feats = decoder_outputs[0]
        
        return {
            "enhanced_text_feats": enhanced_text_feats,
            "text_labels": text_outputs["text_labels"],
            "text_ids": text_outputs["text_ids"],
            "text_masks": text_outputs["text_masks"],
            "decoder_outputs": decoder_outputs,
        }

    def forward(self, batch):
        """
        前向传播，包含ITC + 交叉模态MLM
        """
        outputs = {}
        
        # ITC损失
        image_outputs = self.infer_image(batch)
        text_outputs = self.infer_text(batch)
        
        image_proj_feats = self.itc_image_proj(image_outputs["cls_feats"])
        text_proj_feats = self.itc_text_proj(text_outputs["cls_feats"])
        
        # 归一化特征
        image_proj_feats = F.normalize(image_proj_feats, p=2, dim=-1)
        text_proj_feats = F.normalize(text_proj_feats, p=2, dim=-1)
        
        outputs.update({
            "image_proj_feats": image_proj_feats,
            "text_proj_feats": text_proj_feats,
            "image_encoder_outputs": image_outputs["encoder_outputs"],
            "text_encoder_outputs": text_outputs["encoder_outputs"],
        })
        
        # MLM损失（标准）
        if "text_labels_mlm" in batch:
            text_mlm_outputs = self.infer_text(batch, mask_text=True)
            mlm_logits = self.mlm_score(text_mlm_outputs["text_feats"])
            outputs.update({
                "mlm_logits": mlm_logits,
                "mlm_labels": text_mlm_outputs["text_labels"],
                "mlm_encoder_outputs": text_mlm_outputs["encoder_outputs"],
            })
        
        # 交叉模态MLM损失
        cross_modal_outputs = self.infer_cross_modal_mlm(batch)
        cross_modal_mlm_logits = self.cross_modal_mlm_score(cross_modal_outputs["enhanced_text_feats"])
        outputs.update({
            "cross_modal_mlm_logits": cross_modal_mlm_logits,
            "cross_modal_mlm_labels": cross_modal_outputs["text_labels"],
            "cross_modal_decoder_outputs": cross_modal_outputs["decoder_outputs"],
        })
        
        return outputs

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        self.current_batch_size = batch["text_ids"].size(0)
        outputs = self.forward(batch)
        
        total_loss = 0.0
        loss_dict = {}
        
        # 1. ITC损失
        if "itc_logits_i2t" in outputs and "itc_logits_t2i" in outputs:
            itc_i2t_logits = outputs["itc_logits_i2t"]
            itc_t2i_logits = outputs["itc_logits_t2i"]
            
            # 使用SigLIP损失或标准对比损失
            if self.config.get("use_siglip_loss", False):
                itc_loss = self.compute_siglip_loss(itc_i2t_logits, itc_t2i_logits)
            else:
                itc_loss = self.compute_contrastive_loss(itc_i2t_logits, itc_t2i_logits)
            
            itc_weight = self.config.get("itc_loss_weight", 1.0)
            total_loss += itc_weight * itc_loss
            loss_dict["itc_loss"] = itc_loss
        
        # 2. 交叉模态MLM损失
        if "cross_modal_mlm_logits" in outputs and "text_labels" in outputs:
            cross_modal_mlm_logits = outputs["cross_modal_mlm_logits"]
            mlm_labels = outputs["text_labels"]
            
            cross_modal_mlm_loss = F.cross_entropy(
                cross_modal_mlm_logits.reshape(-1, self.config["vocab_size"]),
                mlm_labels.reshape(-1),
                ignore_index=-100,
            )
            
            # 根据你的描述，交叉熵损失权重为0.1
            cross_modal_mlm_weight = self.config.get("cross_modal_mlm_weight", 0.1)
            total_loss += cross_modal_mlm_weight * cross_modal_mlm_loss
            loss_dict["cross_modal_mlm_loss"] = cross_modal_mlm_loss
        
        # 3. MoE辅助损失
        moe_aux_loss = self.compute_moe_aux_loss_from_outputs(outputs)
        if moe_aux_loss is not None:
            total_loss += moe_aux_loss
            loss_dict["moe_aux_loss"] = moe_aux_loss
        
        # 记录损失
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value, prog_bar=False, logger=True, sync_dist=True)
        
        self.log("train/total_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        
        # 记录准确率
        self._log_accuracies(outputs, "train")
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        self.current_batch_size = batch["text_ids"].size(0)
        outputs = self.forward(batch)
        
        total_loss = 0.0
        loss_dict = {}
        
        # 计算ITC损失
        if "itc_logits_i2t" in outputs and "itc_logits_t2i" in outputs:
            itc_i2t_logits = outputs["itc_logits_i2t"]
            itc_t2i_logits = outputs["itc_logits_t2i"]
            
            if self.config.get("use_siglip_loss", False):
                itc_loss = self.compute_siglip_loss(itc_i2t_logits, itc_t2i_logits)
            else:
                itc_loss = self.compute_contrastive_loss(itc_i2t_logits, itc_t2i_logits)
            
            itc_weight = self.config.get("itc_loss_weight", 1.0)
            total_loss += itc_weight * itc_loss
            loss_dict["itc_loss"] = itc_loss
        
        # 计算交叉模态MLM损失
        if "cross_modal_mlm_logits" in outputs and "text_labels" in outputs:
            cross_modal_mlm_logits = outputs["cross_modal_mlm_logits"]
            mlm_labels = outputs["text_labels"]
            
            cross_modal_mlm_loss = F.cross_entropy(
                cross_modal_mlm_logits.reshape(-1, self.config["vocab_size"]),
                mlm_labels.reshape(-1),
                ignore_index=-100,
            )
            
            cross_modal_mlm_weight = self.config.get("cross_modal_mlm_weight", 0.1)
            total_loss += cross_modal_mlm_weight * cross_modal_mlm_loss
            loss_dict["cross_modal_mlm_loss"] = cross_modal_mlm_loss
        
        # MoE辅助损失
        moe_aux_loss = self.compute_moe_aux_loss_from_outputs(outputs)
        if moe_aux_loss is not None:
            total_loss += moe_aux_loss
            loss_dict["moe_aux_loss"] = moe_aux_loss
        
        # 记录验证损失
        for loss_name, loss_value in loss_dict.items():
            self.log(f"val/{loss_name}", loss_value, prog_bar=False, logger=True, sync_dist=True)
        
        self.log("val/total_loss", total_loss, prog_bar=True, logger=True, sync_dist=True)
        
        # 记录准确率
        self._log_accuracies(outputs, "val")

        return total_loss

    def _log_accuracies(self, outputs, phase):
        """记录准确率指标"""
        # ITC准确率
        if "itc_logits_i2t" in outputs and "itc_logits_t2i" in outputs:
            itc_i2t_logits = outputs["itc_logits_i2t"]
            itc_t2i_logits = outputs["itc_logits_t2i"]
            
            # 计算准确率
            labels = torch.arange(itc_i2t_logits.size(0), device=itc_i2t_logits.device)
            
            i2t_preds = itc_i2t_logits.argmax(dim=-1)
            t2i_preds = itc_t2i_logits.argmax(dim=-1)
            
            # 更新指标
            i2t_metric = getattr(self, f"{phase}_{vlmo_utils.VL_PRETRAIN_ITC_I2T_METRIC_NAME}", None)
            t2i_metric = getattr(self, f"{phase}_{vlmo_utils.VL_PRETRAIN_ITC_T2I_METRIC_NAME}", None)
            
            if i2t_metric:
                i2t_metric.update(i2t_preds, labels)
                self.log(f"{phase}/{vlmo_utils.VL_PRETRAIN_ITC_I2T_METRIC_NAME}", i2t_metric, 
                        prog_bar=False, logger=True, on_step=False, on_epoch=True)
            
            if t2i_metric:
                t2i_metric.update(t2i_preds, labels)
                self.log(f"{phase}/{vlmo_utils.VL_PRETRAIN_ITC_T2I_METRIC_NAME}", t2i_metric,
                        prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
        # 交叉模态MLM准确率
        if "cross_modal_mlm_logits" in outputs and "text_labels" in outputs:
            cross_modal_mlm_logits = outputs["cross_modal_mlm_logits"]
            mlm_labels = outputs["text_labels"]
            
            if mlm_labels is not None:
                cross_modal_preds = cross_modal_mlm_logits.argmax(dim=-1)
                mask = mlm_labels != -100
                
                if mask.sum() > 0:
                    cross_modal_metric = getattr(self, f"{phase}_{vlmo_utils.VL_PRETRAIN_CROSS_MODAL_MLM_METRIC_NAME}", None)
                    if cross_modal_metric:
                        cross_modal_metric.update(cross_modal_preds[mask], mlm_labels[mask])
                        self.log(f"{phase}/{vlmo_utils.VL_PRETRAIN_CROSS_MODAL_MLM_METRIC_NAME}", cross_modal_metric,
                                prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """验证周期结束时的处理"""
        vlmo_utils.epoch_wrapup_for_vision_language_pretraining(self)

    def compute_moe_aux_loss_from_outputs(self, outputs):
        """从输出中计算MoE辅助损失"""
        moe_aux_loss = torch.tensor(0.0, device=self.device)
        
        # 解码器MoE损失
        if "decoder_outputs" in outputs:
            decoder_outputs = outputs["decoder_outputs"]
            if len(decoder_outputs) > 2:  # 包含router_logits
                decoder_router_logits = decoder_outputs[2]  # 假设是第三个输出
                if decoder_router_logits:
                    decoder_moe_loss = self.compute_moe_aux_loss_from_router_logits(decoder_router_logits)
                    moe_aux_loss += decoder_moe_loss
        
        return moe_aux_loss if moe_aux_loss > 0 else None

    def compute_moe_aux_loss_from_router_logits(self, router_logits_tuple):
        """从路由器logits计算MoE损失"""
        if not router_logits_tuple:
            return torch.tensor(0.0, device=self.device)
        
        balance_loss_total = 0.0
        router_z_loss_total = 0.0
        
        for layer_logits in router_logits_tuple:
            if layer_logits is not None:
                balance_loss, router_z_loss = self._layer_moe_losses(layer_logits)
                balance_loss_total += balance_loss
                router_z_loss_total += router_z_loss
        
        n_layer = len(router_logits_tuple)
        if n_layer > 0:
            balance_loss_avg = balance_loss_total / n_layer
            router_z_loss_avg = router_z_loss_total / n_layer
        else:
            balance_loss_avg = router_z_loss_avg = 0.0
        
        coef_b = self.config.get("moe_balance_loss_weight", 0.01)
        coef_z = self.config.get("moe_router_z_loss_weight", 0.001)
        
        return coef_b * balance_loss_avg + coef_z * router_z_loss_avg

    @staticmethod
    def _layer_moe_losses(router_logits):
        """计算单层MoE损失"""
        if router_logits is None:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        # Balance loss
        if router_logits.numel() == 0:
            return torch.tensor(0.0), torch.tensor(0.0)
        
        probs = F.softmax(router_logits, dim=-1)
        num_experts = probs.size(-1)
        
        # 计算每个专家的负载
        counts = probs.sum(dim=0)
        route_prob_max = probs.max(dim=-1)[0]
        
        # Balance loss: 鼓励负载均衡
        balance_loss = num_experts * torch.sum(counts * route_prob_max) / counts.sum()
        
        # Router z-loss: 正则化路由器输出
        router_z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        
        return balance_loss, router_z_loss

    def compute_siglip_loss(self, logits_i2t, logits_t2i):
        """计算SigLIP损失"""
        # SigLIP使用sigmoid而不是softmax
        batch_size = logits_i2t.size(0)
        
        # 创建目标矩阵
        targets = torch.eye(batch_size, device=logits_i2t.device)
        
        # SigLIP损失
        loss_i2t = F.binary_cross_entropy_with_logits(logits_i2t, targets)
        loss_t2i = F.binary_cross_entropy_with_logits(logits_t2i.T, targets)
        
        return (loss_i2t + loss_t2i) / 2.0

    def compute_contrastive_loss(self, logits_i2t, logits_t2i):
        """计算标准对比损失"""
        labels = torch.arange(logits_i2t.size(0), device=logits_i2t.device)
        
        loss_i2t = F.cross_entropy(logits_i2t, labels)
        loss_t2i = F.cross_entropy(logits_t2i, labels)
        
        return (loss_i2t + loss_t2i) / 2.0

    def configure_optimizers(self):
        """配置优化器"""
        return vlmo_utils.set_schedule_for_vision_language_pretraining(self)

# 用于构建和训练一个视觉 - 语言多模态模型
class VLMo(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        # 保存传入的配置参数，方便后续使用
        self.save_hyperparameters()
        self.automatic_optimization = True
        self.test_step_outputs = []
        self.Encoder = TransformerEncoder(
            img_size=config.get("img_size", 384),
            patch_size=config.get("patch_size", 16), # 从config获取或使用默认值
            in_chans=config.get("in_chans", 3),
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            num_kv_heads=config.get("num_kv_heads", None), # GQA heads
            qkv_bias=config.get("qkv_bias", True),
            drop_path_rate=config["drop_path_rate"],
            norm_layer=RMSNorm, # 假设 TransformerEncoder 内部处理 partial
            norm_eps=config.get("norm_eps", 1e-6),
            layer_scale_init_values=config.get("layer_scale_init_values", 1e-5),
            num_experts=config.get("num_experts", 1), # MoE
            num_experts_per_tok=config.get("num_experts_per_tok", 2 if config.get("num_experts", 1) > 1 else 1),
            mlp_ratio=config.get("mlp_ratio", 4.0),
            norm_topk_prob=config.get("norm_topk_prob", False), # MoeSparseMoeBlock param
            moe_hidden_act=config.get("moe_hidden_act", "silu"), # MoeSparseMoeBlock param
            max_seq_len=config["max_text_len"], # For 1D RoPE in TransformerEncoder
            rope_base=config.get("rope_base", 10000),
            num_token_types=config.get("num_token_types", 2)
        )
        self.activation = self.Encoder.moe_hidden_act
        self.img_size = self.Encoder.img_size
        self.patch_size = self.Encoder.patch_size
        self.num_layers = self.Encoder.depth
        self.embed_dim = self.Encoder.embed_dim
        self.head_dim = self.Encoder.head_dim
    
        bert_config = BertConfig(
            vocab_size=config["vocab_size"],
            hidden_size=self.embed_dim,
            max_position_embeddings=config["max_text_len"],
            hidden_dropout_prob=config["drop_path_rate"], 
        )

        self.text_embeddings = BertEmbeddings(bert_config)
        # 应用初始化权重函数
        self.text_embeddings.apply(objectives.init_weights)
        if hasattr(self.text_embeddings, 'position_embeddings') and \
            hasattr(self.text_embeddings.position_embeddings, 'weight'):
            rank_zero_info(">>> Disabling Text Absolute Position Embeddings by zeroing weights <<<")
            with torch.no_grad():
                self.text_embeddings.position_embeddings.weight.zero_()
            # self.text_embeddings.position_embeddings.weight.requires_grad = False
        else:
            rank_zero_info(">>> Warning: Could not find text_embeddings.position_embeddings.weight to zero out. <<<")
      
        self.pooler = heads.Pooler(self.embed_dim)
        self.pooler.apply(objectives.init_weights)

        self.load_pretrained_weight()

        if config["loss_names"]["mlm"] > 0 or config["loss_names"]["textmlm"] > 0:
            self.mlm_score = heads.MLMHead(bert_config)
            self.mlm_score.apply(objectives.init_weights)

        if config["loss_names"]["itm"] > 0:
            self.itm_score = heads.ITMHead(self.embed_dim)
            self.itm_score.apply(objectives.init_weights)
        
        ## contrastive loss (or sampling for global hard negative)
        # 如果配置中 itc 损失大于 0，则初始化图像和文本的投影头
        if config["loss_names"]["itc"] > 0:
            self.itc_text_proj = heads.ITCHead(self.embed_dim)
            self.itc_image_proj = heads.ITCHead(self.embed_dim)
            # 应用初始化权重函数
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            self.itc_vl_text_proj = heads.ITCHead(self.embed_dim)
            self.itc_vl_image_proj = heads.ITCHead(self.embed_dim)
            # 应用初始化权重函数
            self.itc_vl_text_proj.apply(objectives.init_weights)
            self.itc_vl_image_proj.apply(objectives.init_weights)

            # 初始化可学习的 logit 缩放参数
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.logit_vl_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self.siglip_bias = nn.Parameter(torch.zeros(1))

        ## retrieval task ft
        # 如果配置中 irtr 损失大于 0，则再次初始化图像和文本的投影头
        if config["loss_names"]["irtr"] > 0:
            self.itc_text_proj = heads.ITCHead(self.num_features)
            self.itc_image_proj = heads.ITCHead(self.num_features)
            # 应用初始化权重函数
            self.itc_text_proj.apply(objectives.init_weights)
            self.itc_image_proj.apply(objectives.init_weights)

            # 初始化可学习的 logit 缩放参数
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        
        if self.hparams.config["loss_names"]["vqa"] > 0:
            # 获取 VQAv2 任务的标签数量
            vqav2_label_size = self.hparams.config["vqav2_label_size"]
            self.vqa_classifier = nn.Sequential(
                nn.Linear(self.embed_dim, self.embed_dim * 2, bias=False),
                RMSNorm(self.embed_dim * 2, eps=1e-6),
                ACT2FN[self.activation],
                nn.Linear(self.embed_dim * 2, vqav2_label_size),
            )
            # 应用初始化权重函数
            self.vqa_classifier.apply(objectives.init_weights)

        if self.hparams.config["loss_names"]["nlvr2"] > 0:
            self.nlvr2_classifier = nn.Sequential(
                nn.Linear(self.embed_dim * 2, self.embed_dim * 2,bias=False),
                RMSNorm(self.embed_dim * 2,eps=1e-6),
                ACT2FN[self.activation],
                nn.Linear(self.embed_dim * 2, 2, bias=False),
            )

            self.nlvr2_classifier.apply(objectives.init_weights)

        # 设置模型的评估指标
        vlmo_utils.set_metrics(self)
        # 初始化当前任务列表
        self.current_tasks = list()

        if self.hparams.config["load_path"] != "" and self.hparams.config["test_only"]:
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))
            # 加载检查点
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu")

            state_dict = None
            
            for state_dict_key in ("state_dict", "module", "model"):
                if state_dict_key in ckpt:
                    rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                    state_dict = ckpt[state_dict_key]
                    break
            
            if state_dict is None:
                rank_zero_info("Read state dict from ckpt. ")
                state_dict = ckpt

            # 加载状态字典
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
            rank_zero_info("missing_keys: {}".format(missing_keys))
            rank_zero_info("unexpected_keys: {}".format(unexpected_keys))

    def log(self, name, value, *args, **kwargs):
        if "batch_size" not in kwargs:
            bs = getattr(self, "_current_batch_size", 1)
            if bs is not None:
                kwargs["batch_size"] = bs
        return super().log(name, value, *args, **kwargs)

    def load_pretrained_weight(self):
        print("开始加载预训练权重")
        if self.hparams.config["load_path"] != "" and not self.hparams.config["test_only"]:
            # 获取配置文件
            config = self.hparams.config
            # 从指定路径加载检查点，使用 CPU 进行加载
            ckpt = torch.load(self.hparams.config["load_path"], map_location="cpu", weights_only=False)
            # 记录加载的检查点路径
            rank_zero_info("Load ckpt from: {}".format(self.hparams.config["load_path"]))
    
            # 初始化状态字典
            state_dict = None
    
        # 遍历可能的状态字典键
        for state_dict_key in ("state_dict", "module", "model"):
            if state_dict_key in ckpt:
                # 记录读取状态字典的键
                rank_zero_info("Read state dict from ckpt[%s]. " % state_dict_key)
                # 获取状态字典
                state_dict = ckpt[state_dict_key]
                break
        # 如果状态字典为空，则直接使用检查点
        if state_dict is None:
            rank_zero_info("Read state dict from ckpt. ")
            state_dict = ckpt
    
        # 遍历状态字典中的键值对
        for key in state_dict:
            # 获取变量
            var = state_dict[key]
            # 记录变量的键和大小
            rank_zero_info("%s = %s" % (key, str(var.size())))
    
        # 记录损失名称
        rank_zero_info(config["loss_names"])
        # 如果文本掩码语言模型损失大于 0，则转换为文本预训练状态字典
        if config["loss_names"]["textmlm"] > 0:
            rank_zero_info("convert to textpt")
            state_dict = convert_to_textpt_ckpt(state_dict, self)
    
        # 获取最大文本长度
        max_text_len = config["max_text_len"]
        rank_zero_info("RPE processing skipped (using RoPE).")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        rank_zero_info(f"State dict loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

        # 记录缺失的权重类型
        missing_key_types = {}
        for key in missing_keys:
            key_type = key.split('.')[3] if len(key.split('.')) > 3 else key
            if key_type not in missing_key_types:
                missing_key_types[key_type] = 0
            missing_key_types[key_type] += 1
        
        rank_zero_info("Missing key types: {}".format(missing_key_types))
        
        # 初始化文本专家和门控网络
        rank_zero_info("初始化文本专家和门控网络...")
        with torch.no_grad():
            # 遍历所有transformer块
            for block in self.transformer.blocks:
                # 检查是否有门控网络需要初始化
                if hasattr(block, 'gate') and not hasattr(block, '_gate_initialized'):
                    # 初始化门控网络，使两个专家初始有相等选择概率
                    nn.init.zeros_(block.gate.bias)
                    # 使用小的随机值初始化权重，打破对称性
                    nn.init.normal_(block.gate.weight, std=0.01)
                    block._gate_initialized = True
                
                # 初始化文本专家 (expert_mlps[0])
                if hasattr(block, 'expert_mlps') and len(block.expert_mlps) > 0:
                    text_expert = block.expert_mlps[0]
                    for name, param in text_expert.named_parameters():
                        if 'weight' in name:
                            # 文本专家使用标准初始化
                            nn.init.trunc_normal_(param, std=0.02)
                        elif 'bias' in name:
                            nn.init.zeros_(param)
                
                # 初始化文本专家的norm层
                if hasattr(block, 'norm_layers') and len(block.norm_layers) > 0:
                    if hasattr(block.norm_layers[0], 'weight'):
                        nn.init.ones_(block.norm_layers[0].weight)
                
                # 初始化MoE统计变量
                if hasattr(block, 'expert_usage'):
                    block.expert_usage.zero_()
                    block.usage_count = 0
        
        # 验证专家初始化情况
        with torch.no_grad():
            for i, block in enumerate(self.transformer.blocks):
                if i == 0 and hasattr(block, 'expert_mlps') and len(block.expert_mlps) > 1:  # 仅检查第一个块作为示例
                    try:
                        # 比较文本和图像专家的参数
                        text_fc1 = block.expert_mlps[0].fc1.weight.data
                        image_fc1 = block.expert_mlps[1].fc1.weight.data
                        weight_diff = (text_fc1 - image_fc1).abs().mean().item()
                        
                        rank_zero_info(f"Block 0: 文本和图像专家权重差异: {weight_diff}")
                        
                        # 检验差异是否足够明显（表明成功区分了两个专家）
                        if weight_diff < 1e-6:
                            rank_zero_info("警告: 文本和图像专家权重几乎相同，可能未正确初始化!")
                        else:
                            rank_zero_info("成功: 文本和图像专家已正确区分初始化")
                    except Exception as e:
                        rank_zero_info(f"检查专家初始化时出错: {e}")
    

    def collect_moe_losses(self):
        """收集所有Transformer块的MoE辅助损失"""
        balance_loss = 0.0
        router_z_loss = 0.0
        num_blocks = 0
        
        for block in self.transformer.blocks:
            if hasattr(block, 'balance_loss'):
                balance_loss += block.balance_loss
                router_z_loss += block.router_z_loss
                num_blocks += 1
        
        if num_blocks > 0:
            balance_loss = balance_loss / num_blocks
            router_z_loss = router_z_loss / num_blocks
        
        # 从配置中获取损失权重
        balance_coef = self.hparams.config.get("moe_balance_loss_weight", 0.01)
        router_z_coef = self.hparams.config.get("moe_router_z_loss_weight", 0.001)
        
        return balance_coef * balance_loss + router_z_coef * router_z_loss
    
    def infer(
        self,
        batch,
        mask_text=False,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):


        # 确定图像数据在 batch 中的键名
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        # 根据 mask_text 决定是否使用掩码后的文本数据
        do_mlm = "_mlm" if mask_text else ""
        # 从 batch 中获取文本 ID
        text_ids = batch[f"text_ids{do_mlm}"]
        # 从 batch 中获取文本标签
        text_labels = batch[f"text_labels{do_mlm}"]
        # 从 batch 中获取文本掩码
        text_masks = batch[f"text_masks"]
        # 使用文本嵌入层将文本 ID 转换为嵌入向量
        text_embeds = self.text_embeddings(text_ids)

        # 从 batch 中获取图像数据
        img = batch[imgkey][0] if isinstance(batch[imgkey], list) else batch[imgkey]
        # 使用视觉嵌入层将图像转换为嵌入向量和掩码
        image_embeds_vis, image_masks_vis = self.transformer.visual_embed(img) # (B, N_img, C)
        image_masks_vis = image_masks_vis.long().to(image_embeds_vis.device)

        # 为文本嵌入添加文本类型的 token 嵌入
        # 为图像嵌入添加图像类型的 token 嵌入
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        image_embeds_vis = image_embeds_vis + self.token_type_embeddings(
            torch.full_like(image_masks_vis, image_token_type_idx)
        )

        # 在维度 1 上拼接文本和图像的嵌入向量
        co_embeds = torch.cat([text_embeds, image_embeds_vis], dim=1)
        # 在维度 1 上拼接文本和图像的掩码
        co_masks = torch.cat([text_masks, image_masks_vis], dim=1)

        # 初始化输入为拼接后的嵌入向量
        x = co_embeds
        B, N, C = x.shape
        N_text = text_embeds.shape[1]
        N_img = image_embeds_vis.shape[1]
        current_dev = x.device
        cos_1d, sin_1d, text_pos_ids = None, None, None

        if self.rope_1d is not None:
            text_pos_ids = torch.arange(N_text, device=current_dev).unsqueeze(0).expand(B, -1)
            # 确保 rope_1d 在正确设备上
            self.rope_1d.to(current_dev)
            # 调用 rope_1d 获取缓存
            cos_1d, sin_1d = self.rope_1d(x[:, :N_text, :], seq_len=N_text) # 可能需要虚拟输入
        
        image_pos_ids = None
        rope_2d_inst = None
        grid_hw_val = None

        if self.rope_2d is not None:
            image_pos_ids = torch.arange(N_img, device=current_dev).unsqueeze(0).expand(B, -1)
            # 确保 rope_2d 在正确设备上
            self.rope_2d.to(current_dev)
            rope_2d_inst = self.rope_2d
            grid_hw_val = (self.grid_height, self.grid_width)
        # 依次通过 Transformer 块进行前向传播
        for i, blk in enumerate(self.transformer.blocks):
            x = blk(x, mask=co_masks, 
                    N_text=N_text,cos_1d=cos_1d,sin_1d=sin_1d,
                    text_pos_ids=text_pos_ids,rope_2d_instance=rope_2d_inst,
                    image_pos_ids=image_pos_ids,grid_hw=grid_hw_val)

        # 使用归一化层对输出进行归一化
        x = self.transformer.norm(x)
        # 从输出中分离出文本特征
        # 从输出中分离出图像特征
        text_feats, image_feats = (
            x[:, :N_text],
            x[:, N_text:],
        )
        # 使用池化层对输出进行池化，得到 CLS 特征
        cls_feats = self.pooler(x)

        # 构建包含各种特征和输入数据的返回字典
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "image": img, # 可能需要返回原始图像
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
            "image_masks": image_masks_vis # 添加图像掩码到返回字典可能有用
        }

        return ret

    def infer_text(
        self,
        batch,
        mask_text=False,
    ):
        """
        对输入的文本数据进行推理，提取文本特征。

        参数:
            batch (dict): 包含输入数据的字典，如文本 ID、文本标签、文本掩码等。
            mask_text (bool): 是否对文本进行掩码处理，默认为 False。

        返回:
            dict: 包含提取的文本特征、CLS 特征等信息的字典。
        """
        # 根据 mask_text 决定是否使用掩码后的文本数据
        do_mlm = "_mlm" if mask_text else ""
        # 从 batch 中获取文本 ID
        text_ids = batch[f"text_ids{do_mlm}"]
        # 从 batch 中获取文本标签
        text_labels = batch[f"text_labels{do_mlm}"]
        # 从 batch 中获取文本掩码
        text_masks = batch[f"text_masks"]
        # 使用文本嵌入层将文本 ID 转换为嵌入向量
        text_embeds = self.text_embeddings(text_ids)
        # 为文本嵌入添加文本类型的 token 嵌入
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        # 合并后的嵌入即为文本嵌入
        co_embeds = text_embeds
        # 合并后的掩码即为文本掩码
        co_masks = text_masks
        
        # 初始化输入为合并后的嵌入向量
        x = co_embeds
        B, N_text, C = x.shape
        current_dev = x.device
        cos_1d, sin_1d, text_pos_ids = None, None, None
        if self.rope_1d is not None:
            text_pos_ids = torch.arange(N_text, device=current_dev).unsqueeze(0).expand(B, -1)
            self.rope_1d.to(current_dev)
            cos_1d, sin_1d = self.rope_1d(x, seq_len=N_text)
        # 依次通过 Transformer 块进行前向传播
        for i, blk in enumerate(self.transformer.blocks):
            # 经过当前 Transformer 块的计算
            x = blk(x, mask=co_masks, 
                        N_text=N_text, # 表明这是纯文本
                        cos_1d=cos_1d,
                        sin_1d=sin_1d,
                        text_pos_ids=text_pos_ids,
                        # 2D RoPE 参数设为 None
                        rope_2d_instance=None,
                        image_pos_ids=None,
                        grid_hw=None)
            # 记录当前层的隐藏状态
        final_hiddens = x
        final_hiddens = self.transformer.norm(final_hiddens)
        text_feats = final_hiddens
        image_feats = None
        # 使用 ITC 文本投影层处理 CLS 标记的特征
        cls_feats = self.itc_text_proj(final_hiddens[:, 0])
        # 对 CLS 特征进行归一化
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        # 构建包含各种特征和输入数据的返回字典
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats, # 这是最终文本路径的CLS投影
            "raw_cls_feats": final_hiddens[:, 0], # 最终文本路径的原始CLS
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_ft(
        self,
        batch,
        mask_text=False,
    ):
        """
        对输入的文本数据进行微调推理，提取文本特征。

        参数:
            batch (dict): 包含输入数据的字典，如文本 ID、文本标签、文本掩码等。
            mask_text (bool): 是否对文本进行掩码处理，默认为 False。

        返回:
            dict: 包含提取的文本特征、CLS 特征等信息的字典。
        """
        # 根据 mask_text 决定是否使用掩码后的文本数据
        do_mlm = "_mlm" if mask_text else ""
        # 从 batch 中获取文本 ID
        text_ids = batch[f"text_ids{do_mlm}"]
        # 从 batch 中获取文本标签
        text_labels = batch[f"text_labels{do_mlm}"]
        # 从 batch 中获取文本掩码
        text_masks = batch[f"text_masks"]
        # 使用文本嵌入层将文本 ID 转换为嵌入向量
        text_embeds = self.text_embeddings(text_ids)
        # 为文本嵌入添加文本类型的 token 嵌入
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        # 合并后的嵌入即为文本嵌入
        co_embeds = text_embeds
        # 合并后的掩码即为文本掩码
        co_masks = text_masks

        # 初始化输入为合并后的嵌入向量
        x = co_embeds
        B, N_text, C = x.shape
        current_dev = x.device
        cos_1d, sin_1d, text_pos_ids = None, None, None
        if self.rope_1d is not None:
            text_pos_ids = torch.arange(N_text, device=current_dev).unsqueeze(0).expand(B, -1)
            self.rope_1d.to(current_dev)
            cos_1d, sin_1d = self.rope_1d(x, seq_len=N_text)
        # 依次通过 Transformer 块进行前向传播
        for i, blk in enumerate(self.transformer.blocks):
            # 经过当前 Transformer 块的计算
            x = blk(x, mask=co_masks, 
                        N_text=N_text, # 表明这是纯文本
                        cos_1d=cos_1d,
                        sin_1d=sin_1d,
                        text_pos_ids=text_pos_ids,
                        # 2D RoPE 参数设为 None
                        rope_2d_instance=None,
                        image_pos_ids=None,
                        grid_hw=None)

        # 获取最后一层的隐藏状态
        final_hiddens = x

        # 使用归一化层对最后一层的隐藏状态进行归一化
        final_hiddens = self.transformer.norm(final_hiddens)
        # 文本特征即为归一化后的最后一层隐藏状态
        # 图像特征为空，因为只处理文本
        text_feats = final_hiddens
        image_feats = None

        # 使用 ITC 文本投影层处理 CLS 标记的特征
        cls_feats = self.itc_text_proj(final_hiddens[:, 0])
        # 对 CLS 特征进行归一化
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        # 构建包含各种特征和输入数据的返回字典
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": final_hiddens[:, 0],
            "image_masks": None,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def infer_text_mlm(
        self,
        batch,
        mask_text=False,
    ):
        """
        对输入的文本数据进行掩码语言模型（MLM）推理，提取文本特征。

        参数:
            batch (dict): 包含输入数据的字典，如文本 ID、文本标签、文本掩码等。
            mask_text (bool): 是否对文本进行掩码处理，默认为 False。

        返回:
            dict: 包含提取的文本特征、图像特征（这里为空）、CLS 特征（这里为空）等信息的字典。
        """
        # 根据 mask_text 决定是否使用掩码后的文本数据
        do_mlm = "_mlm" if mask_text else ""
        # 从 batch 中获取文本 ID
        text_ids = batch[f"text_ids{do_mlm}"]
        # 从 batch 中获取文本标签
        text_labels = batch[f"text_labels{do_mlm}"]
        # 从 batch 中获取文本掩码
        text_masks = batch[f"text_masks"]
        # 使用文本嵌入层将文本 ID 转换为嵌入向量
        text_embeds = self.text_embeddings(text_ids)
        # 为文本嵌入添加文本类型的 token 嵌入
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        # 合并后的嵌入即为文本嵌入
        co_embeds = text_embeds
        # 合并后的掩码即为文本掩码
        co_masks = text_masks

        # 初始化输入为合并后的嵌入向量
        x = co_embeds
        B, N_text, C = x.shape
        current_dev = x.device
        # 用于存储每一层的隐藏状态
        cos_1d, sin_1d, text_pos_ids = None, None, None
        if self.rope_1d is not None:
            text_pos_ids = torch.arange(N_text, device=current_dev).unsqueeze(0).expand(B, -1)
            self.rope_1d.to(current_dev)
            cos_1d, sin_1d = self.rope_1d(x, seq_len=N_text)

        # 依次通过 Transformer 块进行前向传播
        for i, blk in enumerate(self.transformer.blocks):
            # 经过当前 Transformer 块的计算
            x = blk(x, mask=co_masks, 
                        N_text=N_text, # 表明这是纯文本
                        cos_1d=cos_1d,
                        sin_1d=sin_1d,
                        text_pos_ids=text_pos_ids,
                        # 2D RoPE 参数设为 None
                        rope_2d_instance=None,
                        image_pos_ids=None,
                        grid_hw=None)
            # 记录当前层的隐藏状态


        # 获取最后一层的隐藏状态
        final_hiddens = x

        # 使用归一化层对最后一层的隐藏状态进行归一化
        final_hiddens = self.transformer.norm(final_hiddens)
        # 文本特征即为归一化后的最后一层隐藏状态
        # 图像特征为空，因为只处理文本
        text_feats = final_hiddens
        image_feats = None
        
        # 构建包含各种特征和输入数据的返回字典
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            # 这里 CLS 特征设为 None
            "cls_feats": None,  # MLM 通常不直接使用投影后的CLS
            # 原始的 CLS 特征
            "raw_cls_feats": final_hiddens[:, 0], # 返回归一化前的CLS可能更合适，或归一化后的
            # 图像掩码设为 None
            "image_masks": None,
            # 文本标签
            "text_labels": text_labels,
            # 文本 ID
            "text_ids": text_ids,
            # 文本掩码
            "text_masks": text_masks,
        }

        return ret

    def infer_image(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        """
        对输入的图像数据进行推理，提取图像特征。

        参数:
            batch (dict): 包含输入数据的字典，如文本 ID、文本标签、文本掩码、图像等。
            mask_image (bool): 是否对图像进行掩码处理，默认为 False。
            image_token_type_idx (int): 图像 token 类型的索引，默认为 1。
            image_embeds (torch.Tensor): 可选的图像嵌入，默认为 None。
            image_masks (torch.Tensor): 可选的图像掩码，默认为 None。

        返回:
            dict: 包含提取的图像特征、CLS 特征等信息的字典。
        """
        # 确定图像数据在 batch 中的键名
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        # 从 batch 中获取图像数据
        img = batch[imgkey][0] if isinstance(batch[imgkey], list) else batch[imgkey]
        # 使用视觉嵌入层将图像转换为嵌入向量和掩码
        image_embeds_vis, image_masks_vis = self.transformer.visual_embed(img) # (B, N_img, C)

        # 将图像掩码转换为 long 类型
        image_masks_vis = image_masks_vis.long().to(image_embeds_vis.device)
        # 为图像嵌入添加图像类型的 token 嵌入
        image_embeds_vis = image_embeds_vis + self.token_type_embeddings(
                    torch.full_like(image_masks_vis, image_token_type_idx)
                )

        # 合并后的嵌入即为图像嵌入
        co_embeds = image_embeds_vis
        # 合并后的掩码即为图像掩码
        co_masks = image_masks_vis
        # 初始化输入为合并后的嵌入向量
        x = co_embeds
        B, N_img, C = x.shape
        current_dev = x.device
        cos_1d, sin_1d, image_pos_ids = None, None, None
        if self.rope_2d is not None:
            image_pos_ids = torch.arange(N_img, device=current_dev).unsqueeze(0).expand(B, -1)
            self.rope_2d.to(current_dev)
            rope_2d_inst = self.rope_2d
            grid_hw_val = (self.grid_height, self.grid_width)
        # 依次通过 Transformer 块进行前向传播
        for i, blk in enumerate(self.transformer.blocks):
            # 经过当前 Transformer 块的计算
            x = blk(x, mask=co_masks,  
                            N_text=0, # 或 None，表明无文本
                            # 1D RoPE 参数设为 None
                            cos_1d=None,
                            sin_1d=None,
                            text_pos_ids=None,
                            # 传递 2D RoPE 参数
                            rope_2d_instance=rope_2d_inst,
                            image_pos_ids=image_pos_ids,
                            grid_hw=grid_hw_val)
        final_hiddens = x
        final_hiddens = self.transformer.norm(final_hiddens)
        image_feats = final_hiddens
        text_feats = None

        # 使用 ITC 图像投影层处理 CLS 标记的特征
        cls_feats = self.itc_image_proj(final_hiddens[:, 0])
        # 对 CLS 特征进行归一化
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        # 构建包含各种特征和输入数据的返回字典
        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "image": img, # 可能需要返回原始图像
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def infer_image_ft(
        self,
        batch,
        mask_image=False,
        image_token_type_idx=1,
        image_embeds=None,
        image_masks=None,
    ):
        """
        对输入的图像数据进行微调推理，提取图像特征。

        参数:
            batch (dict): 包含输入数据的字典，如文本 ID、文本标签、文本掩码、图像等。
            mask_image (bool): 是否对图像进行掩码处理，默认为 False。
            image_token_type_idx (int): 图像 token 类型的索引，默认为 1。
            image_embeds (torch.Tensor): 可选的图像嵌入，默认为 None。
            image_masks (torch.Tensor): 可选的图像掩码，默认为 None。

        返回:
            dict: 包含提取的图像特征、CLS 特征等信息的字典。
        """
        # 确定图像数据在 batch 中的键名
        if f"image_{image_token_type_idx - 1}" in batch:
            imgkey = f"image_{image_token_type_idx - 1}"
        else:
            imgkey = "image"

        # 从 batch 中获取图像数据
        img = batch[imgkey][0] if isinstance(batch[imgkey], list) else batch[imgkey]
        # 使用视觉嵌入层将图像转换为嵌入向量和掩码
        image_embeds_vis, image_masks_vis = self.transformer.visual_embed(img)

        # 将图像掩码转换为 long 类型并移动到与图像相同的设备上
        image_masks_vis = image_masks_vis.long().to(image_embeds_vis.device)
        # 为图像嵌入添加图像类型的 token 嵌入
        image_embeds_vis = image_embeds_vis + self.token_type_embeddings(
            torch.full_like(image_masks_vis, image_token_type_idx)
        )
        
        co_embeds = image_embeds_vis
        # 合并后的掩码即为图像掩码
        co_masks = image_masks_vis

        # 初始化输入为合并后的嵌入向量
        x = co_embeds
        B, N_img, C = x.shape
        current_dev = x.device
        cos_1d, sin_1d, image_pos_ids = None, None, None
        if self.rope_2d is not None:
            image_pos_ids = torch.arange(N_img, device=current_dev).unsqueeze(0).expand(B, -1)
            self.rope_2d.to(current_dev)
            rope_2d_inst = self.rope_2d
            grid_hw_val = (self.grid_height, self.grid_width)

        # 依次通过 Transformer 块进行前向传播
        for i, blk in enumerate(self.transformer.blocks):
            # 经过当前 Transformer 块的计算
            x = blk(x, mask=co_masks, 
                    N_text=0, # 或 None，表明无文本
                    cos_1d=None,
                    sin_1d=None,
                    text_pos_ids=None,
                    # 传递 2D RoPE 参数
                    rope_2d_instance=rope_2d_inst,
                    image_pos_ids=image_pos_ids,
                    grid_hw=grid_hw_val)    
            # 记录当前层的隐藏状态

        # 获取最后一层的隐藏状态
        final_hiddens = x

        # 使用归一化层对最后一层的隐藏状态进行归一化
        final_hiddens = self.transformer.norm(final_hiddens)
        # 文本特征为空，因为只处理图像
        # 图像特征即为归一化后的最后一层隐藏状态
        text_feats, image_feats = (
            None,
            final_hiddens,
        )

        # 使用 ITC 图像投影层处理 CLS 标记的特征
        cls_feats = self.itc_image_proj(final_hiddens[:, 0])
        # 对 CLS 特征进行归一化
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        ret = {
            "text_feats": text_feats,
            "image_feats": image_feats,
            "cls_feats": cls_feats,
            "image": img,
            # "cls_vlffn_feats": None,
            "raw_cls_feats": x[:, 0],
            "image_masks": image_masks,
            "text_labels": None,
            "text_ids": None,
            "text_masks": None,
        }

        return ret

    def forward(self, batch):

        # 初始化返回结果的字典
        ret = dict()
        # 如果当前任务列表为空，则调用 infer 方法进行推理
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Masked Language Modeling
        # 如果当前任务包含 "mlm"，则调用 compute_mlm 方法计算掩码语言模型损失
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        # Textonly Masked Language Modeling
        # 如果当前任务包含 "textmlm"，则调用 compute_textonly_mlm 方法计算仅文本的掩码语言模型损失
        if "textmlm" in self.current_tasks:
            ret.update(objectives.compute_textonly_mlm(self, batch))

        # Contrastive loss for pretraining
        # 如果当前任务包含 "itc"，则调用 compute_itc 方法计算预训练的对比损失
        if "itc" in self.current_tasks:
            if self.hparams.config.get("use_siglip_loss", False):
                ret.update(objectives.compute_itc_siglip(self, batch))
            else:
                ret.update(objectives.compute_itc(self, batch))

        # Contrastive loss for finetuning
        # 如果当前任务包含 "irtr"，则调用 compute_irtr 方法计算微调的对比损失
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch))

        # Image Text Matching with global hard negative, must use with itc
        # 如果当前任务包含 "itm"，则调用 compute_itm_hardneg 方法计算带有全局硬负样本的图像文本匹配损失
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm_hardneg(self, batch, ret["itc_i2t_logits"], ret["itc_t2i_logits"]))

        # Visual Question Answering
        # 如果当前任务包含 "vqa"，则调用 compute_vqa 方法计算视觉问答损失
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))

        # Natural Language for Visual Reasoning 2
        # 如果当前任务包含 "nlvr2"，则调用 compute_nlvr2 方法计算自然语言视觉推理 2 的损失
        if "nlvr2" in self.current_tasks:
            ret.update(objectives.compute_nlvr2(self, batch))

        return ret
    

    def training_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        # 计算主任务损失
        task_loss = sum([v for k, v in output.items() if "loss" in k])
        moe_loss = self.collect_moe_losses()
        total_loss = task_loss + moe_loss
        batch_size = batch["text_ids"].size(0) if "text_ids" in batch else batch["image"].size(0)
        self.log("task_loss", task_loss, prog_bar=False, logger=True, batch_size=batch_size)
        self.log("moe_loss", moe_loss, prog_bar=False, logger=True, batch_size=batch_size)  # self.log("train_loss", total_loss, prog_bar=True, logger=True,batch_size=batch_size)
        return total_loss

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()
                
    def on_after_backward(self):
        try:
            """在反向传播后记录梯度范数"""
            batch_size = getattr(self, "_current_batch_size", 1)
            # 1. 计算全局梯度范数
            total_norm = 0.0
            params_with_grad = 0
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                    params_with_grad += 1
            if params_with_grad > 0:
                total_norm = total_norm ** 0.5
            # 记录全局梯度范数
            self.log("grad/global_norm", total_norm, prog_bar=False, logger=True,batch_size=batch_size)
        
            # 2. 可选: 记录各层梯度范数
            with torch.no_grad():
                key_blocks = [0, len(self.transformer.blocks)//2, len(self.transformer.blocks)-1]
                for i in key_blocks: 
                    if i < len(self.transformer.blocks):# 只记录关键块
                        block = self.transformer.blocks[i]
                        block_params_with_grad = 0
                        block_norm_sum = 0.0

                        for name, p in block.named_parameters():
                            if p.requires_grad and p.grad is not None:
                                block_norm_sum += p.grad.detach().norm(2).item() ** 2
                                block_params_with_grad += 1                    
                        if block_params_with_grad > 0:
                            block_norm = block_norm_sum ** 0.5
                            self.log(f"grad/block_{i}_norm", block_norm, 
                                prog_bar=False, logger=True, batch_size=batch_size)
                    
        except Exception as e:
        # 捕获并记录任何异常，防止中断训练
            print(f"Error in gradient norm logging: {e}")            
        
                
    
    def validation_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        return output

    def on_validation_epoch_end(self):
        vlmo_utils.epoch_wrapup(self)
        the_metric = 0
        computed_metrics = self.trainer.callback_metrics # 获取已计算的 epoch 指标

        # 累加来自召回率的指标 (如果计算了)
        if self.hparams.config["get_recall_metric"]:
             # 从日志中获取 val_avg (确保名称匹配 epoch_wrapup 中的 log 调用)
             val_avg = computed_metrics.get("recalls/val_avg", torch.tensor(0.0))
             the_metric += val_avg.item() # 使用 .item() 获取标量值

        # 累加其他任务的指标 (从 self.log 记录的结果中获取)
        phase = "val"
        for loss_name, v in self.hparams.config["loss_names"].items():
            if v < 1:
                continue

            value = 0
            if loss_name == "vqa":
                # 从日志获取 VQA 分数 (确保名称匹配 objectives.py 中的 log 调用)
                score_key = f"vqa/{phase}/score" # 注意：不再是 score_epoch
                value = computed_metrics.get(score_key, torch.tensor(0.0))
            elif loss_name == "nlvr2":
                # 从日志获取 NLVR2 dev 准确率 (如果这是主要指标)
                # 注意：objectives.py 中记录的是 val/accuracy，dev/accuracy 是单独记录的
                # 确认哪个指标用于 the_metric
                acc_key = f"nlvr2/{phase}/accuracy" # 主要验证指标
                # 或者 acc_key = f"nlvr2/dev/accuracy" # 如果使用 dev 指标
                value = computed_metrics.get(acc_key, torch.tensor(0.0))
            elif loss_name == "irtr":
                # 从日志获取 IRTR 准确率
                i2t_key = f"irtr/{phase}/i2t_accuracy"
                t2i_key = f"irtr/{phase}/t2i_accuracy"
                value_i2t = computed_metrics.get(i2t_key, torch.tensor(0.0))
                value_t2i = computed_metrics.get(t2i_key, torch.tensor(0.0))
                value = value_i2t + value_t2i
            elif loss_name == "itm":
                # 从日志获取 ITM 准确率
                acc_key = f"itm/{phase}/accuracy"
                value = computed_metrics.get(acc_key, torch.tensor(0.0))
            elif loss_name == "itc":
                 # 从日志获取 ITC 准确率
                i2t_key = f"itc/{phase}/i2t_accuracy"
                t2i_key = f"itc/{phase}/t2i_accuracy"
                value_i2t = computed_metrics.get(i2t_key, torch.tensor(0.0))
                value_t2i = computed_metrics.get(t2i_key, torch.tensor(0.0))
                # 考虑是否包含 vl 指标
                vl_i2t_key = f"itc/{phase}/vl_i2t_accuracy"
                vl_t2i_key = f"itc/{phase}/vl_t2i_accuracy"
                value_vl_i2t = computed_metrics.get(vl_i2t_key, torch.tensor(0.0))
                value_vl_t2i = computed_metrics.get(vl_t2i_key, torch.tensor(0.0))
                value = value_i2t + value_t2i # + value_vl_i2t + value_vl_t2i # 根据需要添加
            else: # MLM, TextMLM 等
                # 从日志获取准确率
                acc_key = f"{loss_name}/{phase}/accuracy"
                value = computed_metrics.get(acc_key, torch.tensor(0.0))

            the_metric += value.item() # 累加标量值

        # 记录最终的 the_metric
        self.log(f"{phase}/the_metric", the_metric, rank_zero_only=True)
        rank_zero_info(f"Epoch {self.current_epoch} Validation 'the_metric': {the_metric}")

        torch.cuda.empty_cache()


    def test_step(self, batch, batch_idx):
        vlmo_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))
        self.test_step_outputs.append(ret) 
        return ret

    def on_test_epoch_end(self):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(self.test_step_outputs, model_name, self.hparams.config["log_dir"])
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return vlmo_utils.set_schedule(self)
