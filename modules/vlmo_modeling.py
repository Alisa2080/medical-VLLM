import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
from functools import partial
import warnings
import os
from dataclasses import dataclass, field
import numpy as np

from transformers.modeling_utils import PreTrainedModel
# --- 添加 GenerationMixin 导入 ---
from transformers.generation import GenerationMixin
from transformers.utils import ModelOutput, logging
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPast, SequenceClassifierOutput, BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, BaseModelOutputWithPooling
from pytorch_lightning.utilities import rank_zero_info
import torch.distributed as dist
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

# 导入本地模块
from .vlmo_config import VLMoEncoderDecoderConfig
from .Encoder_Decoder import VLMoEncoderDecoder
# 尝试导入 RMSNorm
try:
    from modules.RMSNorm import RMSNorm
except ImportError:
    RMSNorm = nn.LayerNorm
    warnings.warn("RMSNorm not found, using nn.LayerNorm as fallback for config reconstruction.")

logger = logging.get_logger(__name__)

# 定义输出类以匹配 Hugging Face 风格
@dataclass
class VLMoSeq2SeqLMOutput(ModelOutput):
    """
    VLMo 序列到序列模型的输出类，包含损失、logits、MoE损失和KV缓存。
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None # 注意力权重（如果需要）
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None # 交叉注意力权重（如果需要）
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None # 编码器隐藏状态（如果需要）
    encoder_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None # 编码器注意力权重（如果需要）
    moe_balance_loss: Optional[torch.FloatTensor] = None # MoE 负载均衡损失
    moe_router_z_loss: Optional[torch.FloatTensor] = None # MoE 路由器 Z 损失

class VLMoEncoderDecoderPreTrainedModel(PreTrainedModel):
    """
    VLMo 预训练模型的基类，处理权重初始化和加载/保存。
    """
    config_class = VLMoEncoderDecoderConfig
    base_model_prefix = "model" # 指向 VLMoEncoderDecoder 实例
    supports_gradient_checkpointing = True # 假设支持梯度检查点
    _no_split_modules = ["DecoderBlock_GQA_ROPE"] # 防止在 FSDP 中分割这些模块

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.2)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.2)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
             if hasattr(module, 'bias') and module.bias is not None:
                 module.bias.data.zero_()
             module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, VLMoEncoderDecoder):
            module.gradient_checkpointing = value


class VLMoEncoderDecoderForConditionalGeneration(VLMoEncoderDecoderPreTrainedModel, GenerationMixin): # <-- 添加 GenerationMixin
    """
    用于条件生成的 VLMo 编码器-解码器模型。(恢复为原始生成功能)
    """
    # --- 确保 main_input_name 在类级别定义 ---
    main_input_name = "decoder_input_ids" # 或者根据您的主要输入调整

    def __init__(self, config: VLMoEncoderDecoderConfig, freeze_encoder: bool = False):
        super().__init__(config)
        self.model = VLMoEncoderDecoder(
            encoder_config=config.encoder.to_dict(),
            decoder_config=config.decoder.to_dict(),
            encoder_checkpoint_path=config.encoder_checkpoint_path,
            freeze_encoder=freeze_encoder,
            max_seq_len=config.max_seq_len, # 512
            rope_base=config.rope_base,
            image_size=config.image_size, # 384
            patch_size=config.encoder.patch_size, # 16
        )
        # decoder 已自带 tok_embeddings 和 lm_head，此处仅关闭自动权重同名绑定
        self.config.tie_word_embeddings = False
        self.post_init()
        # ---- 断开可能由旧 ckpt 带来的权重共享 ----
        if id(self.model.decoder.lm_head.weight) == id(self.model.decoder.tok_embeddings.weight):
            self.model.decoder.lm_head.weight = nn.Parameter(
            self.model.decoder.lm_head.weight.detach().clone()
            )
            logger.info("lm_head 与 tok_embeddings 已解除共享")

    def get_input_embeddings(self):
        return self.model.decoder.tok_embeddings

    def set_input_embeddings(self, value):
        self.model.decoder.tok_embeddings = value

    def get_output_embeddings(self):
        return self.model.decoder.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.decoder.lm_head = new_embeddings
  

    def get_encoder(self):
        return self.model.encoder

    def get_decoder(self):
        return self.model.decoder

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, VLMoSeq2SeqLMOutput]:
        """
        pixel_values: Optional[torch.Tensor]：输入的图像张量（如视觉特征、图片 patch），用于视觉编码器。
        text_ids: Optional[torch.LongTensor]：输入的文本 ID 张量，用于文本编码器。
        attention_mask: Optional[torch.Tensor]：文本输入的 attention mask，1 表示有效 token，0 表示 padding。
        decoder_input_ids: Optional[torch.LongTensor]：解码器的输入 ID 张量。
        decoder_attention_mask: Optional[torch.BoolTensor]：解码器输入的 attention mask，True/1 表示有效 token，False/0 表示 padding。

        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if hasattr(self.config, 'use_cache') else False)
        _image = image if image is not None else pixel_values

        outputs = self.model(
            image=_image,
            text_ids=text_ids,
            text_masks=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_mask=decoder_attention_mask.bool() if decoder_attention_mask is not None else None,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_hidden_states=output_hidden_states,
        )

        if use_cache:
            logits, decoder_hidden_states, decoder_moe_losses, current_past_key_values = outputs
        else:
            logits, decoder_hidden_states, decoder_moe_losses = outputs
            current_past_key_values = None

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.decoder.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        moe_balance_loss, moe_router_z_loss = self.model.get_moe_losses() or (None, None)
        if loss is not None:
             if moe_balance_loss is not None:
                 safe_balance_loss = torch.clamp(moe_balance_loss, -10.0, 10.0)
                 loss = loss + self.config.moe_balance_loss_weight * safe_balance_loss
             if moe_router_z_loss is not None:
                 safe_router_z_loss = torch.clamp(moe_router_z_loss, -10.0, 10.0)
                 loss = loss + self.config.moe_router_z_loss_weight * safe_router_z_loss

        if not return_dict:
            output_list = [loss] if loss is not None else []
            output_list.extend([logits, current_past_key_values, decoder_hidden_states])
            return tuple(output for output in output_list if output is not None)

        _encoder_last_hidden_state = None
        if encoder_outputs is not None and isinstance(encoder_outputs, tuple) and len(encoder_outputs) > 0:
            _encoder_last_hidden_state = encoder_outputs[0]
        elif hasattr(self.model, '_last_encoder_context'):
            _encoder_last_hidden_state = self.model._last_encoder_context

        return VLMoSeq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=current_past_key_values,
            decoder_hidden_states=decoder_hidden_states,
            encoder_last_hidden_state=_encoder_last_hidden_state,
            moe_balance_loss=moe_balance_loss,
            moe_router_z_loss=moe_router_z_loss,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        _encoder_outputs = None
        if encoder_outputs is not None:
            if isinstance(encoder_outputs, tuple) and len(encoder_outputs) >= 2 and \
               isinstance(encoder_outputs[0], torch.Tensor) and isinstance(encoder_outputs[1], torch.Tensor):
                _encoder_outputs = (encoder_outputs[0], encoder_outputs[1])
            elif isinstance(encoder_outputs, (BaseModelOutputWithPastAndCrossAttentions, BaseModelOutput)):
                 _context_mask = kwargs.get("encoder_attention_mask", None)
                 if _context_mask is not None:
                     _encoder_outputs = (encoder_outputs.last_hidden_state, _context_mask)
                 else:
                     _encoder_outputs = (encoder_outputs.last_hidden_state, 
                        torch.ones(encoder_outputs.last_hidden_state.shape[0], 
                               encoder_outputs.last_hidden_state.shape[1], 
                               device=encoder_outputs.last_hidden_state.device))
            else:
                rank_zero_info(f"Warning: Unexpected type for encoder_outputs in prepare_inputs_for_generation: {type(encoder_outputs)}")
                

        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        model_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": _encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_attention_mask": decoder_attention_mask,
            "use_cache": use_cache,
            "pixel_values": kwargs.get("pixel_values", None),
            "image": kwargs.get("image", None),
            "text_ids": kwargs.get("text_ids", None),
            "attention_mask": attention_mask,
        }
        model_inputs = {k: v for k, v in model_inputs.items() if v is not None}

        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]],
        beam_idx: torch.Tensor
    ) -> List[Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]]:
        reordered_past = []
        for layer_past in past_key_values:
            if layer_past is None or len(layer_past) != 2:
                reordered_past.append(None)
                continue

            self_attn_cache, cross_attn_cache = layer_past

            reordered_self_attn_cache = None
            if self_attn_cache is not None and len(self_attn_cache) == 2:
                # 正确处理自注意力缓存
                reordered_self_attn_cache = (
                self_attn_cache[0].index_select(0, beam_idx),
                self_attn_cache[1].index_select(0, beam_idx)
            )
            reordered_cross_attn_cache = None
            if cross_attn_cache is not None and len(cross_attn_cache) == 2:
                reordered_cross_attn_cache = (
                cross_attn_cache[0].index_select(0, beam_idx),
                cross_attn_cache[1].index_select(0, beam_idx)
                    )
            reordered_past.append((reordered_self_attn_cache, reordered_cross_attn_cache))

        return reordered_past

    def get_encoder_outputs(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
         return self.model.get_encoder_context(pixel_values)


@dataclass
class VLMoForNLVR2Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    moe_balance_loss: Optional[torch.FloatTensor] = None
    moe_router_z_loss: Optional[torch.FloatTensor] = None

class VLMoForNLVR2(VLMoEncoderDecoderPreTrainedModel):
    """VLMo 模型，带有用于 NLVR2 的二分类头。"""
    def __init__(self, config: VLMoEncoderDecoderConfig, freeze_encoder: bool = False):
        super().__init__(config)
        self.model = VLMoEncoderDecoder(
            encoder_config=config.encoder.to_dict(),
            decoder_config=config.decoder.to_dict(),
            encoder_checkpoint_path=config.encoder_checkpoint_path,
            freeze_encoder=freeze_encoder,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base
        )
        fused_dim = config.encoder.embed_dim * 2
        self.nlvr_classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.GELU(),
            nn.Linear(fused_dim * 2, 2),
        )
        self.post_init()

    def forward(
        self,
        pixel_values_1: torch.Tensor,
        pixel_values_2: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, VLMoForNLVR2Output]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeds_1, _ = self.model.get_encoder_context(pixel_values_1)
        image_embeds_2, _ = self.model.get_encoder_context(pixel_values_2)
        image_cls_1 = image_embeds_1[:, 0]
        image_cls_2 = image_embeds_2[:, 0]

        fused_features = torch.cat([image_cls_1, image_cls_2], dim=-1)

        logits = self.nlvr_classifier(fused_features)

        loss = None

        moe_balance_loss = None
        moe_router_z_loss = None

        if not return_dict:
            output = (logits,)
            if moe_balance_loss is not None:
                output += (moe_balance_loss, moe_router_z_loss)
            return output

        return VLMoForNLVR2Output(
            loss=loss,
            logits=logits,
            moe_balance_loss=moe_balance_loss,
            moe_router_z_loss=moe_router_z_loss,
        )

@dataclass
class VLMoForRetrievalOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    image_embeds: Optional[torch.FloatTensor] = None
    text_embeds: Optional[torch.FloatTensor] = None
    logit_scale: Optional[torch.FloatTensor] = None
    moe_balance_loss: Optional[torch.FloatTensor] = None
    moe_router_z_loss: Optional[torch.FloatTensor] = None

class VLMoForRetrieval(VLMoEncoderDecoderPreTrainedModel):
    """VLMo 模型，用于图像-文本检索 (输出图像和文本嵌入)。"""
    def __init__(self, config: VLMoEncoderDecoderConfig, freeze_encoder: bool = False):
        super().__init__(config)
        self.model = VLMoEncoderDecoder(
            encoder_config=config.encoder.to_dict(),
            decoder_config=config.decoder.to_dict(),
            encoder_checkpoint_path=config.encoder_checkpoint_path,
            freeze_encoder=freeze_encoder,
            max_seq_len=config.max_seq_len,
            rope_base=config.rope_base
        )
        embed_dim = config.encoder.embed_dim
        proj_dim = config.get("projection_dim", embed_dim)
        self.image_projection = nn.Linear(embed_dim, proj_dim, bias=False)
        self.text_projection = nn.Linear(embed_dim, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.post_init()

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        return_loss: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Union[Tuple, VLMoForRetrievalOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_embeds_full, _ = self.model.get_encoder_context(pixel_values)
        image_cls_embed = image_embeds_full[:, 0]
        image_embeds_proj = self.image_projection(image_cls_embed)
        image_embeds_norm = F.normalize(image_embeds_proj, p=2, dim=-1)

        text_cls_embed = image_cls_embed
        text_embeds_proj = self.text_projection(text_cls_embed)
        text_embeds_norm = F.normalize(text_embeds_proj, p=2, dim=-1)

        loss = None

        moe_balance_loss = None
        moe_router_z_loss = None

        logit_scale = self.logit_scale.exp()

        if not return_dict:
            output = (image_embeds_norm, text_embeds_norm, logit_scale)
            if moe_balance_loss is not None:
                output += (moe_balance_loss, moe_router_z_loss)
            return output

        return VLMoForRetrievalOutput(
            loss=loss,
            image_embeds=image_embeds_norm,
            text_embeds=text_embeds_norm,
            logit_scale=logit_scale,
            moe_balance_loss=moe_balance_loss,
            moe_router_z_loss=moe_router_z_loss,
        )

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def contrastive_loss_ita(logits: torch.Tensor) -> torch.Tensor:
    return (contrastive_loss(logits) + contrastive_loss(logits.t())) / 2

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
