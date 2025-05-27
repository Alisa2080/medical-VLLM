import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
import os
from functools import partial
import warnings

# 从项目模块导入
from .Encoder import MultiWayTransformer, vlmo_base_patch16, vlmo_large_patch16, vlmo_base_plus_patch16 # 假设这些函数返回 MultiWayTransformer 实例
from .Decoder import TransformerDecoder
from .rope import Rope2DPosEmb, DeepseekV3RotaryEmbedding, apply_rotary_pos_emb, apply_rope,SimpleQwen3RotaryEmbedding # Ensure apply_rope is imported
from .RMSNorm import RMSNorm
from timm.models import create_model # 用于创建 Encoder

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.cache_utils import Cache, DynamicCache

class VLMoEncoderDecoder(nn.Module):
    """
    结合 VLMo Encoder (MultiWayTransformer) 和 TransformerDecoder 的模型。
    用于序列到序列的任务，如图像描述生成。
    """
    def __init__(
        self,
        encoder_config: Dict[str, Any],
        decoder_config: Dict[str, Any],
        encoder_checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = True,
        max_seq_len: int = 512, 
        rope_base: int = 10000,
        image_size: int = 384,
        patch_size: int = 16,
        **kwargs, # 其他参数
    ):

        super().__init__()
        self.freeze_encoder = freeze_encoder # Store freeze status
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
        # --- 1. 初始化 Encoder ---
        encoder_model_name = self.encoder_config.get("model_name", "vlmo_base_patch16") 
        encoder_args = {k: v for k, v in self.encoder_config.items() if k != 'model_name'}
        self.encoder: MultiWayTransformer = create_model(encoder_model_name, **encoder_args)

        # 加载预训练权重
        if encoder_checkpoint_path and os.path.exists(encoder_checkpoint_path):
            print(f"Loading encoder weights from: {encoder_checkpoint_path}")
            try:
                checkpoint = torch.load(encoder_checkpoint_path, map_location='cpu')
                # --- 修改权重加载逻辑 ---
                full_state_dict = None
                # Handle nested state dicts (common in PL checkpoints)
                for key in ('state_dict', 'module', 'model'): # state_dict
                    if key in checkpoint:
                        full_state_dict = checkpoint[key]
                        print(f"Extracted state_dict using key: '{key}'")
                        break
                if full_state_dict is None:
                    full_state_dict = checkpoint # Assume checkpoint is the state_dict itself

                # Filter for encoder keys (starting with 'transformer.') and remove prefix
                encoder_state_dict = {}
                prefix = 'transformer.'
                prefix_len = len(prefix)
                found_encoder_keys = False
                for k, v in full_state_dict.items():
                    if k.startswith(prefix):
                        encoder_state_dict[k[prefix_len:]] = v
                        found_encoder_keys = True

                if not found_encoder_keys:
                     print(f"Warning: No keys starting with '{prefix}' found in the checkpoint. Attempting to load the full state_dict directly into the encoder (might cause mismatches).")
                     encoder_state_dict = full_state_dict # Fallback: try loading everything (original behavior)

                # Load the filtered and prefixed state dict
                missing_keys, unexpected_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)

                if missing_keys:
                    print(f"Encoder missing keys after filtering/prefixing: {missing_keys}")
                if unexpected_keys:
                    print(f"Encoder unexpected keys after filtering/prefixing: {unexpected_keys}")
            except Exception as e:
                print(f"Error loading encoder weights: {e}")
        else:
            print("No valid encoder checkpoint path provided or file not found. Initializing encoder from scratch.")
            self.encoder.apply(self._init_weights)

        # 冻结 Encoder
        if self.freeze_encoder:
            print("Freezing encoder weights.")
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
        else:
            self.encoder.train() # 确保编码器在训练模式（如果参与训练）

        # --- 2. 初始化 Decoder ---
        print("Initializing new TransformerDecoder...")
        decoder_params = {
            "vocab_size": decoder_config['vocab_size'],
            "depth": decoder_config['depth'],
            "dim": decoder_config['dim'],
            "num_heads": decoder_config['num_heads'],
            "mlp_ratio": decoder_config.get('mlp_ratio', 4.0),
            "num_kv_heads": decoder_config.get('num_kv_heads', None),
            "context_dim": self.encoder.embed_dim, # *** 关键：设置 context_dim ***
            "norm_layer": RMSNorm, # 假设使用 RMSNorm
            "qkv_bias": decoder_config.get('qkv_bias', True),
            "norm_eps": decoder_config.get('norm_eps', 1e-6),
            "drop_rate": decoder_config.get('drop_rate', 0.0),
            "attn_drop_rate": decoder_config.get('attn_drop_rate', 0.0),
            "drop_path_rate": decoder_config.get('drop_path_rate', 0.0),
            "layer_scale_init_values": decoder_config.get('layer_scale_init_values', 1e-5),
            "num_experts": decoder_config.get('num_experts', 2),
            "padding_idx": decoder_config.get('padding_idx', 0),
            "max_seq_len": max_seq_len,
            "rope_base": rope_base,
            "rope_scaling": decoder_config.get('rope_scaling', None),
            "act_layer": getattr(nn, decoder_config.get('act_layer', 'GELU'), nn.GELU), # 动态获取激活层
        }
        
        self.decoder = TransformerDecoder(**decoder_params)

        # --- 3.2 Encoder RoPE ---
        self.encoder_head_dim = self.encoder.embed_dim // self.encoder.num_heads
        encoder_max_len_for_rope = encoder_args['config']['max_text_len'] + self.encoder.patch_embed.num_patches + 1
        self.encoder_rope_1d = DeepseekV3RotaryEmbedding(
            dim=self.encoder_head_dim,
            max_position_embeddings=encoder_max_len_for_rope,
            base=rope_base
        )
        grid_size = self.encoder.patch_embed.grid_size
        self.encoder_rope_2d = Rope2DPosEmb(
             dim=self.encoder_head_dim,
             max_height=grid_size[0],
             max_width=grid_size[1]
        )
        self.grid_height = grid_size[0]
        self.grid_width = grid_size[1]

        self.max_seq_len = max_seq_len

        # --- 4. 添加 Token Type Embeddings ---
        hidden_size = self.encoder.embed_dim
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        self.token_type_embeddings.apply(self._init_weights)

    def get_encoder_context(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """通过 Encoder 处理图像，获取上下文表示和掩码。"""
        vis_embed, vis_mask = self.encoder.visual_embed(image)
        B, N_vis_plus_1, C = vis_embed.shape

        # Prepare RoPE arguments for Encoder Blocks
        image_pos_ids = torch.arange(N_vis_plus_1, device=vis_embed.device).unsqueeze(0).expand(B, -1)
        grid_hw_val = (self.grid_height, self.grid_width)

        # Pass through Encoder Blocks with RoPE arguments
        encoder_hidden_state = vis_embed
        for blk in self.encoder.blocks:
             encoder_hidden_state = blk(
                 encoder_hidden_state,
                 mask=vis_mask,
                 N_text=0,
                 cos_1d=None,
                 sin_1d=None,
                 text_pos_ids=None,
                 rope_2d_instance=self.encoder_rope_2d,
                 image_pos_ids=image_pos_ids,
                 grid_hw=grid_hw_val
             )

        context = self.encoder.norm(encoder_hidden_state)
        context_mask = vis_mask.bool() # 仅使用视觉掩码

        return context, context_mask

    def forward_text_only(self, text_ids: torch.Tensor, text_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """仅使用文本通过 Encoder 获取上下文。"""
        text_embeds = self.decoder.tok_embeddings(text_ids) # 使用 Decoder 的 embedding
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))

        x = text_embeds
        B, N_text, C = x.shape
        current_dev = x.device
        cos_1d, sin_1d, text_pos_ids = None, None, None
        if self.encoder_rope_1d is not None:
            text_pos_ids = torch.arange(N_text, device=current_dev).unsqueeze(0).expand(B, -1)
            self.encoder_rope_1d.to(current_dev)
            cos_1d, sin_1d = self.encoder_rope_1d(x, seq_len=N_text)

        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x, mask=text_masks,
                    N_text=N_text, cos_1d=cos_1d, sin_1d=sin_1d,
                    text_pos_ids=text_pos_ids, rope_2d_instance=None,
                    image_pos_ids=None, grid_hw=None)

        encoder_context = self.encoder.norm(x)
        context_mask = text_masks.bool() # 仅使用文本掩码
        return encoder_context, context_mask

    def encode(
        self,
        image: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        text_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据输入类型对输入进行编码。

        Args:
            image: 图像输入 (B, C, H, W)。
            text_ids: 文本输入 IDs (B, N_txt)。
            text_masks: 文本输入掩码 (B, N_txt)。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: encoder_context 和 context_mask。
        """
        if image is not None and text_ids is not None:
            # --- 图文融合编码 (VQA/Captioning like) ---
            text_embeds = self.decoder.tok_embeddings(text_ids)
            vis_embeds, vis_masks = self.encoder.visual_embed(image)
            fused_context = self.forward_encoder_with_fusion(
                text_embeds, text_masks, vis_embeds, vis_masks
            )
            # 创建融合后的掩码
            fused_mask = torch.cat([text_masks, vis_masks.long()], dim=1)
            return fused_context, fused_mask
        elif image is not None:
            # --- 纯图像编码 ---
            return self.get_encoder_context(image)
        elif text_ids is not None:
            # --- 纯文本编码 ---
            return self.forward_text_only(text_ids, text_masks)
        else:
            raise ValueError("At least one of image or text_ids must be provided to encode.")

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None, # 添加 text_ids 用于纯文本编码
        text_masks: Optional[torch.Tensor] = None, # 添加 text_masks 用于纯文本编码
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_inputs_embeds:Optional[torch.FloatTensor] = None,
        encoder_outputs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_hidden_states: bool = False, # 控制是否返回所有层的隐藏状态（hidden states
        **kwargs, # 其他参数
    ):
        """
        Encoder-Decoder 模型的前向传播。
        现在可以根据 image, text_ids, text_masks 自动进行编码。
        """
        # 1. 获取编码器上下文
        if encoder_outputs is None:
            # 使用新的 encode 方法
            encoder_context, context_mask = self.encode(image=image, text_ids=text_ids, text_masks=text_masks)
        else:
            # 使用预计算的 encoder_outputs
            encoder_context, context_mask = encoder_outputs


        # 4. 通过解码器
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            context=encoder_context,
            context_mask=context_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=return_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        if use_cache:
            logits, hidden_states, moe_losses, present_key_values = decoder_outputs
        else:
            logits, hidden_states, moe_losses = decoder_outputs
            present_key_values = None

        if use_cache:
            return logits, hidden_states, moe_losses, present_key_values
        else:
            return logits, hidden_states, moe_losses

    def get_moe_losses(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # 基准张量：与 decoder 参数同 device / dtype
        base = self.decoder.tok_embeddings.weight
        zero = torch.zeros((), device=base.device, dtype=base.dtype)
    
        # ---------- Encoder ----------
        if not self.freeze_encoder and hasattr(self, "get_encoder_moe_losses"):
            enc_balance, enc_router = self.get_encoder_moe_losses()
        else:
            enc_balance = enc_router = zero
    
        # ---------- Decoder ----------
        if hasattr(self.decoder, "get_moe_losses"):
            dec_losses = self.decoder.get_moe_losses() or (zero, zero)
            dec_balance, dec_router = dec_losses
        else:
            dec_balance = dec_router = zero
    
        total_balance = enc_balance + dec_balance
        total_router  = enc_router  + dec_router
    
        if torch.equal(total_balance, zero) and torch.equal(total_router, zero):
            return None
        return total_balance, total_router
    
    def forward_encoder_with_fusion(
        self,
        text_embeds: torch.Tensor,
        text_masks: torch.Tensor,
        image_embeds: torch.Tensor,
        image_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用融合的文本和图像嵌入通过编码器。
        """
        text_embeds = text_embeds + self.token_type_embeddings(torch.zeros_like(text_masks))
        image_embeds = image_embeds + self.token_type_embeddings(torch.full_like(image_masks, 1))

        co_embeds = torch.cat([text_embeds, image_embeds], dim=1)
        co_masks = torch.cat([text_masks, image_masks], dim=1)

        x = co_embeds
        B, N, C = x.shape
        N_text = text_embeds.shape[1]
        N_img = image_embeds.shape[1]
        current_dev = x.device
        cos_1d, sin_1d, text_pos_ids = None, None, None
        if self.encoder_rope_1d is not None:
            text_pos_ids = torch.arange(N_text, device=current_dev).unsqueeze(0).expand(B, -1)
            self.encoder_rope_1d.to(current_dev)
            cos_1d, sin_1d = self.encoder_rope_1d(x[:, :N_text, :], seq_len=N_text)

        image_pos_ids = None
        rope_2d_inst = None
        grid_hw_val = None
        if self.encoder_rope_2d is not None:
            image_pos_ids = torch.arange(N_img, device=current_dev).unsqueeze(0).expand(B, -1)
            self.encoder_rope_2d.to(current_dev)
            rope_2d_inst = self.encoder_rope_2d
            grid_hw_val = (self.grid_height, self.grid_width)

        for i, blk in enumerate(self.encoder.blocks):
            x = blk(x, mask=co_masks,
                    N_text=N_text, cos_1d=cos_1d, sin_1d=sin_1d,
                    text_pos_ids=text_pos_ids, rope_2d_instance=rope_2d_inst,
                    image_pos_ids=image_pos_ids, grid_hw=grid_hw_val)

        x = self.encoder.norm(x)
        return x

    def get_encoder_moe_losses(self):
        """收集编码器所有 Transformer 块的 MoE 辅助损失"""
        balance_loss = 0.0
        router_z_loss = 0.0
        num_blocks = 0
        if hasattr(self.encoder, 'blocks'):
            for block in self.encoder.blocks:
                if hasattr(block, 'get_moe_losses'):
                    b_loss, r_loss = block.get_moe_losses()
                    if b_loss is not None and r_loss is not None:
                        balance_loss += b_loss
                        router_z_loss += r_loss
                        num_blocks += 1

        if num_blocks > 0:
            balance_loss = balance_loss / num_blocks
            router_z_loss = router_z_loss / num_blocks
        else:
            balance_loss = torch.tensor(0.0, device=self.token_type_embeddings.weight.device)
            router_z_loss = torch.tensor(0.0, device=self.token_type_embeddings.weight.device)

        return balance_loss, router_z_loss

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
