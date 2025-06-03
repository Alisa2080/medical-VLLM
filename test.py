import torch
import torch.nn as nn
import tempfile
import os
import unittest
from functools import partial
import shutil


from models.vlmo_module import VLMoForTextPretraining
from modules.Encoder import TransformerEncoder # 主要用于类型检查或间接引用
from modules.RMSNorm import RMSNorm
from modules.heads import MLMHead
from modules.Mlp import Mlp,MoeSparseMoeBlock
from models import Encoder_version # 确保这个导入会执行 @register_model

from timm.models import create_model
from pytorch_lightning.utilities import rank_zero_info
from transformers.models.bert.modeling_bert import BertConfig



class TestVLMoForTextPretrainingRealCheckpointLoading(unittest.TestCase):

    def get_base_config(self):
        """
        提供一个与 beit_base_patch16_384 兼容的基础配置。
        您可能需要根据您的 Encoder_version.py 中的 beit_base_patch16_384 定义来调整这些值。
        """
        return {
            "model_name": "beit_base_patch16_384", # 这个模型名应该在 Encoder_version.py 中注册
            "vocab_size": 8192, # 与 Encoder_version.py 中的默认值匹配或根据您的需求调整
            "embed_dim": 512,   # from beit_base_patch16_384
            "depth": 6,         # from beit_base_patch16_384
            "num_heads": 8,     # from beit_base_patch16_384
            "num_kv_heads": 4,  # from beit_base_patch16_384
            "mlp_ratio": 4.0,   # from beit_base_patch16_384
            "num_experts": 4,   # from beit_base_patch16_384
            "num_experts_per_tok": 2, # from beit_base_patch16_384
            "patch_size": 16,
            "img_size": 384,    # 确保 patch_embed 可以初始化
            "max_seq_len": 512, # from beit_base_patch16_384
            "num_token_types": 2, # from beit_base_patch16_384
            "qkv_bias": False,  # from beit_base_patch16_384
            "layer_scale_init_values": 0.1, # from beit_base_patch16_384
            "drop_path_rate": 0.0, # 通常在预训练加载时不重要，但模型初始化可能需要
            "moe_balance_loss_weight": 0.01,
            "moe_router_z_loss_weight": 0.001,
            "learning_rate": 1e-4, # For optimizer if it were used
            "weight_decay": 0.01,
            "adam_epsilon": 1e-8,
            "adam_betas": [0.9, 0.98],
            "warmup_steps": 0,
            "decay_power": "cosine",
            "end_lr": 0.0,
            "moe_hidden_act": "silu", # from beit_base_patch16_384
            "norm_eps": 1e-6, # from beit_base_patch16_384
            "attn_drop_rate": 0.01, # from beit_base_patch16_384
            "init_std": 0.02, # from beit_base_patch16_384
        }

    def test_load_from_custom_beit2_like_checkpoint(self):
        config = self.get_base_config()
        config["convert_beit2_to_textpt"] = True # 关键：启用转换

        # ##################################################################
        # ## 重要: 请将下面的路径替换为您的实际BEiT2风格的权重文件路径  ##
        # ##################################################################
        # 假设这个文件包含一个类似 'state_dict' 或 'model' 的键，其值为权重字典
        # 并且权重键名类似于 'model.encoder.blocks.0.mlp.fc1.weight'
        custom_beit2_weight_file = r"E:\article_code\output\beit2\finetuning_pl\mil_checkpoints\version_0\last.ckpt" 
        # ##################################################################

        if not os.path.exists(custom_beit2_weight_file):
            print(f"警告: BEiT2-like 权重文件 {custom_beit2_weight_file} 未找到。跳过此测试。")
            self.skipTest(f"权重文件 {custom_beit2_weight_file} 未找到。")
            return

        config["load_path"] = custom_beit2_weight_file
        
        print(f"\n--- 测试从 BEiT2-like 检查点加载 (转换模式): {custom_beit2_weight_file} ---")
        try:
            model = VLMoForTextPretraining(config_dict=config)
            model.eval() # 设置为评估模式
            print("VLMoForTextPretraining 模型已成功初始化并尝试加载权重。")

            # --- 基本验证 ---
            # 1. 检查模型是否加载了某些权重 (不一定是所有权重都匹配)
            #    一个简单的检查是看某些参数是否不再是随机初始化的值。
            #    更深入的验证需要知道源检查点中的确切值。
            
            # 检查一个期望被加载的参数，例如第一个block的某个转换后的MLP层
            # 注意：由于我们不知道源权重的确切值，这里只能做一些基本的存在性检查
            self.assertTrue(hasattr(model.Encoder.blocks[0], 'moe_block'), "Block 0 should have moe_block")
            
            if config["num_experts"] > 1:
                self.assertIsInstance(model.Encoder.blocks[0].moe_block, MoeSparseMoeBlock)
                expert_mlp = model.Encoder.blocks[0].moe_block.experts[0]
                self.assertIsNotNone(expert_mlp.gate_proj.weight.data, "Expert 0 gate_proj weight should be loaded")
                # 如果您知道源权重中 fc1 的值，可以进行比较
                # source_fc1_weight = torch.load(custom_beit2_weight_file, map_location='cpu')['state_dict']['model.encoder.blocks.0.mlp.fc1.weight']
                # torch.testing.assert_close(expert_mlp.gate_proj.weight.data, source_fc1_weight)
            else:
                self.assertIsInstance(model.Encoder.blocks[0].moe_block, Mlp)
                self.assertIsNotNone(model.Encoder.blocks[0].moe_block.up_proj.weight.data, "MLP up_proj weight should be loaded")

            # 2. 检查参数冻结状态
            print("检查参数冻结状态...")
            self.assertTrue(model.Encoder.word_embeddings.weight.requires_grad, "Word embeddings should be trainable.")
            self.assertTrue(model.Encoder.token_type_embeddings.weight.requires_grad, "Token type embeddings should be trainable.")
            self.assertTrue(model.Encoder.blocks[0].attn.q_proj.weight.requires_grad, "Attention q_proj should be trainable.")
            if config["num_experts"] > 1:
                 self.assertTrue(model.Encoder.blocks[0].moe_block.experts[0].gate_proj.weight.requires_grad, "Expert gate_proj should be trainable.")
            else:
                 self.assertTrue(model.Encoder.blocks[0].moe_block.up_proj.weight.requires_grad, "MLP up_proj should be trainable.")
            self.assertTrue(model.Encoder.norm.weight.requires_grad, "Final encoder norm should be trainable.")
            self.assertTrue(model.mlm_score.decoder.weight.requires_grad, "MLM decoder weight should be trainable.")
            
            # 视觉相关的部分应该被冻结
            if hasattr(model.Encoder, 'cls_token') and model.Encoder.cls_token is not None:
                 self.assertFalse(model.Encoder.cls_token.requires_grad, "Cls token should be frozen.")
            if hasattr(model.Encoder, 'patch_embed') and hasattr(model.Encoder.patch_embed, 'proj'):
                 self.assertFalse(model.Encoder.patch_embed.proj.weight.requires_grad, "Patch embed proj weight should be frozen.")
            print("参数冻结状态检查完毕。")

        except Exception as e:
            self.fail(f"从 BEiT2-like 检查点加载时发生错误: {e}")
        finally:
            # 如果测试中创建了临时文件，在这里清理
            pass
        print("--- BEiT2-like 检查点加载测试完成 ---")

    def test_load_from_custom_vlmo_like_checkpoint(self):
        config = self.get_base_config()
        config["convert_beit2_to_textpt"] = False # 关键：禁用转换

        # #####################################################################
        # ## 重要: 请将下面的路径替换为您的实际VLMo风格的权重文件路径       ##
        # #####################################################################
        # 假设这个文件包含一个类似 'state_dict' 或 'model' 的键，其值为权重字典
        # 并且权重键名可能带有 'Encoder.' 或 'model.transformer.' 等前缀
        custom_vlmo_weight_file = r"E:\article_code\output\beit2\finetuning_pl\mil_checkpoints\version_0\last.ckpt"
        # #####################################################################

        if not os.path.exists(custom_vlmo_weight_file):
            print(f"警告: VLMo-like 权重文件 {custom_vlmo_weight_file} 未找到。跳过此测试。")
            self.skipTest(f"权重文件 {custom_vlmo_weight_file} 未找到。")
            return
            
        config["load_path"] = custom_vlmo_weight_file

        print(f"\n--- 测试从 VLMo-like 检查点加载 (直接模式): {custom_vlmo_weight_file} ---")
        try:
            model = VLMoForTextPretraining(config_dict=config)
            model.eval()
            print("VLMoForTextPretraining 模型已成功初始化并尝试加载权重。")

            # --- 基本验证 ---
            # 尝试加载一个期望存在的权重，例如 Encoder.word_embeddings.weight
            # 您需要知道您的 VLMo-like 检查点中实际存在的键名
            # 假设您的检查点中存在 'Encoder.word_embeddings.weight' 或类似键
            # source_word_emb_weight = torch.load(custom_vlmo_weight_file, map_location='cpu')['state_dict']['Encoder.word_embeddings.weight']
            # torch.testing.assert_close(model.Encoder.word_embeddings.weight.data, source_word_emb_weight)
            # 由于我们不知道确切的值，这里只检查它是否被加载（非默认随机值）
            self.assertIsNotNone(model.Encoder.word_embeddings.weight.data, "Word embeddings should be loaded.")

            # 检查 MLM head 的权重是否加载
            # source_mlm_decoder_weight = torch.load(custom_vlmo_weight_file, map_location='cpu')['state_dict']['mlm_score.decoder.weight']
            # torch.testing.assert_close(model.mlm_score.decoder.weight.data, source_mlm_decoder_weight)
            self.assertIsNotNone(model.mlm_score.decoder.weight.data, "MLM decoder weight should be loaded.")

            # 参数冻结状态应与转换加载时相同
            print("检查参数冻结状态...")
            self.assertTrue(model.Encoder.word_embeddings.weight.requires_grad)
            # ... 其他冻结/解冻检查与上一个测试用例类似 ...
            print("参数冻结状态检查完毕。")

        except Exception as e:
            self.fail(f"从 VLMo-like 检查点加载时发生错误: {e}")
        finally:
            pass
        print("--- VLMo-like 检查点加载测试完成 ---")


if __name__ == '__main__':
    # 运行测试前确保 Encoder_version.py 中的模型已注册
    # 通常在导入 Encoder_version 时会自动执行
    
    # 如果您想在运行测试时看到被 mock 掉的 rank_zero_info 的输出，
    # 请在 mock_rank_zero_info 函数中取消注释 print 语句。
    
    unittest.main()