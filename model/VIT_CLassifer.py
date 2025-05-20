import torch
import torch.nn as nn
from modules.VITForMIM import VisionTransformerForMaskedImageModeling 
from modules.AttentionSeries import GatedAttention 
from timm.models import create_model
from typing import Optional, Tuple,Callable
from model import pretrained_model

class ViTClassifier(nn.Module):
    def __init__(self,
                 model_name: str, # 用于从timm或本地加载ViT模型的名称或配置
                 pretrained_path: Optional[str] = None, # 预训练ViT权重的路径
                 num_classes: int = 2, # MIL分类任务的类别数
                 embed_dim: Optional[int] = None, # ViT输出的特征维度
                 hidden_dim_att: int = 384,
                 intermediate_dim: int = 512,
                 freeze: bool = False, # 是否冻结ViT的权重
                 ):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = create_model(model_name,pretrained=False,)
        
        # 获取ViT的输出维度
        # 如果vit_params中没有显式给出dim，并且模型有dim属性
        if embed_dim is None:
            if hasattr(self.encoder, 'dim'):
                self.embed_dim = self.encoder.dim
            else:
                raise ValueError("Cannot automatically determine embed_dim. Please provide it.")
        else:
            self.embed_dim = embed_dim

        if pretrained_path:
            try:
                checkpoint = torch.load(pretrained_path, map_location='cpu')
                # 根据您的权重文件结构加载权重
                # 可能是 'model', 'state_dict', 或者直接是权重字典
                if 'model' in checkpoint:
                    state_dict = checkpoint['model']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 处理可能的 'module.' 前缀 (来自DataParallel/DDP训练)
                new_state_dict = {}
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                
                # 过滤掉不匹配的键 (例如，预训练的lm_head在微调时可能不需要)
                model_dict = self.vit_encoder.state_dict()
                pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
                missing_keys, unexpected_keys = self.vit_encoder.load_state_dict(pretrained_dict, strict=False)
                print(f"Loaded ViT weights from {pretrained_path}")
                if missing_keys:
                    print(f"Missing keys in ViT: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys in ViT: {unexpected_keys}")

            except Exception as e:
                print(f"Error loading ViT pretrained weights from {pretrained_path}: {e}")
        else:
            print("No ViT pretrained path provided, ViT will be initialized randomly (or by its own init).")

        # 冻结ViT层 (根据您的要求，先不冻结)
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("ViT encoder weights are frozen.")
        else:
            print("ViT encoder weights are NOT frozen (all layers will be fine-tuned).")


        # 2. MIL聚合模块
        self.mil_aggregator = GatedAttention(
            input_dim=self.embed_dim,
            num_classes=self.num_classes,
            hidden_dim_att=hidden_dim_att,
            intermediate_dim=intermediate_dim
        )

    def forward(self, slide_patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播函数。

        Args:
            slide_patches (torch.Tensor): 一个slide的所有patch图像。
                                         形状: (num_patches, C, H, W)
                                         注意：这里假设输入是一个slide的所有patch，
                                         batch_size隐式地为1（对于slide级别）。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - logits (torch.Tensor): 分类 logits。形状: (1, num_classes_mil)
                - attention_scores (torch.Tensor): 注意力权重。形状: (1, num_patches, 1)
        """
        num_patches, C, H, W = slide_patches.shape

        patch_features_cls = self.encoder(slide_patches, return_cls_feature=True)
        # patch_features_cls 的形状应该是 (num_patches, vit_embed_dim)

        # 将patch特征重塑为 (1, num_patches, vit_embed_dim) 以适应MIL模块
        patch_features_for_mil = patch_features_cls.unsqueeze(0)

        # 通过MIL模块进行聚合和分类
        logits, attention_scores = self.mil_aggregator(patch_features_for_mil)

        return logits, attention_scores
