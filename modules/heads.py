import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from transformers.activations import ACT2FN

class Pooler(nn.Module):
    def __init__(self, hidden_size, hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(self.hidden_size, self.hidden_size,bias=False)
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation function: {hidden_act}. Supported: {list(ACT2FN.keys())}")
        self.activation = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor,**kwargs) -> torch.Tensor:
        first_token_tensor = hidden_states[:, 0]
        output = self.activation(self.linear(first_token_tensor))
        return output


class ITMHead(nn.Module):
    def __init__(self, hidden_size):

        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, 2, bias=False)

    def forward(self, hidden_states: torch.Tensor,**kwargs) -> torch.Tensor:
        output = self.linear(hidden_states)
        return output

class ITCHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor,**kwargs) -> torch.Tensor:
        output = self.linear(hidden_states)
        return output


class MLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        # 使用BERT的预测头变换层进行特征转换
        self.transform = BertPredictionHeadTransform(config)
        # 定义解码器线性层，将隐藏层特征映射到词汇表大小
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 定义可学习的偏置参数，初始化为全零
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 如果提供了预训练权重，则用其初始化decoder层
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, hidden_states: torch.Tensor,**kwargs) -> torch.Tensor:
        hidden_states = self.transform(hidden_states)
        output = self.decoder(hidden_states) + self.bias
        return output
