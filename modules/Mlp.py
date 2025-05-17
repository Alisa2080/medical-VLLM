import torch
import torch.nn as nn
from typing import Optional,Tuple
from transformers.activations import ACT2FN

class Mlp_original(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x





class Mlp(nn.Module):
    def __init__(self, hidden_size, intermediate_size,hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation function: {hidden_act}. Supported: {list(ACT2FN.keys())}")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output = self.down_proj(self.act_fn(self.up_proj(hidden_states)))
        return output
