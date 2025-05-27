import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation function: {hidden_act}. Supported: {list(ACT2FN.keys())}")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class MoeMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act="silu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        if hidden_act not in ACT2FN:
            raise ValueError(f"Unsupported activation function: {hidden_act}. Supported: {list(ACT2FN.keys())}")
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class MoeSparseMoeBlock(nn.Module):
    def __init__(self, 
                 hidden_size:int,
                 num_experts:int,
                 num_experts_per_tok:int,
                 moe_intermediate_size:int,
                 norm_topk_prob:bool = True, 
                 moe_hidden_act:str = "silu"): 
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = min(num_experts_per_tok, num_experts) # 若请求的 top-k 大于专家数，会触发运行时错误；取两者最小值
        self.norm_topk_prob = norm_topk_prob # Qwen3 specific
        self.moe_intermediate_size = moe_intermediate_size
        self.hidden_act = moe_hidden_act

        # gating
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoeMLP(self.hidden_size, self.moe_intermediate_size, self.hidden_act) for _ in range(self.num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim) # (B*N, D)
        # router_logits: (B*N, num_experts)
        router_logits = self.gate(hidden_states_reshaped)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.numel() == 0:
                continue

            current_state = hidden_states_reshaped[top_x] # (num_tokens_for_expert, D)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits # router_logits shape: (B*N, num_experts)
