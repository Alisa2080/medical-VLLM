import torch
import torch.nn as nn


class ScalingLayerForClip(nn.Module):
    def __init__(self):
        super(ScalingLayerForClip, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255. # rescale to [0, 1.]
        return (inp - self.shift) / self.scale

class ScalingLayerForIM(nn.Module):
    def __init__(self):
        super(ScalingLayerForIM, self).__init__()
        self.register_buffer('shift', torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]) # scale for tokenizer with default prosscess type \in [-1, 1]
        self.register_buffer('scale', torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, inp):
        inp = ((inp + 1.) * 127.5).clamp(0, 255.) / 255. # rescale to [0, 1.]
        return (inp - self.shift) / self.scale
