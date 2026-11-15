"""包内的最小层实现（让 VAE 模块不依赖 npt 目录）。"""
import torch
import torch.nn as nn


class Lin(nn.Module):
    """线性层：权重持为 Parameter，forward 只用 matmul。"""

    def __init__(self, w, b=None):
        super().__init__()
        self.w = nn.Parameter(w.detach().float().clone())
        self.b = nn.Parameter(b.detach().float().clone()) if b is not None else None

    def forward(self, x):
        y = torch.matmul(x, self.w.T)
        return y if self.b is None else y + self.b
