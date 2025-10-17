import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, scale: int = 4):
        super().__init__()
        
        self.c_fc = nn.Linear(config.n_embd, scale * config.n_embd)
        self.gelu = nn.GELU() # works better in practice compared to ReLU
        self.c_proj = nn.Linear(scale * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
