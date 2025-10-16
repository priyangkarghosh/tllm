import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import GPTConfig

class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        # make sure config setup properly so dims are divisible
        assert config.n_embd % config.n_head == 0
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd) # combine K, Q, V
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head,  self.n_embd = config.n_head, config.n_embd
        
        self.register_buffer(
            "bias", torch.tril(torch.ones(config.block_size, config.block_size).view(
                1, 1, config.block_size, config.block_size
            ))
        )
        
    def forward(self, x):
        B, T, C = x.size()
        H = C // self.n_head
        
        # grab kqv projections
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, H).tranpose(1, 2)
        q = q.view(B, T, self.n_head, H).tranpose(1, 2)
        v = v.view(B, T, self.n_head, H).tranpose(1, 2)

        # calc attention
        att = (q @ k.tranpose(-2, 1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y