import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import GPTConfig


class CasualSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        H = C // self.n_head

        # project to q, k, v
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, H).transpose(1, 2)
        k = k.view(B, T, self.n_head, H).transpose(1, 2)
        v = v.view(B, T, self.n_head, H).transpose(1, 2)

        # scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(H))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, n_head, T, H)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
