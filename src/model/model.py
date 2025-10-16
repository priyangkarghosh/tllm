import torch
import torch.nn as nn
from torch.nn import functional as F
from model.attention import CasualSelfAttention
from model.config import GPTConfig
from model.mlp import MLP

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x += self.attn(self.ln1(x))
        x += self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config   
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # positional embedding
            h = nn.ModuleList([Block(self.config) for _ in range(config.n_layer)]), # hidden layers
            ln_f = nn.LayerNorm(config.n_embd) # 
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)