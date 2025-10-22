import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import GPTConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class MLA(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()

        assert config.embd_dim % config.n_heads == 0, "n_embd must be divisible by n_head"
       
        self.n_heads = config.n_heads
        self.embd_dim = config.embd_dim
    
        # compression weights (to latent space)
        self.wkv_cmp = nn.Linear(config.embd_dim, config.n_latent + config.n_rope)
        self.kv_norm = RMSNorm(config.n_latent) # normalizes ONLY the kvs (not the positional embeddings)

        # expansion weights
        head_dim = config.embd_dim // config.n_heads
        self.qk_rope_head_dim = config.n_rope # rope dim (rotated part)
        self.qk_nope_head_dim = head_dim - self.qk_rope_head_dim # non-rope dim (not rotated part)
        self.wkv_exp = nn.Linear(config.n_latent, config.n_heads * (self.qk_nope_head_dim * head_dim))
        
        # output weights
        self.wo = nn.Linear(config.n_heads * head_dim, self.embd_dim)

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        assert config.embd_dim % config.n_heads == 0, "n_embd must be divisible by n_head"
        
        self.c_attn = nn.Linear(config.embd_dim, 3 * config.embd_dim)
        
        self.c_proj = nn.Linear(config.embd_dim, config.embd_dim)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # no idea what this does
        
        self.n_head = config.n_heads
        self.embd_dim = config.embd_dim

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
        q, k, v = qkv.split(self.embd_dim, dim=2)
        q = q.view(B, T, self.n_head, H).transpose(1, 2)
        k = k.view(B, T, self.n_head, H).transpose(1, 2)
        v = v.view(B, T, self.n_head, H).transpose(1, 2)

        # scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
