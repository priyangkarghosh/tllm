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

        assert config.n_embd % config.n_heads == 0, "n_embd must be divisible by n_head"
       
        head_dim = config.n_embd // config.n_heads
        self.qk_rope_head_dim = config.n_rope # rope dim (rotated part)
        self.qk_nope_head_dim = head_dim - self.qk_rope_head_dim # non-rope dim (not rotated part)
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim

        # q latent space compression/expansion
        if config.nq_latent == 0:
            self.wq = nn.Linear(config.n_embd, config.n_heads * self.qk_head_dim)
        else:
            self.wq_cmp = nn.Linear(config.n_embd, config.nq_latent)
            self.q_norm = RMSNorm(config.nq_latent)
            self.wq_exp = nn.Linear(config.nq_latent, config.n_heads * self.qk_head_dim)
    
        # kv latent space compression/expansion
        self.wkv_cmp = nn.Linear(config.n_embd, config.nkv_latent + config.n_rope)
        self.kv_norm = RMSNorm(config.nkv_latent) # normalizes ONLY the kvs (not the positional embeddings)
        self.wkv_exp = nn.Linear(config.nkv_latent, config.n_heads * (self.qk_nope_head_dim + head_dim))
        
        # output weights
        self.wo = nn.Linear(config.n_heads * head_dim, config.n_embd)
        self.softmax_scale = self.qk_head_dim ** -0.5
        if config.max_seq_len > config.block_size:  # extends context?
            mscale = 0.1 * config.mscale * math.log(config.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # cache
        self.register_buffer("kv_cache", torch.zeros(config.block_size, config.max_seq_len, config.nkv_latent), persistent=False)
        self.register_buffer("pe_cache", torch.zeros(config.block_size, config.max_seq_len, self.qk_rope_head_dim), persistent=False)
        
class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        assert config.n_embd % config.n_heads == 0, "n_embd must be divisible by n_head"
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1 # no idea what this does
        
        self.n_embd = config.n_embd
        self.n_heads = config.n_heads

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.size()
        H = C // self.n_heads

        # project to q, k, v
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_heads, H).transpose(1, 2)
        k = k.view(B, T, self.n_heads, H).transpose(1, 2)
        v = v.view(B, T, self.n_heads, H).transpose(1, 2)

        # scaled dot-product attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y
