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
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
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
        
        # weight sharing scheme
        # -> saves params and improves model since it explicitly tells the model
        # -> that these two layers should have similar weights
        self.transformer.wte.weight = self.lm_head.weight
    
    # magic numbers from gpt2 code
    def _init_weights(self, module: nn.Linear | nn.Embedding):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward input sequence of length {T}, block size is too small"
        
        # create the input seq
        tok_emb = self.transformer.wte(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb
        
        # forward transformer blocks
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # calculate loss
        loss = (F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) 
                if targets is not None else None)
        return logits, loss
        