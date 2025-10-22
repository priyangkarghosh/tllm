from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50304 # number of tokens in vocab

    n_layer: int = 12 # number of layers
    n_heads: int = 12 # number of heads
    embd_dim: int = 768 # embedding dimension

    n_latent: int = 64 # latent space dimension
    n_rope: int = 32 # rotary positional encoding dim