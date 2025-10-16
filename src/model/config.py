from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    embedded_dim: int = 384