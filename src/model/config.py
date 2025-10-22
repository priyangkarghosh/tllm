from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- Model dimensions ---
    n_layer: int = 12
    n_heads: int = 12
    n_embd: int = 768

    # --- Sequence parameters ---
    block_size: int = 1024
    max_seq_len: int = 1024

    # --- Tokenization ---
    vocab_size: int = 50304

    # --- MLA ---
    nkv_latent: int = 64
    nq_latent: int = 48
    n_rope: int = 32
    mscale: float = 0.34

    @property
    def rope_factor(self) -> float:
        return self.max_seq_len / self.block_size