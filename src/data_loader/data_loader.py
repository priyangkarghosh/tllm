import torch
from tokenizer import Tokenizer

class DataLoader():
    def __init__(
        self, 
        device: torch.device,
        B: int, 
        T: int, 
        tok: Tokenizer,
        data_path: str,
    ) -> None:
        self.device = device
        self.B, self.T = B, T
        
        # create token tensor
        with open(data_path, 'r') as f: text = f.read()
        self.tokens = torch.tensor(tok.encode(text), dtype=torch.long)  # load on cpu initially to save vram
        print(f"Data loaded with {len(self.tokens)} tokens, with {len(self.tokens) // (B * T)} batches per epoch.")
        
        # pre-allocate x, y tensors on selected device
        self.x = torch.empty((B, T), dtype=torch.long, device=self.device)
        self.y = torch.empty((B, T), dtype=torch.long, device=self.device)

        # current batch
        self.batch_offset = 0
    
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, P = self.B, self.T, self.B * self.T
        
        # create buffer views for new batch
        buf = self.tokens[self.batch_offset : self.batch_offset + P + 1]
        self.x.copy_(buf[:-1].view(B, T))
        self.y.copy_(buf[1:].view(B, T))

        # reset batch offset if we reach the end of our tokens
        self.batch_offset += P
        if self.batch_offset + P + 1 > len(self.tokens):
            self.batch_offset = 0
        return self.x, self.y