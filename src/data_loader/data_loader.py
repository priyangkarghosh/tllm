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
        self.tokens = torch.tensor(tok.encode(text), device=device)
        print(f"Data loaded with {len(self.tokens)} tokens, with {len(self.tokens) // (B * T)} tokens per epoch.")
        
        # current batch
        self.batch_offset = 0
    
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, P = self.B, self.T, B * P
        buf = torch.tensor(self.tokens[self.batch_offset : self.batch_offset + P + 1], device=self.device)
        x, y = buf[:-1].view(B, T), buf[1:].view(B, T)

        # reset batch offset if we reach the end of our tokens
        self.batch_offset += P
        if self.batch_offset + P + 1 > len(self.tokens):
            self.batch_offset = 0
        return x, y