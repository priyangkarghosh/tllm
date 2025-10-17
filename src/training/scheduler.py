import math
from torch.optim import Optimizer

class CosineDecayLR:
    def __init__(
        self, 
        optimizer: Optimizer, 
        max_lr: float = 1e-3, 
        min_lr: float = 1e-4, 
        warmup_steps: int = 100, 
        total_steps: int = 1000
    ) -> None:
        self.optimizer = optimizer
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        self.current_step: int = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self, it):
        # linear warmup
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps

        # reached total steps
        if it > self.total_steps: return self.min_lr
        
        # cosine decay
        decay = (it - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        assert 0 <= decay <= 1
        coeff = 0.5 * (1 + math.cos(math.pi * decay))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)
    
