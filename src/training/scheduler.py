import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class CosineDecayLR(LRScheduler):
    def __init__(
        self, 
        optimizer: Optimizer, 
        max_lr: float = 1e-3, 
        min_lr: float = 1e-4, 
        warmup_steps: int = 100, 
        total_steps: int = 1000,
        previous_step: int = -1,
    ) -> None:
        self.optimizer = optimizer
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # set initial learning rates for all param groups
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', max_lr)
        
        # init parent
        super().__init__(optimizer, previous_step)
        
    @property
    def current_step(self): # pytorch weird naming so use this for clarity
        return self.last_epoch
    
    def get_lr(self):
        # linear warmup
        if self.last_epoch < self.warmup_steps:
            lr = self.max_lr * (self.last_epoch + 1) / self.warmup_steps
        
        # reached total steps
        elif self.last_epoch > self.total_steps: 
            lr = self.min_lr 
    
        # cosine decay
        else: 
            decay = (self.last_epoch - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            assert 0 <= decay <= 1
            coeff = 0.5 * (1 + math.cos(math.pi * decay))
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        
        # return lr for every parameter group
        return [lr for _ in self.optimizer.param_groups]
    
