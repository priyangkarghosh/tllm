import torch
import torch.nn as nn
from torch.nn import functional as F
from model.config import GPTConfig

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config