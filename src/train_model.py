import math
import torch
import torch.optim as optim
from training import DataLoader, CosineDecayLR, train
from model import GPTConfig, GPT
from tokenizer import Tokenizer

# torch init
torch.set_float32_matmul_precision("high")

# ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


config = GPTConfig()
tok = Tokenizer.load('./snapshots/tkz.pkl')

# create batches
#batch_size, B, T = 524288, 4, 1024
batch_size, B, T = 1024, 4, 64
assert batch_size % (B * T) == 0, "Batch size is not divisible by (B * T)"
grad_accum_steps = batch_size // (B * T)
train_loader = DataLoader(device, B, T, tok, "./data/tiny_shakespeare.txt")
print(f"Requested batch size of {batch_size}")
print(f"-> Using {grad_accum_steps} gradient accumulation steps")

# create model
model = GPT(config)
model.to(device)
# model = torch.compile(model) <- isn't always supported by windows

# learning rate and optimizer
optimizer = model.configure_optimizer(device, 0.1, 6e-4)
scheduler = CosineDecayLR(optimizer, 3e-4, 0.1 * 3e-4, 10, 50)
train(device, model, train_loader, optimizer, scheduler, 50, grad_accum_steps)