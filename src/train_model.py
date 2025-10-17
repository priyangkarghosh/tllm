import math
import torch
import torch.optim as optim
from training import DataLoader, CosineDecayLR
from model import GPTConfig, GPT
from tokenizer import Tokenizer

# torch init
torch.set_float32_matmul_precision("high")

# ------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


config = GPTConfig()
tok = Tokenizer.load('./checkpoints/tkz.pkl')

# create batches
tl = DataLoader(device, 4, 1024, tok, "./data/tiny_shakespeare.txt")
model = GPT(config)
model.to(device)
# model = torch.compile(model) <- isn't always supported by windows

# learning rate and optimizer
optimizer = model.configure_optimizer(device, 0.1, 6e-4)
scheduler = CosineDecayLR(optimizer, 3e-4, 0.1 * 3e-4, 10, 50)

num_steps = 50 
for i in range(num_steps):    
    x, y = tl.next_batch()
    optimizer.zero_grad()
    
    logits, loss = model(x, y)
    
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    lr = scheduler.step()
    optimizer.step()

    print(f"step: {i} | loss: {loss.item():.6f} | norm: {norm: .4f} | lr: {lr:.6f}")
