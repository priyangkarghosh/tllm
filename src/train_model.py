import torch
import torch.optim as optim
from data_loader.data_loader import DataLoader
from model import GPTConfig, GPT
from tokenizer import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
config = GPTConfig()
tok = Tokenizer.load('./checkpoints/tkz.pkl')

# create batches
tl = DataLoader(device, 4, 32, tok, "./data/tiny_shakespeare.txt")
model = GPT(config)
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    x, y = tl.next_batch()
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step: {i}, loss: {loss.item()}")