import torch
import torch.optim as optim
from model import GPTConfig, GPT
from tokenizer import Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
    
config = GPTConfig()
tok = Tokenizer.load('./checkpoints/tkz.pkl')

# encode training data
with open("./data/tiny_shakespeare.txt", 'r') as f: 
    text = f.read()
text = text[:1000]
tokens = tok.encode(text)

# create batches
B, T = 4, 32
buf = torch.tensor(tokens[:B*T + 1], device=device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model = GPT(config)
model.to(device)

optimizer = optim.AdamW(model.parameters, lr=3e-4)
for i in range(50):
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step: {i}, loss: {loss.item()}")