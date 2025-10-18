import torch
from model import GPT, GPTConfig
from tokenizer import Tokenizer

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
        
        probs = torch.softmax(logits, dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, next_tokens), dim=1)
    return idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load tokenizer
tok = Tokenizer.load("./snapshots/tkz.pkl")

# load checkpoint
model_path = "./snapshots/tllm_base124M.pt"
checkpoint = torch.load(model_path, map_location=device)

# reconstruct config
config = GPTConfig(**checkpoint["config"]) if "config" in checkpoint else GPTConfig()
model = GPT(config).to(device)

# load weights
model.load_state_dict(checkpoint, strict=True)
model.eval()

# ===============================
prompts = [
    "The meaning of life is",
    "Once upon a time",
    "In a distant future, humanity"
]

# encode all prompts into a batch tensor
encoded = [tok.encode(p) for p in prompts]
max_len = max(len(e) for e in encoded)

# pad to equal length for batching
batch = [e + tok.encode(' ') * (max_len - len(e)) for e in encoded]
idx = torch.tensor(batch, dtype=torch.long).to(device)

# generate new tokens
out = generate(model, idx, max_new_tokens=100, temperature=0.8, top_k=50)

# decode each output sequence
for i, seq in enumerate(out):
    print(f"\nPrompt {i+1}:")
    print(tok.decode(seq.tolist()))
