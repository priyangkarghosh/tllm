from helpers import Timer
from model import GPTConfig
from tokenizer import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# init config
config = GPTConfig()

# train tokenizer
tok = Tokenizer()
tok.register_special_tokens(["<|endoftext|>"])

# load openwebtext and stream text
DATASET_ITEMS = 10_000
dataset = load_dataset(
    "roneneldan/TinyStories", 
    split="train", 
    streaming=True,
)

items = []
for i, item in tqdm(enumerate(dataset, start=1), total=DATASET_ITEMS, desc="Loading text"):
    items.append(item.get('text', ''))
    if i >= DATASET_ITEMS: break

text = " ".join(items)
print(f"Total characters: {len(text):,}")

# train tokenizer
with Timer() as t: tok.train(text, config.vocab_size)
print(f"Training took {t.elapsed:.4f}s.")
tok.save('snapshots/tkz.pkl')