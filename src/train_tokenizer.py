from helpers import Timer
from model import GPTConfig
from tokenizer import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

def main():
    # init config
    config = GPTConfig()

    # create tokenizer instance
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
    print(f"Training took {t.elapsed:.4f} seconds.")
    tok.save('snapshots/tkz.pkl')
    
    # make sure tokenizer is tokenizing properly
    test = "The quick brown fox jumps over the lazy dog.*21nk..d180)_)9zujz\n\n\n//...."
    print(val := tok.decode(tok.encode(test)))
    print(test == val)

if __name__ == '__main__':
    main()