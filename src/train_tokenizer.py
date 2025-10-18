import json
from model import GPTConfig
from tokenizer import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

BATCH_SIZE = 10_000
NUM_BATCHES = 15

def main():
    # init config
    config = GPTConfig()

    # create tokenizer instance
    tok = Tokenizer()
    tok.register_special_tokens(["<|endoftext|>"])

    # load dataset
    dataset = load_dataset(
        "roneneldan/TinyStories", 
        split="train", streaming=True
    )

    # go through each batch
    dataset_iter = iter(dataset)
    for batch in range(NUM_BATCHES):
        batch_data = []
        for _ in tqdm(range(BATCH_SIZE), desc=f"Processing batch {batch + 1}"):
            try: item = next(dataset_iter)
            except StopIteration: break
            batch_data.append(item.get("text", ""))
        batch_text = " ".join(batch_data)

        # train tokenizer on this batch
        # -> stop if training errors
        if not tok.train(batch_text, config.vocab_size):
            break
        
        # check if we reached the end of the dataset
        if len(batch_data) != BATCH_SIZE:
            break
    
    # make sure tokenizer is tokenizing properly
    tok.save()
    test = "The quick brown fox jumps over the lazy dog.*21nk..d180)_)9zujz\n\n\n//...."
    
    enc = tok.encode(test)
    print(len(enc), val := tok.decode(enc))
    print(test == val)

if __name__ == '__main__':
    main()