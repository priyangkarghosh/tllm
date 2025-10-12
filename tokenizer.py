import regex as re
from collections import Counter

GPT_REG_SPLIT = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
   
    def _pair_stats(self, data: list[list[int]]):
        pairs = Counter()
        for chunk in data:
            for i in range(len(chunk) - 1):
                pairs[(chunk[i], chunk[i+1])] += 1
        return pairs
   
    def _merge(self, data: list[int], pair: tuple[int, int], token_id: int):
        a, b = pair
        merged = []
        
        i = 0
        dlen = len(data)
        while i < dlen:
            if i < dlen - 1 and data[i] == a and data[i+1] == b:
                merged.append(token_id)
                i += 2
            else:
                merged.append(data[i])
                i += 1
        return merged
   
    def train(self, data_path: str, vocab_size: int, debug_hook: int = 100):
        # read and encode training data
        print("Reading data...")
        with open(data_path, "r", encoding="utf-8") as f: text = f.read()
        data = [list(chunk.encode("utf-8")) for chunk in GPT_REG_SPLIT.findall(text)]
        
        print("Starting training...")
        num_merges = vocab_size - 256
        for i in range(num_merges):
            pairs = self._pair_stats(data)
            if not pairs: break
            
            # find the most common pair globally
            token = pairs.most_common(1)[0][0]
            token_id = 256 + i
            
            # store merge info
            a, b = token
            self.vocab[token_id] = self.vocab[a] + self.vocab[b]
            self.merges[token] = token_id
            
            # apply merge to ALL chunks
            data = [self._merge(chunk, token, token_id) for chunk in data]
            if i % debug_hook == 0: print(f"Merge {i}: {token} → {token_id}, count: {pairs[token]}")
        
        print(f'Finished training with {256 + i} total vocabulary size.')
    
    def decode(self, data: list[int]):
        tokens = b"".join(self.vocab[token_id] for token_id in data)
        return tokens.decode("utf-8", errors="replace")
   
    def encode(self, text: str):
        # split text into chunks
        chunks = GPT_REG_SPLIT.findall(text)
        
        tokens = []
        for chunk in chunks:
            chunk_tokens = list(chunk.encode("utf-8"))
            
            # apply merges to this chunk
            while True:
                pairs = list(zip(chunk_tokens, chunk_tokens[1:]))
                if not pairs: break
                
                # find the pair with the earliest merge
                token = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
                if token not in self.merges: break
                
                token_id = self.merges[token]
                chunk_tokens = self._merge(chunk_tokens, token, token_id)
            tokens.extend(chunk_tokens)
        return tokens