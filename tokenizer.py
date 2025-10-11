from collections import Counter

class Tokenizer:
    # vocab size must be at least 256
    def __init__(self, data_path: str, vocab_size: int):
        # load and encode data
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.data = list(map(int, text.encode("utf-8")))

        #
        self.vocab_size = max(vocab_size, 256)
        self.num_merges = self.vocab_size - 256

        self.vocab = {i: i for i in range(256)}
        self.merges = {}
        
    def _pair_stats(self, data):
        pairs = Counter()
        for i in range(len(data) - 1):
            pairs[(data[i], data[i+1])] += 1
        return pairs
    
    def _merge(self, data, pair, token_id):
        a, b = pair
        merged = []

        i = 0
        dlen = len(data)
        while i < dlen:
            da = data[i]
            if i < dlen - 1 and da == a and data[i+1] == b:
                merged.append(token_id)
                i += 2
            else:
                merged.append(da)
                i += 1
        return merged
    
    def train(self, debug_hook: int = 100):
        print("Starting training...")
        
        data = self.data.copy()
        for i in range(self.num_merges):
            pairs = self._pair_stats(data)
            if not pairs: break
            
            # turn the most used pair into a new token
            token = pairs.most_common(1)[0][0]
            token_id = 256 + i

            # store merge info
            self.vocab[token_id] = token
            self.merges[token] = token_id
            
            # complete the merge
            data = self._merge(data, token, token_id)
            if i % debug_hook == 0: 
                print(f"Merge {i}: {token} → {token_id}")
        
        print(f'Finished training with a {256 + i} total vocabulary size.')
