import pickle
import regex as re
import numpy as np
from heapq import *
from collections import Counter, defaultdict

GPT_REG_SPLIT = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
   
    def _pair_stats(self, nodes: np.ndarray, chunk_heads: list) -> tuple[Counter, defaultdict[list]]:
        pair_stats = Counter()
        pair_positions = defaultdict(set)
        
        for head in chunk_heads:
            node = head
            while node != -1:
                next = nodes[node]['next']
                if next != -1:
                    pair = (int(nodes[node]['val']), int(nodes[next]['val']))
                    pair_stats[pair] += 1
                    pair_positions[pair].add(node)
                node = next
       
        return pair_stats, pair_positions
   
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

    def _merge_node(
        self, 
        nodes: np.ndarray, 
        node: int, 
        token_id: int, 
        pair_stats: dict, 
        pair_positions: defaultdict, 
        heap: list
    ):
        next = nodes[node]['next']
        prev = nodes[node]['prev']
        next_next = nodes[next]['next']

        a = int(nodes[node]['val'])
        b = int(nodes[next]['val'])
        
        # fix old pairs
        if prev != -1:
            left = (int(nodes[prev]['val']), a)
            pair_stats[left] -= 1
        
        if next_next != -1:
            right = (b, int(nodes[next_next]['val']))
            pair_stats[right] -= 1
        
        # merge tokens
        # -> current token turns into the combined token
        # -> current token now also points to next next node
        nodes[node]['val'] = token_id
        nodes[node]['next'] = next_next
        if next_next != -1: nodes[next_next]['prev'] = node
        
        # add new pairs
        if prev != -1:
            pair_stats[left := (int(nodes[prev]['val']), token_id)] += 1
            heappush(heap, (-pair_stats[left], left))
            pair_positions[left].add(prev)
        
        if next_next != -1:
            pair_stats[right := (token_id, int(nodes[next_next]['val']))] += 1
            heappush(heap, (-pair_stats[right], right))
            pair_positions[right].add(next_next)
    
    def _build_training_chunks(self, data: list[list[int]]) -> tuple[np.ndarray, list[int]]:
        token_count = sum(map(len, data))
        
        # build the chunks as a double linked list using np arrays
        dtype = [('val', np.int32), ('prev', np.int32), ('next', np.int32)]
        nodes = np.zeros(token_count, dtype=dtype)
        nodes['prev'] = nodes['next'] = -1  # assume -1 as a nullptr
        
        # build chunk heads
        heads = []
        current = 0
        for chunk in data:
            if not chunk: continue
            
            # set curr index as the start of a new chunk
            start = current
            heads.append(start)
            
            # build the linked list for this chunk
            clen = len(chunk)
            for i, token in enumerate(chunk):
                nodes[current]['val'] = token
                if i > 0: nodes[current]['prev'] = current - 1
                if i < clen - 1: nodes[current]['next'] = current + 1
                current += 1
        return nodes, heads
        
    def train(self, data_path: str, vocab_size: int, debug_hook: int = 100):
        # read and encode training data
        print("Reading data...")
        with open(data_path, "r", encoding="utf-8") as f: text = f.read()
        data = [list(chunk.encode("utf-8")) for chunk in GPT_REG_SPLIT.findall(text)]
        
        # building training chunks
        print("Building chunks...")
        nodes, chunk_heads = self._build_training_chunks(data)
        
        # initialize pair data
        print("Initializing pair data...")
        pair_stats, pair_positions = self._pair_stats(nodes, chunk_heads)
        
        # build max heap using pair data
        heap = [(-count, pair) for pair, count in pair_stats.items()]
        heapify(heap)

        # start training
        print("Starting training...")
        num_merges = vocab_size - 256
        for merge in range(num_merges):
            # remove stale entries from the heap
            while heap:
                peek = heap[0]
                if -peek[0] == pair_stats.get(peek[1], 0): break
                heappop(heap)
            if not heap: break
            
            # find the most common pair globally
            count, token = heappop(heap)
            if count == 0: break  # make sure the count isn't 0
            token_id = 256 + merge
            
            # store merge info
            a, b = token
            self.vocab[token_id] = self.vocab[a] + self.vocab[b]
            self.merges[token] = token_id
            
            # apply merge to all chunks
            for i in pair_positions[token]:
                # verify this token still exists
                next = nodes[i]['next']
                if next == -1 or nodes[i]['val'] != a or nodes[next]['val'] != b:
                    continue
                self._merge_node(nodes, i, token_id, pair_stats, pair_positions, heap)

            # zero out merged pair count
            pair_stats[token] = 0
            pair_positions[token].clear()
            
            # debugging prints
            if merge % debug_hook == 0: print(f"Merge {merge}: {token} → {token_id}")
        
        print(f'Finished training with {256 + merge} total vocabulary size.')
    
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

    def save(self, path: str = './tkz.pkl'):
        with open(path, "wb") as f:
            pickle.dump({
                "vocab": self.vocab,
                "merges": self.merges
            }, f)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f: data = pickle.load(f)
        
        tok = cls()
        tok.vocab = data["vocab"]
        tok.merges = data["merges"]
        print(f"Tokenizer loaded from {path}")
        return tok
