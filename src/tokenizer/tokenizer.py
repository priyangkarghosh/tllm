import pickle
import regex as re
import numpy as np
from heapq import *
from collections import Counter, defaultdict
from helpers import Timer

GPT_REG_SPLIT = re.compile(
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
)

class Tokenizer:
    def __init__(self) -> None:
        self._vocab = {i: bytes([i]) for i in range(256)}
        self._merges: dict[int | tuple, int] = {}
        
        self._special_tokens: dict[str, int] = {}
        self._special_token_pattern: re.Pattern = re.compile('()')
    
    @property
    def vocab_size(self):
        return len(self._vocab)
    
    def register_special_tokens(self, tokens: list[str]) -> None:
        # update vocab with special tokens
        start = max(self._vocab.keys(), default=-1) + 1
        for token_id, token_str in enumerate(tokens, start=start):
            self._vocab[token_id] = token_str.encode("utf-8")
            self._special_tokens[token_str] = token_id
        self._update_special_token_pattern()
    
    def _update_special_token_pattern(self):
        self._special_token_pattern = '(' + '|'.join(map(re.escape, self._special_tokens.keys())) + ')'
        
    def _pair_stats(self, nodes: np.ndarray, chunk_heads: np.ndarray) -> tuple[Counter, defaultdict[list]]:
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
   
    def _merge(
        self, 
        data: list[int], 
        pair: tuple[int, int], 
        token_id: int
    ) -> list[int]:
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
    ) -> None:
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
    
    def _build_training_chunks(self, data: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        # calculate chunk lengths
        chunk_lengths = np.fromiter((len(c) for c in data), dtype=np.int32, count=len(data))
        
        # calculate chunk offsets
        chunk_offsets = np.empty_like(chunk_lengths)
        np.cumsum(chunk_lengths[:-1], out=chunk_offsets[1:])
        chunk_offsets[0] = 0
        
        # calculate token count
        token_count = int(chunk_offsets[-1] + chunk_lengths[-1])
        
        # pre-allocate nodes array
        dtype = [('val', np.int32), ('prev', np.int32), ('next', np.int32)]
        nodes = np.empty(token_count, dtype=dtype)
        
        # flatten all chunks to set values
        nodes['val'][:] = np.concatenate(data)

        # compute prev/next for all nodes at once
        nodes['prev'] = np.arange(-1, token_count - 1, dtype=np.int32)
        nodes['next'] = np.arange(1, token_count + 1, dtype=np.int32)
        
        # fix chunk boundaries
        nodes['prev'][chunk_offsets] = -1
        chunk_ends = chunk_offsets + chunk_lengths - 1
        nodes['next'][chunk_ends] = -1
        
        # return nodes and chunk heads
        return nodes, chunk_offsets
    
    def train_direct(self, data_path: str, vocab_size: int, debug_hook: int = 100) -> None:
        # read and encode training data
        print("Reading data...")
        with open(data_path, "r", encoding="utf-8") as f: 
            text = f.read()
        
        # train using read data
        self.train(text, vocab_size, debug_hook)
        
    def train(self, text: str, vocab_size: int, debug_hook: int = 100) -> None:      
        data = []
        segments = re.split(self._special_token_pattern, text)
        for segment in segments:
            if segment in self._special_tokens: 
                data.append([self._special_tokens[segment]])
            else: 
                data.extend([
                    list(chunk.encode("utf-8")) 
                    for chunk in GPT_REG_SPLIT.findall(segment)
                ])
        
        # make sure data exists
        if not data: return

        # building training chunks
        print("Building chunks...")
        with Timer() as t: nodes, chunk_heads = self._build_training_chunks(data)
        print(f"Built chunks in {t.elapsed:.4f} seconds")
        
        # initialize pair data
        print("Initializing pair data...")
        pair_stats, pair_positions = self._pair_stats(nodes, chunk_heads)
        
        # build max heap using pair data
        heap = [(-count, pair) for pair, count in pair_stats.items()]
        heapify(heap)

        # start training
        print("Starting training...")
        merge_start = self.vocab_size
        num_merges = vocab_size - merge_start
        if num_merges <= 0: 
            print("Current vocabulary length exceeds given target")
            return
        
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
            token_id = merge_start + merge
            
            # store merge info
            a, b = token
            self._vocab[token_id] = self._vocab[a] + self._vocab[b]
            self._merges[token] = token_id
            
            # apply merge to all chunks
            for i in pair_positions[token]:
                # verify this token still exists
                next = nodes[i]['next']
                if next == -1 or nodes[i]['val'] != a or nodes[next]['val'] != b: continue
                self._merge_node(nodes, i, token_id, pair_stats, pair_positions, heap)

            # zero out merged pair count
            pair_stats[token] = 0
            pair_positions[token].clear()
            
            # debugging prints
            if merge % debug_hook == 0: print(f"Merge {merge}: {token} → {token_id}")
        
        print(f'Finished training with {256 + merge} total vocabulary size.')

    def decode(self, data: list[int]) -> str:
        tokens = b"".join(self._vocab[token_id] for token_id in data)
        return tokens.decode("utf-8", errors="replace")
   
    def encode(self, text: str) -> list[int]:
        # split text into chunks
        chunks = []
        segments = re.split(self._special_token_pattern, text)
        for segment in segments:
            if segment in self._special_tokens: chunks.append(segment)
            else: chunks.extend(GPT_REG_SPLIT.findall(segment))
                    
        tokens = []
        for chunk in chunks:
            # check if this is a special token and handle it as such
            if chunk in self._special_tokens: 
                tokens.append(self._special_tokens[chunk])
                continue
            
            # otherwise, treat like a normal token
            chunk_tokens = list(chunk.encode("utf-8"))
            
            # apply merges to this chunk
            while True:
                pairs = list(zip(chunk_tokens, chunk_tokens[1:]))
                if not pairs: break
                
                # find the pair with the earliest merge
                token = min(pairs, key=lambda p: self._merges.get(p, float('inf')))
                if token not in self._merges: break
                
                token_id = self._merges[token]
                chunk_tokens = self._merge(chunk_tokens, token, token_id)
            tokens.extend(chunk_tokens)
        return tokens

    def save(self, path: str = './tkz.pkl') -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "vocab": self._vocab,
                "merges": self._merges,
                "special_tokens": self._special_tokens
            }, f)
        print(f"Tokenizer saved to {path}")

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        with open(path, "rb") as f: data = pickle.load(f)
        
        tok = cls()
        tok._vocab = data["vocab"]
        tok._merges = data["merges"]
        tok._special_tokens = data["special_tokens"]
        tok._update_special_token_pattern()
        print(f"Tokenizer loaded from {path}")
        return tok
