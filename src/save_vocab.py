import json
from tokenizer import Tokenizer

tok = Tokenizer.load('./snapshots/tkz.pkl')

vocab_str = {k: v.decode("latin-1") for k, v in tok._vocab.items()}
with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab_str, f, indent=4, ensure_ascii=False)
