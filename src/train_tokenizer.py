from model import GPTConfig, GPT
from tokenizer import Tokenizer

# init config
config = GPTConfig()

# train tokenizer
tok = Tokenizer()
tok.register_special_tokens(['< |bos>', '< |eos>'])
tok.train("./data/tiny_shakespeare.txt", config.vocab_size)
