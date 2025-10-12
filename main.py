from tokenizer import Tokenizer


test = Tokenizer("test_data.txt", 512)
test.train()
print(val := test.encode('HOW MUCH fire are there in woods that woodchucks cant chuck?'))
print(test.decode(val))
