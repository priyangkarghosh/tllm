from tokenizer import Tokenizer


test = Tokenizer("test_data.txt", 512)
test.train()
print(val := test.encode('HOW MUCH fire are theresadwadbj"kndw1hbs9218bn*&2199bSBmzm in woods that woodchucks cant chuck?'))
print(test.decode(val))
