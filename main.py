from tokenizer import Tokenizer


test = Tokenizer()
test.train("test_data.txt", 512)
print(val := test.encode('HOW MUCH fire are theresadwadbj"kndw1hbs9218bn*&2199bSBmzm in woods that woodchucks cant chuck? btw my name is pri and i have an rtx3090'))
print(test.decode(val))
test.save()

test.train("test_data.txt", 256)
