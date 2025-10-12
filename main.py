from tokenizer import Tokenizer


test = Tokenizer()
test.register_special_tokens([
    '< |bos>', '< |eos>', '< |unk>',
    '< |pad>', '< |cls>', '< |mask>'
])
test.train("test_data.txt", 512)

#print(test._vocab)
print(val := test.encode('< |bos>HOW MUCH fire are theresadwadbj"kndw1hbs9218bn*&2199bSBmzm in woods that woodchucks cant chuck?< |eos>< |bos>btw my name is pri and i have an rtx3090< |eos>'))
print(test.decode(val))
test.save()
