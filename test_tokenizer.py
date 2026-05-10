from tokenizer.tokenizer import SimpleTokenizer

texts = [
    "hello human",
    "how are you",
    "tell me a joke"
]

tokenizer = SimpleTokenizer()

tokenizer.build_vocab(texts)

encoded = tokenizer.encode("hello human")

decoded = tokenizer.decode(encoded)

print("Vocabulary:", tokenizer.word_to_index)
print("Encoded:", encoded)
print("Decoded:", decoded)