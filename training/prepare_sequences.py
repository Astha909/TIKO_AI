import torch # type: ignore
from tokenizer.tokenizer import SimpleTokenizer
from preprocessing.clean_text import clean_text

dataset_path = "dataset/humor_dataset.txt"

texts = []

with open(dataset_path, "r", encoding="utf-8") as file:
    lines = file.readlines()[1:]

for line in lines:
    category, user_input, response = line.strip().split("|")

    texts.append(clean_text(user_input))
    texts.append(clean_text(response))

tokenizer = SimpleTokenizer()

tokenizer.build_vocab(texts)

encoded_sequences = [
    tokenizer.encode(text)
    for text in texts
]

print("Vocabulary:")
print(tokenizer.word_to_index)

print("\nEncoded Sequences:")
print(encoded_sequences)

training_pairs = []

for sequence in encoded_sequences:
    for i in range(1, len(sequence)):
        input_sequence = sequence[:i]
        target_word = sequence[i]

        training_pairs.append((input_sequence, target_word))

print("\nTraining Pairs:")

for pair in training_pairs:
    print(pair)

input_tensors = []
target_tensors = []

for input_sequence, target_word in training_pairs:
    input_tensor = torch.tensor(input_sequence)
    target_tensor = torch.tensor(target_word)

    input_tensors.append(input_tensor)
    target_tensors.append(target_tensor)

print("\nFirst Input Tensor:")
print(input_tensors[0])

print("\nFirst Target Tensor:")
print(target_tensors[0])