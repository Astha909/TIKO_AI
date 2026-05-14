from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from models.lstm_model import LSTMModel
from tokenizer.tokenizer import SimpleTokenizer


with open(
    "dataset/humor_dataset.txt",
    "r",
    encoding="utf-8"
) as file:
    texts = file.readlines()


tokenizer = SimpleTokenizer()

tokenizer.build_vocab(texts)


encoded_data = []

for text in texts:

    encoded_text = tokenizer.encode(text)

    if (
        len(encoded_text) > 1
        and len(encoded_text) <= 50
    ):

        encoded_data.append(encoded_text)

encoded_data = encoded_data[:5000]

input_sequences = []
target_sequences = []

for sequence in encoded_data:

    input_seq = sequence[:-1]
    target_seq = sequence[1:]

    input_sequences.append(input_seq)
    target_sequences.append(target_seq)


input_tensors = [
    torch.tensor(seq)
    for seq in input_sequences
]

target_tensors = [
    torch.tensor(seq)
    for seq in target_sequences
]


input_padded = pad_sequence(
    input_tensors,
    batch_first=True,
    padding_value=0
)

target_padded = pad_sequence(
    target_tensors,
    batch_first=True,
    padding_value=0
)


print(input_padded.shape)
print(target_padded.shape)


VOCAB_SIZE = tokenizer.vocab_size
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2


model = LSTMModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS
)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


output = model(input_padded)

print(output.shape)

output = output.reshape(
    -1,
    VOCAB_SIZE
)

target_padded = target_padded.reshape(-1)

loss = criterion(
    output,
    target_padded
)

print(loss.item())

EPOCHS = 50

for epoch in range(EPOCHS):

    output = model(input_padded)

    output = output.reshape(
        -1,
        VOCAB_SIZE
    )

    targets = target_padded.reshape(-1)

    loss = criterion(
        output,
        targets
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}"
    )

torch.save(
    model.state_dict(),
    "checkpoints/lstm_model.pth"
)