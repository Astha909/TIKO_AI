from torch.nn.utils.rnn import pad_sequence

import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from models.transformer import TransformerModel
from tokenizer.tokenizer import SimpleTokenizer


with open(
    "dataset/balanced_dataset.txt",
    "r",
    encoding="utf-8"
) as file:

    texts = file.readlines()


texts = texts[:1000]


tokenizer = SimpleTokenizer()

tokenizer.build_vocab(texts)


encoded_data = []


for text in texts:

    encoded_text = tokenizer.encode(text)

    if len(encoded_text) > 1:

        encoded_data.append(
            encoded_text
        )


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


VOCAB_SIZE = tokenizer.vocab_size

EMBEDDING_DIM = 128

NUM_HEADS = 4

HIDDEN_DIM = 256

NUM_LAYERS = 2

MAX_SEQ_LENGTH = (
    input_padded.shape[1]
)


model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_seq_length=MAX_SEQ_LENGTH
)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


print(input_padded.shape)

print(target_padded.shape)

output = model(input_padded)

print(output.shape)


output = output.reshape(
    -1,
    VOCAB_SIZE
)

targets = target_padded.reshape(-1)


loss = criterion(
    output,
    targets
)

print(loss.item())

EPOCHS = 10


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
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Loss: {loss.item():.4f}"
    )

torch.save(
    model.state_dict(),
    "checkpoints/transformer_model.pth"
)