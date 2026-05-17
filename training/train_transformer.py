from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(device)


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


texts = texts[:2000]


tokenizer = SimpleTokenizer()

tokenizer.build_vocab(texts)


encoded_data = []

MAX_LENGTH = 128


for text in texts:

    encoded_text = tokenizer.encode(text)

    encoded_text = encoded_text[:MAX_LENGTH]

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


dataset = TensorDataset(
    input_padded,
    target_padded
)

BATCH_SIZE = 32


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


VOCAB_SIZE = tokenizer.vocab_size

EMBEDDING_DIM = 256

NUM_HEADS = 4

HIDDEN_DIM = 512

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

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


print(input_padded.shape)

print(target_padded.shape)


EPOCHS = 20


for epoch in range(EPOCHS):

    print(f"Starting Epoch {epoch+1}")

    total_loss = 0

    for batch_inputs, batch_targets in dataloader:

        batch_inputs = batch_inputs.to(device)

        batch_targets = batch_targets.to(device)

        output = model(batch_inputs)

        output = output.reshape(
            -1,
            VOCAB_SIZE
        )

        targets = batch_targets.reshape(-1)

        loss = criterion(
            output,
            targets
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    print(
        f"Epoch {epoch+1}/{EPOCHS}, "
        f"Loss: {avg_loss:.4f}"
    )


torch.save(
    model.state_dict(),
    "checkpoints/transformer_model.pth"
)