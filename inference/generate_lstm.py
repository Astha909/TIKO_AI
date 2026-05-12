import torch

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

model.load_state_dict(
    torch.load(
        "checkpoints/lstm_model.pth"
    )
)

model.eval()

input_text = "funny hello yo"

encoded_input = tokenizer.encode(input_text)

input_tensor = torch.tensor(
    [encoded_input],
    dtype=torch.long
)

with torch.no_grad():

    output = model(input_tensor)

    predicted_token = torch.argmax(
        output[0, -1]
    ).item()

predicted_word = tokenizer.decode(
    [predicted_token]
)

print("Input:", input_text)
print("Predicted:", predicted_word)