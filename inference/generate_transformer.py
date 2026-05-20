import torch

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


texts = texts[:2000]


tokenizer = SimpleTokenizer()

tokenizer.build_vocab(texts)


VOCAB_SIZE = tokenizer.vocab_size

EMBEDDING_DIM = 256

NUM_HEADS = 4

HIDDEN_DIM = 512

NUM_LAYERS = 2

MAX_SEQ_LENGTH = 109


model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_seq_length=MAX_SEQ_LENGTH
)


model.load_state_dict(
    torch.load(
        "checkpoints/transformer_model.pth"
    )
)

model.eval()


prompt = "hello"


encoded_prompt = tokenizer.encode(
    prompt
)


input_tensor = torch.tensor(
    [encoded_prompt]
)


with torch.no_grad():

    output = model(input_tensor)

    logits = output[0, -1]

    temperature = 0.8

    logits = logits / temperature

    top_k = 20

    top_logits, top_indices = torch.topk(
        logits,
        top_k
    )

    filtered_probabilities = torch.softmax(
        top_logits,
        dim=-1
    )

    sampled_index = torch.multinomial(
        filtered_probabilities,
        num_samples=1
    ).item()

    predicted_token = top_indices[
        sampled_index
    ].item()

    generated_word = tokenizer.index_to_word.get(
        predicted_token,
        "<UNK>"
    )


print("Prompt:", prompt)

print(
    "Generated:",
    generated_word
)