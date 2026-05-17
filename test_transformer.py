import torch

from models.transformer import TransformerModel


VOCAB_SIZE = 1000
EMBEDDING_DIM = 128
NUM_HEADS = 4
HIDDEN_DIM = 256
NUM_LAYERS = 2
MAX_SEQ_LENGTH = 50


model = TransformerModel(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    num_heads=NUM_HEADS,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    max_seq_length=MAX_SEQ_LENGTH
)


sample_input = torch.randint(
    0,
    VOCAB_SIZE,
    (2, 10)
)

output = model(sample_input)

print(output.shape)