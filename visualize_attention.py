import torch

import matplotlib.pyplot as plt

from models.attention import Attention


batch_size = 1
sequence_length = 5
hidden_dim = 8

hidden = torch.randn(
    batch_size,
    hidden_dim
)

encoder_outputs = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim
)

attention = Attention()

context_vector, attention_weights = attention(
    hidden,
    encoder_outputs
)

weights = attention_weights[0].detach().numpy()

plt.imshow([weights])

plt.colorbar()

plt.title("Attention Scores")

plt.xlabel("Sequence Position")

plt.show()