import torch # pyright: ignore[reportMissingImports]

import torch.nn as nn

import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self):

        super().__init__()
    def forward(self, hidden, encoder_outputs):

        attention_scores = torch.bmm(
            encoder_outputs,
            hidden.unsqueeze(2)
        ).squeeze(2)

        attention_weights = F.softmax(
            attention_scores,
            dim=1
        )

        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs
        ).squeeze(1)

        return context_vector, attention_weights
    


if __name__ == "__main__":

    batch_size = 2
    sequence_length = 4
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

    print("Context Vector Shape:")
    print(context_vector.shape)

    print("Attention Weights Shape:")
    print(attention_weights.shape)
