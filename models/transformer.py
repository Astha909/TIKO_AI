import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):

        super().__init__()

        self.embedding_dim = embedding_dim

        self.num_heads = num_heads

        self.head_dim = (
            embedding_dim // num_heads
        )

        self.query = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.key = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.value = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.fc_out = nn.Linear(
            embedding_dim,
            embedding_dim
        )

    def forward(self, x):

        batch_size = x.shape[0]

        seq_length = x.shape[1]

        Q = self.query(x)

        K = self.key(x)

        V = self.value(x)

        Q = Q.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_length,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        attention_scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        )

        attention_scores = (
            attention_scores
            / (self.head_dim ** 0.5)
        )


        mask = torch.tril(
            torch.ones(
                seq_length,
                seq_length
            )
        ).to(x.device)

        attention_scores = attention_scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        out = torch.matmul(
            attention_weights,
            V
        )

        out = out.transpose(1, 2).contiguous()

        out = out.reshape(
            batch_size,
            seq_length,
            self.embedding_dim
        )

        out = self.fc_out(out)

        return out


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        embedding_dim,
        max_seq_length
    ):

        super().__init__()

        pe = torch.zeros(
            max_seq_length,
            embedding_dim
        )

        position = torch.arange(
            0,
            max_seq_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(
                0,
                embedding_dim,
                2
            ) * (
                -torch.log(
                    torch.tensor(10000.0)
                ) / embedding_dim
            )
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        x = x + self.pe[:, :x.size(1)]

        return x
    

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        hidden_dim
    ):

        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embedding_dim,
            num_heads
        )

        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(
                embedding_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                embedding_dim
            )
        )

        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

    def forward(self, x):

        attention_output = self.attention(x)

        x = self.norm1(
            x + attention_output
        )

        feed_forward_output = (
            self.feed_forward(x)
        )

        x = self.norm2(
            x + feed_forward_output
        )

        return x

class TransformerModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        num_heads,
        hidden_dim,
        num_layers,
        max_seq_length
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        self.positional_encoding = PositionalEncoding(
            embedding_dim,
            max_seq_length
        )

        self.attention = MultiHeadSelfAttention(
            embedding_dim,
            num_heads
        )

        self.transformer_blocks = nn.ModuleList(

            [
                TransformerBlock(
                    embedding_dim,
                    num_heads,
                    hidden_dim
                )

                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(
            embedding_dim,
            vocab_size
        )
    def forward(self, x):

        embedded = self.embedding(x)

        embedded = self.positional_encoding(
            embedded
        )

        attention_output = self.attention(
            embedded
        )

        output = embedded

        for block in self.transformer_blocks:

            output = block(output)

            logits = self.fc_out(output)

            return logits   