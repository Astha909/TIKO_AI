import torch

from tokenizer.bpe_tokenizer import BPETokenizer

from models.attention import Attention


texts = [
    "hello world",
    "chatbot chatting",
    "friendly chatbot",
    "advanced tokenizer system"
]

tokenizer = BPETokenizer()

tokenizer.train(
    texts,
    num_merges=15
)

test_word = "chatbot"

encoded_tokens = tokenizer.encode(
    test_word
)

print("Encoded Tokens:")
print(encoded_tokens)


batch_size = 1
sequence_length = len(encoded_tokens)
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

print("\nContext Vector Shape:")
print(context_vector.shape)

print("\nAttention Weights:")
print(attention_weights)