from collections import Counter


class BPETokenizer:

    def __init__(self):

        self.vocab = {}

        self.merges = {}

    def build_initial_vocab(self, texts):

        words = []

        for text in texts:

            words.extend(
                text.strip().split()
            )

        self.vocab = Counter(words)

    def get_stats(self):

        pairs = Counter()

        for word, freq in self.vocab.items():

            symbols = word.split()

            for i in range(len(symbols) - 1):

                pairs[
                    (
                        symbols[i],
                        symbols[i + 1]
                    )
                ] += freq

        return pairs
    
    def merge_vocab(self, pair):

        merged_vocab = {}

        bigram = " ".join(pair)

        replacement = "".join(pair)

        for word in self.vocab:

            new_word = word.replace(
                bigram,
                replacement
            )

            merged_vocab[new_word] = self.vocab[word]

        self.vocab = merged_vocab

        self.merges[pair] = replacement

    def train(self, texts, num_merges=10):

        words = []

        for text in texts:

            for word in text.strip().split():

                words.append(
                    " ".join(list(word))
                )

        self.vocab = Counter(words)

        for _ in range(num_merges):

            pairs = self.get_stats()

            if not pairs:
                break

            best_pair = max(
                pairs,
                key=pairs.get
            )

            self.merge_vocab(best_pair)

    def encode(self, word):

        tokens = list(word)

        while len(tokens) > 1:

            pairs = [
                (
                    tokens[i],
                    tokens[i + 1]
                )
                for i in range(len(tokens) - 1)
            ]

            merge_found = False

            for pair in pairs:

                if pair in self.merges:

                    i = pairs.index(pair)

                    tokens = (
                        tokens[:i]
                        + [self.merges[pair]]
                        + tokens[i + 2:]
                    )

                    merge_found = True

                    break

            if not merge_found:
                break

        return tokens
    
if __name__ == "__main__":

    with open(
        "dataset/balanced_dataset.txt",
        "r",
        encoding="utf-8"
    ) as file:

        texts = file.readlines()

    tokenizer = BPETokenizer()

    tokenizer.train(
        texts,
        num_merges=10
    )

    encoded = tokenizer.encode(
        "chatting"
    )

    print("Encoded Tokens:")

    print(encoded)

    print("\nLearned Merges:")

    for pair, merge in tokenizer.merges.items():

        print(pair, "->", merge)