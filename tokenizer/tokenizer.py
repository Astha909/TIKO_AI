class SimpleTokenizer:
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocab_size = 0
    def build_vocab(self, texts):
        unique_words = set()

        for text in texts:
            words = text.split()
            unique_words.update(words)

        for index, word in enumerate(sorted(unique_words)):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        self.vocab_size = len(self.word_to_index)
    def encode(self, text):
        words = text.split()

        return [
            self.word_to_index[word]
            for word in words
            if word in self.word_to_index
        ]
    def decode(self, indices):
        return " ".join(
            self.index_to_word[index]
            for index in indices
            if index in self.index_to_word
        )