from utils import complaint_to_words


class Lang:
    def __init__(self, complaints):
        self.complaints = complaints
        self.word2index = dict()
        self.index2word = dict()
        self.vocabulary = 0

    def create_index(self):
        words = list()
        for comp in self.complaints:
            words += complaint_to_words(comp)
        words = set(words)
        self.vocabulary = len(words)
        for index, word in enumerate(words):
            self.index2word[index] = word
            self.word2index[word] = index
        return self.word2index, self.index2word
