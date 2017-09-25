from utils import complaint_to_words


class Lang:
    def __init__(self, complaints, labels):
        self.complaints = complaints
        self.labels = labels
        self.word2index = dict()
        self.index2word = dict()
        self.label2index = dict()
        self.index2label = dict()
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

    def create_label_index(self):
        target = list(self.labels.unique())
        for index, value in enumerate(target):
            self.label2index[value] = index
            self.index2label[index] = value
