from utils import complaint_to_words
from nltk.probability import FreqDist


class Lang:
    def __init__(self, complaints, labels):
        self.complaints = complaints
        self.labels = labels
        self.word2index = dict()
        self.index2word = dict()
        self.label2index = dict()
        self.index2label = dict()
        self.vocabulary = 60000

    def create_index(self):
        words = list()
        for comp in self.complaints:
            words += complaint_to_words(comp)
        words = FreqDist(words)
        words = words.most_common(self.vocabulary)
        words = [word[0] for word in words]
        words.insert(0, 'UNK')
        for index, word in enumerate(words):
            self.index2word[index] = word
            self.word2index[word] = index

    def create_label_index(self):
        target = list(self.labels.unique())
        for index, value in enumerate(target):
            self.label2index[value] = index
            self.index2label[index] = value
