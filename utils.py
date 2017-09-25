from nltk.tokenize import RegexpTokenizer


def complaint_to_words(comp):
    words = RegexpTokenizer('\w+').tokenize(comp)
    num = RegexpTokenizer('\d+').tokenize(comp)
    words = [w.lower() for w in words if w not in num]
    return words
