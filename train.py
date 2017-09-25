from preprocessing import Lang
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split


df = pd.read_csv('input/complaints.csv')


def create_lang():
    complaints = df['Consumer complaint narrative']
    labels = df['Product']
    lang = Lang(complaints, labels)
    lang.create_index()
    lang.create_label_index()
    return lang.word2index, lang.index2word, lang.label2index, lang.index2label


word2index, index2word, label2index, index2label = create_lang()


def index_complaint():
    data, target = list(), list()
    for comp, label in zip(df['Consumer complaint narrative'], df['Product']):
        index_comp = [word2index[word] for word in comp]
        data.append(index_comp)
        target.append(label2index[label])
    data = np.array(data)
    target = np.array(target)
    return data, target


data, target = index_complaint()
target_binary = to_categorical(target)

X_train, X_test, y_train, y_test = train_test_split(data, target_binary, test_size=0.4, random_state=32)
