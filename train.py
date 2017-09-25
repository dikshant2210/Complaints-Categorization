from preprocessing import Lang
import pandas as pd
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from models.models import get_gru_model
from utils import complaint_to_words
import pickle as pkl


embedding_vector_length = 256
hidden_size = 128
max_length = 600
df = pd.read_csv('input/complaints.csv')


def create_lang():
    complaints = df['Consumer complaint narrative']
    labels = df['Product']
    lang = Lang(complaints, labels)
    lang.create_index()
    lang.create_label_index()
    return lang.word2index, lang.index2word, lang.label2index, lang.index2label, lang.vocabulary


word2index, index2word, label2index, index2label, vocabulary = create_lang()
num_classes = len(label2index)


def index_complaint():
    data, target = list(), list()
    print(num_classes)
    print(len(word2index), len(label2index))
    for comp, label in zip(df['Consumer complaint narrative'], df['Product']):
        index_comp = list()
        for word in complaint_to_words(comp):
            try:
                index_comp.append(word2index[word])
            except KeyError:
                index_comp.append(0)
        data.append(index_comp)
        target.append(label2index[label])
    data = np.array(data)
    target = np.array(target)
    return data, target


data, target = index_complaint()
target_binary = to_categorical(target)

X_train, X_val, y_train, y_val = train_test_split(data, target_binary, test_size=0.4, random_state=32)
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_val = sequence.pad_sequences(X_val, maxlen=max_length)

model = get_gru_model(vocabulary, embedding_vector_length, max_length, hidden_size, num_classes)
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_val, y_val), nb_epoch=3, batch_size=128)

# saving the model
model.save_weights('weights/weights.hd5')
with open('weights/word2index.pkl') as file:
    pkl.dump(word2index, file)
with open('weights/index2word.pkl') as file:
    pkl.dump(index2word, file)
with open('weights/label2index.pkl') as file:
    pkl.dump(label2index, file)
with open('weights/index2label.pkl') as file:
    pkl.dump(index2label, file)
