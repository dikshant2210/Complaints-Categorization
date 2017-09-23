from nltk.tokenize import RegexpTokenizer
import pandas as pd
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical

numpy.random.seed(7)

df = pd.read_csv('complaints.csv')

def complaint_to_words(comp):
    words = RegexpTokenizer('\w+').tokenize(comp)
    num = RegexpTokenizer('\d+').tokenize(comp)
    words = [w for w in words if w not in num]
    words = [w.lower() for w in words]
    return words

all_words = list()
for comp in df['Consumer complaint narrative']:
    for w in complaint_to_words(comp):
        all_words.append(w)
print "List of all words created, total number of different words: %0.2f%." % len(set(all_words))

index_dict = dict()
count = 0
for word in set(all_words):
    index_dict[word] = count
    count += 1
print "All words indexed."

del all_words

data_list = list()
for comp in df['Consumer complaint narrative']:
    l = list()
    for w in complaint_to_words(comp):
        l.append(index_dict[w])
    data_list.append(l)
print "Complaint data indexed."

del index_dict

le = preprocessing.LabelEncoder()
le.fit(df['Product'])
df['Target'] = le.transform(df['Product'])
y_binary = to_categorical(df['Target'].values)
print "Target variable data transformed."

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(numpy.array(data_list), y_binary,
    test_size=0.4, random_state=0)
print "Cross validation split done."

del data_list, le, df, y_binary

# truncate and pad input sequences
max_review_length = 750
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the model
print "Creating the model...."
top_words = 62943
embedding_vecor_length = 100
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(GRU(32, dropout_W=0.2, dropout_U=0.2))
model.add(Dropout(0.2))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print "Model Created!"
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=1, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%." % (scores[1]*100))
