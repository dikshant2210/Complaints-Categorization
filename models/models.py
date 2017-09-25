from keras.layers import GRU, Dropout, Embedding, Dense
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy


def get_gru_model(vocabulary, embedding_vector_length, max_length, hidden_size, num_classes):
    model = Sequential()
    model.add(Embedding(vocabulary, embedding_vector_length, input_length=max_length))
    model.add(Dropout(0.2))
    model.add(GRU(hidden_size))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=[categorical_accuracy])
    return model
