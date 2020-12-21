import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D


def get_features(data):
    def select_feature(data, ix):
        return [d[ix] for d in data]

    X = []
    for i in range(len(data[0]) - 1):
        X.append(np.array(select_feature(data, i)).reshape(-1, 1))
    X = np.column_stack(X)
    Y = np.array(select_feature(data, -1)).reshape(-1, 1)
    return X, Y


def main(data):
    X, Y = get_features(data)
    split = len(data)//2

    X_train, X_test =X[:split], X[split:]
    y_train, y_test = Y[:split], Y[split:]

    model = keras.Sequential()

    # Building the CNN Model
    model = Sequential()
    model.add(Conv1D(8, activation='relu', kernel_size=2, input_shape=(60, 1)))
    model.add(Conv1D(8, activation='relu', kernel_size=2))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(16, activation='relu', kernel_size=3))
    model.add(Conv1D(16, activation='relu', kernel_size=3))

    # Fitting the data onto model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train.ravel(), validation_data=(X_train, y_train.ravel()), epochs=2, batch_size=128, verbose=2)
    model.summary()

    scores = model.evaluate(X_test, y_test, verbose=0)
    # Displays the accuracy of correct sentiment prediction over test data
    print("Accuracy: %.2f%%" % (scores[1] * 100))

data = pd.read_csv('dataset.csv').values.tolist()
main(data)
