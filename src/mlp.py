from constants import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
import time

DATA_FILE_STANDARD = '..\data\converted_std.csv'
DATA_FILE_3_CLASS = '..\data\converted_3c.csv'

def filter_data(data, index):
    return np.array([d[index] for d in data])


def read_data(data_file, indexes):
    data = pd.read_csv(data_file).values.tolist()
    cols = []
    for index in indexes:
        cols.append(filter_data(data, index).reshape(-1, 1))
    X = np.column_stack(tuple(cols))
    Y = filter_data(data, len(QUESTIONS))
    return X, Y


def cross_val(X, Y):
    means, stds = [], []
    alpha_range = [0.001, 0.01, 0.1]
    for a in alpha_range:
        model = MLPClassifier(random_state=1, max_iter=200, alpha=a)
        scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
        means.append(scores.mean())
        stds.append(scores.std())
        print(f'alpha = {a} >> ({means[-1]}, {stds[-1]})')


def mlp(X, Y):
    start = time.time()
    clf = MLPClassifier(random_state=1, max_iter=50, alpha=0.001)
    clf.fit(X, Y.ravel())
    y_pred = clf.predict(X)
    end = time.time()

    print( "−−−−−−−−−−−ExecutionTime: % fs −−−−−−−−−−−−−" % (end - start))
    print(classification_report(Y.ravel(), y_pred))
    print(confusion_matrix(Y.ravel(), y_pred))
    print("Accuracy %f" %(cross_val_score(clf, X, Y.ravel(), cv=3)))
    return


def main():
    all_indexes = list(range(len(QUESTIONS)))
    X_std, Y_std = read_data(DATA_FILE_STANDARD, all_indexes)
    X_3c, Y_3c = read_data(DATA_FILE_3_CLASS, all_indexes)
    # mlp(X_std, Y_std)
    # mlp(X_3c, Y_3c)
    cross_val(X_3c, Y_3c)

main()

