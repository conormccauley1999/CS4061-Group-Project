from constants import *
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, RidgeClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
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

def select_features(X, Y, C=1):
    model = Lasso(alpha=1/(2*C)).fit(X, Y)
    indexes = []
    for index, coef in enumerate(model.coef_):
        if coef != 0: indexes.append(index)

def mlp(X, Y):
    start = time.time()
    clf = MLPClassifier(random_state=1, max_iter=150, alpha=0.001)
    clf.fit(X, Y.ravel())
    y_pred = clf.predict(X)

    end = time.time()

    print( "−−−−−−−−−−−ExecutionTime: % fs −−−−−−−−−−−−−" % (end - start))
    print(classification_report(Y, y_pred))
    print(confusion_matrix(Y, y_pred))
    print("Accuracy %f" %(accuracy_score(Y, y_pred)))
    return

def main():
    all_indexes = list(range(len(QUESTIONS)))
    X_std, Y_std = read_data(DATA_FILE_STANDARD, all_indexes)
    X_3c, Y_3c = read_data(DATA_FILE_3_CLASS, all_indexes)
    #lin_reg_cv(X, Y)
    mlp(X_std, Y_std)
    mlp(X_3c, Y_3c)

main()

