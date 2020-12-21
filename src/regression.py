from constants import *
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, RidgeClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    return indexes

def lin_reg_cv(X, Y):
    means, stds = [], []
    C_range = [0.01, 0.1, 1, 10, 100, 1000]
    for C in C_range:
        model = Lasso(alpha=1/(2*C)).fit(X, Y)
        scores = cross_val_score(model, X, Y, cv=5, scoring='neg_mean_squared_error')
        means.append(scores.mean())
        stds.append(scores.std())
        print(f'C = {C} >> ({means[-1]}, {stds[-1]})')
    return

def log_reg_cv(X, Y):
    means, stds = [], []
    C_range = [0.1, 1, 10]
    for C in C_range:
        model = LogisticRegression(C=C, penalty='l2', solver='saga', max_iter=1000).fit(X, Y)
        scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
        means.append(scores.mean())
        stds.append(scores.std())
        print(f'C = {C} >> ({means[-1]}, {stds[-1]})')
    #Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)
    #model = LogisticRegression(C=1, penalty='l2', solver='lbfgs', max_iter=2000).fit(Xtr, Ytr)
    #Ypr = model.predict(Xte)
    #print(confusion_matrix(Yte, Ypr))
    #model = LogisticRegression(C=C, solver='saga', max_iter=2000).fit(X, Y)
    return

def main():
    all_indexes = list(range(len(QUESTIONS)))
    X_std, Y_std = read_data(DATA_FILE_STANDARD, all_indexes)
    X_3c, Y_3c = read_data(DATA_FILE_3_CLASS, all_indexes)
    #lin_reg_cv(X, Y)
    log_reg_cv(X_std, Y_std)
    log_reg_cv(X_3c, Y_3c)

main()
