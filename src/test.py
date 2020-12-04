from constants import *
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    C = 10
    model = Lasso(alpha=1/(2*C)).fit(X, Y)
    coefs = [*enumerate(list(model.coef_))]
    coefs.sort(key=lambda x: x[1], reverse=True)
    for i, v in coefs:
        tag = 'Left' if v < 0 else 'Right' if v > 0 else 'Neither'
        print('"%.3f","%s","%s"' % (abs(v), tag, QUESTIONS[i]))
    return

data = pd.read_csv('dataset.csv').values.tolist()
main(data)
