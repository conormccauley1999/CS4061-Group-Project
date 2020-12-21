import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import SVC
import time

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
    start = time.time()
    clf = MLPClassifier(random_state=1, max_iter=150, alpha=0.001)
    clf.fit(X, Y.ravel())
    y_pred = clf.predict(X)

    end = time.time()

    print( "−−−−−−−−−−−ExecutionTime: % fs −−−−−−−−−−−−−" % (end - start))
    print(classification_report(Y, y_pred))
    print(confusion_matrix(Y, y_pred))
    print("Accuracy %f" %(accuracy_score(Y, y_pred)))

data = pd.read_csv('dataset.csv').values.tolist()
main(data)
