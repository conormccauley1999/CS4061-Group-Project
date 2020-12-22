from constants import *
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import plot_confusion_matrix, roc_curve, auc
import time
import matplotlib.pyplot as plt
from itertools import cycle
from scipy import interp


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


def select_C(X, Y):
    means, stds, times = [], [], []
    alpha_range = [0.01, 0.1, 1, 10, 100]
    for a in alpha_range:
        start = time.time()
        model = MLPClassifier(max_iter=500, alpha=a)
        scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
        end = time.time()
        means.append(scores.mean())
        stds.append(scores.std())
        times.append(end-start)
        print(f'alpha = {a} >> ({means[-1]}, {stds[-1]}) >> {times[-1]}s')


def select_hl(X, Y):
    means, stds, times = [], [], []
    hl_range = [(5,)(10,)]
    for hl in hl_range:
        start = time.time()
        model = MLPClassifier(max_iter=500, hidden_layer_sizes=hl, alpha=1)
        scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
        end = time.time()
        means.append(scores.mean())
        stds.append(scores.std())
        times.append(end - start)
        print(f'hidden_layers = {hl} >> ({means[-1]}, {stds[-1]}) >> {times[-1]}s')


def roc(X_train,y_train, z):
    if z==1:
        classes = [1,2,3,4,5,6,7]
    else:
        classes = [1,2,3]

    y_train = label_binarize(y_train, classes=classes)
    n_classes = y_train.shape[1]

    model = MLPClassifier(max_iter=500, hidden_layer_sizes=(5,), alpha=1)
    y_score = model.fit(X_train, y_train).predict(X_train)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_train[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_train.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'yellow', 'lime', 'crimson'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def mlp(X, Y):
    start = time.time()
    a = 0.1
    clf = MLPClassifier(random_state=1, max_iter=500, hidden_layer_sizes=41, alpha=a)
    clf.fit(X, Y.ravel())
    y_pred = clf.predict(X)
    end = time.time()
    print(classification_report(Y.ravel(), y_pred))
    print(confusion_matrix(Y.ravel(), y_pred))
    scores = cross_val_score(clf, X, Y, cv=5, scoring='accuracy')
    print(f'alpha = {a} >> ({scores.mean()}, {scores.std()})')

    print("−−−−−−−−−−−ExecutionTime: % fs −−−−−−−−−−−−−" % (end - start))
    return

def confusion_matrix(X,Y):
    model = MLPClassifier(max_iter=500, hidden_layer_sizes=(5,), alpha=1).fit(X,Y)
    plot_confusion_matrix(model, X, Y)
    plt.show()

def main():
    all_indexes = list(range(len(QUESTIONS)))
    X_std, Y_std = read_data(DATA_FILE_STANDARD, all_indexes)
    X_3c, Y_3c = read_data(DATA_FILE_3_CLASS, all_indexes)

    # Cross Validation
    # select_hl(X_std, Y_std)
    # select_hl(X_3c, Y_3c)
    # select_c(X_std,Y_std)
    # select_c(X_3c, Y_3c)

    # ROC for 3 class and 7 class
    #roc(X_3c, Y_3c, 0)
    #roc(X_std, Y_std, 1)

    # Confusion Matrix
    confusion_matrix(X_std, Y_std)
    confusion_matrix(X_3c, Y_3c)

main()

