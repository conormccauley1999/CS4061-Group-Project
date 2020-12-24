from constants import *
from scipy import interpolate
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import auc, classification_report, plot_confusion_matrix, roc_curve
from sklearn.model_selection import cross_validate, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_FILE_STANDARD = '..\data\converted_std.csv'
DATA_FILE_3_CLASS = '..\data\converted_3c.csv'

def filter_data(data, index):
    return np.array([d[index] for d in data])

def read_data(data_file):
    data = pd.read_csv(data_file).values.tolist()
    cols = []
    for index in range(len(QUESTIONS)):
        cols.append(filter_data(data, index).reshape(-1, 1))
    X = np.column_stack(tuple(cols))
    Y = filter_data(data, len(QUESTIONS))
    return X, Y

def cross_validation(X, Y, is_std=True):
    print('=== Baselines ===')
    C_rng_bl = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    model = DummyClassifier(strategy='most_frequent')
    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    blf_means = [scores.mean()] * len(C_rng_bl)
    blf_stds = [scores.std()] * len(C_rng_bl)
    print('Frequent >> (%.4f, %.4f)' % (blf_means[-1], blf_stds[-1]))
    model = DummyClassifier(strategy='uniform')
    scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
    blr_means = [scores.mean()] * len(C_rng_bl)
    blr_stds = [scores.std()] * len(C_rng_bl)
    print('Random >> (%.4f, %.4f)' % (blr_means[-1], blr_stds[-1]))

    print('=== Logistic Regression (L1) ===')
    lr1_means, lr1_stds = [], []
    C_rng_lr1 = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    for C in C_rng_lr1:
        start = time()
        model = LogisticRegression(C=C, penalty='l1', solver='saga', max_iter=500).fit(X, Y)
        scores = cross_validate(model, X, Y, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
        end = time()
        test_mean = scores['test_score'].mean()
        test_std = scores['test_score'].std()
        train_mean = scores['train_score'].mean()
        lr1_means.append(test_mean)
        lr1_stds.append(test_std)
        print('C = %f >> (%.4f, %.4f) >> (test=%.4f, train=%.4f) >> %ds' % (C, test_mean, test_std, test_mean, train_mean, round(end - start)))
    
    print('=== Logistic Regression (L2) ===')
    lr2_means, lr2_stds = [], []
    C_rng_lr2 = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    for C in C_rng_lr2:
        start = time()
        model = LogisticRegression(C=C, penalty='l2', solver='saga', max_iter=500).fit(X, Y)
        scores = cross_validate(model, X, Y, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
        end = time()
        test_mean = scores['test_score'].mean()
        test_std = scores['test_score'].std()
        train_mean = scores['train_score'].mean()
        lr2_means.append(test_mean)
        lr2_stds.append(test_std)
        print('C = %f >> (%.4f, %.4f) >> (test=%.4f, train=%.4f) >> %ds' % (C, test_mean, test_std, test_mean, train_mean, round(end - start)))
    
    print('=== Ridge Classifier ===')
    rc_means, rc_stds = [], []
    C_rng_rc = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for C in C_rng_rc:
        start = time()
        model = RidgeClassifier(alpha=1/(2*C)).fit(X, Y)
        scores = cross_validate(model, X, Y, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True)
        end = time()
        test_mean = scores['test_score'].mean()
        test_std = scores['test_score'].std()
        train_mean = scores['train_score'].mean()
        rc_means.append(test_mean)
        rc_stds.append(test_std)
        print('C = %f >> (%.4f, %.4f) >> (test=%.4f, train=%.4f) >> %ds' % (C, test_mean, test_std, test_mean, train_mean, round(end - start)))

    plt.errorbar(C_rng_lr1, lr1_means, yerr=lr1_stds)
    plt.errorbar(C_rng_lr2, lr2_means, yerr=lr2_stds)
    plt.errorbar(C_rng_rc, rc_means, yerr=rc_stds)
    plt.errorbar(C_rng_bl, blf_means, yerr=blf_stds)
    plt.errorbar(C_rng_bl, blr_means, yerr=blr_stds)
    plt.xticks(C_rng_bl)
    plt.xlabel('$C$')
    plt.ylabel('Mean accuracy score')
    plt.xscale('log')
    plt.legend(
        ['Logistic (L1)', 'Logistic (L2)', 'Ridge', 'Baseline (frequent)', 'Baseline (random)'],
        loc='lower right'
    )
    if is_std:
        plt.title('Cross-validation results for 7-class dataset')
    else:
        plt.title('Cross-validation results for 3-class dataset')
    plt.show()

def multiclass_roc_curves(X, Y, is_std=True):
    if is_std:
        classes = list(range(1, 8))
        legend = ['Far left', 'Left', 'Moderately left', 'Neither', 'Moderately right', 'Right', 'Far right', 'Micro-average', 'Random classifier']
    else:
        classes = list(range(1, 4))
        legend = ['Left', 'Centre', 'Right', 'Micro-average', 'Random classifier']
    
    classifier = OneVsRestClassifier(LogisticRegression(C=0.001, penalty='l2', solver='saga', max_iter=500))
    Y = label_binarize(Y, classes=classes)
    N = len(classes)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)
    Ysc = classifier.fit(Xtr, Ytr).decision_function(Xte)
    
    fpr, tpr = {}, {}
    for i in range(N):
        fpr[i], tpr[i], _ = roc_curve(Yte[:,i], Ysc[:,i])
    fpr['micro'], tpr['micro'], _ = roc_curve(Yte.ravel(), Ysc.ravel())
    
    for i in range(N):
        plt.plot(fpr[i], tpr[i])
    plt.plot(fpr['micro'], tpr['micro'], linestyle=':')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(legend, loc='lower right')
    if is_std:
        plt.title('ROC curves for 7-class dataset')
    else:
        plt.title('ROC curves for 3-class dataset')
    plt.show()

def logistic_regression(X, Y, is_std=True):
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2)
    model = LogisticRegression(C=0.001, penalty='l2', solver='saga', max_iter=500).fit(Xtr, Ytr)
    
    print('=== Test Report ===')
    Ypr1 = model.predict(Xte)
    print(classification_report(Yte, Ypr1))
    
    print('=== Train Report ===')
    Ypr2 = model.predict(Xtr)
    print(classification_report(Ytr, Ypr2))
    
    plot_confusion_matrix(model, Xte, Yte)
    plt.show()
    multiclass_roc_curves(X, Y, is_std)

def main():
    X_std, Y_std = read_data(DATA_FILE_STANDARD)
    X_3c, Y_3c = read_data(DATA_FILE_3_CLASS)
    #cross_validation(X_std, Y_std, is_std=True)
    #cross_validation(X_3c, Y_3c, is_std=False)
    logistic_regression(X_std, Y_std, is_std=True)
    logistic_regression(X_3c, Y_3c, is_std=False)

main()
