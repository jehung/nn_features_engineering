# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 17:17:14 2017

@author: JTay
"""

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import utility
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
pd.set_option('display.max_columns', None)


def gridSearch_nn(X, y):
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    parameters = {
        'NN__hidden_layer_sizes': [(111,), (111, 111, 111), (111, 111, 111, 111, 111), (111, 111, 111, 111, 111, 111, 111),
                               (111,), (111, 100, 100), (111, 100, 100, 100, 100), (111, 100, 100, 100, 100, 100, 100),
                               (111,), (111, 89, 89), (111, 89, 89, 89, 89), (111, 89, 89, 89, 89, 89, 89),
                               (111,), (111, 78, 78), (111, 78, 78, 78, 78), (111, 78, 78, 78, 78, 78, 78)]}
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('NN', mlp)])
    gs = GridSearchCV(estimator=pipe, param_grid=parameters, n_jobs=6, cv=sss, scoring='roc_auc', verbose=10)
    gs.fit(X, y)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)
    mat = clf.predict_proba(X)
    print(mat)

    return clf, gs.best_score_, gs

if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    clf, score, gs = gridSearch_nn(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac NN.csv')

    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    clf, score, gs = gridSearch_nn(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('BloodDonation NN.csv')


