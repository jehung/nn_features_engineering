

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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pylab as pl
import clustering
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns', None)
from sklearn.decomposition import FastICA

out = './results/ICA/'



def ica(X, problem):
    dims = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    ica = FastICA(random_state=5)
    if 'Blood' in problem:
        dims = range(2, len(X[0]))
    kurt = {}
    for dim in dims:
        ica.set_params(n_components=dim)
        tmp = ica.fit_transform(X)
        tmp = pd.DataFrame(tmp)
        tmp = tmp.kurt(axis=0)
        kurt[dim] = tmp.abs().mean()

    kurt = pd.Series(kurt)
    kurt.to_csv(out+problem+'ICA.csv')


def visualize_ica(X, y, problem):
    pl.figure()
    colors = ['navy',  'darkorange']
    if 'Freddie' in problem:
        target_names = ['default', 'no default']
    else:
        target_names = ['donated', 'not donated']
    lw = 2

    ica = FastICA(n_components=2)
    X_ica = ica.fit(X).transform(X)


    for color, i, target_name in zip(colors, [0, 1], target_names):
        pl.scatter(X_ica[y == i, 0], X_ica[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    pl.legend(loc='best', shadow=False, scatterpoints=1)
    pl.title('ICA of ' + problem)
    pl.show()


def reduction_clustering(X, y, k, problem):
    """choose given an appropriate k components, project the data in new space, and perform clustering"""
    ica = FastICA(n_components=k)
    X_ica = ica.fit(X).transform(X)

    return clustering.evaluate_kmeans(X_ica, y, problem, out='./results/ICA/')
    #TODO: should we perform plotting?



def ica_nn(X, y, problem):
    n = len(X[0])
    sm = SMOTE()
    ica = FastICA(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)
    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n, n, n)],
        'ica__n_components': [2, 5, 10, 15, 20]}

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('ica',ica),('NN',mlp)])
    gs = GridSearchCV(pipe,parameters,verbose=10,cv=sss)

    gs.fit(X_res, y_res)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)

    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+problem+' dim_red_nn.csv')

    return clf, gs.best_score_, gs


def reduction_cluster_nn(X,y,problem):
    n = len(X[0])
    sm = SMOTE()
    ica = FastICA(random_state=5)
    km = KMeans(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)
    if 'Freddie' in problem:
        ica__n_components = [2, 5, 10, 15, 20]
    else:
        ica__n_components = [2, 3]

    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n, n, n)],
        'ica__n_components': ica__n_components,
        'km__n_clusters': [2, 3, 4, 5, 6],
    }

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('ica', ica), ('km', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, parameters, verbose=10, cv=sss)

    gs.fit(X_res, y_res)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)

    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + problem + ' dr_cluster_nn.csv')

    return clf, gs.best_score_, gs


if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    #visualize_ica(train, target, 'FreddieMac')
    #ica(train, 'FreddieMac')
    #reduction_clustering(train, target, 20, 'FreddiMac')
    clf, score, gs = ica_nn(train, target, 'FreddieMac')
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac ICA NN.csv')
    clf, score, gs = reduction_cluster_nn(train, target, 'FreddieMac')
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac ICA_cluster_NN.csv')


    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    #visualize_ica(train, target, 'BloodDonation')
    #ica(train, 'BloodDonation')
    #reduction_clustering(train, target, 2, 'BloodDonation')
    #clf, score, gs = ica_nn(train, target, 'BloodDonation')
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('BloodDonation NN.csv')
    #clf, score, gs = reduction_cluster_nn(train, target, 'BloodDonation')
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('FreddieMac ICA_cluster_NN.csv')
