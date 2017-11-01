# -*- coding: utf-8 -*-
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
import collections
pd.set_option('display.max_columns', None)


out = './results/PCA/'
#cmap = cm.get_cmap('Spectral')

clusters =  [2,5,10]
dims = [2,5,10,15,20,30,40,50,60]



def visualize_pca(X, y, problem):
    pl.figure()
    colors = ['navy',  'darkorange']
    if 'Freddie' in problem:
        target_names = ['default', 'no default']
    else:
        target_names = ['donated', 'not donated']
    lw = 2

    pca = PCA(n_components=2)
    X_pca = pca.fit(X).transform(X)

    print('explained variance ratio (first two components): %s'
          % str(pca.explained_variance_ratio_))

    for color, i, target_name in zip(colors, [0, 1], target_names):
        pl.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    pl.legend(loc='best', shadow=False, scatterpoints=1)
    pl.title('PCA of ' + problem)
    pl.show()


def pca(X, problem):
    dims = [2, 5, 10, 15, 20, 30, 40, 50, 60]
    pca = PCA(random_state=5)
    if 'Blood' in problem:
        dims = range(2, len(X[0]))
    eigen = collections.defaultdict(dict)
    for dim in dims:
        pca.set_params(n_components=dim)
        pca.fit(X)
        eigen[dim]['explained_var']=np.sum(pca.fit(X).explained_variance_ratio_)
        eigen[dim]['eigenval'] = pca.explained_variance_

    eigen = pd.DataFrame.from_dict(eigen)
    eigen.to_csv(out+problem+'PCA.csv')


def reduction_clustering(X, y, k, problem):
    """choose given an appropriate k components, project the data in new space, and perform clustering"""
    pca = PCA(n_components=k)
    X_pca = pca.fit(X).transform(X)

    return clustering.evaluate_kmeans(X_pca, y, problem, out='./results/PCA/')
    #TODO: should we perform plotting?


def pca_nn(X, y, problem):
    n = len(X[0])
    sm = SMOTE()
    pca = PCA(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)
    if 'Freddie' in problem:
        pca__n_components=[2,5,10,15,20]
    else:
        pca__n_components=[2, 3]

    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n), (n, n, n, n, n)],
        'pca__n_components': pca__n_components}

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('pca',pca),('NN',mlp)])
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
    pca = PCA(random_state=5)
    km = KMeans(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)
    if 'Freddie' in problem:
        pca__n_components = [2,5,10,15,20]
    else:
        pca__n_components = [2, 3]

    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n), (n, n, n, n, n)],
        'pca__n_components': pca__n_components,
        'km__n_clusters': [2, 3, 4, 5, 6],
    }

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('pca', pca), ('km', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, parameters, verbose=10, cv=sss)

    gs.fit(X_res, y_res)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)

    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + problem + ' dim_red_nn.csv')

    return clf, gs.best_score_, gs


if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    #visualize_pca(train, target, 'FreddieMac')
    #pca(train, 'FreddieMac')
    #reduction_clustering(train, target, 60, 'FreddiMac')
    clf, score, gs = pca_nn(train, target, 'FreddieMac')
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac PCA NN.csv')
    #visualize_data(5, train, target, 'FreddieMac')
    clf, score, gs = reduction_cluster_nn(train, target, 'FreddieMac')
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac dr_cluster_NN.csv')

    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    #visualize_pca(train, target, 'BloodDonation')
    #pca(train, 'BloodDonation')
    #reduction_clustering(train, target, 2, 'BloodDonation')
    clf, score, gs = pca_nn(train, target, 'BloodDonation')
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('BloodDonation PCA NN.csv')
    #visualize_data(5, train, target, 'BloodDonation')
    clf, score, gs = reduction_cluster_nn(train, target, 'FreddieMac')
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac dr_cluster_NN.csv')