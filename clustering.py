# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
import utility



def clustering():
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    parameters = {
        'NN__hidden_layer_sizes': [(111,), (111, 111, 111), (111, 111, 111, 111, 111),
                                   (111, 111, 111, 111, 111, 111, 111),
                                   (111,), (111, 100, 100), (111, 100, 100, 100, 100),
                                   (111, 100, 100, 100, 100, 100, 100),
                                   (111,), (111, 89, 89), (111, 89, 89, 89, 89), (111, 89, 89, 89, 89, 89, 89),
                                   (111,), (111, 78, 78), (111, 78, 78, 78, 78), (111, 78, 78, 78, 78, 78, 78)],
        'KM__n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]}

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    km = KMeans(random_state=5)
    pipe = Pipeline([('KM', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, parameters, verbose=10)
    gs.fit(X, y)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)
    mat = clf.predict_proba(X)
    print(mat)

    return clf, gs.best_score_, gs





'''
# %% For chart 4/5
madelonX2D = TSNE(verbose=10,random_state=5).fit_transform(madelonX)
digitsX2D = TSNE(verbose=10,random_state=5).fit_transform(digitsX)

madelon2D = pd.DataFrame(np.hstack((madelonX2D,np.atleast_2d(madelonY).T)),columns=['x','y','target'])
digits2D = pd.DataFrame(np.hstack((digitsX2D,np.atleast_2d(digitsY).T)),columns=['x','y','target'])

madelon2D.to_csv(out+'madelon2D.csv')
digits2D.to_csv(out+'digits2D.csv')
'''


def evaluate_kmeans(X, y, problem):
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = KMeans(random_state=5)
    gmm = GMM(random_state=5)

    st = clock()
    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    for k in clusters:
        km.set_params(n_clusters=k)
        # gmm.set_params(n_components=k)
        km.fit(X)
        # gmm.fit(madelonX)
        SSE[k][problem] = km.score(X)
        #ll[k][problem] = gmm.score(X)
        acc[k][problem]['Kmeans'] = cluster_acc(y, km.predict(X))
        # acc[k]['Madelon']['GMM'] = cluster_acc(madelonY,gmm.predict(madelonX))
        adjMI[k][problem]['Kmeans'] = ami(y, km.predict(X))
        # adjMI[k]['Madelon']['GMM'] = ami(madelonY,gmm.predict(madelonX))

    print(k, clock() - st)

    return SSE, ll, acc, adjMI

    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)

    SSE.to_csv(out + 'SSE.csv')
    ll.to_csv(out + 'logliklihood.csv')
    acc.ix[:, :, 'Digits'].to_csv(out + 'Digits acc.csv')
    acc.ix[:, :, 'Madelon'].to_csv(out + 'Madelon acc.csv')
    adjMI.ix[:, :, 'Digits'].to_csv(out + 'Digits adjMI.csv')
    adjMI.ix[:, :, 'Madelon'].to_csv(out + 'Madelon adjMI.csv')



if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    SSE, ll, acc, adjMI = evaluate_kmeans(train, target, 'FreddieMac')
    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)
    SSE.to_csv(out + 'SSE.csv')
    ll.to_csv(out + 'logliklihood.csv')
    acc.iloc[:, :, 'FreddieMac'].to_csv(out + 'FreddieMac acc.csv')
    adjMI.iloc[:, :, 'FreddieMac'].to_csv(out + 'FreddieMac adjMI.csv')
    clf, score, gs = gridSearch_nn(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac NN.csv')


    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    SSE, ll, acc, adjMI = evaluate_kmeans(train, target, 'BloodDonation')
    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)
    SSE.to_csv(out + 'SSE.csv')
    ll.to_csv(out + 'logliklihood.csv')
    acc.iloc[:, :, 'FreddieMac'].to_csv(out + 'FreddieMac acc.csv')
    adjMI.iloc[:, :, 'FreddieMac'].to_csv(out + 'FreddieMac adjMI.csv')
    clf, score, gs = gridSearch_nn(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('BloodDonation NN.csv')
