# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 10:38:28 2017

@author: jtay
"""

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GM
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
import utility





'''
# %% For chart 4/5
madelonX2D = TSNE(verbose=10,random_state=5).fit_transform(madelonX)
digitsX2D = TSNE(verbose=10,random_state=5).fit_transform(digitsX)

madelon2D = pd.DataFrame(np.hstack((madelonX2D,np.atleast_2d(madelonY).T)),columns=['x','y','target'])
digits2D = pd.DataFrame(np.hstack((digitsX2D,np.atleast_2d(digitsY).T)),columns=['x','y','target'])

madelon2D.to_csv(out+'madelon2D.csv')
digits2D.to_csv(out+'digits2D.csv')
'''


'''
def bench_k_means(estimator, name, data, labels):
    t0 = clock()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (clock() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
'''



def evaluate_kmeans(X, y, problem):
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = KMeans(random_state=5)
    gm = GM(random_state=5)

    st = clock()
    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    for k in clusters:
        print('now doing k=' + str(k))
        km.set_params(n_clusters=k)
        gm.set_params(n_components=k)
        km.fit(X)
        gm.fit(X)
        SSE[k][problem] = km.score(X)
        ll[k][problem] = gm.score(X)
        print('km score:', SSE[k][problem])
        print('gm score:', ll[k][problem])
        acc[k][problem]['Kmeans'] = cluster_acc(y, km.predict(X))
        acc[k][problem]['GM'] = cluster_acc(y,gm.predict(X))
        adjMI[k][problem]['Kmeans'] = metrics.ami(y, km.predict(X))
        adjMI[k][problem]['GM'] = metrics.ami(y,gm.predict(X))

    print(k, clock() - st)

    SSE = (-pd.DataFrame(SSE)).T
    SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
    ll = pd.DataFrame(ll).T
    ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
    acc = pd.Panel(acc)
    adjMI = pd.Panel(adjMI)

    SSE.to_csv(problem+ ' SSE.csv')
    ll.to_csv(problem+ ' logliklihood.csv')
    acc.ix[:,:,problem].to_csv(problem+' acc.csv')
    acc.ix[:,:,problem,].to_csv(problem + ' acc.csv')
    adjMI.ix[:,:,problem] .to_csv(problem+' adjMI.csv')
    adjMI.ix[:,:,problem].to_csv(problem + ' adjMI.csv')

    return SSE, ll, acc, adjMI



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


if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    SSE, ll, acc, adjMI = evaluate_kmeans(train, target, 'FreddieMac')
    clf, score, gs = clustering(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac NN.csv')


    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    SSE, ll, acc, adjMI = evaluate_kmeans(train, target, 'BloodDonation')
    clf, score, gs = clustering(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('BloodDonation NN.csv')
