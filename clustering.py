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
from sklearn.decomposition import PCA
from collections import defaultdict
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
import utility
import pylab as pl
from scipy.spatial.distance import cdist
from imblearn.over_sampling import SMOTE





def evaluate_kmeans(X, y, problem):
    """Also evaluate kmeans and em both"""
    SSE = defaultdict(dict)
    ll = defaultdict(dict)
    distort_km = []
    distort_gm = []
    acc = defaultdict(lambda: defaultdict(dict))
    adjMI = defaultdict(lambda: defaultdict(dict))
    km = KMeans(random_state=5)
    gm = GM(random_state=5)

    st = clock()
    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for k in clusters:
        print('now doing k=' + str(k))
        km.set_params(n_clusters=k)
        gm.set_params(n_components=k)
        km.fit(X)
        gm.fit(X)
        distort_km.append(sum(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        #distort_gm.append(sum(np.min(cdist(X, gm.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        SSE[k][problem] = km.score(X)
        ll[k][problem] = gm.score(X)
        print('km score:', SSE[k][problem])
        print('gm score:', ll[k][problem])
        acc[k][problem]['Kmeans'] = cluster_acc(y, km.predict(X))
        acc[k][problem]['GM'] = cluster_acc(y,gm.predict(X))
        adjMI[k][problem]['Kmeans'] = metrics.adjusted_mutual_info_score(y, km.predict(X))
        adjMI[k][problem]['GM'] = metrics.adjusted_mutual_info_score(y,gm.predict(X))

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

    pl.plot(clusters, distort_km, 'bx-')
    pl.xlabel('k')
    pl.ylabel('Distortion')
    pl.title('The Elbow Method showing the optimal k')
    pl.show()

    return SSE, ll, acc, adjMI, km, gm


def visualize_clusters(data, target, problem):
    pca = PCA(n_components=2).fit(data)
    pca_2d = pca.transform(data)

    '''
    # visualize actual class in new projected space
    for i in range(0, pca_2d.shape[0]):
        if target[i] == 0:
            c1 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r',
                         marker='+')
        elif target[i] == 1:
            c2 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g',
                         marker='o')
        elif target[i] == 2:
            c3 = pl.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b',
                         marker='*')
    #pl.legend([c1, c2, c3], ['Setosa', 'Versicolor',
    #                              'Virginica'])
    pl.title('Insert title here')
    pl.show()
    '''
    # now visualize classified data in new projected space
    pl.figure('Reference Plot ' + problem)
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=['navy', 'darkorange'])
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    pl.figure('K-means with 2 clusters ' + problem)
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=['navy', 'darkorange'])
    pl.legend()
    pl.show()


def clustering_nn(X, y):
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    sm = SMOTE
    X_res, y_res = sm.fit_sample(X, y)
    parameters = {
        'NN__hidden_layer_sizes': [(n,), (n, n, n), (n, n, n, n, n),(n,n, n, n, n, n, n),
                                   (n,), (n, int(0.9*n),int(0.9*n)), (n, int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n)),
                                   (n, int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n),int(0.9*n)),
                                   (n,), (n, int(0.8 *n), int(0.8 * n)), (n, int(0.8 *n), int(0.8 * n), int(0.8 * n), int(0.8 * n)),
                                   (n, int(0.8 *n), int(0.8 * n), int(0.8 * n), int(0.8 * n), int(0.8 * n), int(0.8 * n)),
                                   (n,), (n, int(0.7 *n), int(0.7 * n)), (n, int(0.7 *n), int(0.7 * n), int(0.7 * n), int(0.7 * n)),
                                   (n, int(0.7 *n), int(0.7 * n), int(0.7 * n), int(0.7 * n), int(0.7 * n), int(0.7 * n)),],
        'KM__n_clusters': [2, 3, 4, 5, 6]}

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    km = KMeans(random_state=5)
    pipe = Pipeline([('smote', sm), ('KM', km), ('NN', mlp)])
    gs = GridSearchCV(pipe, parameters, verbose=10)
    gs.fit(X_res, y_res)
    clf = gs.best_estimator_
    print(clf)
    print(gs.best_score_)

    return clf, gs.best_score_, gs



if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    SSE, ll, acc, adjMI, km, gm = evaluate_kmeans(train, target, 'FreddieMac')
    visualize_clusters(train, target, 'FreddieMac')
    #clf, score, gs = clustering_nn(train, target)
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('FreddieMac NN.csv')


    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    SSE, ll, acc, adjMI, km, gm = evaluate_kmeans(train, target, 'BloodDonation')
    visualize_clusters(train, target, 'Blood Donation')
    #clf, score, gs = clustering_nn(train, target)
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('BloodDonation NN.csv')
