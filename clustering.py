# -*- coding: utf-8 -*-

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
from scipy.cluster.vq import kmeans,vq




out = './results/Clustering/'

def evaluate_kmeans(X, y, problem):
    '''
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

        #distort_km.append(sum(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        ##distort_gm.append(sum(np.min(cdist(X, gm.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
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

    SSE.to_csv(out+problem+ ' SSE.csv')
    ll.to_csv(out+problem+ ' logliklihood.csv')
    acc.ix[:,:,problem].to_csv(out+problem+' acc.csv')
    acc.ix[:,:,problem,].to_csv(out+problem + ' acc.csv')
    adjMI.ix[:,:,problem] .to_csv(out+problem+' adjMI.csv')
    adjMI.ix[:,:,problem].to_csv(out+problem + ' adjMI.csv')
    '''
    ## evaluate using elbow method
    K_MAX = 10
    KK = range(1, K_MAX + 1)

    KM = [kmeans(X, k) for k in KK]
    centroids = [cent for (cent, var) in KM]
    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D, axis=1) for D in D_k]
    dist = [np.min(D, axis=1) for D in D_k]

    tot_withinss = [sum(d ** 2) for d in dist]  # Total within-cluster sum of squares
    totss = sum(pdist(X) ** 2) / X.shape[0]  # The total sum of squares
    betweenss = totss - tot_withinss  # The between-cluster sum of squares

    ##### plots #####
    kIdx = 9  # K=10
    clr = cm.spectral(np.linspace(0, 1, 10)).tolist()
    mrk = 'os^p<dvh8>+x.'

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(KK, betweenss / totss * 100, 'b*-')
    ax.plot(KK[kIdx], betweenss[kIdx] / totss * 100, marker='o', markersize=12,
            markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    ax.set_ylim((0, 100))
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained (%)')

    pl.plot(clusters, distort_km, 'bx-')
    pl.xlabel('k')
    pl.ylabel('Distortion')
    pl.title('The Elbow Method showing the optimal k')
    pl.show()

    #return SSE, ll, acc, adjMI, km, gm


def visualize_clusters(data, target, problem, k):
    '''
    pca = PCA(n_components=2).fit(data)
    pca_2d = pca.transform(data)
    # now visualize classified data in new projected space
    pl.figure('Reference Plot ' + problem)
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=['black'])
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(data)
    pl.figure('K-means with 2 clusters ' + problem)
    pl.scatter(pca_2d[:, 0], pca_2d[:, 1], c=['navy', 'darkorange', 'green'], alpha=0.4)
    pl.legend()
    pl.show()
    '''

    reduced_data = PCA(n_components=2).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    pl.figure(1)
    pl.clf()
    pl.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=pl.cm.Paired,
               aspect='auto', origin='lower')

    pl.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    pl.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    pl.title('K-means clustering on the ' + problem + ' dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    pl.xlim(x_min, x_max)
    pl.ylim(y_min, y_max)
    pl.xticks(())
    pl.yticks(())
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
    #visualize_clusters(train, target, 'FreddieMac', 6)
    #clf, score, gs = clustering_nn(train, target)
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('FreddieMac NN.csv')


    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    SSE, ll, acc, adjMI, km, gm = evaluate_kmeans(train, target, 'BloodDonation')
    #visualize_clusters(train, target, 'Blood Donation', 3)
    #clf, score, gs = clustering_nn(train, target)
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('BloodDonation NN.csv')
