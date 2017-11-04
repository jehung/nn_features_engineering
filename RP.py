

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from itertools import product
import utility
from sklearn.metrics.pairwise import euclidean_distances
import pylab as pl
import clustering
from sklearn.metrics.pairwise import pairwise_distances
from time import time
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.cluster import KMeans


out = './results/RP/'


def visualize_rp(X, y, problem):
    pl.figure()
    colors = ['navy',  'darkorange']
    if 'Freddie' in problem:
        target_names = ['default', 'no default']
    else:
        target_names = ['donated', 'not donated']
    lw = 2

    rp = SparseRandomProjection(n_components=2)
    X_rp = rp.fit(X).transform(X)

    for color, i, target_name in zip(colors, [0, 1], target_names):
        pl.scatter(X_rp[y == i, 0], X_rp[y == i, 1], color=color, alpha=.8, lw=lw,
                    label=target_name)
    pl.legend(loc='best', shadow=False, scatterpoints=1)
    pl.title('RP of ' + problem)
    pl.show()


def rp(X, problem):
    dims = [2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    tmp = defaultdict(dict)
    if 'Blood' in problem:
        dims = range(2, len(X[0]))
    for i, dim in product(range(10), dims):
        rp = SparseRandomProjection(n_components=dim)
        print(i, dim)
        #rp.fit(X)
        #tmp[dim][i] = euclidean_distances(rp.fit_transform(X))
        tmp[dim][i] = reconstructionError(rp, X)
    tmp = pd.DataFrame(tmp).T
    tmp.to_csv(out+problem+'_RP.csv')


def rp_johnson(X, problem):
    '''
    eps_range = np.linspace(0.1, 0.99, 5)
    n_samples_range = np.logspace(2, 6, 5)
    colors = pl.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

    pl.figure()
    for n_samples, color in zip(n_samples_range, colors):
        min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
        pl.semilogy(eps_range, min_n_components, color=color)

    pl.legend(["n_samples = %d" % n for n in n_samples_range], loc="upper right")
    pl.xlabel("Distortion eps")
    pl.ylabel("Minimum number of dimensions")
    pl.title("Johnson-Lindenstrauss bounds:\nn_components vs eps")
    pl.show()
    '''

    X = X[:5000]
    n_samples, n_features = X.shape
    print("Embedding %d samples with dim %d using various random projections"
          % (n_samples, n_features))

    n_components_range = np.array([2, 5, 10, 20, 60, 80, 100, 120])
    dists = euclidean_distances(X, squared=True).ravel()

    # select only non-identical samples pairs
    nonzero = dists != 0
    dists = dists[nonzero]

    for n_components in n_components_range:
        t0 = time()
        rp = SparseRandomProjection(n_components=n_components)
        projected_data = rp.fit_transform(X)
        print("Projected %d samples from %d to %d in %0.3fs"
              % (n_samples, n_features, n_components, time() - t0))
        if hasattr(rp, 'components_'):
            n_bytes = rp.components_.data.nbytes
            n_bytes += rp.components_.indices.nbytes
            print("Random matrix with size: %0.3fMB" % (n_bytes / 1e6))

        projected_dists = euclidean_distances(
            projected_data, squared=True).ravel()[nonzero]

        pl.figure()
        pl.hexbin(dists, projected_dists, gridsize=100, cmap=pl.cm.PuBu)
        pl.xlabel("Pairwise squared distances in original space")
        pl.ylabel("Pairwise squared distances in projected space")
        pl.title("Pairwise distances distribution for n_components=%d" %
                  n_components)
        cb = pl.colorbar()
        cb.set_label('Sample pairs counts')

        rates = projected_dists / dists
        print("Mean distances rate: %0.2f (%0.2f)"
              % (np.mean(rates), np.std(rates)))

        pl.figure()
        pl.hist(rates, bins=50, normed=True, range=(0., 2.), edgecolor='k')
        pl.xlabel("Squared distances rate: projected / original")
        pl.ylabel("Distribution of samples pairs")
        pl.title("Histogram of pairwise distance rates for n_components=%d" % n_components)
        pl.show()


def reduction_clustering(X, y, k, problem):
    """choose given an appropriate k components, project the data in new space, and perform clustering"""
    rp = SparseRandomProjection(n_components=k)
    X_rp = rp.fit(X).transform(X)

    return clustering.evaluate_kmeans(X_rp, y, problem, out='./results/RP/')
    #TODO: should we perform plotting?


def rp_nn(X, y, problem):
    n = len(X[0])
    sm = SMOTE()
    rp = SparseRandomProjection(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)
    if 'Freddie' in problem:
        rp__n_components=[2, 5, 10, 20, 60, 80, 100, 120]
    else:
        rp__n_components=[2, 3]

    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n, n, n)],
        'rp__n_components': rp__n_components}

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('rp',rp),('NN',mlp)])
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
    rp = SparseRandomProjection(random_state=5)
    km = KMeans(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)
    if 'Freddie' in problem:
        rp__n_components = [2, 5, 10, 20, 60, 80, 100, 120]
    else:
        pca__n_components = [2, 3]

    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n, n, n)],
        'rp__n_components': [2, 5, 10, 20, 60, 80, 100, 120],
        'km__n_clusters': [2, 3, 4, 5, 6],
    }

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('rp', rp), ('km', km), ('NN', mlp)])
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
    #rp(train, 'Freddie Mac')
    #rp_johnson(train, 'Freddie Mac')
    #visualize_rp(train, target, 'FreddieMac')
    #pca(train, 'FreddieMac')
    #reduction_clustering(train, target, 60, 'FreddiMac')
    clf, score, gs = rp_nn(train, target, 'FreddieMac')
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('FreddieMac NN.csv')
    #visualize_data(5, train, target, 'FreddieMac')
    clf, score, gs = reduction_cluster_nn(train, target, 'FreddieMac')

    #all_data = utility.get_all_data_bloodDonation()
    #train, target = utility.process_data_bloodDonation(all_data)
    #rp(train, 'Blood Donation')
    #visualize_rp(train, target, 'BloodDonation')
    #pca(train, 'BloodDonation')
    #reduction_clustering(train, target, 2, 'BloodDonation')
    #clf, score, gs = pca_nn(train, target, 'BloodDonation')
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('BloodDonation NN.csv')
    #visualize_data(5, train, target, 'BloodDonation')
    #clf, score, gs = reduction_cluster_nn(train, target, 'FreddieMac')
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('FreddieMac dr_cluster_NN.csv')