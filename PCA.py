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
import pylab as pl
import clustering
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns', None)

out = './results/PCA/'
#cmap = cm.get_cmap('Spectral')

clusters =  [2,5,10,15,20,25,30,35,40]
dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]



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
    pca = PCA(random_state=5)
    pca.fit(X)
    tmp = pd.Series(data = pca.explained_variance_,index=range(len(X[0])))
    tmp.to_csv(out+problem+'.csv')


def reduction_clustering(X, y, k, problem):
    """choose given an appropriate k components, project the data in new space, and perform clustering"""
    pca = PCA(n_components=k)
    X_pca = pca.fit(X).transform(X)

    return clustering.evaluate_kmeans(X_pca, y, problem)
    #TODO: should we perform plotting?



def pca_nn(X, y, problem):
    n = len(X[0])
    sm = SMOTE()
    pca = PCA(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)
    if 'Freddie' in problem:
        pca__n_components =[2, 3, 4, 5, 6, 7]
    else:
        pca__n_components= [2, 3]

    parameters = {
        'NN__hidden_layer_sizes': [(n,), (n, n, n), (n, n, n, n, n),
                                   (n,), (n, int(0.9 * n), int(0.9 * n)),
                                   (n, int(0.9 * n), int(0.9 * n), int(0.9 * n), int(0.9 * n)),
                                   (n,), (n, int(0.8 * n), int(0.8 * n)),
                                   (n, int(0.8 * n), int(0.8 * n), int(0.8 * n), int(0.8 * n)),
                                   (n,), (n, int(0.7 * n), int(0.7 * n)),
                                   (n, int(0.7 * n), int(0.7 * n), int(0.7 * n), int(0.7 * n)) ],
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


'''  ## what is this?
def visualize_data(dim, X, y, problem):
    dim = 5
    pca = PCA(n_components=dim,random_state=10)

    dataX2 = pca.fit_transform(X)
    data2 = pd.DataFrame(np.hstack((dataX2,np.atleast_2d(y).T)))
    cols = list(range(data2.shape[1]))
    cols[-1] = 'Class'
    data2.columns = cols
    data2.to_hdf(out+'datasets.hdf',problem,complib='blosc',complevel=9)

    dim = 60
    pca = PCA(n_components=dim,random_state=10)
    digitsX2 = pca.fit_transform(digitsX)
    digits2 = pd.DataFrame(np.hstack((digitsX2,np.atleast_2d(digitsY).T)))
    cols = list(range(digits2.shape[1]))
    cols[-1] = 'Class'
    digits2.columns = cols
    digits2.to_hdf(out+'datasets.hdf','digits',complib='blosc',complevel=9)
'''



if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    #visualize_pca(train, target, 'FreddieMac')
    #pca(train, 'FreddieMac')
    #reduction_clustering(train, target, 20, 'FreddiMac')
    clf, score, gs = pca_nn(train, target, 'FreddieMac')
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('FreddieMac NN.csv')
    #visualize_data(5, train, target, 'FreddieMac')

    all_data = utility.get_all_data_bloodDonation()
    train, target = utility.process_data_bloodDonation(all_data)
    #visualize_pca(train, target, 'BloodDonation')
    #pca(train, 'BloodDonation')
    #reduction_clustering(train, target, 2, 'BloodDonation')
    clf, score, gs = pca_nn(train, target, 'BloodDonation')
    #tmp = pd.DataFrame(gs.cv_results_)
    #tmp.to_csv('BloodDonation NN.csv')
    #visualize_data(5, train, target, 'BloodDonation')