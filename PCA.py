# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:51:37 2017

@author: jtay
"""

#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from helpers import  nn_arch,nn_reg
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA

out = './PCA/'
cmap = cm.get_cmap('Spectral') 

clusters =  [2,5,10,15,20,25,30,35,40]
dims = [2,5,10,15,20,25,30,35,40,45,50,55,60]


def pca(X, problem):
    pca = PCA(random_state=5)
    pca.fit(X)
    tmp = pd.Series(data = pca.explained_variance_,index = range(1,501))
    tmp.to_csv(out+problem+' .csv')




def clustering(X, y, problem):
    grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    pca = PCA(random_state=5)
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('pca',pca),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(X,y)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+problem+' dim red.csv')


    grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
    pca = PCA(random_state=5)
    mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
    pipe = Pipeline([('pca',pca),('NN',mlp)])
    gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

    gs.fit(digitsX,digitsY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out+'digits dim red.csv')



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




if __name__ == '__main__':
    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    pca(train, 'FreddieMac')
    clf, score, gs = clustering(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('FreddieMac NN.csv')
    visualize_data(5, train, target, 'FreddieMac')

    all_data = utility.get_all_data()
    train, target = utility.process_data(all_data)
    pca(train, 'BloodDonation')
    clf, score, gs = clustering(train, target)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv('BloodDonation NN.csv')
    visualize_data(5, train, target, 'BloodDonation')