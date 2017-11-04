import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import   nn_arch,nn_reg,ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
import pylab as pl
import clustering
from imblearn.over_sampling import SMOTE
import utility
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit


out = './results/RF/'



def rf(X, y, problem):
    rfc = RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=5,n_jobs=7)
    feat_import = rfc.fit(X,y).feature_importances_
    
    tmp = pd.Series(np.sort(feat_import)[::-1])
    tmp.to_csv(out+problem+' feat_import.csv')


def reduction_clustering(X, y, k, problem):
    """choose given an appropriate k components, project the data in new space, and perform clustering"""
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    X_rf = rf.fit(X).transform(X)

    return clustering.evaluate_kmeans(X_rf, y, problem, out='./results/PCA/')
    #TODO: should we perform plotting?


def rf_nn(X, y, problem):
    n = len(X[0])
    sm = SMOTE()
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    filtr = ImportanceSelect(rf)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)

    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n, n, n)],
        'filtr__n': [2,5,10,15,20]
    }

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('filtr',filtr),('NN',mlp)])
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
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7)
    filtr = ImportanceSelect(rf)
    km = KMeans(random_state=5)
    mlp = MLPClassifier(solver='adam', alpha=1e-5, shuffle=True, early_stopping=True, activation='relu',
                        verbose=True)
    X_res, y_res = sm.fit_sample(X, y)

    parameters = {
        'NN__hidden_layer_sizes': [(n, n, n, n, n)],
        'filtr__n': [2, 5, 10, 15, 20],
        'km__n_clusters': [2, 3, 4, 5, 6],
    }

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2)  ## no need for this given 50000 random sample
    pipe = Pipeline([('filtr', filtr), ('km', km), ('NN', mlp)])
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
    # visualize_pca(train, target, 'FreddieMac')
    # pca(train, 'FreddieMac')
    # reduction_clustering(train, target, 60, 'FreddiMac')
    clf, score, gs = rf_nn(train, target, 'FreddieMac')
    # visualize_data(5, train, target, 'FreddieMac')
    clf, score, gs = reduction_cluster_nn(train, target, 'FreddieMac')

    #all_data = utility.get_all_data_bloodDonation()
    #train, target = utility.process_data_bloodDonation(all_data)
    # visualize_pca(train, target, 'BloodDonation')
    # pca(train, 'BloodDonation')
    # reduction_clustering(train, target, 2, 'BloodDonation')
    # clf, score, gs = pca_nn(train, target, 'BloodDonation')
    # tmp = pd.DataFrame(gs.cv_results_)
    # tmp.to_csv('BloodDonation PCA NN.csv')
    # visualize_data(5, train, target, 'BloodDonation')
    # clf, score, gs = reduction_cluster_nn(train, target, 'FreddieMac')
    # tmp = pd.DataFrame(gs.cv_results_)
    # tmp.to_csv('FreddieMac dr_cluster_NN.csv')