import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale, StandardScaler




def get_all_data():
    dir = 'data/'
    files = ['sample_orig_2016.txt', 'sample_orig_2015.txt', 'sample_orig_2014.txt', 'sample_orig_2013.txt',
             'sample_orig_2012.txt', 'sample_orig_2011.txt',
             'sample_orig_2010.txt', 'sample_orig_2009.txt', 'sample_orig_2008.txt', 'sample_orig_2007.txt']

    files1 = ['sample_svcg_2016.txt', 'sample_svcg_2015.txt', 'sample_svcg_2014.txt', 'sample_svcg_2013.txt',
              'sample_svcg_2012.txt', 'sample_svcg_2011.txt',
              'sample_svcg_2010.txt', 'sample_svcg_2009.txt', 'sample_svcg_2008.txt', 'sample_svcg_2008.txt']

    merged_data = pd.DataFrame()
    for i in [0]:
        print(dir+files[i])
        raw = pd.read_csv(dir+files[i], sep='|', header=None, low_memory=False)
        raw.columns = ['credit_score', 'first_pmt_date', 'first_time', 'mat_date', 'msa', 'mi_perc', 'units',
                       'occ_status', 'ocltv', 'odti', 'oupb', 'oltv', 'oint_rate', 'channel', 'ppm', 'fixed_rate',
                       'state', 'prop_type', 'zip', 'loan_num', 'loan_purpose', 'oterm', 'num_borrowers', 'seller_name',
                       'servicer_name', 'exceed_conform']

        raw1 = pd.read_csv(dir+files1[i], sep='|', header=None, low_memory=False)
        raw1.columns = ['loan_num', 'yearmon', 'curr_upb', 'curr_delinq', 'loan_age', 'remain_months', 'repurchased',
                        'modified', 'zero_bal', 'zero_date', 'curr_rate', 'curr_def_upb', 'ddlpi', 'mi_rec',
                        'net_proceeds',
                        'non_mi_rec', 'exp', 'legal_costs', 'maint_exp', 'tax_insur', 'misc_exp', 'loss', 'mod_exp']

        data = pd.merge(raw, raw1, on='loan_num', how='inner')

        merged_data = merged_data.append(data)

    merged_data.drop(['seller_name', 'servicer_name', 'first_pmt_date', 'mat_date', 'msa', 'net_proceeds'], axis=1,
                     inplace=True)

    # all data must have the following: credit_score, ocltv, odti, oltv, oint_rate, curr_upb
    # remove any datapoints with missing values from the above features
    merged_data.dropna(subset=['credit_score', 'odti', 'oltv', 'oint_rate', 'curr_upb'], how='any', inplace=True)
    merged_data.credit_score = pd.to_numeric(data['credit_score'], errors='coerce')
    merged_data.yearmon = pd.to_datetime(data['yearmon'], format='%Y%m')
    merged_data.fillna(value=0, inplace=True, axis=1)

    merged_data.sort_values(['loan_num'], ascending=True).groupby(['yearmon'],
                                                                  as_index=False)  ##consider move this into the next func
    merged_data.set_index(['loan_num', 'yearmon'], inplace=True)  ## consider move this into the next func

    return merged_data




def process_data(data):
    # data.sort_values(['loan_num'], ascending=True).groupby(['yearmon'], as_index=False)  ##consider move this out
    # data.set_index(['loan_num', 'yearmon'], inplace=True) ## consider move this out
    y = data['curr_delinq']
    y = y.apply(lambda x: 1 if x not in (0, 1) else 0)
    # data['prev_delinq'] = data.curr_delinq.shift(1) ## needs attention here
    # data['prev_delinq'] = data.groupby(level=0)['curr_delinq'].shift(1)
    # print(sum(data.prev_delinq.isnull()))
    data.fillna(value=0, inplace=True, axis=1)
    data.drop(['curr_delinq'], axis=1, inplace=True)
    print(y.shape)
    ## how many classes are y?
    ## remove y from X

    X = pd.get_dummies(data)
    # X.net_proceeds = X.net_proceeds.apply(lambda x:0 if x == 'C' else x)
    # y = label_binarize(y, classes=[0, 1, 2, 3]) ## do we really have to do this?
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #X[['credit_score', 'mi_perc', 'units', 'ocltv', 'oupb', 'oltv', 'oint_rate', 'zip',
    #   'curr_upb', 'loan_age', 'remain_months', 'curr_rate', 'curr_def_upb', 'ddlpi', 'mi_rec',
    #   'non_mi_rec', 'exp', 'legal_costs', 'maint_exp', 'tax_insur', 'misc_exp', 'loss', 'mod_exp']] = \
    #    scale(X[['credit_score', 'mi_perc', 'units', 'ocltv', 'oupb', 'oltv', 'oint_rate', 'zip',
    #             'curr_upb', 'loan_age', 'remain_months', 'curr_rate', 'curr_def_upb', 'ddlpi', 'mi_rec',
    #             'non_mi_rec', 'exp', 'legal_costs', 'maint_exp', 'tax_insur', 'misc_exp', 'loss', 'mod_exp']],
    #          with_mean=False)
    #return X, y
    return X, y.values



def get_all_data_bloodDonation():
    file = 'data/train_cleaned.csv'
    data = pd.read_csv(file)

    return data



def process_data_bloodDonation(data):
    numerical_features = ['Number of Donations', 'Months since First Donation', 'Months since Last Donation']

    y = data['Made Donation in March 2007']
    X = data.loc[:, numerical_features]
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def merge_dict(dicts):
    """dicts: a list of dicts"""
    super_dict = {}
    for d in dicts:
        for k, v in d.items():  # d.items() in Python 3+
            super_dict.setdefault(k, []).append(v)

    df = pd.DataFrame.from_dict(super_dict)
    df.plot()
    plt.show()

    return df




