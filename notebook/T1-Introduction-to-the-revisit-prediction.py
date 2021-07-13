import pandas as pd
import numpy as np
import os
import random
import time
import sys
import xgboost as xgb
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


""" 
This script is an aggregated version of T1-Introduction-to-the-revisit-prediction.ipynb. We used this script to execute the tutorial multiple times - to remain the execution records.
"""


def add_infos(df):  
    """ 
    Before feature engineering, querying some useful information from wifi-sessions data, and add to the dataframe.
    """
    tst = time.time()
    df['l_index'] = df['indices'].apply(lambda x: [int(y) for y in x.split(';')])
    t1 = time.time()
#     print(t1-tst)
    
    newidx = [item for sublist in list(df.l_index) for item in sublist]
    tmpdf = wifi_sessions.loc[newidx]
    traj_lens = df.l_index.apply(len)

    tmp_areas = list(tmpdf['area'])
    tmp_dt = list(tmpdf['dwell_time'])

    rslt_dt = []
    rslt_areas = []

    i = 0
    for x in traj_lens:
        rslt_dt.append(tmp_dt[i:i+x])
        rslt_areas.append(tmp_areas[i:i+x])
        i += x
        
    df['dwell_times'] = rslt_dt
    df['areas'] =  rslt_areas
    
    t2 = time.time()
#     print(t2-t1)
    return df 


def statistical_feature_generator(x):
    fs = []

    total_dwell_time = sum(x['dwell_times'])   # total dwell time
    num_area_trajectory_have = len(x['dwell_times'])  # the number of area
    num_unique_area_sensed = len(set(x['areas']))  # the number of unique areas
    
    fs.append(total_dwell_time)
    fs.append(num_area_trajectory_have)  
    fs.append(num_unique_area_sensed)     
    
    return fs


def add_statistical_features(train_visits):
    """
    Sample code to generate features 
    """
    df = train_visits.copy()
    
    features = df.apply(lambda x: statistical_feature_generator(x), axis=1)
    featureName = ['total_dwell_time', 'num_area', 'num_unique_area']
    
    fdf = pd.DataFrame(list(np.asarray(features)), index=features.index, columns = featureName)
    
    # Combine feature values to the dataframe
    df = pd.concat([df, fdf], axis=1)
    del fdf
    
    return df


def remove_unnecessary_features(df):
    """
    Retain only feature values
    """
    unnecessary_attributes = ['visit_id', 'wifi_id', 'date', 'indices', 'l_index', 'dwell_times', 'areas']
    all_attributes = list(df.columns)
    for attribute in unnecessary_attributes:
        try:
            all_attributes.remove(attribute)
        except:
            pass
    df = df[all_attributes]
    return df


def show_intention_classification_result(y_pred, y_test):
    return metrics.accuracy_score(y_test, y_pred)


def show_interval_regression_result(y_pred, y_test):
    return metrics.mean_squared_error(y_test, y_pred)


def label_balancing(df, name_target_column):
    ## 1:1 Downsampling
    minimum_label_num = list(df[name_target_column].value_counts())[-1]
    
    df_list = []
    for value in df[name_target_column].unique():
        sub_dfs = df.loc[df[name_target_column] == value]
        new_sub_dfs = sub_dfs.iloc[np.random.permutation(len(sub_dfs))][:minimum_label_num]  ## Random Downsampling according to smallest label size
        df_list.append(new_sub_dfs)
        del sub_dfs
        
    new_df = pd.concat(df_list).sort_index()
    
    return new_df



if __name__ == "__main__":
    
    # Load data
    store_id = sys.argv[1]
    pre_release_path = '../data/indoor/store_'+store_id+'/'

    train_labels = pd.read_csv(pre_release_path+'train_labels.tsv', sep='\t')
    test_labels = pd.read_csv(pre_release_path+'test_labels.tsv', sep='\t')
    train_visits = pd.read_csv(pre_release_path+'train_visits.tsv', sep='\t')
    test_visits = pd.read_csv(pre_release_path+'test_visits.tsv', sep='\t')
    wifi_sessions = pd.read_csv(pre_release_path+'wifi_sessions.tsv', sep='\t')

    # Generate Features
    wifi_sessions = wifi_sessions.set_index('index')

    train_visits = add_infos(train_visits)
    test_visits = add_infos(test_visits)

    train_visits = add_statistical_features(train_visits)
    test_visits = add_statistical_features(test_visits)

    # Revisit prediction
    df_train = remove_unnecessary_features(train_visits)
    df_test = remove_unnecessary_features(test_visits)

    df_train = pd.concat([df_train, train_labels['revisit_intention']], axis=1)
    df_test = pd.concat([df_test, test_labels['revisit_intention']], axis=1)


    acc = []

#     print('Class label distribution before downsampling - Train data: revisit_intention 0: {}, 1: {}'.format(
#             df_train.revisit_intention.value_counts()[0],
#             df_train.revisit_intention.value_counts()[1]))
#     print('Class label distribution before downsampling - Test data: revisit_intention 0: {}, 1: {}'.format(
#             df_test.revisit_intention.value_counts()[0],
#             df_test.revisit_intention.value_counts()[1]))
#     print()
#     print('-----------   Experiments Begin   -------------')
#     print()

    for i in range(5):    

        ## Making downsampled dataset for measuring binary classification accuracy - baseline = 0.5
        whole_balanced_train = label_balancing(df_train, 'revisit_intention') 
        whole_balanced_test = label_balancing(df_test, 'revisit_intention') 
#         print('Class label distribution after downsampling - Train data: revisit_intention 0: {}, 1: {}'.format(
#             whole_balanced_train.revisit_intention.value_counts()[0],
#             whole_balanced_train.revisit_intention.value_counts()[1]))
#         print('Class label distribution after downsampling - Test data: revisit_intention 0: {}, 1: {}'.format(
#             whole_balanced_test.revisit_intention.value_counts()[0],
#             whole_balanced_test.revisit_intention.value_counts()[1]))

        for (train_data, test_data, ref) in [(whole_balanced_train, whole_balanced_test, 'Downsampled')]:
            train_array = np.asarray(train_data)  
            test_array = np.asarray(test_data)  

            # Dividing features and labels
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_test, y_test = test_array[:, :-1], test_array[:, -1].astype(int)

            # Training
            start = time.time()
            clf = Pipeline([
              ('classification', XGBClassifier(max_depth=5, learning_rate=0.1))
            ])
            clf = clf.fit(X_train, y_train)

            # Prediction
            y_pred = clf.predict(X_test)
            clas_rslt = metrics.accuracy_score(y_test, y_pred)
            done = time.time()
            elapsed = done-start

            # Result
#             print('Classification result', round(clas_rslt, 4))
#     #         print('Elapsed time:', round(elapsed, 4))
#             print()
            acc.append(clas_rslt)

    print()
    print('-----------  Performance of our model (5-time averaged) ({})  -------------'.format(store_id))
    print()
    print('Average accuracy: {:.4f}'.format(np.mean(acc)))