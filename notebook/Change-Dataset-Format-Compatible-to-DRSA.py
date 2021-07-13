import pandas as pd
import csv
import numpy as np
import time
import math
import sys

""" 
This script is to generate a DRSA-compatible data format from the open benchmark survival dataset.
Preliminary version: 190121-Change-Dataset-Format-Compatible-to-DRSA-StoreA.ipynb
Bash script: Change-Dataset-Format-Compatible-to-DRSA.sh
"""

def add_infos(df):  
    tst = time.time()
    df['l_index'] = df['indices'].apply(lambda x: [int(y) for y in x.split(';')])
    
    newidx = [item for sublist in list(df.l_index) for item in sublist]
    tmpdf = wifi_sessions.loc[newidx]
    traj_lens = df.l_index.apply(len)

    tmp_areas = list(tmpdf['area'])
    tmp_dt = list(tmpdf['dwell_time'])
    tmp_ts_end = list(np.array(tmpdf['ts'])+np.array(tmp_dt))  # end time
    
    rslt_dt = []
    rslt_areas = []
    rslt_ts_end = []
    
    i = 0
    for x in traj_lens:
        rslt_dt.append(tmp_dt[i:i+x])
        rslt_areas.append(tmp_areas[i:i+x])
        rslt_ts_end.append(max(tmp_ts_end[i:i+x]))
        i += x
        
    df['dwell_times'] = rslt_dt
    df['areas'] =  rslt_areas
    df['ts_end'] = rslt_ts_end
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
    df = train_visits.copy()
    
    features = df.apply(lambda x: statistical_feature_generator(x), axis=1)
    featureName = ['total_dwell_time', 'num_area', 'num_unique_area']
    
    fdf = pd.DataFrame(list(np.asarray(features)), index=features.index, columns = featureName)
    
    # Combine feature values to the dataframe
    df = pd.concat([df, fdf], axis=1)
    del fdf
    
    return df


def generate_suppress_time_col(df):
    last_ts_end = max(df['ts_end'])
    df['observation_time'] = [(last_ts_end-x)/86400 for x in df['ts_end']]
    df['suppress_time'] = np.maximum(df['revisit_interval'].fillna(0), df['revisit_interval'].isnull()*df['observation_time'])
    return df


def remove_unnecessary_features(df):
    unnecessary_attributes = ['visit_id', 'wifi_id', 'indices', 'l_index', 'dwell_times', 'areas', 'ts_end']
    all_attributes = list(df.columns)
    for attribute in unnecessary_attributes:
        try:
            all_attributes.remove(attribute)
        except:
            pass
    df = df[all_attributes]
    return df


def simplify_feature_values(df):
    df['total_dwell_time'] = df['total_dwell_time'].apply(lambda x: math.ceil(math.log(x,2)))
    return df


def generate_featindex(dftrain, dftest, path):
    namecol = {}
    featindex = {}
    maxindex = 0
    
    with open(path+'featindex.txt', "w") as file:
        for i, col in enumerate(dftrain.columns[:-4]):
            namecol[col] = i
            featindex[namecol[col]] = {}
            featvals = ['other']+sorted(set(dftrain[col]).union(set(dftest[col])))
            for val in featvals:
                featindex[namecol[col]][val] = maxindex
                maxindex += 1

        for key in featindex.keys():
            for key2 in featindex[key]:
                file.write('{}:{}\t{}\n'.format(key,key2,featindex[key][key2]))
    
    return featindex


def save_to_DRSA_format(df, path):
    with open(path+'.yzbx.txt', "w") as file:
        for i in df.iterrows():
            item = list(i[1])
#             true_event_time = str(int(item[-3]))
#             observation_time = str(int(item[-2]))
            suppress_time = str(int(item[-1]))
            observation_time = str(int(item[-2]))
            assert len(featindex.keys()) == len(item[:-4])
            converted_featvals = [featindex[i][j] for i,j in zip(featindex.keys(),item[:-4])]
            features = ' '.join([str(int(x))+':1' for x in converted_featvals])
            dam = ' '.join(['0',suppress_time,observation_time,features])
            file.write(dam+'\n')
    with open(path+'.bid.txt',"w") as file2:
        for i in df.iterrows():
            item = list(i[1])
            suppress_time = str(int(item[-1]))
            observation_time = str(int(item[-2]))
            ri = str(int(item[-4]))
            dam2 = ' '.join([observation_time,suppress_time,ri])
            file2.write(dam2+'\n')
            

if __name__ == '__main__':
    """
    possible argument: A or B or C or D or E
    """
    
    # Load data
    store_id = sys.argv[1]            
    pre_release_path = '../data/indoor/store_{}/'.format(store_id)
    print('Preparing DRSA-compatible dataset of store {}'.format(store_id))

    # Load dataset
    train_labels = pd.read_csv(pre_release_path+'train_labels.tsv', sep='\t')
    test_labels = pd.read_csv(pre_release_path+'test_labels.tsv', sep='\t')
    train_visits = pd.read_csv(pre_release_path+'train_visits.tsv', sep='\t')
    test_visits = pd.read_csv(pre_release_path+'test_visits.tsv', sep='\t')
    wifi_sessions = pd.read_csv(pre_release_path+'wifi_sessions.tsv', sep='\t')

    wifi_sessions = wifi_sessions.set_index('index')

    ### Before feature engineering, querying some useful information from wifi-sessions data, and add to the dataframe.
    train_visits = add_infos(train_visits)
    test_visits = add_infos(test_visits)

    ### Sample code to generate features 
    train_visits = add_statistical_features(train_visits)
    test_visits = add_statistical_features(test_visits)

    train_visits['date_rel'] = train_visits['date']-min(train_visits.date)
    test_visits['date_rel'] = test_visits['date']-min(train_visits.date)

    df_train = pd.concat([train_visits, train_labels[['revisit_intention','revisit_interval']]], axis=1)
    df_test = pd.concat([test_visits, test_labels[['revisit_intention','revisit_interval']]], axis=1)

    ## Generate 'suppress_time' column for evaluation  
    df_train = generate_suppress_time_col(df_train)
    df_test = generate_suppress_time_col(df_test)

    ### Retain only feature values

    df_train = remove_unnecessary_features(df_train)
    df_test = remove_unnecessary_features(df_test)

    df_train = simplify_feature_values(df_train)
    df_test = simplify_feature_values(df_test)

    featindex = generate_featindex(df_train, df_test, '../drsa/data/drsa-data/INDOOR/Store_{}/'.format(store_id))

    ### int로 저장해야 km.py가 빨리 끝남. round 4로 저장해서 돌렸을 때는 굉장히 오래 걸림           
    save_to_DRSA_format(df_train, '../drsa/data/drsa-data/INDOOR/Store_{}/train'.format(store_id))
    save_to_DRSA_format(df_test, '../drsa/data/drsa-data/INDOOR/Store_{}/test'.format(store_id))
    print('Finished preparing DRSA-compatible dataset of store {}, please check ../drsa/data/drsa-data/INDOOR/Store_{}'.format(store_id, store_id))