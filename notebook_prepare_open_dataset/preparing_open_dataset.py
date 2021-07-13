import pandas as pd
import numpy as np
import reindex
import random
import pickle
import math
import sys
import os
import re



""" 
This script is to generate the open benchmark survival dataset from ZOYI indoor data which were used in ICDM'18. 
Preliminary version: preparing_open_dataset.ipynb
Bash script: preparing_open_dataset.sh

Store Name
A: 319 (L_GA)
B: 1157 (L_MD)
C: 1143 (O_MD)
D: 1627 (E_GN)
E: 1552 (E_SC)

Input & Output
- input: Wi-Fi session dump files from ZOYI (part-000xx)

- intermediary output: logs_60s_10area.csv file
    * columns = wifiId, ts, area, dwellTime
    * which only contains user session logs who spend more than 60 seconds and having greater than and equal to 10 areas.
   
Summary of the preprocessing steps
* Remove unnecessary data (5 steps)
* Additional cleaning of wifi session logs (7 steps)
* Reindexing visits from cleared session datas (5 steps)
* Reindexed data to generate train/test data for survival analysis
"""

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def anon_id(uid):
    return id_anon[uid]

def divide1000(ts):
    return int(ts/1000)

def checkValidity(x, dict_wid_lastlog):
    try:
        if x['index'] <= dict_wid_lastlog.get(x['wifi_id']):
            return True
        else:
            return False
    except TypeError:
        return False




if __name__ == '__main__':
    """
    possible argument: A or B or C or D or E
    """
    
    print('Preparing open dataset')
    
    # Load data
    store_id = sys.argv[1]
    data_path = '../data_raw/indoor/store_{}/'.format(store_id)
    raw_area_path = '../data_raw/indoor/other_APIs/n_shop_sensor_areas.xlsx'
    d_place_num = {'A':'319','B':'1157','C':'1143','D':'1627','E':'1552'}
    print('\nLoading data - data_path:{}'.format(data_path))
    
    
    # Store zones: unique real sensors
    if store_id in ['A','B','C']:
        areadf = pd.read_excel(raw_area_path, sheet_name = d_place_num[store_id])
        zones = list(areadf.area.unique())
        for pass_zone in ['in', 'out', 'max', '1f', '2f', '3f', 'b1']:
            try:  
                zones.remove(pass_zone)
            except ValueError:
                pass
        zones = sorted(zones, key=natural_key)       
    elif store_id in ['D']:
        zones = list(map(lambda x: '1f-sensor'+str(x+1), range(7))) + list(map(lambda x: '2f-sensor'+str(x+1), range(5))) + list(map(lambda x: '3f-sensor'+str(x+1), range(8))) + list(map(lambda x: '4f-sensor'+str(x+1), range(10))) + list(map(lambda x: '1f-FR'+str(x+1), range(2))) + list(map(lambda x: '2f-FR'+str(x+1), range(2))) + list(map(lambda x: '3f-FR'+str(x+1), range(2))) + list(map(lambda x: '4f-FR'+str(x+1), range(2)))
        assert len(zones) == 38
    elif store_id in ['E']: 
        zones = list(map(lambda x: 'sensor'+str(x+1), range(20)))
        assert len(zones) == 20
    else:
        pass
        
    
    # Read Wi-Fi session files and generate a dataframe
    if store_id in ['A', 'B', 'C']:
        fps = [x for x in sorted(os.listdir(data_path)) if 'part-' in x]
        rawcols = ['shopId', 'wifiId', 'ts', 'area', 'dwellTime', 'isLocal', 'rts', 'revisitCount', 'revisitPeriod','isDeny']
        # Read relevant columns
        usecols = [1,2,3,4,5]
        colnames = [rawcols[i] for i in usecols]
        # initialize dataframe
        df = pd.DataFrame(columns = colnames) 
        # load each dataset
        for path in fps:
            df_each = pd.read_csv(data_path+path, header=None, usecols=usecols)
            df_each.columns = colnames
            df = pd.concat([df, df_each])
    elif store_id in ['D', 'E']:
        fps = [x for x in sorted(os.listdir(data_path), key=natural_key) if '.p' in x]
        rawcols = ['area', 'deny', 'dwell_time', 'local', 'reivisit_period', 'revisit_count', 'row_key', 'ts', 'wifi_id']
        # Read relevant columns
        usecols = [8,7,0,2,3]
        colnames = [rawcols[i] for i in usecols]
        # initialize dataframe
        df = pd.DataFrame(columns = colnames) 
        # load each dataset
        for path in fps:
            df_each = pd.read_pickle(data_path+path)[colnames]
            df_each.columns = colnames
            df = pd.concat([df, df_each])
        df = df.rename(columns={"wifi_id": "wifiId", "dwell_time": "dwellTime", "local": "isLocal"})
    else:
        pass
    print(df.area.value_counts())
    
    """
    Preprocessing for data release - 1) Remove unnecessary data
    """
    print('\nPreprocessing - 1. Remove Unnecessary data')
    
    # Step 1: Cut 1 year data
    # - Time zone (1 year: 20170101 00:00:00 ~ 20171231 23:59:59)
    # - After conversion: 1483196400.000 - 1514732399.999
    print(len(df))
    df = df[(df.ts >= 1483196400000) & (df.ts <= 1514732399999)]
    print(len(df))

    # Step 2: Remove all places with dwellTime == 0
    df = df[df.dwellTime > 0]
    print(len(df))

    # Step 3: Only retain customers who has a 'in' log more than 60 sec
    uids = list(set(df[(df.area == 'in') & (df.dwellTime >= 60)].wifiId))
    print('The number of UIDs staying in-store more than 60 sec is {}'.format(len(uids)))
    df = df[df.wifiId.isin(uids)]
    print(len(df))
    
    # Step 4: Only retain local=False signal, since local=True signals are mac-randomized 
    df = df[df.isLocal == False]
    del df['isLocal']
    print(len(df))
    
    # Step 5: Only retaining customers who has more than 10 logs (in total)
    g = df.wifiId.value_counts()
    dnumareas = {}
    for i in range(1,100):
        dnumareas[i] = len(g[g>=i])
    print('# of customers whose logs remain greater than or equal to i --> (i, # of customers)')
    print([i for i in dnumareas.items()][:100])
    uid10 = list(g[:dnumareas[10]].index)
    df = df[df.wifiId.isin(uid10)]
    print('# of customers whose logs remain greater than {} --> {}'.format(10, len(df)))
    
    ### save intermediate raw data
    df.to_csv(data_path+'step1_logs_60s_10area.csv', index=False)
    
    
    
    """
    Preprocessing for data release - 2) Additional cleaning of wifi session logs 
    """
    print('\nPreprocessing - 2. Additional cleaning of wifi session logs')
          
    directory_path = '../data_raw/indoor/store_{}/'.format(store_id)
    
    # Load and Check the intermediary data
    df = pd.read_csv(directory_path+'step1_logs_60s_10area.csv')
        
    # Step 1: Change Wi-Fi ID to sorted index (for step 3)
    id_anon = {}
    for i, j in enumerate(list(df.wifiId.value_counts().index)):
        id_anon[j] = i+1
    df['wifiId'] = df['wifiId'].apply(anon_id)

    # Step 2: Change the unit of ts (milisecond -> second)
    df['ts'] = df['ts'].apply(divide1000)

    # Step 3: Remove top 100 frequent visitors - maybe workers
    print(len(df), len(df[df.wifiId > 100]))
    df = df[df.wifiId > 100]

    # Step 4: Select randomly 50,000 users for open dataset
    print(len(set(df.wifiId)), len(df))
    uids = list(set(df.wifiId))
    random.shuffle(uids)
    df = df[df.wifiId.isin(uids[:50000])]

    # Step 5: Change column name and column type to use previous codes.
    print(len(set(df.wifiId)), len(df))
    df = df.rename(columns={"wifiId": "wifi_id", "dwellTime": "dwell_time"})
    df['wifi_id'] = df['wifi_id']    # .apply(str)
        
    df = df.sort_values(by='ts').reset_index()
    del df['index']

    # Step 6: Re-anonymized wifi-id, since previous Wi-Fi ID is a sorted index of # of sessions.
    id_anon = {}
    ids_new = list(df.wifi_id.value_counts().index)
    random.shuffle(ids_new)
    for i, j in enumerate(ids_new):
        id_anon[j] = i+1
    df['wifi_id'] = df['wifi_id'].apply(anon_id)

    # Step 7: Store two different session data 
    # 1) df: session df with all logs
    # 2) dfin: session df without areas such as [out, max, center..] by using only zones to construct valid in-store visits.
    df = df.reset_index()
    dfin = df[df.area.isin(zones)]
    print(df.shape, dfin.shape)
    
    
    """
    Preprocessing for data release - 3) Reindexing visits from cleared session datas
    """
    print('\nPreprocessing - 3. Reindexing visits from cleared session datas')
    
    # Step 1: Add time difference between neighboring logs
    rdf1 = reindex.update_session_data_before_reindex(dfin)
    
    # Step 2: Defining visits (600s interval = new visit)
    rdf2 = reindex.reindex_session_to_each_visit(rdf1)
    
    # Step 3: Add dates
    rdf3 = reindex.add_enter_leave_date_for_visit(rdf2)
    
    # Step 4: Merge multiple visits if they happen at the same day
    rdf4 = reindex.merge_multiple_sameday_visits_into_daily_trajectory(rdf3)
    
    # Step 5: Generate binary labels and save some additional infos
    rdf4['revisit_interval'] /= 86400
    rdf4['revisit_interval'] = rdf4['revisit_interval'].apply(lambda x: np.around(x, decimals=2)) 
    rdf4['revisit_intention'] = rdf4['revisit_interval'].notnull().astype(int)
    rdf4['ts_end_max'] = rdf4.ts_end.apply(lambda x: np.max(x))
    rdf4['ts_min'] = rdf4.ts.apply(lambda x: np.min(x))
    
    
    """
    Preprocessing for data release - 4) Reindexed data to generate train/test data for survival analysis
    """
    print('\nPreprocessing - 4. Reindexed data to generate train/test data for survival analysis')
    
    # Generate a train set
    train_days_list = sys.argv[2:]
    for train_days in train_days_list:
        release_path = '../data/indoor/store_{}/train_{}days/'.format(store_id, train_days)
        pre_release_path = '../data_sample/indoor/store_{}/train_{}days/'.format(store_id, train_days)
        if not os.path.exists(release_path):
            os.makedirs(release_path)
        if not os.path.exists(pre_release_path):
            os.makedirs(pre_release_path)
        
        train = rdf4[rdf4.date <= min(rdf4.date)+int(train_days)][['wifi_id', 'date', 'indices', 'ts_min','ts_end_max','revisit_interval','revisit_intention']].sort_values(by=['wifi_id', 'date'])
        train['indices'] = train.indices.apply(lambda x: ';'.join(str(e) for e in x))
        print(len(train), len(train[train.revisit_interval > 0]))
        train_length = len(train)
        train['visit_id'] = ['v'+str(i) for i in range(train_length)]

        # Generate a test set
        test = rdf4[rdf4.date > min(rdf4.date)+int(train_days)][['wifi_id', 'date', 'indices', 'ts_min', 'ts_end_max','revisit_interval', 'revisit_intention']].sort_values(by=['wifi_id', 'date'])
        print(len(test), len(test[test.revisit_interval > 0]))
        # For test set, retain only the first appearance. This process is done since if there are two visits available for each wifi_id, it can cause data leakage in test, people knows whether or not the user revisited. (Since we also deal with train_censored case, we only consider test samples that does not appear in training timeframe.
        test = test.drop_duplicates(subset=['wifi_id'], keep='first', inplace=False)
        test = test[~test.wifi_id.isin(list(train.wifi_id))]
        test['indices'] = test.indices.apply(lambda x: ';'.join(str(e) for e in x))
        print(len(test), len(test[test.revisit_interval > 0]))
        test_length = len(test)
        test['visit_id'] = ['v'+str(i+train_length) for i in range(test_length)]

        # To remove revisit interval & intention of some training cases where revisit happens at testing timeframe (If not eliminated, this could be an implicit cheating - JG) When revisit_interval == math.nan, label_validity is False
        train['label_validity'] = train.ts_end_max + train.revisit_interval*86400 < np.min(test.ts_min)

        # For longitudinal study, evaluation of forecasting result of censored dataset is important, for censored dataset label_validity is False
        train_censored = train[train.label_validity == False]
        # train_censored = train.drop_duplicates(subset=['wifi_id'], keep='last', inplace=False) # or this way

        # Remove the label of those cases
        train['revisit_interval'] = np.where(train['label_validity'] == False, math.nan, train['revisit_interval'])
        train['revisit_intention'] = np.where(train['label_validity'] == False, 0, train['revisit_intention'])


        # Save train, test dataset -> Train/test data further divides into two parts: labels data and visits data
        # Labels data only save visit_id and labels, Trajectory information are saved in visits data

        # Saving
        print('Saving train, test labels, visits files...\n')
        train[['visit_id','revisit_interval','revisit_intention']].to_pickle(release_path+'train_labels.pkl')
        test[['visit_id','revisit_interval','revisit_intention']].to_pickle(release_path+'test_labels.pkl')
        train_censored[['visit_id','revisit_interval','revisit_intention']].to_pickle(release_path+'train_censored_actual_labels.pkl')

        train[['visit_id','wifi_id','date','indices']].to_pickle(release_path+'train_visits.pkl')
        test[['visit_id','wifi_id','date','indices']].to_pickle(release_path+'test_visits.pkl')

        # Saving - csv
        train[['visit_id','revisit_interval','revisit_intention']].to_csv(release_path+'train_labels.tsv', sep='\t', index=False)
        test[['visit_id','revisit_interval','revisit_intention']].to_csv(release_path+'test_labels.tsv', sep='\t', index=False)

        train_censored[['visit_id','revisit_interval','revisit_intention']].to_csv(release_path+'train_censored_actual_labels.tsv', sep='\t', index=False)

        train[['visit_id','wifi_id','date','indices']].to_csv(release_path+'train_visits.tsv', sep='\t', index=False)
        test[['visit_id','wifi_id','date','indices']].to_csv(release_path+'test_visits.tsv', sep='\t', index=False)


        # Preparing wifi_sessions data for public 
        print('\nPreparing wifi_session data for public')

        train_used_indices = []
        for visit in train['indices']:
            train_used_indices.extend(visit.split(';'))

        test_used_indices = []
        for visit in test['indices']:
            test_used_indices.extend(visit.split(';'))

        # 1) Track the last and the first index of train & test, respectively
        train_last_idx = max([int(x) for x in train_used_indices])
        test_first_idx = min([int(x) for x in test_used_indices])

        print(train_last_idx, test_first_idx)
        print(len(train_used_indices), len(test_used_indices))

        # 2) Since the original wifi_session logs are sorted by timestamp, we save the last index for each wifi_ids used in our data
        dict_wid_lastlog = {}
        for wid, visit in zip(test['wifi_id'], test['indices']):
            last_index = max([int(x) for x in visit.split(';')])
            dict_wid_lastlog[wid] = last_index

        # 3) Remove the wifi session data which is saved after our train/test. (ex. session logs of the 2nd visits of the same customer in testing period) 
        df_train = df.iloc[:train_last_idx+1]
        df_test = df.iloc[train_last_idx+1:]    
        df_test['valid'] = df_test.apply(lambda x: checkValidity(x, dict_wid_lastlog), axis=1)

        df_final = pd.concat([df_train, df_test[df_test.valid==True]])
        df_final = df_final[['index', 'wifi_id', 'ts', 'area', 'dwell_time']]

        print(df_final.shape)
        print(len(set(df_final.wifi_id)), len(set(train.wifi_id).union(set(test.wifi_id))))

        print('# of wifi_id having logs without any valid zones'.format(len(set(df_final.wifi_id) - (set(train.wifi_id).union(set(test.wifi_id))))))
        assert len(set(zones).intersection(set(df_final[df_final.wifi_id.isin(set(df_final.wifi_id) - (set(train.wifi_id).union(set(test.wifi_id))))].area.value_counts().index))) == 0
        df_final[df_final.wifi_id.isin(set(df_final.wifi_id) - (set(train.wifi_id).union(set(test.wifi_id))))].area.value_counts()

        # Check that our final wifi-session data has a same wifi_id set of train + test dataset
        df_final = df_final[df_final.wifi_id.isin(set(train.wifi_id).union(set(test.wifi_id)))]
        assert set(df_final.wifi_id) == set(train.wifi_id).union(set(test.wifi_id))

        ## Saving a relevant wifi-session data 
        df_final.to_pickle(release_path+'wifi_sessions.pkl')
        df_final.to_csv(release_path+'wifi_sessions.tsv', sep='\t', index=False)


        print('\n. Saving 1,000 users sample dataset')      
        ## Release 1000-users sample data for fast testing
        all_wifi_ids = list(set(train.wifi_id).union(set(test.wifi_id)))
        random.shuffle(all_wifi_ids)
        sample_wifi_ids = all_wifi_ids[:1000]
        print(len(all_wifi_ids), len(sample_wifi_ids))

        ## Saving - csv
        train_sample = train[train.wifi_id.isin(sample_wifi_ids)]
        test_sample = test[test.wifi_id.isin(sample_wifi_ids)]
        train_sample_censored = train_censored[train.wifi_id.isin(sample_wifi_ids)]
        df_final_sample = df_final[df_final.wifi_id.isin(sample_wifi_ids)]
        print(len(set(train_sample.wifi_id)), len(set(test_sample.wifi_id)), len(set(df_final_sample.wifi_id)))

        train_sample[['visit_id','revisit_interval','revisit_intention']].to_csv(pre_release_path+'train_labels.tsv', sep='\t', index=False)
        test_sample[['visit_id','revisit_interval','revisit_intention']].to_csv(pre_release_path+'test_labels.tsv', sep='\t', index=False)
        train_sample_censored[['visit_id','revisit_interval','revisit_intention']].to_csv(pre_release_path+'train_censored_actual_labels.tsv', sep='\t', index=False)

        train_sample[['visit_id','wifi_id','date','indices']].to_csv(pre_release_path+'train_visits.tsv', sep='\t', index=False)
        test_sample[['visit_id','wifi_id','date','indices']].to_csv(pre_release_path+'test_visits.tsv', sep='\t', index=False)

        df_final_sample.to_csv(pre_release_path+'wifi_sessions.tsv', sep='\t', index=False)
        print(len(train_sample), len(test_sample), len(train_sample_censored), len(df_final_sample))
        print('\nFinish preprocessing, check outputs on {}'.format(release_path))

    
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        