'''
Name: reindex.py
Date: 2016-10-04
Description: 새로 moving patterns indexing


Input: 786.p - Output: 786_mpframe_160923.p
TO DO: try-catch statement according to input data availability
'''
__author__ = 'Sundong Kim: sundong.kim@kaist.ac.kr'

import pandas as pd
import datetime
import numpy as np
import re
import time
# import plotly.plotly as py
# from plotly.tools import FigureFactory as FF

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def finddata(logs, df, data):   # x is a list of logs
   return df.ix[logs][data].tolist()

def calend(x, y):
    return list(map(lambda i, j: i+j, x, y ))

def printall(mp, i, columnss):
    for column in columnss:
        print(mp[column].ix[i])

## dataframe.ix로 매번 집어내는 것보다 dictionary로 바꾼 후 찾는 게 빠름. 
def finddata3(logs, dfdict):   
    a = []
    b = []
    c = []
    d = []
  
    for idx in logs:
        bb = dfdict[idx]['ts']
        cc = dfdict[idx]['dwell_time']
        a.append(dfdict[idx]['area'])
        b.append(bb)
        c.append(cc)
        d.append(bb+cc)
    
    pp = [a,b,c,d]

    return pp




##### DEPRECATED(2017-09-22)
# def reindex_by_moving_pattern(df):
    
#     df = df.reset_index()
#     df.loc[:, 'ts'] = df['ts'] / 1000
#     df.loc[:, 'date'] = (df['ts']+32400) // 86400
#     df['date'] = df['date'].astype(int)
#     df.loc[:, 'date_wifi_id'] = df.date.map(str) + "_" + df.wifi_id
    
    
#     mplogs = df.sort_index(ascending=False).groupby('date_wifi_id', sort=False).apply(lambda x: x.index.tolist()).sort_index()
#     mpframe = mplogs.to_frame(name='logs').reset_index()
    
    
#     dfdict = df.transpose()
#     del df 

#     infos = mpframe['logs'].apply(lambda x: finddata3(x, dfdict))
#     infonames = ['traj', 'ts', 'dwell_time', 'ts_end']
    
    
#     fdf = pd.DataFrame(list(np.asarray(infos)), index=infos.index, columns = infonames)
#     mpframe = pd.concat([mpframe, fdf], axis=1)
    

#     mpframe['wifi_id'] = mpframe['date_wifi_id'].apply(lambda x: x[6:])
#     mpframe['date'] = mpframe['date_wifi_id'].apply(lambda x: int(x[:5]))
#     del mpframe['logs']
#     return mpframe



##### DEPRECATED(2017-07-18)
# @timing.timing
# def reindex_by_moving_pattern(df):
# 	### read data and reindex by date and wifi_id
# 	df = df.reset_index()
# 	df.loc[:, 'date'] = (df['ts']+32400000) // 86400000
# 	# df = df.loc[(df['dwell_time'] > 0)]
# 	df.loc[:, 'date_wifi_id'] = df.date.map(str) + "_" + df.wifi_id

# 	df3 = df

# 	# ### collect valid trajectories (ex. trajecties containing only 'out' area, such as 'out-out-out-out' are deleted)
# 	# traj = df.groupby(['date_wifi_id'])['area']
# 	# ar = ['out1']
# 	# ar2 = ['out2']
# 	# temp = traj.unique()
# 	# temp2 = temp[-temp.apply(lambda x: (x == ar).all())]
# 	# temp3 = temp2[-temp2.apply(lambda x: (x == ar2).all())]
	

# 	# print('Ratio of valid trajectory having not only \'out\' area: ', len(temp3)/len(temp))
# 	# df2 = df[df.date_wifi_id.isin(temp3.index.tolist())]


# 	# ### erase other useless trajectories having 'in' shorter than mintime seconds
# 	# dfin = df2[df2['area'] == 'in']
# 	# time = dfin.groupby(['date_wifi_id'])['dwell_time'].sum()
# 	# time2 = time[time >= mintime]
# 	# print('Ratio of trajectory staying \'in\' longer than 0 seconds: ', len(time2)/len(time))
# 	# df3 = df2[df2.date_wifi_id.isin(time2.index.tolist())]

# 	# del df, df2

# 	### erase other useless trajectories having 


# 	### make a list of logs for corresponding each indoor moving trajectories 
# 	mplogs = df3.sort_index(ascending=False).groupby('date_wifi_id', sort=False).apply(lambda x: x.index.tolist()).sort_index()
# 	mpframe = mplogs.to_frame(name='logs').reset_index()


# 	mpframe.loc[:, 'traj'] = mpframe['logs'].apply(lambda x: finddata(x, df3, 'area'))
# 	mpframe.loc[:, 'ts'] = mpframe['logs'].apply(lambda x: finddata(x, df3, 'ts'))
# 	mpframe.loc[:, 'dwell_time'] = mpframe['logs'].apply(lambda x: finddata(x, df3, 'dwell_time'))
# 	mpframe.loc[:, 'ts'] = mpframe['ts'].apply(lambda x: [int(y/1000) for y in x])
# 	# mpframe.loc[:, 'hour_start'] = mpframe['ts'].apply(lambda x: [int(datetime.datetime.fromtimestamp(y).strftime('%H')) for y in x])
# 	# mpframe.loc[:, 'time_start'] = mpframe['ts'].apply(lambda x: [datetime.datetime.fromtimestamp(y).strftime('%H:%M:%S') for y in x])  # %Y-%m-%d 
# 	mpframe.loc[:, 'ts_end'] = mpframe[['ts', 'dwell_time']].apply(lambda x: calend(*x), axis=1)
# 	# mpframe.loc[:, 'hour_end'] = mpframe['ts_end'].apply(lambda x: [int(datetime.datetime.fromtimestamp(y).strftime('%H')) for y in x])
# 	# mpframe.loc[:, 'time_end'] = mpframe['ts_end'].apply(lambda x: [datetime.datetime.fromtimestamp(y).strftime('%H:%M:%S') for y in x])

# 	del df3
	
# 	mpframe['wifi_id'] = mpframe['date_wifi_id'].apply(lambda x: x[6:])
# 	mpframe['date'] = mpframe['date_wifi_id'].apply(lambda x: int(x[:5]))
# 	return mpframe




######################################################################
##### NEW VERSION OF REINDEXING -- MUCH FASTER(2017-09-22)
######################################################################



### Add revisit interval for session data (re-sensed after 600 seconds after leaving any zone = new visit)

def update_session_data_before_reindex(df):
    df['ts_end'] = df['ts'] + df['dwell_time']
    df = df.sort_values(by=['wifi_id', 'ts']).reset_index()
    del df['index']

    c1 = df.groupby(['wifi_id'])['ts']
    c2 = df.groupby(['wifi_id'])['ts_end']
    df['time_difference'] = c1.shift(periods=-1) - c2.shift(periods=0)
    df['ending_point'] = df['time_difference'].apply(lambda x: 1 if (x >= 600) or np.isnan(x) else 0)   # 600,000 ms = 600 sec

    df['ending_point2'] = df['ts'] * df['ending_point']
    df['ending_point2'] = df['ending_point2'].replace(0, np.nan)
    df = df.iloc[::-1]
    df['ending_point2'] = df['ending_point2'].fillna(method='ffill').astype(int)
    df = df.iloc[::-1]

    df['revisit_interval_sec'] = df['time_difference']*df['ending_point']
    df['key'] = df['wifi_id'].map(str)+'_'+df['ending_point2'].map(str)
    del df['ending_point']; del df['ending_point2']
    
    return df



### Reform session data to make a visit

def reindex_session_to_each_visit(raw):
    gbyobj = raw.groupby('key')
    df = pd.DataFrame({'traj':gbyobj['area'].apply(list), 'ts':gbyobj['ts'].apply(list), 'ts_end':gbyobj['ts_end'].apply(list), 'dwell_time':gbyobj['dwell_time'].apply(list), 'indices':gbyobj['level_0'].apply(list), 'revisit_interval':gbyobj['revisit_interval_sec'].apply(list)}).reset_index()
    df = df.rename(columns={"key":"wifi_id_ts"})
    df['revisit_interval'] = df['revisit_interval'].apply(lambda x: x[-1])
    df['wifi_id'] = df['wifi_id_ts'].apply(lambda x: int(x[:-11]))
    return df



### Add date information before merging into daily basis

def add_enter_leave_date_for_visit(df):
	### leave_date도 일단은 필요 없으므로 코멘트 처리 
	# df_date = pd.DataFrame(dict(enter_date = df['ts'].apply(lambda x: (x[0]/1000+32400)//86400), leave_date = df['ts_end'].apply(lambda x: (x[-1]/1000+32400)//86400))).reset_index()
	df_date = pd.DataFrame(dict(enter_date = df['ts'].apply(lambda x: (x[0]+32400)//86400).astype(int))).reset_index()
	del df_date['index']
	#### date_mid와 date_diff를 고려할 정도로 detail하게 안 봐도 되므로 일단 코멘트 처리 - 용량 줄이기 위함 
	# df_date['date_mid'] = df.apply(lambda x: ((x['ts'][0]+ x['ts_end'][-1])/(2*1000)+32400)//86400, axis=1)
	# df_date['date_diff'] = df_date['leave_date'] - df_date['enter_date']
	df_before_integration = pd.concat([df, df_date], axis=1)
	df_before_integration['date_wifi_id'] = df_before_integration.apply(lambda x: str(int(x['enter_date']))+'_'+str(x['wifi_id']), axis=1)
	del df_before_integration['wifi_id_ts']  ### 세션의 이름에 ts가 들어가는데 필요 없으므로 제거   

	df_before_integration['cnt'] = 1
	df_before_integration['total_visit_count'] = df_before_integration[['wifi_id', 'cnt']].groupby('wifi_id').cumsum()
	del df_before_integration['cnt']

	return df_before_integration



### Merge multiple sessions of trajectories into daily basis (merge if there exists multiple visits in single day)

def merge_multiple_sameday_visits_into_daily_trajectory(df_before_integration):

	prev_ddid = ''
	prev_item = ''
	updated_item_list = []

	for item in df_before_integration.iterrows():
	    current_ddid = item[1]['date_wifi_id'] 
	    current_item = item[1]
	    
	    if prev_ddid == current_ddid:
	        current_item['dwell_time'] = prev_item['dwell_time'] + current_item['dwell_time']
	        current_item['traj'] = prev_item['traj'] + current_item['traj']
	        current_item['ts'] = prev_item['ts'] + current_item['ts']
	        current_item['ts_end'] = prev_item['ts_end'] + current_item['ts_end']
	        current_item['indices'] = prev_item['indices'] + current_item['indices']
	        current_item['enter_date'] = prev_item['enter_date']     ### 시작 date는 이전 하루 단위 코드와의 일치를 위해 첫 date를 택함.   
	        
	    else:
	        pass
	    
	    prev_ddid = current_ddid
	    prev_item = current_item
	    updated_item_list.append(prev_item)
	    
	df_integrated = pd.DataFrame(updated_item_list)  ## df_integrated = dataframe before removing unnecessary duplicates
	df_daily_reindexed = df_integrated[df_integrated.date_wifi_id != df_integrated.date_wifi_id.shift(-1)].sort_values(by='date_wifi_id')
	df_daily_reindexed = df_daily_reindexed.rename(index=str, columns={"enter_date": "date"})


	df_daily_reindexed['cnt'] = 1
	df_daily_reindexed['total_daily_visit_count'] = df_daily_reindexed[['wifi_id', 'cnt']].groupby('wifi_id').cumsum()
	del df_daily_reindexed['cnt']

	return df_daily_reindexed


### Reindexing framework summary
def reindexing_framework(session_df):
	print('reindex.py - checkpoint1')
	enriched_session_df = update_session_data_before_reindex(session_df)
	print('reindex.py - checkpoint2')
	trajectories_each_visit = reindex_session_to_each_visit(enriched_session_df)
	print('reindex.py - checkpoint3')
	trajectories_each_visit = add_enter_leave_date_for_visit(trajectories_each_visit)
	print('reindex.py - checkpoint4')
	trajectories_daily = merge_multiple_sameday_visits_into_daily_trajectory(trajectories_each_visit)
	trajectories_daily['revisit_intention'] = trajectories_daily['revisit_interval'].notnull().astype(int)
	trajectories_daily = trajectories_daily[['date_wifi_id','traj','ts','dwell_time','ts_end','wifi_id','date','total_visit_count','total_daily_visit_count','revisit_interval', 'revisit_intention']]
	print('reindex.py - checkpoint5')
	return trajectories_daily

### 



if __name__ == '__main__':

	### load cleaned-up session data
	session_df = pd.read_pickle("../data/20170918/1552/experiment/1552_0_rawdata_cleaned.p")

	trajectories_daily = reindexing_framework(session_df)

	trajectories_daily.to_pickle('../data/20170918/1552/experiment/1552_1_reindexed_faster_ver.p')

