### README
This directory is for preprocessing ZOYI dataset(ICDM'18) suitable for survival analysis.

#### preparing_open_dataset.py
This script is to generate the open benchmark survival dataset from ZOYI indoor data which were used in ICDM'18. 

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

#### Preparing-open-dataset_step1,2_store_E.ipynb
This notebooks is a preliminary version of preparing_open_dataset.py. We provide this notebook file for readers to check the intermediary results while generating final datasets.


#### preparing_open_dataset.sh
Bash script to generate datasets for store A-E

#### revisit.py
A library which has several methods of preprocessing 