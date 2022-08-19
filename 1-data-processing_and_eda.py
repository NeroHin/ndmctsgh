# import pandas as pd 

# if you have cudf, it would be faster to use pandas x10+
import cudf as pd 

import os, glob, sys
from tqdm import  tqdm
import time


train_data_path = 'train.csv'
test_data_path = 'test.csv'
ecg_data_folder = 'ecg/'

# read ecg folder files
ecg_file_path = []

def load_data(filename):
    df = pd.read_csv(filename)
    return df
  
def add_uid_col_in_data(df, filename: str):
    df['UID'] = filename.split('.')[0]
    return df

def get_ecg_files_path(path):
    for root, dirs, files in tqdm(os.walk(path)):
        for file in tqdm(files):
            fullpath = os.path.join(root, file)
            ecg_file_path.append(fullpath)
            

# main function

start = time.time()

train_df = load_data(train_data_path)
print("=========================================")
print(train_df.head())

test_df = load_data(test_data_path)
print("=========================================")
print(test_df.head())

get_ecg_files_path(ecg_data_folder)


final_ecg_df = pd.DataFrame(columns=['leadI', 'leadII', 'leadIII', 'leadaVR', 'leadaVL', 'leadaVF', 'leadV1',
       'leadV2', 'leadV3', 'leadV4', 'leadV5', 'leadV6', 'UID'])

for ecg in tqdm(ecg_file_path):
    # print(ecg)
    ecg_df = load_data(ecg)
    ecg_df['UID'] = ecg.split('/')[1].split('.')[0]
    final_ecg_df = final_ecg_df.append(ecg_df, ignore_index=True)
    
# final_ecg_df.to_csv('final_ecg.csv', index=False)

print("=========================================")
print(final_ecg_df.head())


train_df = pd.merge(train_df, final_ecg_df, on='UID')
test_df = pd.merge(test_df, final_ecg_df, on='UID')

print("=========================================")
print(train_df.head())
print("=========================================")
print(test_df.head())

test_df.to_csv('test_with_ecg.csv', index=False)
train_df.to_csv('train_with_ecg.csv', index=False)

end = time.time()

print(f'Finished the data pre-processing & merge, used: { end - start} sec.')





