### 1-data-processing

'''
1-data-processing

action：

1. 合併資料到每一個病人的資料中
2. 檢查資料的完整性


'''

# import pandas as pd 

# if you have cudf, it would be faster to use pandas x10+
import cudf as pd 
import sweetviz as sv
# import pandas as pd
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
    df.columns = df.columns.str.replace(' ', '')
    return df
  
def add_uid_col_in_data(df, filename: str):
    df['UID'] = filename.split('.')[0]
    return df

def get_ecg_files_path(path):
    for root, dirs, files in tqdm(os.walk(path)):
        for file in tqdm(files):
            fullpath = os.path.join(root, file)
            ecg_file_path.append(fullpath)
            
def label_data(df):
    df['label'] = df['EF'].apply(lambda x: 1 if x <= 35 else 0)
            

# main function

start = time.time()

train_df = load_data(train_data_path)
print("=========================================")

label_data(train_df)
print("=========================================")
print("Finised label data")
print(train_df.head())

test_df = load_data(test_data_path)
print("=========================================")
print(test_df.head())

get_ecg_files_path(ecg_data_folder)


final_ecg_df = pd.DataFrame(columns=['leadI', 'leadII', 'leadIII', 'leadaVR', 'leadaVL', 'leadaVF', 'leadV1',
       'leadV2', 'leadV3', 'leadV4', 'leadV5', 'leadV6', 'UID','index'])

for ecg in tqdm(ecg_file_path):
    # print(ecg)
    ecg_df = load_data(ecg)
    ecg_df['index'] = ecg_df.index
    ecg_df['UID'] = ecg.split('/')[1].split('.')[0]
    final_ecg_df = final_ecg_df.append(ecg_df, ignore_index=True).sort_values(by=['UID', 'index'])
    
# final_ecg_df.to_csv('final_ecg.csv', index=False)

print("=========================================")
print(final_ecg_df.head())


train_df = pd.merge(train_df, final_ecg_df, on='UID').sort_values(by=['PID','UID', 'index'])
test_df = pd.merge(test_df, final_ecg_df, on='UID').sort_values(by=['PID','UID', 'index'])



print("=========================================")
print(train_df.head())
print("=========================================")
print(test_df.head())



test_df.to_csv('test_with_ecg.csv', index=False)
train_df.to_csv('train_with_ecg.csv', index=False)

end = time.time()

print(f'Finished the data pre-processing & merge, used: { end - start} sec.')

# start to analyze the data with EDA

print("=========================================")
print("start to analyze the data with EDA")

start = time.time()
train_with_ecg_df = pd.read_csv(filepath_or_buffer='train_with_ecg.csv').to_pandas()

data_report = sv.analyze([train_with_ecg_df, 'train'])
end = time.time()

data_report.show_html(filepath='data_report.html')
print("=========================================")
print("finished the EDA")
print(f"used: { end - start} sec.")