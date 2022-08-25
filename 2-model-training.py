# import cudf as pd
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, precision_recall_curve, average_precision_score, accuracy_score, recall_score, precision_score, f1_score

df_path = 'dataset/train_with_ecg.csv'


def pre_process(df):
    # remove older label
    df.drop(columns=['label'], inplace=True)
    
    # create new label
    df['label'] = df['EF'].apply(lambda x: 1 if x <= 35 else 0)
    
    return df
    

def normalize(df):
    # remove label, PID, UID, index
    cols = df.drop(columns=['label', 'PID','UID', 'index']).columns
    
    # normalize data
    for item in tqdm(cols):
        mean_tmp = np.mean(np.array(df[item]))
        std_tmp = np.std(np.array(df[item]))
        if(std_tmp):
            df[item] = df[item].apply(lambda x: (x - mean_tmp) / std_tmp)
            
    return df
            

def univariate_data(df, history_size, target_size):
    target = df.values[:,-1]
    df = df.drop(['label'],axis=1).values
    data = []
    labels = []

    start_index = history_size
    end_index = len(df)
    
    # 多特徵
    for i in tqdm(range(start_index, end_index)):

        indices = range(i-history_size, i,10) # step表示滑动步长
        data.append(df[indices])
        labels.append(target[indices])
        
    return np.array(data), np.array(labels)

def buildManyToOneModel(shape, layer=1):

    input_nodes = shape[1] * shape[2]
    output_nodes = 1
    hidden_nodes = int(round(2/3 * (input_nodes + output_nodes)))

    model = Sequential()

    if layer == 1:
        model.add(GRU(hidden_nodes, input_length=shape[1], input_dim=shape[2], kernel_initializer='normal'))
        model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    PRC = tf.keras.metrics.AUC(curve='PR')
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=[tf.keras.metrics.binary_accuracy,tf.keras.metrics.Recall(), PRC , tf.keras.metrics.AUC(curve='ROC')],run_eagerly=True)
    model.summary()
    
    return model



# mian function

df = pd.read_csv(df_path).sort_values(by=['PID', 'UID', 'index'])

print("=========================================")
print("start to pre-process data")
pre_process(df=df)
# drop ef column
df = df.drop(columns=['EF'])
print(df.head())

print("=========================================")
print("start to normalize data")
normalize(df=df)
print(df.head())

print("=========================================")
print("start to split data")
    # list of pid 
pid_list = df.PID.unique().tolist()

# random shuffle pid list, select no repeat 0.6 from train, 0.2 from test, 0.2 from validation
# ref: https://stackoverflow.com/questions/4211209/remove-all-the-elements-that-occur-in-one-list-from-another

train_id = np.random.choice(pid_list, int(len(pid_list) * 0.6), replace=False).tolist()
print(f"train id list length: {len(train_id)}")

test_id  = np.random.choice([x for x in tqdm(pid_list) if x not in train_id], int(len(pid_list) * 0.2), replace=False).tolist()
print(f"test id list length: {len(test_id)}")

# sum of train and test id array
train_test_id_list = train_id + test_id

vali_id = np.random.choice([x for x in tqdm(pid_list) if x not in train_test_id_list], int(len(pid_list) * 0.2), replace=False).tolist()
print(f"test id list length: {len(test_id)}")


train_df = df[df['PID'].isin(train_id)].reset_index(drop=True).drop(['PID'],axis=1).sort_values(by=['UID', 'index']).drop(columns=['UID', 'index'])
test_df = df[df['PID'].isin(test_id)].reset_index(drop=True).drop(['PID'],axis=1).sort_values(by=['UID', 'index']).drop(columns=['UID', 'index'])
validation_df = df[df['PID'].isin(vali_id)].reset_index(drop=True).drop(['PID'],axis=1).sort_values(by=['UID', 'index']).drop(columns=['UID', 'index'])


print("=========================================")

x_train_single, y_train_single = univariate_data(train_df,1000,0)
x_val_single, y_val_single = univariate_data(validation_df,1000,0)
x_test,y_test = univariate_data(test_df,1000,0)

print(f'x_train_uni.shape -- {x_train_single.shape}')
print(f'y_train_uni.shape -- {y_train_single.shape}')
print(f'x_val_uni.shape -- {x_val_single.shape}')
print(f'y_val_uni.shape -- {y_val_single.shape}')


BUFFER_SIZE = 10000
BATCH_SIZE = 256
EPOCHS = 150

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().batch(BATCH_SIZE).shuffle(BUFFER_SIZE)#.repeat().shuffle(BUFFER_SIZE)

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE)


callbacks_list = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
]


buildManyToOneModel(shape=x_train_single.shape[-2:])



history = model.fit(train_data_single, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    validation_data=val_data_single, callbacks=callbacks_list, verbose=1)

Y_testPred = model.predict(X_test_shuffled)


# AUC-ROC
fpr, tpr, threshold = roc_curve(y_true=y_test, y_score=Y_testPred)
# np.save(
#     NNModel_folderpath + f'NN_Classifier_{classifier_name}_{layer}hiddenlayer-fprtpr-features_{feature_set}-target_{classify_label}-window_{input_window * 2}mins-sampling_{sampling_name}-round_{round_id}-{timestamp}.npy',
#     np.array([fpr, tpr]))
auc_score = auc(fpr, tpr)
# AUC-ROC best threshold
best_thre, best_tpr, best_fpr = opt_threshold(fpr, tpr, threshold, auc_score)
# AUC-PRC & Average precision socre
prc_precision, prc_recall, prc_threstholds = precision_recall_curve(y_true=y_test, probas_pred=Y_testPred)
ap_score = average_precision_score(y_true=y_test, y_score=Y_testPred)

Y_testPred = [1 if y >= best_thre else 0 for y in Y_testPred]
cm = confusion_matrix(y_pred=Y_testPred, y_true=y_test)

# metrics
f1 = f1_score(y_pred=Y_testPred, y_true=y_test)
sensitivity = cm[1][1] / (cm[1][1] + cm[1][0])  # TP/(TP+FN)
specificity = cm[0][0] / (cm[0][0] + cm[0][1])  # TN/(TN+FP)
precision_1 = cm[1][1] / (cm[1][1] + cm[0][1])  # TP/(TP+FP)
recall_1 = cm[1][1] / (cm[1][1] + cm[1][0])  # TP/(TP+FN)
accuracy = (cm[1][1] + cm[0][0]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])  # (TP+TN)/(TP+FP+FN+TN)
# model.save(NNModel_folderpath + f'NN_Classifier_{classifier_name}_{layer}hiddenlayer-features_{feature_set}-target_{classify_label}-window_{input_window * 2}mins-sampling_{sampling_name}-round_{round_id}-f1_{f1:.3f}-{timestamp}.h5')

# delete model to release RAM
del model
gc.collect()

df_res = pd.DataFrame([[auc_score, best_thre, f1, sensitivity, specificity,
                        precision_1, recall_1, accuracy, ap_score,cm[0][0], cm[0][1], cm[1][0], cm[1][1]]],
                        columns=['AUC', 'best_threshold', 'f1_score', 'sensitivity', 'specificity',
                                'precision_label1', 'recall_label1', 'accuracy', 'ap_score','TN', 'FP', 'FN', 'TP'])
print(df_res.T)
print(f'Confusion Matrix: \n{cm}')
print(f'Classification Report: \n{classification_report(y_true=y_test, y_pred=Y_testPred)}')
