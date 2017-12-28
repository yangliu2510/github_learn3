import numpy as np
import pandas as pd
import math

#prepare source_pos dictionary
source_pos = {'X1':['11.6-5.5.log_608'],
              'X2':['14.5-3.5.log_760'],
              'X3':['20.2-5.5.log_615'],
              'X4':['23.5-5.0.log_637'],
              'X5':['26-12.log_981'],
              'X6':['26.5-0.5.log_297'],
              'X7':['36-13.log_804'],
              'X8':['38.2-5.5.log_709'],
              'X9':['7.5-9.5.log_342']
             }

for i in source_pos.keys():
    index1 = source_pos[i][0].find('-')
    index2 = source_pos[i][0].find('.log')
    source_pos[i].append([float(source_pos[i][0][0:index1]),float(source_pos[i][0][index1+1:index2])])

#prepare probe_pos dictionary
probe_pos_x = {"e4956e410ac2": 8.5, 
               "e4956e4e540a": 12.0,
               "e4956e410abd": 17.5,
               "e4956e410b4c": 20.5,
               "e4956e410b32": 23.5,
               "e4956e410ac0": 26.8,
               "e4956e4e53e4": 26.8, 
               "e4956e410acf": 35.5,
               "e4956e4e53e7": 38.2
              }

probe_pos_y = {"e4956e410ac2": 5.7, 
               "e4956e4e540a": 9.0,
               "e4956e410abd": 5.5,
               "e4956e410b4c": 7.2,
               "e4956e410b32": 5.2,
               "e4956e410ac0": 8.0,
               "e4956e4e53e4": 5.5, 
               "e4956e410acf": 5.5,
               "e4956e4e53e7": 7.4
              }

namelist = ['Timestamp','ProbeMac','SourceMac','DestinationMac','BSSID','FrameType','RSSI','Channel','SSID']
columns = ['Timestamp','ProbeMac','SourceMac','FrameType','RSSI','source_x','source_y','probe_x','probe_y','distance']
dataset_train = pd.DataFrame(columns = columns)
dataset_test = pd.DataFrame(columns = columns)

for i in source_pos.keys():
    raw_dataset = pd.read_csv('/home/hadoop/sdl/hdfs_data//' + source_pos[i][0], header=None, names = namelist, encoding='utf-8')

    #select columns
    dataset_up = pd.DataFrame(raw_dataset, columns = ['Timestamp','ProbeMac','SourceMac','FrameType','RSSI'])

    #select rows
    odd_row = dataset_up.shape[0]
    for j in xrange(odd_row):
        if dataset_up['ProbeMac'][j] not in probe_pos_x.keys():
            dataset_up = dataset_up.drop(j)

    #add source position
    dataset_up['source_x'] = source_pos[i][1][0]
    dataset_up['source_y'] = source_pos[i][1][1]

    #add probe position
    new_row = dataset_up.shape[0]
    dataset_up['probe_x'] = dataset_up['ProbeMac'].map(probe_pos_x)
    dataset_up['probe_y'] = dataset_up['ProbeMac'].map(probe_pos_y)
    dist_sq = (dataset_up['source_x'] - dataset_up['probe_x'])**2 + (dataset_up['source_y'] - dataset_up['probe_y'])**2
    dataset_up['distance'] = np.sqrt(dist_sq)
    
    #deal with Timestamp
    dataset_up['Timestamp'] = dataset_up['Timestamp'].astype(str)
    dataset_up['Timestamp'] = dataset_up['Timestamp'].map(lambda x: x[0:10])
    dataset_up['Timestamp'] = dataset_up['Timestamp'].astype(int)
    
    if i != 'X5':
        dataset_train = dataset_train.append(dataset_up)
        print (i + ' train ready!')
    else:
        dataset_test = dataset_test.append(dataset_up)
        print (i + ' test ready!')
        
#data preprocessing
train = dataset_train

#deleted duplicated
train.drop_duplicates()

#exclude outliers
train = train[(train['RSSI'] > -90)]
train = train[(train['RSSI'] < -40)]

#add dumpy variable
probe_dummies = pd.get_dummies(train['ProbeMac'])
source_dummies = pd.get_dummies(train['SourceMac'])
frame_dummies = pd.get_dummies(train['FrameType'])

new_train = pd.concat([train[['RSSI','distance']],probe_dummies,source_dummies,frame_dummies], axis = 1)

#XGBoost Regression Model
import xgboost as xgb  
from sklearn.cross_validation import train_test_split

#prepare training set and testing set
train_xy,val = train_test_split(new_train, test_size = 0.1,random_state=1)

tra_y = train_xy.distance
tra_X = train_xy.drop(['distance'],axis=1)
tra_X = tra_X.sort_index(axis=1)
val_y = val.distance
val_X = val.drop(['distance'],axis=1)
val_X = val_X.sort_index(axis=1)

#xgb矩阵赋值
xgb_val = xgb.DMatrix(val_X,label=val_y)
xgb_train = xgb.DMatrix(tra_X, label=tra_y)
params={
        'booster':'gbtree',
        'objective': 'reg:linear', 
        'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth':3, # 构建树的深度，越大越容易过拟合
        'lambda':1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'min_child_weight':1, 
        'silent':1 ,#设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.8, # 如同学习率
        'seed':1000,
        'nthread':-1,# cpu 线程数
        'eval_metric': 'rmse'
        }

plst = list(params.items())
num_rounds = 500 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
model.save_model('/home/hadoop/sdl/liuyang/xgb.model') # 用于存储训练出的模型
print "best best_ntree_limit",model.best_ntree_limit 

import os
import numpy as np
import pandas as pd
import math
import xgboost as xgb  
from sklearn.cross_validation import train_test_split

def get_location_internal(log_file):
    lat = 0.0
    lng = 0.0
    variance = 0.0
   
    #initialize index
    probe_pos_x, probe_pos_y, column_name_list = index_initializer()
    
    #prepare test dataset
    test_logfile, test_dataset = prepare_test_data (log_file, probe_pos_x,  probe_pos_y, column_name_list)
    
    #predict distance
    test_preds = xgb_prediction(testset = test_logfile)
    
    #calculate location estimation and variance
    lat, lng, std_dist, std_lse, variance = postion_estimation(dataset = test_dataset, preds = test_preds)
    print (lat, lng, std_dist, std_lse, variance)


def get_location(log_file):
    if not os.path.exists(log_file):
        print("Cannot find file: ".format(log_file))
    get_location_internal(log_file)


if __name__ == "__main__":
    get_location("/home/hadoop/sdl/hdfs_data//26-12.log_981")
    
def index_initializer():
    probe_pos_x = {"e4956e410ac2": 8.5, "e4956e4e540a": 12.0, "e4956e410abd": 17.5,
                   "e4956e410b4c": 20.5, "e4956e410b32": 23.5, "e4956e410ac0": 26.8,
                   "e4956e4e53e4": 26.8, "e4956e410acf": 35.5,"e4956e4e53e7": 38.2
                  }

    probe_pos_y = {"e4956e410ac2": 5.7, "e4956e4e540a": 9.0, "e4956e410abd": 5.5,
                   "e4956e410b4c": 7.2, "e4956e410b32": 5.2, "e4956e410ac0": 8.0,
                   "e4956e4e53e4": 5.5, "e4956e410acf": 5.5, "e4956e4e53e7": 7.4
                  }
    column_name_list = ['58:1f:28:16:37:ab','6c:5c:14:28:b4:03',
                        '80:13:82:f5:fd:c7','ac:37:43:52:d4:dc',
                        'b4:ef:fa:87:0f:37','e4956e410abd',
                        'e4956e410ac0','e4956e410ac2',
                        'e4956e410acf','e4956e410b32',
                        'e4956e410b4c','e4956e4e53e7',
                        'ACK','ACTION','ASOCRQ','AUTH',
                        'BACK','BACKRQ','CFEND','CTS','DEAUTH',
                        'PROBRQ','QDATA','QDNULL','RSSI','RTS']
    
    #columns = ['Timestamp','ProbeMac','SourceMac','FrameType','RSSI','probe_x','probe_y']
    
    return probe_pos_x, probe_pos_y, column_name_list 
def prepare_test_data(log_file,  probe_pos_x,  probe_pos_y, column_name_list):
    
    namelist = ['Timestamp','ProbeMac','SourceMac','DestinationMac','BSSID','FrameType','RSSI','Channel','SSID']
    raw_dataset = pd.read_csv(log_file, header=None, names = namelist, encoding='utf-8')
    
    #select columns
    dataset = pd.DataFrame(raw_dataset, columns = ['Timestamp','ProbeMac','SourceMac','FrameType','RSSI'])
    
    #select rows
    odd_row = dataset.shape[0]
    for j in xrange(odd_row):
        if dataset['ProbeMac'][j] not in probe_pos_x.keys():
            dataset = dataset.drop(j)
            
    #add probe position
    new_row = dataset.shape[0]
    dataset['probe_x'] = dataset['ProbeMac'].map(probe_pos_x)
    dataset['probe_y'] = dataset['ProbeMac'].map(probe_pos_y)

    #deal with time
    dataset['Timestamp'] = dataset['Timestamp'].astype(str)
    dataset['Timestamp'].map(lambda x: x[0:10])
    dataset['Timestamp'] = dataset['Timestamp'].astype(int)

    #data preprocessing
    #deleted duplicated
    dataset.drop_duplicates()

    #exclude outliers
    dataset = dataset[(dataset['RSSI'] > -90)]
    dataset = dataset[(dataset['RSSI'] < -40)]

    #add dumpy variable
    probe_dummies = pd.get_dummies(dataset['ProbeMac'])
    source_dummies = pd.get_dummies(dataset['SourceMac'])
    frame_dummies = pd.get_dummies(dataset['FrameType'])

    test_logfile = pd.concat([dataset[['RSSI']],probe_dummies,source_dummies,frame_dummies], axis = 1)

    #matching columns with train dataset
    for i in list(test_logfile):
        if i not in column_name_list:
            test_logfile = test_logfile.drop(i,axis=1)
    for i in column_name_list:
        if i not in list(test_logfile):
            test_logfile[i] = 0

    test_logfile = test_logfile.sort_index(axis = 1)
    
    return test_logfile, dataset

def xgb_prediction(testset):
    bst = xgb.Booster(model_file = '/home/hadoop/sdl/liuyang/xgb.model') 
    xgb_test = xgb.DMatrix(testset)
    preds = bst.predict(xgb_test)
    return preds

def postion_estimation(dataset,preds):
    #prepare dataset
    cal_pre1 = pd.DataFrame(dataset, columns = ['probe_x','probe_y'])
    cal_pre1['dist'] = preds
    cal_pre2 = cal_pre1.groupby(['probe_x','probe_y'])['dist'].median()
    cal_pre2 = cal_pre2.sort_index(axis = 0)[0:4] #select closest 4 probes
    row_number = cal_pre2.shape[0]

    #prepare matrix
    cal_pre2 = cal_pre2.reset_index()  
    cal_pre3 = pd.DataFrame()
    cal_pre3['2x_2xn'] = 2*(cal_pre2['probe_x'] - cal_pre2['probe_x'][row_number - 1])
    cal_pre3['2y_2yn'] = 2*(cal_pre2['probe_y'] - cal_pre2['probe_y'][row_number - 1])
    cal_pre3['x_xn_sq'] = cal_pre2['probe_x']**2 - cal_pre2['probe_x'][row_number - 1]**2
    cal_pre3['y_yn_sq'] = cal_pre2['probe_y']**2 - cal_pre2['probe_y'][row_number - 1]**2
    cal_pre3['d_dn_sq'] = cal_pre2['dist']**2 - cal_pre2['dist'][row_number - 1]**2
    cal_pre3['b'] = cal_pre3['x_xn_sq'] + cal_pre3['y_yn_sq'] - cal_pre3['d_dn_sq']

    A = np.array(cal_pre3.ix[:row_number - 2,['2x_2xn','2y_2yn']])
    b = np.array(cal_pre3.ix[:row_number - 2,['b']])

    #calculate postion 
    X,lat,lng = lse_calculation(A, b)
    
    #calculate variance
    std_dist = (cal_pre1.groupby(['probe_x','probe_y'])['dist'].std()).mean()
    std_lse = np.sqrt(abs(np.dot(A,X) - b)).mean()
    std = (std_dist + std_lse)/2
    var = math.pow(std,2)
    
    return lat, lng, std_dist, std_lse, var

def lse_calculation(A, b):
    A_inv = np.linalg.pinv(A)
    X = np.dot(A_inv, b )
    lat = X[0,0]
    lng = X[1,0]
    return X, lat, lng
