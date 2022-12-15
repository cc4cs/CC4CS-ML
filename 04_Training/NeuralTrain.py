# Regression Example With Boston Dataset: Standardized
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
from keras.layers.activation import LeakyReLU
import tensorflow as tf
import math
from keras import layers
import csv
import time
import pandas as pd
import os
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.svm import SVR
import pickle
from sklearn.ensemble import RandomForestRegressor


feaature=['Total_Operands',  'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Assignment', 'CyclomaticComplexity', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT', 'cInstr', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_22', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_32', 'Sloc', 'PointerDereferencing', 'SCALAR_INPUT', 'RANGE_SCALAR_VALUES', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_5', 'V_VAL_6', 'V_VAL_7', 'V_VAL_8', 'V_VAL_9', 'V_VAL_10', 'V_VAL_11', 'V_VAL_12', 'V_VAL_13', 'V_VAL_14', 'V_VAL_15', 'V_VAL_16', 'V_VAL_17', 'V_VAL_18', 'V_VAL_19', 'V_VAL_20', 'V_VAL_21', 'V_VAL_22', 'V_VAL_23', 'Goto', 'ExitPoint', 'Function', 'FunctionCall', 'SCALAR_INDEX_INPUT', 'RANGE_SCALAR_INDEX_VALUES']
MSE=[]
MAE=[]
NRMSE=[]
R2=[]
iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
feature=[]
for p in iss:
# load dataset
        feaature.append('assemblyInstr')
        feaature.append('clockCycles')
        feaature.append('text')
        feaature.append('data')
        feaature.append('data')
        feaature.append('bss')
        dataframe = read_csv("TotalParameterMatrix"+p+"Train.csv", skipinitialspace=True,sep=',', header = 0)
        for f in dataframe.columns.values:
                if f not in feaature:
                        feature.append(f)
        dataframeFeature=dataframe.drop(feature,axis=1)
        with open("TotalParameterMatrix"+p+"Train.csv") as f:
            firstline = f.readline().rstrip().split(',')

        

        #x_train = np.concatenate((dataframe.to_numpy()[:,8:37],dataframe.to_numpy()[:,42:106]), axis=1)
        #y_train=dataframe.to_numpy()[:,38:42]
        
        with open(os.getcwd()+"\\"+p+"\\NN\\resultsTrain.csv",'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id','MSE','MAE','RMSE','time'])

                """ MSE=[]
                MAE=[]
                NRMSE=[]
                R2=[]
                
                for depth in [8,10,12,14,16]:
                        # define the keras model
                        training_t=time.time()
                        reg = Sequential()
                        reg.add(Dense(12, input_shape=(92,), activation='relu'))
                        reg.add(Dense(92, activation='relu'))
                        reg.add(Dense(1, activation='sigmoid'))
                        reg.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                        reg.fit(x_train.astype(np.float32),y_train.astype(np.float32), epochs=150, batch_size=10)
                        pp=reg.predict(x_train.astype(np.float32))
                        training_t=time.time()-training_t
                        filename = os.getcwd()+'\\'+p+'\\NN\depth'+str(depth)+'_Multi.sav'
                        pickle.dump(reg, open(filename, 'wb'))

                        timet=training_t
                        MSE.append(np.round(metrics.mean_squared_error(y_train[:,0], pp[:,0]), 2))
                        NRMSE.append(np.divide(MSE[0],np.std(y_train[:,0])))
                        MAE.append(np.round(metrics.mean_absolute_error(y_train[:,0], pp[:,0]), 2))
                        R2.append(np.round(metrics.r2_score(y_train[:,0], pp[:,0]), 2))
                        
                        MSE.append(np.round(metrics.mean_squared_error(y_train[:,1], pp[:,1]), 2))
                        NRMSE.append(np.divide(MSE[1],np.std(y_train[:,1])))
                        MAE.append(np.round(metrics.mean_absolute_error(y_train[:,1], pp[:,1]), 2))
                        R2.append(np.round(metrics.r2_score(y_train[:,1], pp[:,1]), 2))

                        MSE.append(np.round(metrics.mean_squared_error(y_train[:,2], pp[:,2]), 2))
                        NRMSE.append(np.divide(MSE[2],np.std(y_train[:,2])))
                        MAE.append(np.round(metrics.mean_absolute_error(y_train[:,2], pp[:,2]), 2))
                        R2.append(np.round(metrics.r2_score(y_train[:,2], pp[:,2]), 2))

                        MSE.append(np.round(metrics.mean_squared_error(y_train[:,3], pp[:,3]), 2))
                        NRMSE.append(np.divide(MSE[3],np.std(y_train[:,3])))
                        MAE.append(np.round(metrics.mean_absolute_error(y_train[:,3], pp[:,3]), 2))
                        R2.append(np.round(metrics.r2_score(y_train[:,3], pp[:,3]), 2))

                d={1: 'Assembly',2:'Clock',3:'text',4:'data',5:'bss'}
                for depth in [8,10,12,14,16]:
                        for x in range(1,5):
                                writer.writerow(['multi, feature: '+d[x],depth,MSE[x-1],NRMSE[x-1],MAE[x-1],R2[x-1],timet]) """

        #single target
                cvscores = []
                for i in [0,1,2,3,4]:
                        x_train = np.concatenate((dataframe.to_numpy()[:,8:(37+i)],dataframe.to_numpy()[:,(38+i):106]), axis=1)
                        y_train=dataframe.to_numpy()[:,(37+i)]
                        #for ep in [150,250,500,1000]:
                        training_t=time.time()
                        reg = tf.keras.Sequential()
                        reg.add(Dense(12, input_shape=(96,), activation='relu'))
                        reg.add(Dense(8, activation='relu'))
                        reg.add(Dense(1, activation='sigmoid'))
                        reg.compile(loss='mean_squared_error', optimizer='SGD', metrics=['MeanAbsoluteError','RootMeanSquaredError'])
                        reg.fit(np.asarray(x_train).astype(np.float32), np.asarray(y_train).astype(np.float32), epochs=64,batch_size=12000)
                        scores = reg.evaluate(np.asarray(x_train).astype(np.float32), np.asarray(y_train).astype(np.float32), verbose=0)
                        print(scores)
                        training_t=time.time()-training_t
                        filename = os.getcwd()+'\\'+p+'\\NN\\depth'+'_'+firstline[37+i]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))


                        writer.writerow(['Single:'+firstline[(37+i)],scores[0],scores[1],scores[2],training_t])

                        x_train = np.concatenate((dataframeFeature.to_numpy()[:,0:(29+i)],dataframeFeature.to_numpy()[:,(30+i):len(dataframeFeature)]), axis=1)
                        y_train=dataframe.to_numpy()[:,29+i]
                        #for ep in [150,250,500,1000]:
                        training_t=time.time()
                        reg = tf.keras.Sequential()
                        reg.add(Dense(12, input_shape=(81,), activation='relu'))
                        reg.add(Dense(8, activation='relu'))
                        reg.add(Dense(1, activation='sigmoid'))
                        reg.compile(loss='mean_squared_error', optimizer='SGD', metrics=['MeanAbsoluteError','RootMeanSquaredError'])
                        reg.fit(np.asarray(x_train).astype(np.float32), np.asarray(y_train).astype(np.float32), epochs=64,batch_size=12000)
                        scores = reg.evaluate(np.asarray(x_train).astype(np.float32), np.asarray(y_train).astype(np.float32), verbose=0)
                        print(scores)
                        training_t=time.time()-training_t
                        filename = os.getcwd()+'\\'+p+'\\NN\\depth'+'_'+firstline[37+i]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))


                        writer.writerow(['Single, feature removed:'+firstline[(37+i)],scores[0],scores[1],scores[2],training_t])