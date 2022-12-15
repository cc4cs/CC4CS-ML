
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
import math
import csv
import time
import pandas as pd
import os
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
from sklearn import svm
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

        
            with open(os.getcwd()+"\\"+p+"\\SVM\\resultsTrain.csv",'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id','MSE','NRMSE','MAE','R2','time'])
                    for i in [0,1,2,3,4]:
                
                #single target
                    
                        reg = svm.LinearSVR()
                        x_train = np.concatenate((dataframe.to_numpy()[:,8:(37+i)],dataframe.to_numpy()[:,(38+i):106]), axis=1)
                        
                        y_train=dataframe.to_numpy()[:,(37+i)]
                        training_t=time.time()
                        reg.fit(x_train,y_train)
                        filename = os.getcwd()+'\\'+p+'\SVM\\'+firstline[37+i]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
                        MSE=np.round(metrics.mean_squared_error(y_train, pp), 2)
                        NRMSE=np.divide(MSE,np.std(y_train))
                        MAE=np.round(metrics.mean_absolute_error(y_train, pp), 2)
                        R2=np.round(metrics.r2_score(y_train, pp), 2)

                        writer.writerow(['Single:'+firstline[(37+i)],MSE,NRMSE,MAE,R2,training_t])
                
                #single target
                    
                        reg = svm.LinearSVR()
                        x_train = np.concatenate((dataframeFeature.to_numpy()[:,0:(29+i)],dataframeFeature.to_numpy()[:,(30+i):len(dataframeFeature)]), axis=1)
                        y_train=dataframe.to_numpy()[:,29+i]
                        training_t=time.time()
                        reg.fit(x_train,y_train)
                        filename = os.getcwd()+'\\'+p+'\SVM\FEATURE'+firstline[(37+i)]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
                        MSE=np.round(metrics.mean_squared_error(y_train, pp), 2)
                        NRMSE=np.divide(MSE,np.std(y_train))
                        MAE=np.round(metrics.mean_absolute_error(y_train, pp), 2)
                        R2=np.round(metrics.r2_score(y_train, pp), 2)

                        writer.writerow(['Single, feature removed:'+firstline[(37+i)],MSE,NRMSE,MAE,R2,training_t])
                        

                        