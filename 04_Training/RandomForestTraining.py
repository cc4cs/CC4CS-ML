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
import math
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


iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]

for p in iss:
# load dataset
        dataframe = read_csv("TotalParameterMatrix"+p+"Train.csv", skipinitialspace=True,sep=',', header = 0)
        with open("TotalParameterMatrix"+p+"Train.csv") as f:
                firstline = f.readline().rstrip().split(',')

        

        x_train = np.concatenate((dataframe.to_numpy()[:,8:39],dataframe.to_numpy()[:,43:106]), axis=1)
        y_train=dataframe.to_numpy()[:,39:42]
        d={1:'text',2:'data',3:'bss'}
        with open(os.getcwd()+"\\"+p+"\\RandomForest\\resultsTrain.csv",'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id','depth','MSE','NRMSE','MAE','R2','time'])

                MSE=[]
                MAE=[]
                NRMSE=[]
                R2=[]
                
                for depth in [8,10,12,14,16]:
                        reg = RandomForestRegressor(n_estimators= depth)
                        training_t=time.time()
                        reg.fit(x_train, y_train)
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
                        filename = os.getcwd()+'\\'+p+'\\RandomForest\depth'+str(depth)+'_Multi.sav'
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


                
                        for x in range(1,4):
                                writer.writerow(['multi, feature: '+d[x],depth,MSE[x-1],NRMSE[x-1],MAE[x-1],R2[x-1],training_t])
                        MSE=[]
                        MAE=[]
                        NRMSE=[]
                        R2=[]

        #single target
                for i in [0,1,2,3,4]:
                        x_train = np.concatenate((dataframe.to_numpy()[:,8:(37+i)],dataframe.to_numpy()[:,(38+i):106]), axis=1)
                        y_train=dataframe.to_numpy()[:,(37+i)]
                        for depth in [8,10,12,14,16]:
                                reg = RandomForestRegressor(n_estimators = depth)
                                training_t=time.time()
                                reg.fit(x_train, y_train)
                                pp=reg.predict(x_train)
                                training_t=time.time()-training_t

                                filename = os.getcwd()+'\\'+p+'\RandomForest\depth'+str(depth)+'_'+firstline[37+i]+'.sav'
                                pickle.dump(reg, open(filename, 'wb'))
                                MSE=np.round(metrics.mean_squared_error(y_train, pp), 2)
                                NRMSE=np.divide(MSE,np.std(y_train))
                                MAE=np.round(metrics.mean_absolute_error(y_train, pp), 2)
                                R2=np.round(metrics.r2_score(y_train, pp), 2)

                                writer.writerow(['Single:'+firstline[(37+i)],depth,MSE,NRMSE,MAE,R2,training_t])
        

