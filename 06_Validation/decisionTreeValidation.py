
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
        dataframe = read_csv("TotalParameterMatrix"+p+".csv", skipinitialspace=True,sep=';', header = 0)
        with open("TotalParameterMatrix"+p+".csv") as f:
                firstline = f.readline().rstrip().split(';')
        
        x_test = np.concatenate((dataframe.to_numpy()[:,6:35],dataframe.to_numpy()[:,41:len(dataframe.to_numpy()[0,:])]), axis=1)
        y_test=dataframe.to_numpy()[:,38:41]
        d={1:'text',2:'data',3:'bss'}
        with open(os.getcwd()+"\\"+p+"\\REG_TREE\\resultsValidation.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id','depth','MSE','NRMSE','RMSPE','MAE','R2','time'])

            MSE=[]
            MAE=[]
            NRMSE=[]
            R2=[]
            rmspe=[]
            for depth in [8,10,12,16]:
                filename = os.getcwd()+'\\'+p+'\\REG_TREE\depth'+str(depth)+'_Multi.sav'
                reg=pickle.load(open(filename, 'rb'))
                testing_t=time.time()
                pp=reg.predict(x_test)
                testing_t=time.time()-testing_t
                
                MSE.append(np.round(metrics.mean_squared_error(y_test[:,0], pp[:,0]), 2))
                NRMSE.append(np.divide(MSE[0],np.std(y_test[:,0])))
                MAE.append(np.round(metrics.mean_absolute_error(y_test[:,0], pp[:,0]), 2))
                R2.append(np.round(metrics.r2_score(y_test[:,0], pp[:,0]), 2))
                y_train0=[]
                pp0=[]
                for c in range(0,len(y_test)):
                    if y_test[c,0]!=0:
                        y_train0.append(y_test[c,0])
                        pp0.append(pp[c,0])
                rmspe.append((np.sqrt(np.mean(np.square(np.divide((np.array(y_train0) - np.array(pp0)), np.array(y_train0)))))) * 100)

                MSE.append(np.round(metrics.mean_squared_error(y_test[:,1], pp[:,1]), 2))
                NRMSE.append(np.divide(MSE[1],np.std(y_test[:,1])))
                MAE.append(np.round(metrics.mean_absolute_error(y_test[:,1], pp[:,1]), 2))
                R2.append(np.round(metrics.r2_score(y_test[:,1], pp[:,1]), 2))
                y_train0=[]
                pp0=[]
                for c in range(0,len(y_test)):
                    if y_test[c,1]!=0:
                        y_train0.append(y_test[c,1])
                        pp0.append(pp[c,1])
                rmspe.append((np.sqrt(np.mean(np.square(np.divide((np.array(y_train0) - np.array(pp0)), np.array(y_train0)))))) * 100)

                MSE.append(np.round(metrics.mean_squared_error(y_test[:,2], pp[:,2]), 2))
                NRMSE.append(np.divide(MSE[2],np.std(y_test[:,2])))
                MAE.append(np.round(metrics.mean_absolute_error(y_test[:,2], pp[:,2]), 2))
                R2.append(np.round(metrics.r2_score(y_test[:,2], pp[:,2]), 2))
                y_train0=[]
                pp0=[]
                for c in range(0,len(y_test)):
                    if y_test[c,2]!=0:
                        y_train0.append(y_test[c,2])
                        pp0.append(pp[c,2])
                rmspe.append((np.sqrt(np.mean(np.square(np.divide((np.array(y_train0) - np.array(pp0)), np.array(y_train0)))))) * 100)

                
                for x in range(1,4):
                    writer.writerow(['multi, feature: '+d[x],depth,MSE[x-1],NRMSE[x-1],rmspe[x-1],MAE[x-1],R2[x-1],testing_t])
                MSE=[]
                MAE=[]
                NRMSE=[]
                R2=[]

        #single target
            for i in [0,1,2,3,4]:
                x_test = np.concatenate((dataframe.to_numpy()[:,6:35],dataframe.to_numpy()[:,41:len(dataframe.to_numpy()[0,:])]), axis=1)
                y_test=dataframe.to_numpy()[:,(36+i)]

                for depth in [8,10,12,16]:
                    testing_t=time.time()
                    filename = os.getcwd()+'\\'+p+'\REG_TREE\depth'+str(depth)+'_'+firstline[36+i]+'.sav'
                    reg=pickle.load(open(filename, 'rb'))
                    pp=reg.predict(x_test)
                    testing_t=time.time()-testing_t
                    MSE=np.round(metrics.mean_squared_error(y_test, pp), 2)
                    NRMSE=np.divide(MSE,np.std(y_test))
                    MAE=np.round(metrics.mean_absolute_error(y_test, pp), 2)
                    R2=np.round(metrics.r2_score(y_test, pp), 2)
                    y_train0=[]
                    pp0=[]
                    for c in range(0,len(y_test)):
                        if y_test[c]!=0:
                            y_train0.append(y_test[c])
                            pp0.append(pp[c])
                    
                    rmspe=(np.sqrt(np.mean(np.square(np.divide((np.array(y_train0) - np.array(pp0)), np.array(y_train0)))))) * 100

                    writer.writerow(['Single:'+firstline[(36+i)],depth,MSE,NRMSE,rmspe,MAE,R2,testing_t])