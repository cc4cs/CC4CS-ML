
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

    
MSE=[]
MAE=[]
NRMSE=[]
R2=[]

iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
for p in iss:
# load dataset
        dataframe = read_csv("TotalParameterMatrix"+p+"Test.csv", skipinitialspace=True,sep=',', header = 0)
        with open("TotalParameterMatrix"+p+"Train.csv") as f:
                firstline = f.readline().rstrip().split(',')

        

        x_test = np.concatenate((dataframe.to_numpy()[:,8:37],dataframe.to_numpy()[:,42:106]), axis=1)
        y_test=dataframe.to_numpy()[:,38:42]
        print(y_test[:,0])
        
        with open(os.getcwd()+"\\"+p+"\\SVM\\results.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id','depth','MSE','NRMSE','MAE','R2'])

            
                
            
            d={1: 'Assembly',2:'Clock',3:'text',4:'data',5:'bss'}
            for depth in [2,4,8,16]:
                for x in range(1,5):
                    writer.writerow(['multi, feature: '+d[x],depth,MSE[x-1],NRMSE[x-1],MAE[x-1],R2[x-1]])

        #single target
            for i in [0,1,2,3,4]:
                x_test = np.concatenate((dataframe.to_numpy()[:,8:(36+i)],dataframe.to_numpy()[:,(38+i):106]), axis=1)
                y_test=dataframe.to_numpy()[:,(37+i)]
                testing_t=time.time()
                filename = os.getcwd()+'\\'+p+'\SVM\depth'+str(depth)+'_'+firstline[37+i]+'.sav'
                reg=pickle.load(open(filename, 'rb'))
                pp=reg.predict(x_test)
                testing_t=time.time()-testing_t
                print('training time for SVM'+p+'_depth'+str(depth)+'_'+firstline[37+i]+':'+str(testing_t))
                MSE=np.round(metrics.mean_squared_error(y_test, pp), 2)
                NRMSE=np.divide(MSE,np.std(y_test))
                MAE=np.round(metrics.mean_absolute_error(y_test, pp), 2)
                R2=np.round(metrics.r2_score(y_test, pp), 2)

                writer.writerow(['Single:'+firstline[(37+i)],depth,MSE,NRMSE,MAE,R2])