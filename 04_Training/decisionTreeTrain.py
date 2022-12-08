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

    
def splitting(results):
        col=results.columns
        functions=['banker_algorithm','bellmanford','gcd','insertionsort','selectionsort','kruskal','bubble_sort','bfs','fibcall','binary_search','matrix_mult','qsort','quicksort','park_miller','mergesort']
        training = pd.DataFrame(columns=col)
        test = pd.DataFrame(columns=col)
        data = pd.DataFrame(columns=col)
        train=pd.DataFrame(columns=col)
        test=pd.DataFrame(columns=col)
        for f in functions:
                if f=='matrixmult':
                        t,tt=train_test_split(dataframe[dataframe['FUNCTION']=='matrix_mult'][0:2000], test_size=0.2)
                t,tt=train_test_split(dataframe[dataframe['FUNCTION']==f][0:1000], test_size=0.2)
                train=pd.merge(train,t,how='outer')
                test=pd.merge(test,tt,how='outer')
        return train,test

iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
for p in iss:
# load dataset
        dataframe = read_csv("TotalParameterMatrix"+p+"Train.csv", skipinitialspace=True,sep=',', header = 0)
        with open("TotalParameterMatrix"+p+"Train.csv") as f:
                firstline = f.readline().rstrip().split(',')

        

        x_train = np.concatenate((dataframe.to_numpy()[:,8:37],dataframe.to_numpy()[:,42:106]), axis=1)
        y_train=dataframe.to_numpy()[:,38:42]
        
        
        for depth in [2,4,8,16]:
                reg = DecisionTreeRegressor(max_depth= depth)
                training_t=time.time()
                reg.fit(x_train, y_train)
                training_t=time.time()-training_t
                filename = os.getcwd()+'\\'+p+'\\REG_TREE\depth'+str(depth)+'_Multi.sav'
                pickle.dump(reg, open(filename, 'wb'))
                print('training time for randomForest_'+p+'_depth'+str(depth)+'_Multi:'+str(training_t))
                

        #single target
        for i in [0,1,2,3,4]:
    
                
                x_train = np.concatenate((dataframe.to_numpy()[:,8:(36+i)],dataframe.to_numpy()[:,(38+i):106]), axis=1)
                y_train=dataframe.to_numpy()[:,(37+i)]

                for depth in [2,4,8,16]:
                        reg = DecisionTreeRegressor(max_depth = depth)
                        training_t=time.time()
                        reg.fit(x_train, y_train)
                        training_t=time.time()-training_t
                        filename = os.getcwd()+'\\'+p+'\REG_TREE\depth'+str(depth)+'_'+firstline[37+i]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        print('training time for randomForest_'+p+'_depth'+str(depth)+'_'+firstline[37+i]+':'+str(training_t))
        

    






