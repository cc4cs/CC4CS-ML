# Regression Example With Boston Dataset: Standardized
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import math
import csv
import pickle
import time
import os
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.svm import SVR

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

def learning_curves(x_train, y_train, x_test, y_test,depth):
# Create 6 different models based on max_depth
        reg = depth
# Iteratively increase training set size
        training_t=time.time()
        reg.fit(x_train, y_train)
        training_t=time.time()-training_t

        test_t=time.time()
        p=reg.predict(x_test)
        test_t=time.time()-test_t

        #RMSE
        
        MSE = np.round(metrics.mean_squared_error(y_test, p), 2)
        print("RMSE for DTReg (All features): " , np.round(np.sqrt(MSE)))


        #NRMSE
        NRMSE=np.divide(MSE,np.std(y_test))
        print("NRMSE for DTReg (All features): " , NRMSE)

        #MAE
        MAE= np.round(metrics.mean_absolute_error(y_test, p), 2)
        print("MAE for DTReg (All features): " , MAE )

        #R2
    
        R2=np.round(metrics.r2_score(y_test, p), 2)
        print("RSquared for DTReg (All features): " , R2)
        
        return [MSE,NRMSE,MAE,R2,training_t,test_t]

                
        


iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
for p in iss:
# load dataset
    dataframe = read_csv("TotalParameterMatrix"+p+".csv", skipinitialspace=True,sep=';', header = 0)
    with open("TotalParameterMatrixArmv4t.csv") as f:
        firstline = f.readline().rstrip().split(';')

    with open(os.getcwd()+"\\"+p+"\\SVM\\results.csv",'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id','depth','MSE','NRMSE','MAE','R2','training_t','test_t'])
        train, test = splitting(dataframe)

        x_train = np.concatenate((train.to_numpy()[:,6:35],train.to_numpy()[:,40:104]), axis=1)
        y_train=train.to_numpy()[:,36:40]
        x_test = np.concatenate((test.to_numpy()[:,6:35],test.to_numpy()[:,40:104]), axis=1)
        y_test = test.to_numpy()[:,36:40]

        svr_lin = SVR(kernel="linear")
        #svr_rbf1 = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
        #svr_rbf3 = SVR(kernel="rbf", C=100, gamma=0.3, epsilon=0.1)
        #svr_rbf6 = SVR(kernel="rbf", C=100, gamma=0.6, epsilon=0.1)
        #svr_rbf9 = SVR(kernel="rbf", C=100, gamma=0.9, epsilon=0.1)
        svr_poly2 = SVR(kernel="poly", C=100, gamma="auto", degree=2, epsilon=0.1, coef0=1)
        svr_poly3 = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
        svr_poly4 = SVR(kernel="poly", C=100, gamma="auto", degree=4, epsilon=0.1, coef0=1)
        t=[svr_lin,svr_poly2,svr_poly3,svr_poly4]#svr_rbf1,svr_rbf3,svr_rbf6,svr_rbf9]

        
        #for depth in t:
        #        [MSE,NRMSE,MAE,R2,training_t,test_t]=learning_curves(x_train, y_train, x_test, y_test,depth)
        #        writer.writerow(['multi',str(depth),MSE,NRMSE,MAE,R2,training_t,test_t])
                

        #single target
        for i in [0,1,2,3,4]:
            train, test = train_test_split(dataframe, test_size=0.2)
            x_train = np.concatenate((train.to_numpy()[:,6:(35+i)],train.to_numpy()[:,(37+i):104]), axis=1)
            y_train=train.to_numpy()[:,(36+i)]
            x_test = np.concatenate((test.to_numpy()[:,6:(35+i)],test.to_numpy()[:,(37+i):104]), axis=1)
            y_test = test.to_numpy()[:,(36+i)]
            
            for depth in t:
                [MSE,NRMSE,MAE,R2,training_t,test_t]=learning_curves(x_train, y_train, x_test, y_test,depth)
                writer.writerow(['Single:'+firstline[(36+i)],depth,MSE,NRMSE,MAE,R2,training_t,test_t])

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
