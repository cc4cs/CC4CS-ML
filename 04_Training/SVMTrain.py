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
                

        #single target
        for i in [0,1,2,3,4]:
            x_train = np.concatenate((dataframe.to_numpy()[0:5,8:(36+i)],dataframe.to_numpy()[0:5,(38+i):106]), axis=1)
            y_train=dataframe.to_numpy()[0:5,(37+i)]
            reg = SVR(kernel="linear")
            training_t=time.time()
            reg.fit(x_train, y_train)
            training_t=time.time()-training_t
            filename = os.getcwd()+'\\'+p+'\SVM\\'+firstline[37+i]+'.sav'
            pickle.dump(reg, open(filename, 'wb'))
            print('training time for SVM_'+p+firstline[37+i]+':'+str(training_t))