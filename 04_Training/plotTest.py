
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

featureMin=[]

MSE=[]
MAE=[]
NRMSE=[]
R2=[]
iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]

for p in iss:
    DecisionTest = read_csv(os.getcwd()+'\\'+p+'\\REG_TREE\\resultsTest.csv', skipinitialspace=True,sep=',', header = 0)
    RandomTest = read_csv(os.getcwd()+'\\'+p+'\\RandomForest\\resultsTest.csv', skipinitialspace=True,sep=',', header = 0)
    SVMTest = read_csv(os.getcwd()+'\\'+p+'\\SVM\\resultsTest.csv', skipinitialspace=True,sep=',', header = 0)
    NNTest = read_csv(os.getcwd()+'\\'+p+'\\DNeuralNetwork\\resultsTest.csv', skipinitialspace=True,sep=',', header = 0)

    for i in range(0,5):
        title=DecisionTest.to_numpy()[i,0]

        MSEd=DecisionTest.to_numpy()[i,2]
        RMSPEd=DecisionTest.to_numpy()[i,4]
        NRMSEd=DecisionTest.to_numpy()[i,3]
        MAEd=DecisionTest.to_numpy()[i,5]
        timed=DecisionTest.to_numpy()[i,7]
        MAPEd=DecisionTest.to_numpy()[i,8]

        MSEr=RandomTest.to_numpy()[i,2]
        RMSPEr=RandomTest.to_numpy()[i,4]
        NRMSEr=RandomTest.to_numpy()[i,3]
        MAEr=RandomTest.to_numpy()[i,5]
        timer=RandomTest.to_numpy()[i,7]
        MAPEr=RandomTest.to_numpy()[i,8]

        MSEs=SVMTest.to_numpy()[i,2]
        RMSPEs=SVMTest.to_numpy()[i,4]
        NRMSEs=SVMTest.to_numpy()[i,3]
        MAEs=SVMTest.to_numpy()[i,5]
        times=SVMTest.to_numpy()[i,7]
        MAPEs=SVMTest.to_numpy()[i,8]

        MSEn=NNTest.to_numpy()[i,2]
        RMSPEn=NNTest.to_numpy()[i,4]
        NRMSEn=NNTest.to_numpy()[i,3]
        MAEn=NNTest.to_numpy()[i,5]
        timen=NNTest.to_numpy()[i,7]
        MAPEn=NNTest.to_numpy()[i,8]

        plt.figure(figsize=(9, 3))
        plt.subplot(2,3,1)
        
        bars = ('RMSPE Random Forest','RMSPE Decision Tree')
        y_pos = np.arange(len(bars))

        # Create bars
        plt.bar(y_pos, [RMSPEr,RMSPEd])

        # Create names on the x-axis
        plt.xticks(y_pos, bars)


        plt.subplot(2,3,2)
        bars = ('NRMSE Random Forest','NRMSE Decision Tree')
        y_pos = np.arange(len(bars))

        # Create bars
        plt.bar(y_pos, [NRMSEr,NRMSEd])

        # Create names on the x-axis
        plt.xticks(y_pos, bars)


        plt.subplot(2,3,3)
        bars = ('MAPE Random Forest','MAPE Decision Tree')
        y_pos = np.arange(len(bars))
        # Create bars
        plt.bar(y_pos, [RMSPEs,RMSPEn])

        # Create names on the x-axis
        plt.xticks(y_pos, bars)

        plt.subplot(2,3,4)
        
        bars = ('RMSPE SVM','RMSPE NN')
        y_pos = np.arange(len(bars))

        # Create bars
        plt.bar(y_pos, [RMSPEs,RMSPEn])

        # Create names on the x-axis
        plt.xticks(y_pos, bars)


        plt.subplot(2,3,5)
        bars = ('NRMSE SVM','NRMSE NN')
        y_pos = np.arange(len(bars))

        # Create bars
        plt.bar(y_pos, [RMSPEs,RMSPEn])

        # Create names on the x-axis
        plt.xticks(y_pos, bars)


        plt.subplot(2,3,6)
        bars = ('MAPE SVM','MAPE NN')
        y_pos = np.arange(len(bars))

        plt.suptitle('Processor: '+p+', Target: '+title, fontsize=10)

        # Create bars
        plt.bar(y_pos, [RMSPEs,RMSPEn])

        # Create names on the x-axis
        plt.xticks(y_pos, bars)

        # Show graphic
        plt.show()