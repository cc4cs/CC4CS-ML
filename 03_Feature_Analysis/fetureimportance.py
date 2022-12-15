

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

feature=['Assembly','Clock','Text','Data','Bss']
fea=[]
for f in feature:
    
    df = pd.read_csv('allCLFscore'+f+'.csv', sep='\t')
    df=df.to_numpy()
    avg=np.mean(df[:,6])
    for i in range(0,91):
        if df[i,5]>avg:#+(avg*20)/100:
            if df[i,0] not in fea:
                fea.append(df[i,0])

iss=['Leon3','Atmega328p','Armv4t','Armv6-m']
print(fea)
for i in iss:
    print(i)
    df = pd.read_csv(i+'Paersonresults.csv', sep=',')
    df=df.to_numpy()
    for i in range(0,9506):
        if df[i,0] in fea and df[i,1] in fea and df[i,2]>0.9 and df[i,3]<0.05:
            print(df[0,0]+df[0,1])
        

            


