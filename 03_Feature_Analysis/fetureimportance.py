

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import io
import time
import collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

feature=['Assembly','Clock','Text','Data','Bss']
fea=[]
avgV=[]
for f in feature:
    statistics = pd.read_csv('generalStatistics.csv', sep=',')
    df = pd.read_csv('allCLFscore'+f+'.csv', sep='\t')
    df=df.to_numpy()
    avgV.append(np.mean(df[:,6]))
    avg=np.mean(df[:,6])
    for i in range(0,91):
        if df[i,5]>avg:#+(avg*20)/100:
            if df[i,0]:
                fea.append(df[i,0])

AVG=np.mean(avgV)

names = list(collections.Counter(fea).keys())
values = list(collections.Counter(fea).values())
print(collections.Counter(fea))
plt.bar(range(len(collections.Counter(fea))), values, tick_label=names)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.show()

fea=[]
for f in feature:
    print('target feature: '+f)
    statistics = pd.read_csv('generalStatistics.csv', sep=',')
    df = pd.read_csv('allCLFscore'+f+'.csv', sep='\t')
    df=df.to_numpy()
    #avg=np.mean(df[:,6])
    for i in range(0,91):
        if df[i,5]>AVG:#+(avg*20)/100:
            if df[i,0]:
                fea.append(df[i,0])

    print(fea)
    fea=[]
    """ iss=['Leon3','Atmega328p','Armv4t','Armv6-m']
    corrLeon3=[]
    corrAtmega328p=[]
    corrArmv4t=[]
    corrArmv6m=[]

    for x in iss:
        corr=[]
        df = pd.read_csv(x+'Paersonresults.csv', sep=',')
        df=df.to_numpy()
        for i in range(0,9506):
            if df[i,0] in fea and df[i,1] in fea and df[i,2]>0.8 and df[i,3]<0.05:
                if df[i,0]+'/'+df[i,1] not in corr and df[i,1]+'/'+df[i,0] not in corr:
                    corr.append(df[i,0]+'/'+df[i,1])
        if x=='Leon3':
            corrLeon3=corr
        elif x=='Atmega328p':
            corrAtmega328p=corr
        elif x=='Armv4t':
            corrArmv4t=corr
        elif x=='Armv6-m':
            corrArmv6m=corr


    statistics=statistics.to_numpy()

    varianza=dict()
    featureMin=[]
    featureMax=[]
    for e in corrArmv6m:
        for i in range(0,392):
            if(statistics[i,0].split('---')[0]=='Armv6-m'):
                if(statistics[i,0].split(':')[1]==e.split('/')[0] or statistics[i,0].split(':')[1]==e.split('/')[1]):
                    varianza[statistics[i,0].split(':')[1]]=statistics[i,1]/statistics[i,5]
            if len(varianza)==2:
                if(varianza[e.split('/')[1]]>varianza[e.split('/')[0]]):
                    featureMax.append(e.split('/')[1])
                    varianza=dict()
                else:
                    featureMin.append(e.split('/')[0])
                    varianza=dict()

    print('feature massimi coefficenti armv6-m :')
    print(set(featureMax))
    print('feature minimi coefficenti armv6-m :')
    print(set(featureMin))

    varianza=dict()
    featureMin=[]
    featureMax=[]
    for e in corrArmv6m:
        for i in range(0,392):
            if(statistics[i,0].split('---')[0]=='Armv4t'):
                if(statistics[i,0].split(':')[1]==e.split('/')[0] or statistics[i,0].split(':')[1]==e.split('/')[1]):
                    varianza[statistics[i,0].split(':')[1]]=statistics[i,1]/statistics[i,5]
            if len(varianza)==2:
                if(varianza[e.split('/')[1]]>varianza[e.split('/')[0]]):
                    featureMax.append(e.split('/')[1])
                    varianza=dict()
                else:
                    featureMin.append(e.split('/')[0])
                    varianza=dict()

    print('feature massimi coefficenti armv4t :')
    print(set(featureMax))
    print('feature minimi coefficenti armv4t :')
    print(set(featureMin))

    varianza=dict()
    featureMin=[]
    featureMax=[]
    for e in corrArmv6m:
        for i in range(0,392):
            if(statistics[i,0].split('---')[0]=='Atmega328P'):
                if(statistics[i,0].split(':')[1]==e.split('/')[0] or statistics[i,0].split(':')[1]==e.split('/')[1]):
                    varianza[statistics[i,0].split(':')[1]]=statistics[i,1]/statistics[i,5]
            if len(varianza)==2:
                if(varianza[e.split('/')[1]]>varianza[e.split('/')[0]]):
                    featureMax.append(e.split('/')[1])
                    varianza=dict()
                else:
                    featureMin.append(e.split('/')[0])
                    varianza=dict()

    print('feature massimi coefficenti Atmega328p :')
    print(set(featureMax))
    print('feature minimi coefficenti Atmega328p :')
    print(set(featureMin))

    varianza=dict()
    featureMin=[]
    featureMax=[]
    for e in corrArmv6m:
        for i in range(0,392):
            if(statistics[i,0].split('---')[0]=='Leon3'):
                if(statistics[i,0].split(':')[1]==e.split('/')[0] or statistics[i,0].split(':')[1]==e.split('/')[1]):
                    varianza[statistics[i,0].split(':')[1]]=statistics[i,1]/statistics[i,5]
            if len(varianza)==2:
                if(varianza[e.split('/')[1]]>varianza[e.split('/')[0]]):
                    featureMax.append(e.split('/')[1])
                    varianza=dict()
                else:
                    featureMin.append(e.split('/')[0])
                    varianza=dict()

    print('feature massimi coefficenti Leon3 :')
    print(set(featureMax))
    print('feature minimi coefficenti Leon3 :')
    print(set(featureMin))



                


 """