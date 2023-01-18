

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
from pandas import read_csv
features=['Assembly','Clock','Text','Data','Bss']

iss=['Armv4t','Armv6-m','Leon3','Atmega328P']
fea=[]
avgV=[]
""" for ii in iss:
    for f in features:
        statistics = pd.read_csv('generalStatistics.csv', sep=',')
        df = pd.read_csv('allCLFscore'+str(f)+ii+'.csv', sep='\t')
        df=df.to_numpy()
        avgV.append(np.mean(df[:,6]))
        avg=np.mean(df[:,6])
        for i in range(0,91):
            if df[i,5]>avg:#+(avg*20)/100:
                if df[i,0]:
                    fea.append(df[i,0])
names = list(collections.Counter(fea).keys())
values = list(collections.Counter(fea).values())
print(collections.Counter(fea))
plt.bar(range(len(collections.Counter(fea))), values, tick_label=names)
plt.xticks(rotation=90)
plt.yticks(rotation=90)
plt.savefig('Counter') """
fea=[]
for ii in iss:
    for f in features:
        statistics = pd.read_csv('generalStatistics.csv', sep=',')
        df = pd.read_csv('allCLFscore'+f+ii+'.csv', sep='\t')
        df=df.to_numpy()
        avg=np.mean(df[:,6])
        for i in range(0,91):
            if df[i,5]>avg:#+(avg*20)/100:
                if df[i,0]:
                    fea.append(df[i,0])
        print('Processor: '+ii+' Target: '+f)
        print(fea)

        feature=[]
        fea.append('assemblyInstr')
        fea.append('clockCycles')
        fea.append('text')
        fea.append('data')
        fea.append('bss')

        dataframe = read_csv("TotalParameterMatrix"+ii+".csv", skipinitialspace=True,sep=';', header = 0)
        for d in dataframe.columns.values:
            if d not in fea:
                feature.append(d)
        dataframeFeatureMin=dataframe.drop(feature,axis=1)  
        corr_df = dataframeFeatureMin.corr()
        heatmap = sns.heatmap(corr_df, yticklabels=corr_df.columns.values,xticklabels=corr_df.columns.values)
        heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':5}, pad=20)
        plt.savefig('heatmap'+f+ii+'.png',bbox_inches='tight')
        plt.close()
        fea=[]

    