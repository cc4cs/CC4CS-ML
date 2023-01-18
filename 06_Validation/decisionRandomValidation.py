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

        with open(os.getcwd()+"\\"+p+"\\REG_TREE\\resultsTest.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id','depth','MSE','NRMSE','RMSPE','MAE','R2','time','MAPE'])

            MSE=[]
            MAE=[]
            NRMSE=[]
            R2=[]
            rmspe=[]
            if(p=='Leon3'):
                dictt={1: 12,2:16,3:16,4:16,5:10 }
            elif(p=='Atmega328p'):
                dictt={1: 12,2:16,3:16,4:16,5:10 }
            elif(p=="Armv4t"):
                dictt={1: 12,2:12,3:16,4:16,5:16 }
            elif(p=="Armv6-M"):
                dictt={1: 12,2:16,3:16,4:16,5:12 }
            for i in [0,1,2,3,4]:
                x_train = np.concatenate((dataframe.to_numpy()[:,8:36],dataframe.to_numpy()[:,42:len(dataframe.to_numpy()[0,:])]), axis=1)
                filename = os.getcwd()+'\\'+p+'\REG_TREE\depth'+str(dictt[i+1])+'_'+firstline[36+i]+'.sav'
                y_train=dataframe.to_numpy()[:,(36+i)]
                reg = pickle.load(open(filename, 'rb'))
                training_t=time.time()
                reg.fit(x_train, y_train)
                pp=reg.predict(x_train)
                training_t=time.time()-training_t
                filename = os.getcwd()+'\\'+p+'\REG_TREE\depth'+str(dictt[i+1])+'_'+firstline[36+i]+'.sav'
                
                MSE=np.round(metrics.mean_squared_error(y_train, pp), 2)
                NRMSE=np.divide(MSE,np.std(y_train))
                MAE=np.round(metrics.mean_absolute_error(y_train, pp), 2)
                R2=np.round(metrics.r2_score(y_train, pp), 2)

                y_train0=[]
                pp0=[]
                for c in range(0,len(y_train)):
                        if y_train[c]!=0:
                                y_train0.append(y_train[c])
                                pp0.append(pp[c])
                rmspe=(np.sqrt(np.mean(np.square(np.divide((np.array(y_train0) - np.array(pp0)), np.array(y_train0)))))) * 100
                MAPE=np.mean(np.abs(np.divide(np.array(y_train0) - np.array(pp0) , np.array(y_train0)))) * 100
                
                plt.figure()
                plt.scatter(y_train,pp)
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.plot(500, 500)
                plt.savefig(os.getcwd()+'\\'+p+'\\REG_TREE\depth'+str(str(dictt[i+1]))+'_Single_'+firstline[36+i]+'TestResidual.png')
                plt.close()

                plt.figure()
                error = y_train - pp
                plt.boxplot(error)
                plt.xlabel("Prediction Error")
                _ = plt.ylabel("Count")
                plt.savefig(os.getcwd()+'\\'+p+'\\REG_TREE\depth'+str(str(dictt[i+1]))+'_Single_'+firstline[36+i]+'TestError.png')
                plt.close()
                
                writer.writerow(['Single:'+firstline[36+i],str(dictt[i+1]),MSE,NRMSE,rmspe,MAE,R2,training_t,MAPE])

        with open(os.getcwd()+"\\"+p+"\\RandomForest\\resultsTest.csv",'w') as f:
            writer = csv.writer(f)
            writer.writerow(['id','depth','MSE','NRMSE','RMSPE','MAE','R2','time','MAPE'])

            MSE=[]
            MAE=[]
            NRMSE=[]
            R2=[]
            rmspe=[]
            if(p=='Leon3'):
                dictt={1: 12,2:16,3:12,4:10,5:10 }
            elif(p=='Atmega328p'):
                dictt={1: 12,2:12,3:12,4:10,5:10 }
            elif(p=="Armv4t"):
                dictt={1: 12,2:12,3:12,4:10,5:12 }
            elif(p=="Armv6-M"):
                dictt={1: 12,2:12,3:12,4:12,5:12 }
            for i in [0,1,2,3,4]:
                x_train = np.concatenate((dataframe.to_numpy()[:,8:36],dataframe.to_numpy()[:,41:len(dataframe.to_numpy()[0,:])]), axis=1)
                filename = os.getcwd()+'\\'+p+'\RandomForest\depth'+str(dictt[i+1])+'_'+firstline[36+i]+'.sav'
                y_train=dataframe.to_numpy()[:,(36+i)]
                reg = pickle.load(open(filename, 'rb'))
                training_t=time.time()
                reg.fit(x_train, y_train)
                pp=reg.predict(x_train)
                training_t=time.time()-training_t
                filename = os.getcwd()+'\\'+p+'\REG_TREE\depth'+str(dictt[i+1])+'_'+firstline[36+i]+'.sav'
                
                MSE=np.round(metrics.mean_squared_error(y_train, pp), 2)
                MAE=np.round(metrics.mean_absolute_error(y_train, pp), 2)
                R2=np.round(metrics.r2_score(y_train, pp), 2)
                NRMSE=np.divide(MSE,np.std(y_train))
                y_train0=[]
                pp0=[]
                for c in range(0,len(y_train)):
                        if y_train[c]!=0:
                                y_train0.append(y_train[c])
                                pp0.append(pp[c])
                rmspe=(np.sqrt(np.mean(np.square(np.divide((np.array(y_train0) - np.array(pp0)), np.array(y_train0)))))) * 100
                MAPE=np.mean(np.abs(np.divide(np.array(y_train0) - np.array(pp0) , np.array(y_train0)))) * 100
                
                plt.figure()
                plt.scatter(y_train,pp)
                plt.xlabel('True Values')
                plt.ylabel('Predictions')
                plt.plot(500, 500)
                plt.savefig(os.getcwd()+'\\'+p+'\\RandomForest\depth'+str(str(dictt[i+1]))+'_Single_'+firstline[36+i]+'TestResidual.png')
                plt.close()

                plt.figure()
                error = y_train - pp
                plt.boxplot(error,whis=1.5)
                plt.xlabel("Prediction Error")
                _ = plt.ylabel("Count")
                plt.savefig(os.getcwd()+'\\'+p+'\\RandomForest\depth'+str(str(dictt[i+1]))+'_Single_'+firstline[36+i]+'TestError.png')
                plt.close()
                
                writer.writerow(['Single:'+firstline[36+i],str(dictt[i+1]),MSE,NRMSE,rmspe,MAE,R2,training_t,MAPE])











