
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
feature=[]
for p in iss:

        with open("TotalParameterMatrix"+p+"Train.csv") as f:
            firstline = f.readline().rstrip().split(',')

        
            with open(os.getcwd()+"\\"+p+"\\SVM\\resultsTrain.csv",'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(['id','kernel','RMSE','NRMSE','RMSPE','MAE','R2','time','MAPE'])
                    for i in [0,1,2,3,4]:
                        if(i==0):
                                featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Assignment', 'CyclomaticComplexity', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT', 'cInstr', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_22', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_32']
                        if(i==1):
                                featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Assignment', 'CyclomaticComplexity', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT', 'cInstr', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_14', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_22', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_32']
                        if(i==2):
                                featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Assignment', 'PointerDereferencing', 'CyclomaticComplexity', 'SCALAR_INPUT', 'RANGE_SCALAR_VALUES', 'ARRAY_INPUT', 'cInstr', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_5', 'V_VAL_6', 'V_VAL_7', 'V_VAL_8', 'V_VAL_9', 'V_VAL_10', 'V_VAL_11', 'V_VAL_12', 'V_VAL_13', 'V_VAL_14', 'V_VAL_15', 'V_VAL_16', 'V_VAL_17', 'V_VAL_18', 'V_VAL_19', 'V_VAL_20', 'V_VAL_21', 'V_VAL_22', 'V_VAL_23', 'M_VAL_1', 'M_VAL_6']
                        if(i==3):
                                featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'TimeToImplement', 'BugsDelivered', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Goto', 'Assignment', 'ExitPoint', 'Function', 'FunctionCall', 'CyclomaticComplexity', 'SCALAR_INDEX_INPUT', 'RANGE_SCALAR_INDEX_VALUES', 'ARRAY_INPUT', 'cInstr', 'V_VAL_1', 'V_VAL_2', 'V_VAL_5', 'M_VAL_1', 'M_VAL_2', 'M_VAL_6', 'M_VAL_9', 'M_VAL_10', 'M_VAL_14', 'M_VAL_17', 'M_VAL_18', 'M_VAL_22', 'M_VAL_25', 'M_VAL_26']
                        if(i==4):
                                featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Goto', 'Assignment', 'ExitPoint', 'Function', 'FunctionCall', 'PointerDereferencing', 'CyclomaticComplexity', 'SCALAR_INDEX_INPUT', 'RANGE_SCALAR_INDEX_VALUES', 'ARRAY_INPUT', 'cInstr', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'M_VAL_1', 'M_VAL_2']
                        featureMin.append('assemblyInstr')
                        featureMin.append('clockCycles')
                        featureMin.append('text')
                        featureMin.append('data')
                        featureMin.append('data')
                        featureMin.append('bss')

                        dataframe = read_csv("TotalParameterMatrix"+p+"Train.csv", skipinitialspace=True,sep=',', header = 0)
                        for f in dataframe.columns.values:
                                if f not in featureMin:
                                        feature.append(f)
                        dataframeFeatureMin=dataframe.drop(feature,axis=1)
                        
                
                
                #single target
                        count=0
                        for x in dataframeFeatureMin.columns:
                                if(x!='assemblyInstr'):
                                        count=count+1
                                else:
                                        break
                        reg = svm.LinearSVR()
                        x_train = np.concatenate((dataframeFeatureMin.to_numpy()[:,0:(count+i)],dataframeFeatureMin.to_numpy()[:,(count+1+i):len(dataframeFeatureMin)]), axis=1)
                        y_train=dataframeFeatureMin.to_numpy()[:,count+i]
                        training_t=time.time()
                        reg.fit(x_train,y_train)
                        filename = os.getcwd()+'\\'+p+'\SVM\linear'+dataframeFeatureMin.columns[count+i]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
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
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\linear'+dataframeFeatureMin.columns[count+i]+'Residual.png')
                        plt.close()

                        error = y_train - pp
                        plt.hist(error)
                        plt.xlabel("Prediction Error")
                        _ = plt.ylabel("Count")
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\linear'+dataframeFeatureMin.columns[count+i]+'Error.png')
                        plt.close()
                        writer.writerow([dataframeFeatureMin.columns[count+i],'linear',MSE,NRMSE,rmspe,MAE,R2,training_t,MAPE])

                        reg = svm.SVR(kernel='poly',degree=2)
                        x_train = np.concatenate((dataframeFeatureMin.to_numpy()[:,0:(count+i)],dataframeFeatureMin.to_numpy()[:,(count+1+i):len(dataframeFeatureMin)]), axis=1)
                        y_train=dataframeFeatureMin.to_numpy()[:,count+i]
                        training_t=time.time()
                        reg.fit(x_train,y_train)
                        filename = os.getcwd()+'\\'+p+'\SVM\poly2'+firstline[(37+i)]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
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
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\poly2'+dataframeFeatureMin.columns[count+i]+'Residual.png')
                        plt.close()

                        error = y_train - pp
                        plt.hist(error)
                        plt.xlabel("Prediction Error")
                        _ = plt.ylabel("Count")
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\poly2'+dataframeFeatureMin.columns[count+i]+'Error.png')
                        plt.close()
                        writer.writerow([dataframeFeatureMin.columns[count+i],2, MSE,NRMSE,rmspe,MAE,R2,training_t,MAPE])

                        reg = svm.SVR(kernel='poly',degree=3)
                        x_train = np.concatenate((dataframeFeatureMin.to_numpy()[:,0:(count+i)],dataframeFeatureMin.to_numpy()[:,(count+1+i):len(dataframeFeatureMin)]), axis=1)
                        y_train=dataframeFeatureMin.to_numpy()[:,count+i]
                        training_t=time.time()
                        reg.fit(x_train,y_train)
                        filename = os.getcwd()+'\\'+p+'\SVM\poly3'+firstline[(37+i)]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
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
                        plt.figure()
                        plt.scatter(y_train,pp)
                        plt.xlabel('True Values')
                        plt.ylabel('Predictions')
                        plt.plot(500, 500)
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\poly3'+dataframeFeatureMin.columns[count+i]+'Residual.png')
                        plt.close()

                        error = y_train - pp
                        plt.hist(error)
                        plt.xlabel("Prediction Error")
                        _ = plt.ylabel("Count")
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\poly3'+dataframeFeatureMin.columns[count+i]+'Error.png')
                        plt.close()
                        writer.writerow([dataframeFeatureMin.columns[count+i],3,MSE,NRMSE,rmspe,MAE,R2,training_t,MAPE])
                        
                        reg = svm.SVR(kernel='poly',degree=4)
                        x_train = np.concatenate((dataframeFeatureMin.to_numpy()[:,0:(count+i)],dataframeFeatureMin.to_numpy()[:,(count+1+i):len(dataframeFeatureMin)]), axis=1)
                        y_train=dataframeFeatureMin.to_numpy()[:,count+i]
                        training_t=time.time()
                        reg.fit(x_train,y_train)
                        filename = os.getcwd()+'\\'+p+'\SVM\poly4'+firstline[(37+i)]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
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
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\poly4'+dataframeFeatureMin.columns[count+i]+'Residual.png')
                        plt.close()

                        error = y_train - pp
                        plt.hist(error)
                        plt.xlabel("Prediction Error")
                        _ = plt.ylabel("Count")
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\poly4'+dataframeFeatureMin.columns[count+i]+'Error.png')
                        plt.close()

                        writer.writerow([dataframeFeatureMin.columns[count+i],4,MSE,NRMSE,rmspe,MAE,R2,training_t,MAPE])

                        reg = svm.SVR(kernel='rbf')
                        x_train = np.concatenate((dataframeFeatureMin.to_numpy()[:,0:(count+i)],dataframeFeatureMin.to_numpy()[:,(count+1+i):len(dataframeFeatureMin)]), axis=1)
                        y_train=dataframeFeatureMin.to_numpy()[:,count+i]
                        training_t=time.time()
                        reg.fit(x_train,y_train)
                        filename = os.getcwd()+'\\'+p+'\SVM\\rbf'+firstline[(37+i)]+'.sav'
                        pickle.dump(reg, open(filename, 'wb'))
                        pp=reg.predict(x_train)
                        training_t=time.time()-training_t
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
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\\rbf'+dataframeFeatureMin.columns[count+i]+'Residual.png')
                        plt.close()

                        error = y_train - pp
                        plt.hist(error)
                        plt.xlabel("Prediction Error")
                        _ = plt.ylabel("Count")
                        plt.savefig(os.getcwd()+'\\'+p+'\SVM\\rbf'+dataframeFeatureMin.columns[count+i]+'Error.png')
                        plt.close()

                        writer.writerow([dataframeFeatureMin.columns[count+i],'rbf',MSE,NRMSE,rmspe,MAE,R2,training_t,MAPE])

                        