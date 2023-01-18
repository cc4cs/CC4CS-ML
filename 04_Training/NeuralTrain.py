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
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score as r2_score_sk
from keras.layers.activation import LeakyReLU
import tensorflow as tf
import math
from keras import layers
import csv
import time
import pandas as pd
import os
import statistics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit 
from sklearn.svm import SVR
from keras import backend as K
import pickle
from sklearn.ensemble import RandomForestRegressor



MSE=[]
MAE=[]
NRMSE=[]
featuremin=[]
R2=[]
iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
feature=[]
for p in iss:
        with open("TotalParameterMatrix"+p+"Train.csv") as f:
            firstline = f.readline().rstrip().split(',')
        with open(os.getcwd()+"\\"+p+"\\DNeuralNetwork\\resultsTrain.csv",'w') as f:
                writer = csv.writer(f)
                writer.writerow(['id','epochs','MSE','MAE','RMSPE','RMSE','R2','time','MAPE'])

                cvscores = []
                for i in [0,1,2,3,4]:
                        if(i==0):
                                if(p=="Armv4t"):
                                        featureMin=['Total_Operands', 'ProgramLength', 'ProgramVolume', 'Effort', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'cInstr']
                                if(p=="Armv6-M"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'GlobalVariables', 'Loop', 'CyclomaticComplexity', 'ARRAY_INPUT', 'cInstr', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_7', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'M_VAL_15', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_21', 'M_VAL_22', 'M_VAL_23', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_31', 'M_VAL_32']
                                if(p=="Atmega328P"):
                                        featureMin=['ProgramLength', 'DifficultyLevel', 'cInstr']
                                if(p=="Leon3"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'GlobalVariables', 'ARRAY_INPUT', 'cInstr', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_7', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'M_VAL_15', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_21', 'M_VAL_22', 'M_VAL_23', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_31', 'M_VAL_32']
                        if(i==1):
                                if(p=="Armv4t"):
                                        featureMin=['Total_Operands', 'ProgramLength', 'ProgramVolume', 'Effort', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'cInstr']
                                if(p=="Armv6-M"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'GlobalVariables', 'ARRAY_INPUT', 'cInstr', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_7', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'M_VAL_15', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_21', 'M_VAL_22', 'M_VAL_23', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_31', 'M_VAL_32']
                                if(p=="Atmega328P"):
                                        featureMin=['ProgramLength', 'Effort', 'cInstr']
                                if(p=="Leon3"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'GlobalVariables', 'Loop', 'ARRAY_INPUT', 'cInstr', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_7', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'M_VAL_15', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_21', 'M_VAL_22', 'M_VAL_23', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_31', 'M_VAL_32']
                        if(i==2):
                                if(p=="Armv4t"):
                                        featureMin=['VocabularySize', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Goto', 'Assignment', 'ExitPoint', 'Function', 'FunctionCall', 'PointerDereferencing', 'CyclomaticComplexity', 'ARRAY_INPUT']
                                if(p=="Armv6-M"):
                                        featureMin=['Total_Operands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Goto', 'Assignment', 'CyclomaticComplexity', 'SCALAR_INPUT', 'RANGE_SCALAR_VALUES', 'SCALAR_INDEX_INPUT', 'RANGE_SCALAR_INDEX_VALUES', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT', 'SV_VAL_1', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_4', 'V_VAL_6', 'M_VAL_1', 'M_VAL_3', 'M_VAL_6', 'M_VAL_22']
                                if(p=="Atmega328P"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'ProgramLevel', 'ProgramLevel.1', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Goto', 'Assignment', 'ExitPoint', 'Function', 'FunctionCall', 'PointerDereferencing', 'CyclomaticComplexity', 'M_VAL_1', 'M_VAL_17']

                                if(p=="Leon3"):
                                        featureMin=['VocabularySize', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Assignment', 'ExitPoint', 'Function', 'FunctionCall', 'PointerDereferencing', 'CyclomaticComplexity', 'ARRAY_INPUT', 'V_VAL_2', 'M_VAL_1', 'M_VAL_3', 'M_VAL_20']
                        if(i==3):
                                if(p=="Armv4t"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'ProgramVolume', 'TimeToImplement', 'BugsDelivered', 'M_VAL_9', 'M_VAL_10', 'M_VAL_14', 'M_VAL_18', 'M_VAL_22']
                                if(p=="Armv6-M"):
                                        featureMin=['DistinctOperands', 'VocabularySize', 'Effort', 'Sloc', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Assignment', 'ExitPoint', 'Function', 'FunctionCall', 'CyclomaticComplexity', 'SCALAR_INDEX_INPUT', 'RANGE_SCALAR_INDEX_VALUES', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_4', 'V_VAL_5', 'V_VAL_6', 'M_VAL_1']
                                if(p=="Atmega328P"):
                                        featureMin=['Total_Operands', 'ProgramLength', 'ProgramVolume', 'BugsDelivered', 'cInstr', 'M_VAL_2', 'M_VAL_6', 'M_VAL_9', 'M_VAL_10', 'M_VAL_14', 'M_VAL_18', 'M_VAL_22', 'M_VAL_25', 'M_VAL_26']
                                if(p=="Leon3"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'GlobalVariables', 'Loop', 'PointerDereferencing', 'SCALAR_INPUT', 'RANGE_SCALAR_VALUES', 'ARRAY_INPUT', 'cInstr', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_4', 'V_VAL_5', 'V_VAL_6', 'V_VAL_7', 'V_VAL_8', 'V_VAL_9', 'V_VAL_11', 'V_VAL_12', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_7', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_13', 'M_VAL_14', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_20', 'M_VAL_21', 'M_VAL_22', 'M_VAL_23', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_31', 'M_VAL_32']

                        if(i==4):
                                if(p=="Armv4t"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'TimeToImplement', 'BugsDelivered', 'Sloc', 'DecisionPoint', 'If', 'CyclomaticComplexity', 'ARRAY_INPUT', 'cInstr', 'M_VAL_9', 'M_VAL_18', 'M_VAL_22', 'M_VAL_25']
                                if(p=="Armv6-M"):
                                        featureMin=['DistinctOperands', 'VocabularySize', 'Effort', 'TimeToImplement', 'Sloc', 'GlobalVariables', 'Loop', 'Assignment', 'FunctionCall', 'SCALAR_INPUT', 'RANGE_SCALAR_VALUES', 'SCALAR_INDEX_INPUT', 'RANGE_SCALAR_INDEX_VALUES', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT', 'SV_VAL_1', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_4', 'V_VAL_5', 'V_VAL_6', 'M_VAL_1']
                                if(p=="Atmega328P"):
                                        featureMin=['Total_Operands', 'DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'TimeToImplement', 'BugsDelivered', 'DecisionPoint', 'If', 'Loop', 'CyclomaticComplexity', 'ARRAY_INPUT', 'M_VAL_9', 'M_VAL_18']  
                                if(p=="Leon3"):
                                        featureMin=['DistinctOperands', 'ProgramLength', 'VocabularySize', 'ProgramVolume', 'Effort', 'DifficultyLevel', 'TimeToImplement', 'BugsDelivered', 'DecisionPoint', 'GlobalVariables', 'If', 'Loop', 'Function', 'PointerDereferencing', 'CyclomaticComplexity', 'SCALAR_INPUT', 'RANGE_SCALAR_VALUES', 'SCALAR_INDEX_INPUT', 'RANGE_SCALAR_INDEX_VALUES', 'ARRAY_INPUT', 'RANGE_ARRAY_INPUT', 'cInstr', 'V_VAL_1', 'V_VAL_2', 'V_VAL_3', 'V_VAL_4', 'V_VAL_5', 'V_VAL_6', 'V_VAL_7', 'V_VAL_8', 'V_VAL_9', 'V_VAL_10', 'V_VAL_11', 'V_VAL_12', 'V_VAL_13', 'V_VAL_14', 'M_VAL_1', 'M_VAL_2', 'M_VAL_3', 'M_VAL_4', 'M_VAL_5', 'M_VAL_6', 'M_VAL_7', 'M_VAL_8', 'M_VAL_9', 'M_VAL_10', 'M_VAL_11', 'M_VAL_12', 'M_VAL_14', 'M_VAL_16', 'M_VAL_17', 'M_VAL_18', 'M_VAL_19', 'M_VAL_22', 'M_VAL_24', 'M_VAL_25', 'M_VAL_26', 'M_VAL_27', 'M_VAL_28', 'M_VAL_29', 'M_VAL_32']

                        featureMin.append('assemblyInstr')
                        featureMin.append('clockCycles')
                        featureMin.append('text')
                        featureMin.append('data')
                        featureMin.append('bss')

                        dataframe = read_csv("TotalParameterMatrix"+p+"Train.csv", skipinitialspace=True,sep=',', header = 0)
                        for f in dataframe.columns.values:
                                if f not in featureMin:
                                        feature.append(f)
                        dataframeFeatureMin=dataframe.drop(feature,axis=1)
                       

                        count=0
                        for x in dataframeFeatureMin.columns:
                                if(x!='assemblyInstr'):
                                        count=count+1
                                else:
                                        break
                                
                        x_train = np.concatenate((dataframeFeatureMin.to_numpy()[:,0:(count+i)],dataframeFeatureMin.to_numpy()[:,(count+1+i):len(dataframeFeatureMin)]), axis=1)
                        y_train=dataframeFeatureMin.to_numpy()[:,count+i]
                        for ep in [150,250,500,1000,1300]:
                                
                                training_t=time.time()
                                reg = tf.keras.Sequential([
                        layers.Dense(12,kernel_initializer='normal', activation='relu', input_shape=[len(x_train[0,:])]),
                        layers.Dense(8, activation='relu'),
                        layers.Dense(1, activation='linear')
                        ])
                                optimizer = tf.keras.optimizers.RMSprop(0.001)
                                
                                reg.compile(loss='mse',optimizer=optimizer,metrics=['mae', 'mse'])
                                
                                reg.fit(np.asarray(x_train).astype(np.float32), np.asarray(y_train).astype(np.float32), epochs=ep,batch_size=600,validation_split = 0.2)
                                scores = reg.evaluate(np.asarray(x_train).astype(np.float32), np.asarray(y_train).astype(np.float32), verbose=0)
                                pp=reg.predict(np.asarray(x_train).astype(np.float32))

                                training_t=time.time()-training_t
                                filename = os.getcwd()+'\\'+p+'\\DNeuralNetwork\\Epoch'+str(ep)+'_'+dataframeFeatureMin.columns[count+i]+'.h5'
                                
                                reg.save(filename)
                                R2=np.round(metrics.r2_score(y_train, pp), 2)
                                y_train0=[]
                                pp0=[]
                                for c in range(0,len(y_train)):
                                        if y_train[c]!=0:
                                                y_train0.append(y_train[c])
                                                pp0.append(pp[c])
                                
                                rmspe=(np.sqrt(np.mean(np.square(np.divide((np.array(y_train0) - np.array(pp0)), np.array(y_train0)))))) * 100
                                MAPE=np.mean(np.abs(np.divide(np.array(y_train0) - np.array(pp0) , np.array(y_train0)))) * 100
                                """ plt.figure()
                                plt.scatter(y_train,pp)
                                plt.xlabel('True Values')
                                plt.ylabel('Predictions')
                                plt.plot(500, 500)
                                plt.savefig(os.getcwd()+'\\'+p+'\\DNeuralNetwork\\Epoch'+str(ep)+'_'+dataframeFeatureMin.columns[count+i]+'residual.png')
                                plt.close()

                                plt.figure()
                                error = y_train - pp
                                plt.boxplot(error)
                                plt.xlabel("Prediction Error")
                                _ = plt.ylabel("Count")
                                plt.savefig(os.getcwd()+'\\'+p+'\\DNeuralNetwork\\Epoch'+str(ep)+'_'+dataframeFeatureMin.columns[count+i]+'error.png')
                                plt.close() 
 """
                                
                                writer.writerow(['Single, feature removed:'+dataframeFeatureMin.columns[count+i],ep,scores[0],scores[1],rmspe,scores[2],R2,training_t,MAPE])

                        