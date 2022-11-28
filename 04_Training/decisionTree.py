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
        

# load dataset
dataframe = read_csv("TotalParameterMatrixARMV6-M.csv", skipinitialspace=True,sep=';', header = 0)
with open("TotalParameterMatrixARMV6-M.csv") as f:
    firstline = f.readline().rstrip().split(';')

print(firstline)

dataset = dataframe.values
# split into input (X) and output (Y) variables

x_train = dataset[1:10000,6:100]
y_train=dataset[1:10000,101:104]
x_test = dataset[10000:17000,6:100]
y_test = dataset[10000:17000,101:104]




def learning_curves(x_train, y_train, x_test, y_test):
# Both X_train and y_train have 354 rows, and X_test and y_test have 152 rows
    fig = plt.figure(figsize=(10,8))
# Generate 40 evenly spaced numbers (rounded to nearest integer) over a specified interval 1 to 354
    datapoints = np.rint(np.linspace(1, len(x_test)-1,40)).astype(int)
#initialise array of shape (40,)
    test_err = []#np.zeros(len(datapoints))
    
# Create 6 different models based on max_depth
    for k, depth in enumerate(range(1,2)):
        max=0
        min=1000000000
        for i in datapoints:
            reg = DecisionTreeRegressor(max_depth = depth) #increasing depth
# Iteratively increase training set size
            reg.fit(x_train, y_train)
# MSE for training and test sets of increasing size
            if(tree_reg.predict(x_test)[i][0]>max):
                max=tree_reg.predict(x_test)[i][0]
            if(tree_reg.predict(x_test)[i][0]<min):
                min=tree_reg.predict(x_test)[i][0]

            test_err.append(y_test[i][0]-tree_reg.predict(x_test)[i][0])

        #RMSE
        MSE = np.square(np.subtract(y_test, tree_reg.predict(x_test))).mean()
        print("RMSE for DTReg (All features): " , np.round(math.sqrt(MSE), 2))

        
        #NRMSE
        NRMSE=math.sqrt(MSE)/(max-min)
        print("NRMSE for DTReg (All features): " , np.round(NRMSE, 2))

        #MAE
        print("MAE for DTReg (All features): " , np.round(metrics.mean_absolute_error(y_test, tree_reg.predict(x_test)), 2))

        #R2
        print("RSquared for DTReg (All features): " , np.round(metrics.r2_score(y_test, tree_reg.predict(x_test)), 2))
        


learning_curves(x_train, y_train, x_test, y_test)
