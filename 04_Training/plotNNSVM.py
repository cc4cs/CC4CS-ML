import os
import matplotlib.pyplot as plt
import csv
from pandas import read_csv
import numpy as np
import matplotlib.patches as mpatches


RMSPEd=[]
NRMSEd=[]
timed=[]
R2d=[]
MAPEd=[]

RMSPEr=[]
NRMSEr=[]
timer=[]
R2r=[]
MAPEr=[]


iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
for p in iss:
    
    depth=[1,2,3,4,5]
    DecisionTest = read_csv(os.getcwd()+'\\'+p+'\\SVM\\resultsTrain.csv', skipinitialspace=True,sep=',', header = 0)
    RandomTest = read_csv(os.getcwd()+'\\'+p+'\\DNeuralNetwork\\resultsTrain.csv', skipinitialspace=True,sep=',', header = 0)
    print("Training for processor: "+p)

    for i in [0,5,10,15,20]:

        RMSPEd.append(DecisionTest.to_numpy()[0+i:5+i,3])
        NRMSEd.append(DecisionTest.to_numpy()[0+i:5+i,2])
        timed.append(DecisionTest.to_numpy()[0+i:5+i,7])
        R2d.append(DecisionTest.to_numpy()[0+i:5+i,6])
        MAPEd.append(DecisionTest.to_numpy()[0+i:5+i,8])
    
        RMSPEr.append(RandomTest.to_numpy()[0+i:5+i,3])
        NRMSEr.append(RandomTest.to_numpy()[0+i:5+i,2])
        timer.append(RandomTest.to_numpy()[0+i:5+i,7])
        R2r.append(RandomTest.to_numpy()[0+i:5+i,6])
        MAPEr.append(RandomTest.to_numpy()[0+i:5+i,8])

    plt.figure(figsize=(20,18))
    plt.subplot(2, 2, 1)
    plt.yscale('log')
    plt.plot(depth, RMSPEd[0],'r',label='SVM: Assembly')
    plt.plot(depth, RMSPEr[0],'r',linestyle='dashed',label='Neural network: Assembly')
    plt.plot(depth, RMSPEd[1],'g',label='SVM: ClockCycles')
    plt.plot(depth, RMSPEr[1],'g',linestyle='dashed',label='Neural network: ClockCycles')
    plt.plot(depth, RMSPEd[2],'yellow',label='SVM: Text')
    plt.plot(depth, RMSPEr[2],'yellow',linestyle='dashed',label='Neural network: Text')
    plt.plot(depth, RMSPEd[3],'violet',label='SVM: Data')
    plt.plot(depth, RMSPEr[3],'violet',linestyle='dashed',label='Neural network: Data')
    plt.plot(depth, RMSPEd[4],'brown',label='SVM: Bss')
    plt.plot(depth, RMSPEr[4],'brown',linestyle='dashed',label='Neural network: Bss')
    plt.legend(loc='upper left')
    plt.xlabel('1 : Linear/ 150, 2: poly 2/ 250 ,3 : Linear/ 500, 4: poly 4 / 1000 , 5: rbf/ 1300')
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
    plt.yscale('log')
    plt.plot(depth, NRMSEd[0],'r',label='SVM: Assembly')
    plt.plot(depth, NRMSEr[0],'r',linestyle='dashed',label='Neural network: Assembly')
    plt.plot(depth, NRMSEd[1],'g',label='SVM: ClockCycles')
    plt.plot(depth, NRMSEr[1],'g',linestyle='dashed',label='Neural network: ClockCycles')
    plt.plot(depth, NRMSEd[2],'yellow',label='SVM: Text')
    plt.plot(depth, NRMSEr[2],'yellow',linestyle='dashed',label='Neural network: Text')
    plt.plot(depth, NRMSEd[3],'violet',label='SVM: Data')
    plt.plot(depth, NRMSEr[3],'violet',linestyle='dashed',label='Neural network: Data')
    plt.plot(depth, NRMSEd[4],'brown',label='SVM: Bss')
    plt.plot(depth, NRMSEr[4],'brown',linestyle='dashed',label='Neural network: Bss')
    plt.legend(loc='upper left')
    plt.xlabel('1 : Linear/ 150, 2: poly 2/ 250 ,3 : Linear/ 500, 4: poly 4 / 1000 , 5: rbf/ 1300')
    plt.title("NRMSE")

    plt.subplot(2, 2, 3)
    plt.plot(depth, timed[0],'r',label='SVM: Assembly')
    plt.plot(depth, timer[0],'r',linestyle='dashed',label='Neural network: Assembly')
    plt.plot(depth, timed[1],'g',label='SVM: ClockCycles')
    plt.plot(depth, timer[1],'g',linestyle='dashed',label='Neural network: ClockCycles')
    plt.plot(depth, timed[2],'yellow',label='SVM: Text')
    plt.plot(depth, timer[2],'yellow',linestyle='dashed',label='Neural network: Text')
    plt.plot(depth, timed[3],'violet',label='SVM: Data')
    plt.plot(depth, timer[3],'violet',linestyle='dashed',label='Neural network: Data')
    plt.plot(depth, timed[4],'brown',label='SVM: Bss')
    plt.plot(depth, timer[4],'brown',linestyle='dashed',label='Neural network: Bss')
    plt.legend(loc='upper left')
    plt.xlabel('1 : Linear/ 150, 2: poly 2/ 250 ,3 : Linear/ 500, 4: poly 4 / 1000 , 5: rbf/ 1300')
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.yscale('log')
    plt.plot(depth, MAPEd[0],'r',label='SVM: Assembly')
    plt.plot(depth, MAPEr[0],'r',linestyle='dashed',label='Neural network: Assembly')
    plt.plot(depth, MAPEd[1],'g',label='SVM: ClockCycles')
    plt.plot(depth, MAPEr[1],'g',linestyle='dashed',label='Neural network: ClockCycles')
    plt.plot(depth, MAPEd[2],'yellow',label='SVM: Text')
    plt.plot(depth, MAPEr[2],'yellow',linestyle='dashed',label='Neural network: Text')
    plt.plot(depth, MAPEd[3],'violet',label='SVM: Data')
    plt.plot(depth, MAPEr[3],'violet',linestyle='dashed',label='Neural network: Data')
    plt.plot(depth, MAPEd[4],'brown',label='SVM: Bss')
    plt.plot(depth, MAPEr[4],'brown',linestyle='dashed',label='Neural network: Bss')
    plt.legend(loc='upper left')
    plt.xlabel('1 : Linear/ 150, 2: poly 2/ 250 ,3 : Linear/ 500, 4: poly 4 / 1000 , 5: rbf/ 1300')
    plt.title("MAPE")
    plt.savefig(os.getcwd()+'\\'+p+'\GGeneral.png')


    count=1
    plt.figure(figsize=(20,18))
    for i in [0,5,10,15,20]:
        title=DecisionTest.to_numpy()[0+i,0]
        MSEd=DecisionTest.to_numpy()[0+i:5+i,2]
        MAEd=DecisionTest.to_numpy()[0+i:5+i,5]
        MSEr=RandomTest.to_numpy()[ 0+i:5+i,2]
        MAEr=RandomTest.to_numpy()[0+i:5+i,4]

        

        plt.subplot(4, 3, count)
        plt.plot(depth, MSEd,'r',label='SVM')
        plt.plot(depth, MSEr,'g',label='Neural network')
        plt.legend(loc='upper left')
        plt.xlabel('1 : Linear/ 150, 2: poly 2/ 250 ,3 : Linear/ 500, 4: poly 4 / 1000 , 5: rbf/ 1300')
        plt.title(title+": RMSE")


        plt.subplot(4, 3, count+1)
        plt.plot(depth, MAEd,'r',label='SVM')
        plt.plot(depth, MAEr,'g',label='Neural network')
        plt.title(title+": MAE")
        plt.legend(loc='upper left')
        plt.xlabel('1 : Linear/ 150, 2: poly 2/ 250 ,3 : Linear/ 500, 4: poly 4 / 1000 , 5: rbf/ 1300')
        plt.plot()

        count=count+2
    plt.savefig(os.getcwd()+'\\'+p+'\SSpecific.png')