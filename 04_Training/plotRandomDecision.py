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
    
    depth=[8,10,12,14,16]
    DecisionTest = read_csv(os.getcwd()+'\\'+p+'\\REG_TREE\\resultsTrain.csv', skipinitialspace=True,sep=',', header = 0)
    RandomTest = read_csv(os.getcwd()+'\\'+p+'\\RandomForest\\resultsTrain.csv', skipinitialspace=True,sep=',', header = 0)
    print("Training for processor: "+p)

    for i in [0,5,10,15,20]:
        RMSPEd.append(DecisionTest.to_numpy()[15+i:20+i,4])
        NRMSEd.append(DecisionTest.to_numpy()[15+i:20+i,3])
        timed.append(DecisionTest.to_numpy()[15+i:20+i,7])
        R2d.append(DecisionTest.to_numpy()[15+i:20+i,6])
        MAPEd.append(DecisionTest.to_numpy()[15+i:20+i,8])
    
        RMSPEr.append(RandomTest.to_numpy()[15+i:20+i,4])
        NRMSEr.append(RandomTest.to_numpy()[15+i:20+i,3])
        timer.append(RandomTest.to_numpy()[15+i:20+i,7])
        R2r.append(RandomTest.to_numpy()[15+i:20+i,6])
        MAPEr.append(RandomTest.to_numpy()[15+i:20+i,8])

    #valori multi
    for i in [0,1,2]:
        NRMSEd.append(np.array([DecisionTest.to_numpy()[0+i,3],DecisionTest.to_numpy()[3+i,3],DecisionTest.to_numpy()[6+i,3],DecisionTest.to_numpy()[9+i,3],DecisionTest.to_numpy()[12+i,3]]))
        RMSPEd.append(np.array([DecisionTest.to_numpy()[0+i,4],DecisionTest.to_numpy()[3+i,4],DecisionTest.to_numpy()[6+i,4],DecisionTest.to_numpy()[9+i,4],DecisionTest.to_numpy()[12+i,4]]))
        timed.append(np.array([DecisionTest.to_numpy()[0+i,7],DecisionTest.to_numpy()[3+i,7],DecisionTest.to_numpy()[6+i,7],DecisionTest.to_numpy()[9+i,7],DecisionTest.to_numpy()[12+i,7]]))
        R2d.append(np.array([DecisionTest.to_numpy()[0+i,6],DecisionTest.to_numpy()[3+i,6],DecisionTest.to_numpy()[6+i,6],DecisionTest.to_numpy()[9+i,6],DecisionTest.to_numpy()[12+i,6]]))
        MAPEd.append(np.array([DecisionTest.to_numpy()[0+i,8],DecisionTest.to_numpy()[3+i,8],DecisionTest.to_numpy()[6+i,8],DecisionTest.to_numpy()[9+i,8],DecisionTest.to_numpy()[12+i,8]]))
        


        NRMSEr.append(np.array([RandomTest.to_numpy()[0+i,3],RandomTest.to_numpy()[3+i,3],RandomTest.to_numpy()[6+i,3],RandomTest.to_numpy()[9+i,3],RandomTest.to_numpy()[12+i,3]]))
        RMSPEr.append(np.array([RandomTest.to_numpy()[0+i,4],RandomTest.to_numpy()[3+i,4],RandomTest.to_numpy()[6+i,4],RandomTest.to_numpy()[9+i,4],RandomTest.to_numpy()[12+i,4]]))
        timer.append(np.array([RandomTest.to_numpy()[0+i,7],RandomTest.to_numpy()[3+i,7],RandomTest.to_numpy()[6+i,7],RandomTest.to_numpy()[9+i,7],RandomTest.to_numpy()[12+i,7]]))
        R2r.append(np.array([RandomTest.to_numpy()[0+i,6],RandomTest.to_numpy()[3+i,6],RandomTest.to_numpy()[6+i,6],RandomTest.to_numpy()[9+i,6],RandomTest.to_numpy()[12+i,6]]))
        MAPEr.append(np.array([RandomTest.to_numpy()[0+i,8],RandomTest.to_numpy()[3+i,8],RandomTest.to_numpy()[6+i,8],RandomTest.to_numpy()[9+i,8],RandomTest.to_numpy()[12+i,8]]))

    plt.figure(figsize=(20,18))
    plt.subplot(2, 2, 1)
    plt.yscale('log')
    plt.plot(depth, RMSPEd[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, RMSPEr[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, RMSPEd[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, RMSPEr[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, RMSPEd[2],'yellow',label='Decision Three: Text')
    plt.plot(depth, RMSPEr[2],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, RMSPEd[3],'violet',label='Decision Three: Data')
    plt.plot(depth, RMSPEr[3],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, RMSPEd[4],'brown',label='Decision Three: Bss')
    plt.plot(depth, RMSPEr[4],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend(loc='upper left')
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
    plt.yscale('log')
    plt.plot(depth, NRMSEd[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, NRMSEr[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, NRMSEd[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, NRMSEr[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, NRMSEd[2],'yellow',label='Decision Three: Text')
    plt.plot(depth, NRMSEr[2],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, NRMSEd[3],'violet',label='Decision Three: Data')
    plt.plot(depth, NRMSEr[3],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, NRMSEd[4],'brown',label='Decision Three: Bss')
    plt.plot(depth, NRMSEr[4],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend(loc='upper left')
    plt.title("NRMSE")

    plt.subplot(2, 2, 3)
    plt.plot(depth, timed[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, timer[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, timed[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, timer[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, timed[2],'yellow',label='Decision Three: Text')
    plt.plot(depth, timer[2],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, timed[3],'violet',label='Decision Three: Data')
    plt.plot(depth, timer[3],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, timed[4],'brown',label='Decision Three: Bss')
    plt.plot(depth, timer[4],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend(loc='upper left')
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.yscale('log')
    plt.plot(depth, MAPEd[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, MAPEr[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, MAPEd[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, MAPEr[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, MAPEd[2],'yellow',label='Decision Three: Text')
    plt.plot(depth, MAPEr[2],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, MAPEd[3],'violet',label='Decision Three: Data')
    plt.plot(depth, MAPEr[3],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, MAPEd[4],'brown',label='Decision Three: Bss')
    plt.plot(depth, MAPEr[4],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend(loc='upper left')
    plt.title("MAPE")
    plt.savefig(os.getcwd()+'\\'+p+'\General.png')

    plt.figure(figsize=(20,18))

    plt.subplot(2, 2, 1)
    plt.yscale('log')
    plt.plot(depth, RMSPEd[2],'black',label='Decision Three: Text Single')
    plt.plot(depth, RMSPEr[2],'black',linestyle='dashed',label='Random Forest: Text Single')
    plt.plot(depth, RMSPEd[3],'violet',label='Decision Three: Data Single')
    plt.plot(depth, RMSPEr[3],'violet',linestyle='dashed',label='Random Forest: Data Single')
    plt.plot(depth, RMSPEd[4],'brown',label='Decision Three: Bss Single')
    plt.plot(depth, RMSPEr[4],'brown',linestyle='dashed',label='Random Forest: Bss Single')
    plt.plot(depth, RMSPEd[5],'yellow',label='Decision Three: Text Multi')
    plt.plot(depth, RMSPEr[5],'yellow',linestyle='dashed',label='Random Forest: Text Multi')
    plt.plot(depth, RMSPEd[6],'violet',label='Decision Three: Data Multi')
    plt.plot(depth, RMSPEr[6],'violet',linestyle='dashed',label='Random Forest: Data Multi')
    plt.plot(depth, RMSPEd[7],'brown',label='Decision Three: Bss Multi')
    plt.plot(depth, RMSPEr[7],'brown',linestyle='dashed',label='Random Forest: Bss Multi')
    plt.legend(loc='upper left')
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
    plt.yscale('log')
    plt.plot(depth, NRMSEd[2],'black',label='Decision Three: Text Single')
    plt.plot(depth, NRMSEr[2],'black',linestyle='dashed',label='Random Forest: Text Single')
    plt.plot(depth, NRMSEd[3],'violet',label='Decision Three: Data Single')
    plt.plot(depth, NRMSEr[3],'violet',linestyle='dashed',label='Random Forest: Data Single')
    plt.plot(depth, NRMSEd[4],'brown',label='Decision Three: Bss Single')
    plt.plot(depth, NRMSEr[4],'brown',linestyle='dashed',label='Random Forest: Bss Single')
    plt.plot(depth, NRMSEd[5],'yellow',label='Decision Three: Text Multi')
    plt.plot(depth, NRMSEr[5],'yellow',linestyle='dashed',label='Random Forest: Text Multi')
    plt.plot(depth, NRMSEd[6],'violet',label='Decision Three: Data Multi')
    plt.plot(depth, NRMSEr[6],'violet',linestyle='dashed',label='Random Forest: Data Multi')
    plt.plot(depth, NRMSEd[7],'brown',label='Decision Three: Bss Multi')
    plt.plot(depth, NRMSEr[7],'brown',linestyle='dashed',label='Random Forest: Bss Multi')
    plt.legend(loc='upper left')
    plt.title("NRMSE")

    plt.subplot(2, 2, 3)
    plt.plot(depth, timed[2],'black',label='Decision Three: Text Single')
    plt.plot(depth, timer[2],'black',linestyle='dashed',label='Random Forest: Text Single')
    plt.plot(depth, timed[3],'violet',label='Decision Three: Data Single')
    plt.plot(depth, timer[3],'violet',linestyle='dashed',label='Random Forest: Data Single')
    plt.plot(depth, timed[4],'brown',label='Decision Three: Bss Single')
    plt.plot(depth, timer[4],'brown',linestyle='dashed',label='Random Forest: Bss Single')
    plt.plot(depth, timed[5],'yellow',label='Decision Three: Text Multi')
    plt.plot(depth, timer[5],'yellow',linestyle='dashed',label='Random Forest: Text Multi')
    plt.plot(depth, timed[6],'violet',label='Decision Three: Data Multi')
    plt.plot(depth, timer[6],'violet',linestyle='dashed',label='Random Forest: Data Multi')
    plt.plot(depth, timed[7],'brown',label='Decision Three: Bss Multi')
    plt.plot(depth, timer[7],'brown',linestyle='dashed',label='Random Forest: Bss Multi')
    plt.legend(loc='upper left')
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.yscale('log')
    plt.plot(depth, MAPEd[2],'black',label='Decision Three: Text Single')
    plt.plot(depth, MAPEr[2],'black',linestyle='dashed',label='Random Forest: Text Single')
    plt.plot(depth, MAPEd[3],'violet',label='Decision Three: Data Single')
    plt.plot(depth, MAPEr[3],'violet',linestyle='dashed',label='Random Forest: Data Single')
    plt.plot(depth, MAPEd[4],'brown',label='Decision Three: Bss Single')
    plt.plot(depth, MAPEr[4],'brown',linestyle='dashed',label='Random Forest: Bss Single')
    plt.plot(depth, MAPEd[5],'yellow',label='Decision Three: Text Multi')
    plt.plot(depth, MAPEr[5],'yellow',linestyle='dashed',label='Random Forest: Text Multi')
    plt.plot(depth, MAPEd[6],'violet',label='Decision Three: Data Multi')
    plt.plot(depth, MAPEr[6],'violet',linestyle='dashed',label='Random Forest: Data Multi')
    plt.plot(depth, MAPEd[7],'brown',label='Decision Three: Bss Multi')
    plt.plot(depth, MAPEr[7],'brown',linestyle='dashed',label='Random Forest: Bss Multi')
    plt.legend(loc='upper left')
    plt.title("MAPE")

    plt.savefig(os.getcwd()+'\\'+p+'\MultiSingle.png')

    count=1
    plt.figure(figsize=(20,18))
    for i in [0,5,10,15,20]:
        title=DecisionTest.to_numpy()[15+i,0]
        MSEd=DecisionTest.to_numpy()[15+i:20+i,2]
        MAEd=DecisionTest.to_numpy()[15+i:20+i,5]
        MSEr=RandomTest.to_numpy()[ 15+i:20+i,2]
        MAEr=RandomTest.to_numpy()[15+i:20+i,5]

        

        plt.subplot(4, 3, count)
        plt.plot(depth, MSEd,'r',label='Decision Three')
        plt.plot(depth, MSEr,'g',label='Random Forest')
        plt.legend(loc='upper left')
        plt.title(title.split(':')[1]+": RMSE")


        plt.subplot(4, 3, count+1)
        plt.plot(depth, MAEd,'r',label='Decision Three')
        plt.plot(depth, MAEr,'g',label='Random Forest')
        plt.title(title.split(':')[1]+": MAE")
        plt.legend(loc='upper left')
        plt.plot()

        count=count+2
    plt.savefig(os.getcwd()+'\\'+p+'\Specific.png')