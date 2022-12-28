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

RMSPEr=[]
NRMSEr=[]
timer=[]
R2r=[]


iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
for p in iss:
    
    DecisionTest = read_csv(os.getcwd()+'\\'+p+'\\REG_TREE\\resultsValidation.csv', skipinitialspace=True,sep=',', header = 0)
    
    RandomTest = read_csv(os.getcwd()+'\\'+p+'\\RandomForest\\resultsValidation.csv', skipinitialspace=True,sep=',', header = 0)

    #single Assembly
    depth=[8,10,12,16]
    print("Testing for processor: "+p)

    #valori single
    for i in [0,4,8,12,16]:

        RMSPEd.append(DecisionTest.to_numpy()[12+i:16+i,3])
        NRMSEd.append(DecisionTest.to_numpy()[12+i:16+i,2])
        timed.append(DecisionTest.to_numpy()[12+i:16+i,7])
        R2d.append(DecisionTest.to_numpy()[12+i:16+i,6])
    
        RMSPEr.append(RandomTest.to_numpy()[12+i:16+i,3])
        NRMSEr.append(RandomTest.to_numpy()[12+i:16+i,2])
        timer.append(RandomTest.to_numpy()[12+i:16+i,7])
        R2r.append(RandomTest.to_numpy()[12+i:16+i,6])
    #valori multi
    for i in [0,1,2]:
        NRMSEd.append(np.array([DecisionTest.to_numpy()[0+i,3],DecisionTest.to_numpy()[3+i,3],DecisionTest.to_numpy()[6+i,3],DecisionTest.to_numpy()[9+i,3]]))
        RMSPEd.append(np.array([DecisionTest.to_numpy()[0+i,4],DecisionTest.to_numpy()[3+i,4],DecisionTest.to_numpy()[6+i,4],DecisionTest.to_numpy()[9+i,4]]))
        timed.append(np.array([DecisionTest.to_numpy()[0+i,7],DecisionTest.to_numpy()[3+i,5],DecisionTest.to_numpy()[6+i,7],DecisionTest.to_numpy()[9+i,7]]))
        R2d.append(np.array([DecisionTest.to_numpy()[0+i,6],DecisionTest.to_numpy()[3+i,6],DecisionTest.to_numpy()[6+i,6],DecisionTest.to_numpy()[9+i,6]]))


        NRMSEr.append(np.array([RandomTest.to_numpy()[0+i,3],RandomTest.to_numpy()[3+i,3],RandomTest.to_numpy()[6+i,3],RandomTest.to_numpy()[9+i,3]]))
        RMSPEr.append(np.array([RandomTest.to_numpy()[0+i,4],RandomTest.to_numpy()[3+i,4],RandomTest.to_numpy()[6+i,4],RandomTest.to_numpy()[9+i,4]]))
        timer.append(np.array([RandomTest.to_numpy()[0+i,5],RandomTest.to_numpy()[3+i,7],RandomTest.to_numpy()[6+i,7],RandomTest.to_numpy()[9+i,7]]))
        R2r.append(np.array([RandomTest.to_numpy()[0+i,6],RandomTest.to_numpy()[3+i,6],RandomTest.to_numpy()[6+i,6],RandomTest.to_numpy()[9+i,6]]))

    print('single target')
    plt.subplot(2, 2, 1)
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
    plt.legend()
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
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
    plt.legend()
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
    plt.legend()
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.plot(depth, R2d[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, R2r[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, R2d[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, R2r[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, R2d[2],'yellow',label='Decision Three: Text')
    plt.plot(depth, R2r[2],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, R2d[3],'violet',label='Decision Three: Data')
    plt.plot(depth, R2r[3],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, R2d[4],'brown',label='Decision Three: Bss')
    plt.plot(depth, R2r[4],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("R2")

    plt.show()

    """ plt.subplot(2, 2, 1)
    plt.plot(depth, RMSPEd[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, RMSPEr[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, RMSPEd[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, RMSPEr[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, RMSPEd[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, RMSPEr[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, RMSPEd[6],'violet',label='Decision Three: Data')
    plt.plot(depth, RMSPEr[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, RMSPEd[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, RMSPEr[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
    plt.plot(depth, NRMSEd[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, NRMSEr[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, NRMSEd[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, NRMSEr[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, NRMSEd[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, NRMSEr[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, NRMSEd[6],'violet',label='Decision Three: Data')
    plt.plot(depth, NRMSEr[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, NRMSEd[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, NRMSEr[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("NRMSE")

    plt.subplot(2, 2, 3)
    plt.plot(depth, timed[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, timer[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, timed[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, timer[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, timed[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, timer[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, timed[6],'violet',label='Decision Three: Data')
    plt.plot(depth, timer[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, timed[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, timer[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.plot(depth, R2d[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, R2r[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, R2d[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, R2r[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, R2d[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, R2r[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, R2d[6],'violet',label='Decision Three: Data')
    plt.plot(depth, R2r[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, R2d[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, R2r[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("R2")

    plt.show() """


    plt.subplot(2, 2, 1)
    plt.plot(depth, RMSPEd[2],'brown',label='Decision Three: Text Single')
    plt.plot(depth, RMSPEr[2],'brown',linestyle='dashed',label='Random Forest: Text Single')
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
    plt.legend()
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
    plt.plot(depth, NRMSEd[2],'brown',label='Decision Three: Text Single')
    plt.plot(depth, NRMSEr[2],'brown',linestyle='dashed',label='Random Forest: Text Single')
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
    plt.legend()
    plt.title("NRMSE")

    plt.subplot(2, 2, 3)
    plt.plot(depth, timed[2],'brown',label='Decision Three: Text Single')
    plt.plot(depth, timer[2],'brown',linestyle='dashed',label='Random Forest: Text Single')
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
    plt.legend()
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.plot(depth, R2d[2],'brown',label='Decision Three: Text Single')
    plt.plot(depth, R2r[2],'brown',linestyle='dashed',label='Random Forest: Text Single')
    plt.plot(depth, R2d[3],'violet',label='Decision Three: Data Single')
    plt.plot(depth, R2r[3],'violet',linestyle='dashed',label='Random Forest: Data Single')
    plt.plot(depth, R2d[4],'brown',label='Decision Three: Bss Single')
    plt.plot(depth, R2r[4],'brown',linestyle='dashed',label='Random Forest: Bss Single')
    plt.plot(depth, R2d[5],'yellow',label='Decision Three: Text Multi')
    plt.plot(depth, R2r[5],'yellow',linestyle='dashed',label='Random Forest: Text Multi')
    plt.plot(depth, R2d[6],'violet',label='Decision Three: Data Multi')
    plt.plot(depth, R2r[6],'violet',linestyle='dashed',label='Random Forest: Data Multi')
    plt.plot(depth, R2d[7],'brown',label='Decision Three: Bss Multi')
    plt.plot(depth, R2r[7],'brown',linestyle='dashed',label='Random Forest: Bss Multi')
    plt.legend()
    plt.title("R2")

    plt.show()

    for i in [0,1,2]:
        
        title=DecisionTest.to_numpy()[0+i,0]

        MSEd=np.array([DecisionTest.to_numpy()[0+i,2],DecisionTest.to_numpy()[3+i,2],DecisionTest.to_numpy()[6+i,2],DecisionTest.to_numpy()[9+i,2]])
        MAEd=np.array([DecisionTest.to_numpy()[0+i,4],DecisionTest.to_numpy()[3+i,4],DecisionTest.to_numpy()[6+i,4],DecisionTest.to_numpy()[9+i,4]])

        MSEr=np.array([RandomTest.to_numpy()[0+i,2],RandomTest.to_numpy()[3+i,2],RandomTest.to_numpy()[6+i,2],RandomTest.to_numpy()[9+i,2]])
        MAEr=np.array([RandomTest.to_numpy()[0+i,4],RandomTest.to_numpy()[3+i,4],RandomTest.to_numpy()[6+i,4],RandomTest.to_numpy()[9+i,4]])

    
        plt.subplot(1, 2, 1)
        plt.plot(depth, MSEd,'r',label='Decision Three')
        plt.plot(depth, MSEr,'g',label='Random Forest')
        plt.title(title+": RMSE")

        plt.subplot(1, 2, 2)
        plt.plot(depth, MAEd,'r',label='Decision Three')
        plt.plot(depth, MAEr,'g',label='Random Forest')
        plt.title(title+": MAE")


        plt.show()

    for i in [0,4,8,12,16]:
        title=DecisionTest.to_numpy()[12+i,0]

        MSEd=DecisionTest.to_numpy()[12+i:16+i,2]
        MAEd=DecisionTest.to_numpy()[12+i:16+i,4]
        MSEr=RandomTest.to_numpy()[12+i:16+i,2]
        MAEr=RandomTest.to_numpy()[12+i:16+i,4]

    
        plt.subplot(1, 2, 1)
        plt.plot(depth, MSEd,'r',label='Decision Three')
        plt.plot(depth, MSEr,'g',label='Random Forest')
        plt.title(title+": RMSE")


        plt.subplot(1, 2, 2)
        plt.plot(depth, MAEd,'r',label='Decision Three')
        plt.plot(depth, MAEr,'g',label='Random Forest')
        plt.title(title+": MAE")
        plt.plot()

        plt.show()
    

    
    DecisionTest = read_csv(os.getcwd()+'\\'+p+'\\REG_TREE\\resultsValidation.csv', skipinitialspace=True,sep=',', header = 0)
    RandomTest = read_csv(os.getcwd()+'\\'+p+'\\RandomForest\\resultsValidation.csv', skipinitialspace=True,sep=',', header = 0)
    print("Training for processor: "+p)

    for i in [0,4,8,12,16]:

        RMSPEd.append(DecisionTest.to_numpy()[12+i:16+i,3])
        NRMSEd.append(DecisionTest.to_numpy()[12+i:16+i,2])
        timed.append(DecisionTest.to_numpy()[12+i:16+i,7])
        R2d.append(DecisionTest.to_numpy()[12+i:16+i,6])
    
        RMSPEr.append(RandomTest.to_numpy()[12+i:16+i,3])
        NRMSEr.append(RandomTest.to_numpy()[12+i:16+i,2])
        timer.append(RandomTest.to_numpy()[12+i:16+i,7])
        R2r.append(RandomTest.to_numpy()[12+i:16+i,6])
    #valori multi
    for i in [0,1,2]:
        NRMSEd.append(np.array([DecisionTest.to_numpy()[0+i,3],DecisionTest.to_numpy()[3+i,3],DecisionTest.to_numpy()[6+i,3],DecisionTest.to_numpy()[9+i,3]]))
        RMSPEd.append(np.array([DecisionTest.to_numpy()[0+i,4],DecisionTest.to_numpy()[3+i,4],DecisionTest.to_numpy()[6+i,4],DecisionTest.to_numpy()[9+i,4]]))
        timed.append(np.array([DecisionTest.to_numpy()[0+i,7],DecisionTest.to_numpy()[3+i,5],DecisionTest.to_numpy()[6+i,7],DecisionTest.to_numpy()[9+i,7]]))
        R2d.append(np.array([DecisionTest.to_numpy()[0+i,6],DecisionTest.to_numpy()[3+i,6],DecisionTest.to_numpy()[6+i,6],DecisionTest.to_numpy()[9+i,6]]))


        NRMSEr.append(np.array([RandomTest.to_numpy()[0+i,3],RandomTest.to_numpy()[3+i,3],RandomTest.to_numpy()[6+i,3],RandomTest.to_numpy()[9+i,3]]))
        RMSPEr.append(np.array([RandomTest.to_numpy()[0+i,4],RandomTest.to_numpy()[3+i,4],RandomTest.to_numpy()[6+i,4],RandomTest.to_numpy()[9+i,4]]))
        timer.append(np.array([RandomTest.to_numpy()[0+i,5],RandomTest.to_numpy()[3+i,7],RandomTest.to_numpy()[6+i,7],RandomTest.to_numpy()[9+i,7]]))
        R2r.append(np.array([RandomTest.to_numpy()[0+i,6],RandomTest.to_numpy()[3+i,6],RandomTest.to_numpy()[6+i,6],RandomTest.to_numpy()[9+i,6]]))


    plt.subplot(2, 2, 1)
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
    plt.legend()
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
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
    plt.legend()
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
    plt.legend()
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.plot(depth, R2d[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, R2r[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, R2d[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, R2r[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, R2d[2],'yellow',label='Decision Three: Text')
    plt.plot(depth, R2r[2],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, R2d[3],'violet',label='Decision Three: Data')
    plt.plot(depth, R2r[3],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, R2d[4],'brown',label='Decision Three: Bss')
    plt.plot(depth, R2r[4],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("R2")

    plt.show()

    """ plt.subplot(2, 2, 1)
    plt.plot(depth, RMSPEd[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, RMSPEr[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, RMSPEd[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, RMSPEr[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, RMSPEd[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, RMSPEr[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, RMSPEd[6],'violet',label='Decision Three: Data')
    plt.plot(depth, RMSPEr[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, RMSPEd[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, RMSPEr[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
    plt.plot(depth, NRMSEd[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, NRMSEr[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, NRMSEd[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, NRMSEr[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, NRMSEd[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, NRMSEr[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, NRMSEd[6],'violet',label='Decision Three: Data')
    plt.plot(depth, NRMSEr[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, NRMSEd[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, NRMSEr[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("NRMSE")

    plt.subplot(2, 2, 3)
    plt.plot(depth, timed[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, timer[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, timed[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, timer[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, timed[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, timer[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, timed[6],'violet',label='Decision Three: Data')
    plt.plot(depth, timer[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, timed[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, timer[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.plot(depth, R2d[0],'r',label='Decision Three: Assembly')
    plt.plot(depth, R2r[0],'r',linestyle='dashed',label='Random Forest: Assembly')
    plt.plot(depth, R2d[1],'g',label='Decision Three: ClockCycles')
    plt.plot(depth, R2r[1],'g',linestyle='dashed',label='Random Forest: ClockCycles')
    plt.plot(depth, R2d[5],'yellow',label='Decision Three: Text')
    plt.plot(depth, R2r[5],'yellow',linestyle='dashed',label='Random Forest: Text')
    plt.plot(depth, R2d[6],'violet',label='Decision Three: Data')
    plt.plot(depth, R2r[6],'violet',linestyle='dashed',label='Random Forest: Data')
    plt.plot(depth, R2d[7],'brown',label='Decision Three: Bss')
    plt.plot(depth, R2r[7],'brown',linestyle='dashed',label='Random Forest: Bss')
    plt.legend()
    plt.title("R2")

    plt.show()
 """
    plt.subplot(2, 2, 1)
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
    plt.legend()
    plt.title("RMSPE")
    
    plt.subplot(2, 2, 2)
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
    plt.legend()
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
    plt.legend()
    plt.title("time")

    plt.subplot(2, 2, 4)
    plt.plot(depth, R2d[2],'black',label='Decision Three: Text Single')
    plt.plot(depth, R2r[2],'black',linestyle='dashed',label='Random Forest: Text Single')
    plt.plot(depth, R2d[3],'violet',label='Decision Three: Data Single')
    plt.plot(depth, R2r[3],'violet',linestyle='dashed',label='Random Forest: Data Single')
    plt.plot(depth, R2d[4],'brown',label='Decision Three: Bss Single')
    plt.plot(depth, R2r[4],'brown',linestyle='dashed',label='Random Forest: Bss Single')
    plt.plot(depth, R2d[5],'yellow',label='Decision Three: Text Multi')
    plt.plot(depth, R2r[5],'yellow',linestyle='dashed',label='Random Forest: Text Multi')
    plt.plot(depth, R2d[6],'violet',label='Decision Three: Data Multi')
    plt.plot(depth, R2r[6],'violet',linestyle='dashed',label='Random Forest: Data Multi')
    plt.plot(depth, R2d[7],'brown',label='Decision Three: Bss Multi')
    plt.plot(depth, R2r[7],'brown',linestyle='dashed',label='Random Forest: Bss Multi')
    plt.legend()
    plt.title("R2")

    plt.show()

    for i in [0,1,2]:
        
        title=DecisionTest.to_numpy()[0+i,0]

        MSEd=np.array([DecisionTest.to_numpy()[0+i,2],DecisionTest.to_numpy()[3+i,2],DecisionTest.to_numpy()[6+i,2],DecisionTest.to_numpy()[9+i,2]])
        MAEd=np.array([DecisionTest.to_numpy()[0+i,4],DecisionTest.to_numpy()[3+i,4],DecisionTest.to_numpy()[6+i,4],DecisionTest.to_numpy()[9+i,4]])

        MSEr=np.array([RandomTest.to_numpy()[0+i,2],RandomTest.to_numpy()[3+i,2],RandomTest.to_numpy()[6+i,2],RandomTest.to_numpy()[9+i,2]])
        MAEr=np.array([RandomTest.to_numpy()[0+i,4],RandomTest.to_numpy()[3+i,4],RandomTest.to_numpy()[6+i,4],RandomTest.to_numpy()[9+i,4]])

    
        plt.subplot(1, 2, 1)
        plt.plot(depth, MSEd,'r',label='Decision Three')
        plt.plot(depth, MSEr,'g',label='Random Forest')
        plt.title(title+": RMSE")

        plt.subplot(1, 2, 2)
        plt.plot(depth, MAEd,'r',label='Decision Three')
        plt.plot(depth, MAEr,'g',label='Random Forest')
        plt.title(title+": MAE")


        plt.show()

    for i in [0,4,8,12,16]:

        title=DecisionTest.to_numpy()[12+i,0]

        MSEd=DecisionTest.to_numpy()[12+i:16+i,2]
        MAEd=DecisionTest.to_numpy()[12+i:16+i,4]
        MSEr=RandomTest.to_numpy()[12+i:16+i,2]
        MAEr=RandomTest.to_numpy()[12+i:16+i,4]

    
        plt.subplot(1, 2, 1)
        plt.plot(depth, MSEd,'r',label='Decision Three')
        plt.plot(depth, MSEr,'g',label='Random Forest')
        plt.title(title+": RMSE")


        plt.subplot(1, 2, 2)
        plt.plot(depth, MAEd,'r',label='Decision Three')
        plt.plot(depth, MAEr,'g',label='Random Forest')
        plt.title(title+": MAE")
        plt.plot()

        plt.show()