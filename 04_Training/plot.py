import os
import matplotlib.pyplot as plt
import csv
from pandas import read_csv
import numpy as np

iss=["Armv4t","Armv6-M","Atmega328P","Leon3"]
for p in iss:
    Decisiontrain = read_csv(os.getcwd()+'\\'+p+'\\REG_TREE\\resultsTrain.csv', skipinitialspace=True,sep=',', header = 0)
    DecisionTest = read_csv(os.getcwd()+'\\'+p+'\\REG_TREE\\resultsTest.csv', skipinitialspace=True,sep=',', header = 0)
    Randomtrain = read_csv(os.getcwd()+'\\'+p+'\\RandomForest\\resultsTrain.csv', skipinitialspace=True,sep=',', header = 0)
    RandomTest = read_csv(os.getcwd()+'\\'+p+'\\RandomForest\\resultsTest.csv', skipinitialspace=True,sep=',', header = 0)

    #single Assembly
    depth=[8,10,12,16]
    print("Testing for processor: "+p)

    for i in [0,1,2]:
        
        title=DecisionTest.to_numpy()[0+i,0]

        MSEd=np.array([DecisionTest.to_numpy()[0+i,2],DecisionTest.to_numpy()[3+i,2],DecisionTest.to_numpy()[6+i,2],DecisionTest.to_numpy()[9+i,2]])
        NRMSEd=np.array([DecisionTest.to_numpy()[0+i,3],DecisionTest.to_numpy()[3+i,3],DecisionTest.to_numpy()[6+i,3],DecisionTest.to_numpy()[9+i,3]])
        MAEd=np.array([DecisionTest.to_numpy()[0+i,4],DecisionTest.to_numpy()[3+i,4],DecisionTest.to_numpy()[6+i,4],DecisionTest.to_numpy()[9+i,4]])
        timed=np.array([DecisionTest.to_numpy()[0+i,5],DecisionTest.to_numpy()[3+i,5],DecisionTest.to_numpy()[6+i,5],DecisionTest.to_numpy()[9+i,5]])
        R2d=np.array([DecisionTest.to_numpy()[0+i,6],DecisionTest.to_numpy()[3+i,6],DecisionTest.to_numpy()[6+i,6],DecisionTest.to_numpy()[9+i,6]])


        MSEr=np.array([RandomTest.to_numpy()[0+i,2],RandomTest.to_numpy()[3+i,2],RandomTest.to_numpy()[6+i,2],RandomTest.to_numpy()[9+i,2]])
        NRMSEr=np.array([RandomTest.to_numpy()[0+i,3],RandomTest.to_numpy()[3+i,3],RandomTest.to_numpy()[6+i,3],RandomTest.to_numpy()[9+i,3]])
        MAEr=np.array([RandomTest.to_numpy()[0+i,4],RandomTest.to_numpy()[3+i,4],RandomTest.to_numpy()[6+i,4],RandomTest.to_numpy()[9+i,4]])
        timer=np.array([RandomTest.to_numpy()[0+i,5],RandomTest.to_numpy()[3+i,5],RandomTest.to_numpy()[6+i,5],RandomTest.to_numpy()[9+i,5]])
        R2r=np.array([RandomTest.to_numpy()[0+i,6],RandomTest.to_numpy()[3+i,6],RandomTest.to_numpy()[6+i,6],RandomTest.to_numpy()[9+i,6]])

    
        plt.subplot(2, 3, 1)
        plt.plot(depth, MSEd,'r')
        plt.plot(depth, MSEr,'g')
        plt.title(title+": MSE")

        plt.subplot(2, 3, 2)
        plt.plot(depth, NRMSEd,'r')
        plt.plot(depth, NRMSEr,'g')
        plt.title(title+": NRMSE")

        plt.subplot(2, 3, 3)
        plt.plot(depth, MAEd,'r')
        plt.plot(depth, MAEr,'g')
        plt.title(title+": MAE")

        plt.subplot(2, 3, 4)
        plt.plot(depth, timed,'r')
        plt.plot(depth, timer,'g')
        plt.title(title+": R2")

        plt.subplot(2, 3, 5)
        plt.plot(depth, R2d,'r')
        plt.plot(depth, R2r,'g')
        plt.title(title+": Time")
        plt.plot()

        plt.show()

    for i in [0,4,8,12,16]:
        title=DecisionTest.to_numpy()[12+i,0]

        MSEd=DecisionTest.to_numpy()[12+i:16+i,2]
        NRMSEd=DecisionTest.to_numpy()[12+i:16+i,3]
        MAEd=DecisionTest.to_numpy()[12+i:16+i,4]
        timed=DecisionTest.to_numpy()[12+i:16+i,5]
        R2d=DecisionTest.to_numpy()[12+i:16+i,6]
    

        MSEr=RandomTest.to_numpy()[12+i:16+i,2]
        NRMSEr=RandomTest.to_numpy()[12+i:16+i,3]
        MAEr=RandomTest.to_numpy()[12+i:16+i,4]
        timer=RandomTest.to_numpy()[12+i:16+i,5]
        R2r=RandomTest.to_numpy()[12+i:16+i,6]

    
        plt.subplot(2, 3, 1)
        plt.plot(depth, MSEd,'r')
        plt.plot(depth, MSEr,'g')
        plt.title(title+": MSE")

        plt.subplot(2, 3, 2)
        plt.plot(depth, NRMSEd,'r')
        plt.plot(depth, NRMSEr,'g')
        plt.title(title+": NRMSE")

        plt.subplot(2, 3, 3)
        plt.plot(depth, MAEd,'r')
        plt.plot(depth, MAEr,'g')
        plt.title(title+": MAE")

        plt.subplot(2, 3, 4)
        plt.plot(depth, timed,'r')
        plt.plot(depth, timer,'g')
        plt.title(title+": R2")

        plt.subplot(2, 3, 5)
        plt.plot(depth, R2d,'r')
        plt.plot(depth, R2r,'g')
        plt.title(title+": Time")
        plt.plot()

        plt.show()
        
    print("Training for processor: "+p)

    for i in [0,1,2]:
        
                title=DecisionTest.to_numpy()[0+i,0]

                MSEd=np.array([DecisionTest.to_numpy()[0+i,2],DecisionTest.to_numpy()[3+i,2],DecisionTest.to_numpy()[6+i,2],DecisionTest.to_numpy()[9+i,2]])
                NRMSEd=np.array([DecisionTest.to_numpy()[0+i,3],DecisionTest.to_numpy()[3+i,3],DecisionTest.to_numpy()[6+i,3],DecisionTest.to_numpy()[9+i,3]])
                MAEd=np.array([DecisionTest.to_numpy()[0+i,4],DecisionTest.to_numpy()[3+i,4],DecisionTest.to_numpy()[6+i,4],DecisionTest.to_numpy()[9+i,4]])
                timed=np.array([DecisionTest.to_numpy()[0+i,5],DecisionTest.to_numpy()[3+i,5],DecisionTest.to_numpy()[6+i,5],DecisionTest.to_numpy()[9+i,5]])
                R2d=np.array([DecisionTest.to_numpy()[0+i,6],DecisionTest.to_numpy()[3+i,6],DecisionTest.to_numpy()[6+i,6],DecisionTest.to_numpy()[9+i,6]])


                MSEr=np.array([RandomTest.to_numpy()[0+i,2],RandomTest.to_numpy()[3+i,2],RandomTest.to_numpy()[6+i,2],RandomTest.to_numpy()[9+i,2]])
                NRMSEr=np.array([RandomTest.to_numpy()[0+i,3],RandomTest.to_numpy()[3+i,3],RandomTest.to_numpy()[6+i,3],RandomTest.to_numpy()[9+i,3]])
                MAEr=np.array([RandomTest.to_numpy()[0+i,4],RandomTest.to_numpy()[3+i,4],RandomTest.to_numpy()[6+i,4],RandomTest.to_numpy()[9+i,4]])
                timer=np.array([RandomTest.to_numpy()[0+i,5],RandomTest.to_numpy()[3+i,5],RandomTest.to_numpy()[6+i,5],RandomTest.to_numpy()[9+i,5]])
                R2r=np.array([RandomTest.to_numpy()[0+i,6],RandomTest.to_numpy()[3+i,6],RandomTest.to_numpy()[6+i,6],RandomTest.to_numpy()[9+i,6]])

            
                plt.subplot(2, 3, 1)
                plt.plot(depth, MSEd,'r')
                plt.plot(depth, MSEr,'g')
                plt.title(title+": MSE")

                plt.subplot(2, 3, 2)
                plt.plot(depth, NRMSEd,'r')
                plt.plot(depth, NRMSEr,'g')
                plt.title(title+": NRMSE")

                plt.subplot(2, 3, 3)
                plt.plot(depth, MAEd,'r')
                plt.plot(depth, MAEr,'g')
                plt.title(title+": MAE")

                plt.subplot(2, 3, 4)
                plt.plot(depth, timed,'r')
                plt.plot(depth, timer,'g')
                plt.title(title+": R2")

                plt.subplot(2, 3, 5)
                plt.plot(depth, R2d,'r')
                plt.plot(depth, R2r,'g')
                plt.title(title+": Time")
                plt.plot()

                plt.show()

    for i in [0,4,8,12,16]:
                    title=DecisionTest.to_numpy()[12+i,0]

                    MSEd=DecisionTest.to_numpy()[12+i:16+i,2]
                    NRMSEd=DecisionTest.to_numpy()[12+i:16+i,3]
                    MAEd=DecisionTest.to_numpy()[12+i:16+i,4]
                    timed=DecisionTest.to_numpy()[12+i:16+i,5]
                    R2d=DecisionTest.to_numpy()[12+i:16+i,6]
                

                    MSEr=RandomTest.to_numpy()[12+i:16+i,2]
                    NRMSEr=RandomTest.to_numpy()[12+i:16+i,3]
                    MAEr=RandomTest.to_numpy()[12+i:16+i,4]
                    timer=RandomTest.to_numpy()[12+i:16+i,5]
                    R2r=RandomTest.to_numpy()[12+i:16+i,6]

                
                    plt.subplot(2, 3, 1)
                    plt.plot(depth, MSEd,'r')
                    plt.plot(depth, MSEr,'g')
                    plt.title(title+": MSE")

                    plt.subplot(2, 3, 2)
                    plt.plot(depth, NRMSEd,'r')
                    plt.plot(depth, NRMSEr,'g')
                    plt.title(title+": NRMSE")

                    plt.subplot(2, 3, 3)
                    plt.plot(depth, MAEd,'r')
                    plt.plot(depth, MAEr,'g')
                    plt.title(title+": MAE")

                    plt.subplot(2, 3, 4)
                    plt.plot(depth, timed,'r')
                    plt.plot(depth, timer,'g')
                    plt.title(title+": R2")

                    plt.subplot(2, 3, 5)
                    plt.plot(depth, R2d,'r')
                    plt.plot(depth, R2r,'g')
                    plt.title(title+": Time")
                    
                    plt.suptitle(title,fontsize=20)

                    plt.show()

                        