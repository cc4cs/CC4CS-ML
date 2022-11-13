import csv
import pandas as pd
import numpy as np
import statistics as stat

iss=['ARMV6-M','ARMV4T','LEON3','Atmega328p']
import csv

with open("TotalParameterMatrixAtmega328p.csv","r") as disco:
# Stiamo creando un oggetto del lettore csv
    csvreader_object=csv.reader(disco)
# La riga salterà la prima riga del file csv (riga di intestazione)
    features=next(csvreader_object)[0].split(';')
    


with open('generalStatistics.csv', 'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['id','mean','mode','median','var','std','median','min','q1','q2','q3','q4','max'])
            for p in iss:
                lettore = pd.read_csv("TotalParameterMatrix"+p+".csv" ,delimiter=";")
                for i in range(6,len(lettore.axes[1])):
                    sum=np.sum(lettore.iloc[:,i])
                    mean = np.mean(lettore.iloc[:,i]) #calcola la media della prima colonna
                    std = np.std(lettore.iloc[:,i])
                    mode=stat.mode(lettore.iloc[:,i])
                    median=np.median(lettore.iloc[:,i])
                    var=np.var(lettore.iloc[:,i])
                    q1=np.quantile(lettore.iloc[:,i],q=0.25)
                    q2=np.quantile(lettore.iloc[:,i],q=0.50)
                    q3=np.quantile(lettore.iloc[:,i],q=0.75)
                    q4=np.quantile(lettore.iloc[:,i],q=0.99)
                    max=lettore.iloc[:,i].max()
                    min=lettore.iloc[:,i].min()
                    writer.writerow([str(p)+"--- in feature:"+str(features[i]),mean,mode,median,var,std,median,min,q1,q2,q3,q4,max])

functions=['gcd','banker_algorithm','bellmanford','bfs','binary_search','fibcall','bubble_sort','insertionsort','selectionsort','matrix_mult','kruskal','park_miller','qsort','quicksort']
for x in iss:
    with open('specificStatistics'+x+'.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id','mean','mode','median','var','std','median','min','q1','q2','q3','q4','max'])
        for f in functions:  
            lettore = pd.read_csv("TotalParameterMatrix"+x+".csv" ,delimiter=";")
            g =  lettore.groupby("BOARD")
            h=g.get_group(x)
            g =  h.groupby("FUNCTION")
            h=g.get_group(f)
            for i in range(6,len(h.axes[1])):
                sum=np.sum(h.iloc[:,i])
                mean = np.mean(h.iloc[:,i]) #calcola la media della prima colonna
                std = np.std(h.iloc[:,i])
                mode=stat.mode(h.iloc[:,i])
                median=np.median(h.iloc[:,i])
                var=np.var(h.iloc[:,i])
                q1=np.quantile(h.iloc[:,i],q=0.25)
                q2=np.quantile(h.iloc[:,i],q=0.50)
                q3=np.quantile(h.iloc[:,i],q=0.75)
                q4=np.quantile(h.iloc[:,i],q=0.99)
                max=h.iloc[:,i].max()
                min=h.iloc[:,i].min()
                writer.writerow(['total of function:'+f+'---feature:'+str(features[i]),mean,mode,median,var,std,median,min,q1,q2,q3,q4,max])