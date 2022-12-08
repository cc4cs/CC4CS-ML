
import pandas as pd
import csv

iss=['Atmega328p','Armv4t','Armv6-m','Leon3']
#create the dictionary, `my_dictionary`, using the fromkeys() method

for x in iss:
    results = pd.read_csv ("TotalParameterMatrix"+x+".csv",sep=';')
    col=results.columns

with open('Validation.csv', 'w') as csvfile:        
        writer = csv.writer(csvfile)
        writer.writerow(['feature','iss','function','val','input'])
        with open("./TotalParameterMatrix"+x+".csv", newline="", encoding="ISO-8859-1") as filecsv:
            lettore = pd.read_csv(filecsv,delimiter=";")
            for xx in range(len(lettore)):
                for i in range(6,104):
                    if lettore.iloc[xx,i]==0:
                        writer.writerow([col[i],x,str(lettore.iloc[xx,3]),0,lettore.iloc[xx,5]])

