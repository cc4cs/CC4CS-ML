
import pandas as pd
import csv
iss=['ARMV4T','Atmega328p','LEON3','ARMV6-M']
import csv

with open("TotalParameterMatrixAtmega328p.csv","r") as disco:
# Stiamo creando un oggetto del lettore csv
    csvreader_object=csv.reader(disco)
# La riga salterà la prima riga del file csv (riga di intestazione)
    features=next(csvreader_object)[0].split(';')

for x in iss:
    with open("./TotalParameterMatrix"+x+".csv", newline="", encoding="ISO-8859-1") as filecsv:
        lettore = pd.read_csv(filecsv,delimiter=";")
        for i in range(6,len(lettore.axes[1])):
            if lettore.iloc[:,i][1]==0:
                print('feature:'+features[i]+'dell iss'+x+' nella function:'+str(lettore.iloc[:,3][1])+'  vale 0')

