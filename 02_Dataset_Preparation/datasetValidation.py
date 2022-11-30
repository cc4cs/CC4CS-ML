
import pandas as pd
import csv
iss=['Armv4T','Atmega328p','Leon3','Armv6-m']
import csv

with open("TotalParameterMatrixAtmega328p.csv","r") as disco:
# Stiamo creando un oggetto del lettore csv
    
    csvreader_object=csv.reader(disco)
# La riga salterà la prima riga del file csv (riga di intestazione)
    features=next(csvreader_object)[0].split(';')



with open('Validation.csv', 'w') as csvfile:
    for x in iss:
        writer = csv.writer(csvfile)
        writer.writerow(['feature','iss','function','val'])
        with open("./TotalParameterMatrix"+x+".csv", newline="", encoding="ISO-8859-1") as filecsv:
            lettore = pd.read_csv(filecsv,delimiter=";")
            for xx in range(len(lettore)):
                for i in range(6,len(lettore.axes[1])-61):
                    if lettore.iloc[:,i][xx]==0:
                        writer.writerow([features[i],x,str(lettore.iloc[:,3][1]),0])

