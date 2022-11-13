
import pandas as pd
import csv
iss=['Leon3']
import csv
for x in iss:
    with open("./TotalParameterMatrix"+x+".csv", newline="", encoding="ISO-8859-1") as filecsv:
        lettore = pd.read_csv(filecsv,delimiter=";")
        print(lettore)
        for linea in lettore:
            if linea[32]==0 or linea[34]==0 or linea[36]==0 or linea[37]==0 or linea[38]==0 or linea[39] == 0 or linea[40]==0:
                print('ERRORE A LINEA'+linea+'dell iss'+x)
                break
                
            for i in range(1,len(linea)):
                if linea[i]==None:
                    print('ERRORE A LINEA'+linea+'dell iss'+x)
                    break
