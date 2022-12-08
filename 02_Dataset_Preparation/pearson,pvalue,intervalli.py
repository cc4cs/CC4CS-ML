import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os
import csv



def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi




iss=['Atmega328p','Armv4t','Armv6-m','Leon3']
#create the dictionary, `my_dictionary`, using the fromkeys() method
my_dictionary = {}

for x in iss:
    results = pd.read_csv ("TotalParameterMatrix"+x+".csv",sep=';')
    col=results.columns
    col=col[6:104]
    for c in col:
        my_dictionary[c]=results[c].values
        
    with open(os.getcwd()+"/"+x+"Paersonresults.csv",'w') as f:
        print(os.getcwd()+x+"Paersonresults.csv")
        writer = csv.writer(f)
        writer.writerow(['feature 1','feature 2','pearson', 'p', 'lo', 'hi'])
        for d in my_dictionary:
            for dd in my_dictionary:
                    if(d!=dd):
                        r, p, lo, hi=pearsonr_ci(my_dictionary[d],my_dictionary[dd])
                        writer.writerow([d,dd, r, p, lo, hi])