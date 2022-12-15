import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iss=['Atmega328p','Armv4t','Armv6-m','Leon3']
results = pd.read_csv ("TotalParameterMatrixWithInputs.csv", skipinitialspace=True,sep=';')

corr_df = results.corr()
print(corr_df.iloc[4:106])

heatmap = sns.heatmap(corr_df.iloc[4:106])
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':5}, pad=20)
plt.title("HeatMap")
plt.show()

        #fig=pd.plotting.scatter_matrix(corr_df.iloc[:, 0:32])
        #plt.title("ScatterMatrix of "+x)
        #plt.show()