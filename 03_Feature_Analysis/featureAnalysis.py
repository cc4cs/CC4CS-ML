##### Feature Importance Code

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, log_loss, accuracy_score, f1_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor

ESTIMATORS = 250


df = pd.read_csv('TotalParameterMatrixWithInputs.csv', sep=';')
del df['Unnamed: 0']

listHeader = list(df.columns.values)
y = df[listHeader[38]].astype(float)
del listHeader[35:41]
del listHeader[:5]
del listHeader[-3]

print(listHeader)

#temporary dataframe for results
dfAllResults = pd.DataFrame()
    
X = df[listHeader].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=False)

def feature_importances_histo(clf, X_train, y_train=None,
                             top_n=10, figsize=(8, 8), print_table=False, title="Feature Importances"):
    
    __name__ = "feature_importances_histo"

    #try:
    if not hasattr(clf, 'feature_importances_'):
        clf.fit(X_train.values, y_train.values.ravel())

        if not hasattr(clf, 'feature_importances_'):
            raise AttributeError("{} does not have feature_importances_ attribute".
                                  format(clf.__class__.__name__))
        
    feat_imp = pd.DataFrame({title: clf.feature_importances_})
    feat_imp['feature'] = X_train.columns
    dfAllResults['feature'] = X_train.columns
    dfAllResults[title] = clf.feature_importances_
    feat_imp = feat_imp.iloc[:top_n]
    feat_imp.sort_values(by='feature', ascending=False, inplace=True)
    feat_imp = feat_imp.set_index('feature', drop=True)

    print("================  Feature Importance  ==============")
    print(title)
    print("____________________________________________________")
    print(feat_imp)
    print("----------------------------------------------------")
    
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.show()

    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='feature', ascending=False))

    return feat_imp

## Main Function

clfs = [ExtraTreesRegressor(),
        GradientBoostingRegressor(learning_rate=0.1, n_estimators=ESTIMATORS),
        AdaBoostRegressor(learning_rate=0.1),
        RandomForestRegressor()
        ]

for clf in clfs:
    try:
        _ = feature_importances_histo(clf, X_train, y_train, top_n=X_train.shape[1], title=clf.__class__.__name__)
    except AttributeError as e:
        print(e)        
print("|------------------------------------|")
dfAllResults.set_index(['feature'], inplace=True)

dfAllResults.to_csv('generalScore.csv', sep=',', encoding='utf-8', index=False)
dfAllResults['sum'] = dfAllResults[dfAllResults.columns].sum(axis=1)
dfAllResults['avg'] = dfAllResults['sum']/(len(dfAllResults.columns) - 1)
file_name = 'allCLFscore.csv'
dfAllResults.to_csv(file_name, sep='\t', encoding='utf-8')
#files.download(file_name)

print(dfAllResults)

dfavg = dfAllResults[['avg']]
file_name = 'scoreMean.csv'
dfavg.to_csv(file_name, sep='\t', encoding='utf-8')
#files.download(file_name)

dfavg.plot.barh(title='Feature Importance Mean Score',logx=True, figsize=(8, 8))
plt.xlabel('Feature Importance Mean Score')
plt.xscale('log')
plt.axvline(x=0.001)
plt.show()

print("=============   Feature Importance Mean Score   =============")
print(dfavg)
dfpruned = dfavg[~(dfavg['avg'] < 0.05)]
print("=============   Feature Pruned by Score   =============")
print(dfpruned)

dfavg.plot.barh(title='Feature Importance Mean Score',logx=True, figsize=(8, 8))
plt.xscale('log')
plt.axvline(x=0.001, color='r', linestyle='--')
plt.show()