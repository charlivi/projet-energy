import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump

dfn=pd.read_csv("df_nettoye.csv", sep=",")
dfn = dfn.drop(columns=['Unnamed: 0','Eolien (MW)','Solaire (MW)','Hydraulique (MW)','Bioénergies (MW)','Production (MW)'])
dfp=pd.read_csv("df_test.csv", sep=",")
dfp = dfp.drop(columns=['Unnamed: 0'])

X_train_p = dfn.drop('Consommation (MW)', axis = 1)
X_test_p = dfp.drop('Consommation (MW)', axis = 1)
y_train_p = dfn['Consommation (MW)']
y_test_p = dfp['Consommation (MW)']

cols_n = X_train_p.columns
cols = X_test_p.columns
sc = StandardScaler()
X_train_p.loc[:,cols_n] = sc.fit_transform(X_train_p[cols_n])
X_test_p.loc[:,cols] = sc.transform(X_test_p[cols])

forest = RandomForestRegressor(n_estimators=413, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features='log2', random_state=42)
forest.fit(X_train_p, y_train_p)
dump(forest, 'forest_pred.py') 

gbr = GradientBoostingRegressor(max_depth=9, random_state=42, n_estimators=1400, learning_rate=0.05)
gbr.fit(X_train_p, y_train_p)
dump(gbr, 'gbr_pred.py') 