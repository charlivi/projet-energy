import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from joblib import dump

dfn=pd.read_csv("df_nettoye.csv", sep=",")
dfn = dfn.drop(columns=['Unnamed: 0'])

df_un = dfn.drop(columns=['Energies renouvelables (MW)','Production (MW)'])
df_un.head()
target_un = df_un['Consommation (MW)']
feats_un = df_un.drop('Consommation (MW)', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(feats_un, target_un, test_size=0.25, random_state = 42)
cols = feats_un.columns
sc = StandardScaler()
X_train.loc[:,cols] = sc.fit_transform(X_train[cols])
X_test.loc[:,cols] = sc.transform(X_test[cols])

lr = LinearRegression()
lr.fit(X_train, y_train)
dump(lr, 'lr.py') 

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train)
dump(dtr, 'dtr.py') 

forest = RandomForestRegressor(random_state=42)
forest.fit(X_train, y_train)
dump(forest, 'forest.py') 

forest_op = RandomForestRegressor(n_estimators=413, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features='log2', random_state=42)
forest_op.fit(X_train, y_train)
dump(forest_op, 'forest_op.py') 

gbr = GradientBoostingRegressor(max_depth=9, random_state=42, n_estimators=1400, learning_rate=0.05)
gbr.fit(X_train, y_train)
dump(gbr, 'gbr.py') 