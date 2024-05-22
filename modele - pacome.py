import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

pb = pd.read_csv('DataFrame.csv', sep = ',')
pb_un = pb
pb_un.head()
y_pb = pb_un['Consommation (MW)']
X_pb = pb_un.drop('Consommation (MW)', axis = 1)
X_train_pb, X_test_pb, y_train_pb, y_test_pb = train_test_split(X_pb, y_pb, test_size=0.25, random_state = 42)
cols = X_pb.columns
sc_pb = StandardScaler()
X_train_pb.loc[:,cols] = sc_pb.fit_transform(X_train_pb[cols])
X_test_pb.loc[:,cols] = sc_pb.transform(X_test_pb[cols])

lr_pb = LinearRegression()
lr_pb.fit(X_train_pb, y_train_pb)
dump(lr_pb, 'model_reg_line.py') 

dtr_pb = DecisionTreeRegressor(random_state=42)
dtr_pb.fit(X_train_pb, y_train_pb)
dump(dtr_pb, 'model_reg_dtr.py') 

forest_pb = RandomForestRegressor(random_state=42)
forest_pb.fit(X_train_pb, y_train_pb)
dump(forest_pb, 'model_reg_forest.py') 