from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC 
import pandas as pd
from joblib import dump

et = pd.read_csv('df_new.csv', sep = ',')
et['Saison']=et['Saison'].replace(to_replace = ['Hiver', 'Printemps', 'Eté', 'Automne'], value= ['0','1','2','3']).astype(int)
et['Consommation (MW)'] = et['Consommation (MW)'].astype(int)
et['Consommation (MW)']= pd.qcut(et['Consommation (MW)'], q=4, labels= False)
et_un = et.drop(columns=['Energies renouvelables (MW)','Production (MW)','Température référence (°C)'])
y_et = et_un['Consommation (MW)']
X_et = et_un.drop('Consommation (MW)', axis = 1)

X_train_et, X_test_et, y_train_et, y_test_et = train_test_split(X_et, y_et, test_size=0.2, random_state = 42)
sc = StandardScaler()
X_train_et = sc.fit_transform(X_train_et)
X_test_et = sc.transform(X_test_et)

le = LabelEncoder()
le.fit_transform(y_train_et)
le.transform(y_test_et)

rl1 = LogisticRegression()
rl1.fit(X_train_et, y_train_et)
dump(rl1, 'reg_logistique_un.py')

knn1 = KNeighborsClassifier()
knn1.fit(X_train_et, y_train_et)
dump(knn1, "knn_un.py")

rfc1 = RandomForestClassifier()
rfc1.fit(X_train_et, y_train_et)
dump(rfc1, "foret_aleatoire_class_un.py")

svc1 = SVC()
svc1.fit(X_train_et, y_train_et)
dump(svc1, "support_vector_class_un.py")
