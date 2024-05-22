import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from joblib import load
from PIL import Image

st.set_page_config(page_title="Projet de Data Analyse", page_icon=":bar_chart:")
st.markdown(
    """
    <style>
         h1 {
          text-align: center;
          color: #3d85c6;
          }
    </style>
    """,
    unsafe_allow_html=True
)

df = pd.read_csv("eco.csv", sep=";")
conso = df.drop(columns = 'Column 30')
variables_numeriques = conso.select_dtypes(include=['float']).columns
conso = conso.dropna(axis=0, how='all', subset=variables_numeriques)
conso['Nucléaire (MW)'] = conso['Nucléaire (MW)'].fillna(0)
conso['Eolien (MW)'] = conso['Eolien (MW)'].fillna(0)
conso['Date'] = pd.to_datetime(conso['Date'], yearfirst=True)
conso['Production (MW)'] = conso['Thermique (MW)'] + conso['Nucléaire (MW)'] + conso['Eolien (MW)'] + conso['Solaire (MW)'] + conso['Hydraulique (MW)'] + conso['Bioénergies (MW)']
conso = conso.groupby('Date').agg({'Consommation (MW)' : 'sum', 'Production (MW)' : 'sum'})

dfn = pd.read_csv("df_nettoye.csv", sep=",")
dfn = dfn.drop(columns=['Unnamed: 0'])

df_un = dfn.drop(columns=['Energies renouvelables (MW)','Production (MW)'])
target_un = df_un['Consommation (MW)']
feats_un = df_un.drop('Consommation (MW)', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(feats_un, target_un, test_size=0.25, random_state = 42)
cols = feats_un.columns
sc = StandardScaler()
X_train.loc[:,cols] = sc.fit_transform(X_train[cols])
X_test.loc[:,cols] = sc.transform(X_test[cols])

dfm = dfn.drop(columns=['Eolien (MW)','Solaire (MW)','Hydraulique (MW)','Bioénergies (MW)','Production (MW)'])
dfp = pd.read_csv("df_test.csv", sep=",")
dfp = dfp.drop(columns=['Unnamed: 0'])

X_train_p = dfm.drop('Consommation (MW)', axis = 1)
X_test_p = dfp.drop('Consommation (MW)', axis = 1)
y_train_p = dfm['Consommation (MW)']
y_test_p = dfp['Consommation (MW)']
cols_n = X_train_p.columns
cols_p = X_test_p.columns
sc = StandardScaler()
X_train_p.loc[:,cols_n] = sc.fit_transform(X_train_p[cols_n])
X_test_p.loc[:,cols_p] = sc.transform(X_test_p[cols_p])

st.title("Projet Energie")
st.sidebar.title("Sommaire")
pages=["Accueil", "Présentation", "Exploration", "Datavisualisation", "Modélisation nationale", "Modélisation régionale", "Prédiction", "Conclusion"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.image(Image.open('Ampoule.png'))

if page == pages[0] : 
  st.markdown(
      """
      <style>
          h2 {
              font-size: 40px;
              text-align: center;
          }
          h3 {
              color: #76a5af;
              font-size: 30px;
              margin-top: 5px;
              text-align: center;
          }
          p {
              text-align: center;
          }
      </style>
      """,
      unsafe_allow_html=True
  )

  st.header("Projet de Data Analyse")
  st.subheader("Formation DataScientest")
  st.subheader("Session septembre 2023")

  col1, col2, col3 = st.columns([5,5,3])
  with col1:
    st.write("")
  with col2:
    st.image(Image.open('Ampoule.png'), width=150)
  with col3:
    st.write("")

  st.write("Damien Lioret, Enzo Tournefier, Charlène Vince, Pacôme Bouadou")

if page == pages[1] : 
  st.write("### Présentation")
  st.write('###### Les objectifs du projet Energie :')
  st.markdown('''
              - Analyser le phasage entre la consommation et la production énergétique au niveau national et régional, avec un accent sur la prévention des risques de black-out.
              - Réaliser une analyse au niveau régional pour déduire des prévisions de consommation.
              - Analyser les différentes filières de production d'énergie, en mettant particulièrement l'accent sur l'énergie nucléaire et les sources d'énergie renouvelable.
  ''')
  st.write('###### La source des données :')
  st.write("Notre jeu de données éCO2mix régionales est issu de l'ODRE (OpenData Réseaux Energies). Cet organisme est une collaboration entre les principaux producteurs de l’énergie français, ainsi que des entreprises qui gèrent son transport dans tout le pays. Ces données permettent d'analyser la consommation et la production d'énergie, tant au niveau national qu'au niveau régional. ") 
  st.write("En savoir plus : https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-tr/information/")

if page == pages[2] : 
  st.write("### Exploration du jeu de données") 
  st.write('###### Les premières lignes du jeu de données :')
  st.dataframe(df.head(5))
  st.write('Le jeu de données éCO2mix régionales couvre une période allant de janvier 2013 à mai 2022, avec une fréquence de collecte toutes les 1/2 heures. Le DataFrame possède',df.shape[0], 'lignes et', df.shape[1], 'colonnes.')

  if st.checkbox("Afficher les valeurs manquantes") :
    taux = pd.DataFrame(df.isna().sum(), columns=["Nombre de NA"])
    taux['Taux de NA'] = taux['Nombre de NA'] / df.shape[0] * 100
    st.dataframe(taux)
    st.write('Les 12 premières lignes du jeu de données sont à supprimer, ainsi que la dernière colonne.')
  
  if st.checkbox("Afficher la matrice de corrélation des variables") :
    st.image(Image.open('Matrice.png'))
    st.write("Si la matrice de corrélation nous a permis d’identifier certaines relations, aucune variable ne semble avoir de relation privilégiée avec notre variable cible (Consommation).")

  if st.checkbox("Afficher un descriptif des données numériques") :
    st.dataframe(df.describe())
    st.write("Ce descriptif nous a permis d'identifier rapidement certaines valeurs extrêmes, voire aberrantes.")  

  if st.checkbox("Afficher les boxplot des principales variables") :
    st.image(Image.open('Boxplot.png'))
    st.write("Les boxplot nous permettent de visualiser les nombreuses valeurs extrêmes de notre jeu de données.")
 
  st.write("### Pré-processing des jeux de données")
  st.markdown('''
              - Suppression de la dernière colonne qui est vide
              - Suppression des 12 premières lignes qui ne possèdent que des NaN 
              - Suppression des colonnes : TCO, TCH, nature, Ech. physiques, Stockage batterie, Déstockage batterie, Éolien terrestre, Éolien offshore, date - heure
              - Suppression de la colonne région, mais conservation de la colonne Code INSEE
              - Création de colonnes jour, mois, année et suppression de la colonne Date
              - Conversion des NAN en 0 dans les colonnes : nucléaire, hydraulique, éolien, solaire, bioénergie 
              - Groupby par jour afin de faire nos prédictions sur une journée
  ''')
  st.write("Nous avons choisi d'étudier les données au niveau national et au niveau régional. Pour cela, nous avons créé deux nouveaux jeux de données qui fusionnent des données de températures également téléchargées sur le site de l'ODRE : au niveau national (période 2013-2022) et régional (période 2016-2022).")

if page == pages[3] : 
  st.write("### Datavisualisation des données")
  st.write('##### Quel lien entre consommation et production d’énergie ?')
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=conso.index, y=conso['Consommation (MW)'], mode='lines', name='Consommation', line=dict(color='royalblue')))
  fig.add_trace(go.Scatter(x=conso.index, y=conso['Production (MW)'], mode='lines', name='Production', line=dict(color='firebrick')))
  fig.update_layout(title='Production vs Consommation d\'électricité en France',
                    xaxis_title='',
                    yaxis_title='Méga Watt par jour')
  st.plotly_chart(fig)
  st.write("Consommation et production d'énergie sont bien liées. Plus les habitants consomment, plus le pays produit de l'électricité. Ce graphique permet aussi d'appréhender la saisonnalité du couple production-consommation d'énergie.")
  st.write('##### Quelle est la répartition de la production d’électricité par région ?')
  if st.checkbox("Afficher la répartition en pourcentage") :
    st.image(Image.open('region.png'))
  st.image(Image.open('barplot.jpg'))
  st.write("La production d’énergie diffère selon les régions. Certaines régions produisent de façon quasi-exclusive de l'électricité à partir d’énergie renouvelable, tandis que d'autres ont une grosse production nucléaire.")
  st.write('##### Où se situent les risques de black-out ?')
  col1, col2, col3 = st.columns([1,4,10])
  with col1:
    st.write("")
  with col2:
    st.image(Image.open('carte.png'), width=600)
  with col3:
    st.write("")
  st.write("Les régions qui produisent seulement de l'électricité verte ne sont pas autonomes. Seule une importante production d’énergie nucléaire permet à des régions d'être indépendantes énergiquement parlant.")
  st.write('##### Quelle est la part d’énergies renouvelables dans la production française ?')
  st.image(Image.open('Renouvelables.png'))
  st.write("La part des énergies renouvelables dans la production d'électricité en France a augmenté de près de 5% entre 2013 et 2021, et cela au détriment de la production nucléaire.")
  if st.checkbox("Afficher la répartition de la production électrique") :
    st.write('##### Comment est répartie la production d’électricité en France ?')
    st.image(Image.open('Filieres.png'))
    st.write("Le nucléaire représente 72 % de la production d'énergie de notre pays sur la période étudiée. Viennent ensuite la production hydraulique et la production thermique des énergies fossiles. La production nucléaire baisse, tandis que celles des énergies renouvelables augmentent.")
  if st.checkbox("Afficher la répartition de la consommation par mois") :
    st.write('##### Les mois ont-ils une influence sur la consommation d’électricité ?')
    st.image(Image.open('Mois.png'))
    st.write("Les Français consomment plus en décembre, janvier, février, qui sont les mois d’hiver, en raison de l’utilisation du chauffage. A l’inverse, le mois d’août est le mois où la consommation est la plus basse. De plus, la production nationale d'électricité semble toujours supérieure à la consommation, donc peu de risque de black-out.")
  if st.checkbox("Afficher la répartition de la consommation par jours et par heures") :
    st.write('##### Les jours et les horaires ont-ils une influence sur la consommation d’électricité ?')
    st.image(Image.open('Jours.png'))
    st.write("La France consomme moins d'électricité les week-ends. Le dimanche est pour la majorité des Français un jour non travaillé et par conséquent de nombreuses entreprises sont à l'arrêt. Du côté des horaires,  les pics de consommation d’électricité se situent entre 11h et 13h, puis entre 19h et 20h.")
  st.write('##### Quel lien entre consommation d’énergie et température ?')
  temperature = dfn
  temperature['date'] = temperature['Année'].astype(str) +'-'+ temperature['Mois'].astype(str) +'-'+ temperature['Jour'].astype(str)
  temperature['date'] = pd.to_datetime(temperature['date'])
  temperature = temperature.groupby('date').agg({'Consommation (MW)' : 'sum', 'TMoy (°C)' : 'mean'})
  fig2 = px.scatter(x=temperature['Consommation (MW)'], y=temperature['TMoy (°C)'])
  fig2.update_layout(title='Température moyenne vs Consommation d\'électricité en France',
                   xaxis_title="Consommation d'électricité",
                   yaxis_title='Température moyenne')
  st.plotly_chart(fig2)
  st.write("Plus la température moyenne baisse et plus la corrélation entre température et consommation d'énergie est forte.")
  
  st.write("### Analyses statistiques")
  st.write('Des tests ont permis de confirmer certains liens entre les variables. Le test Anova a été choisi pour étudier une interaction entre une variable catégorielle et une variable continue. Voici les liens observés :')
  st.markdown('''
              - La production d'énergie et la région
              - La consommation d'énergie et le mois
              ''')
  st.write("Le test valide la corrélation si la p.value est inférieure à 0,05.")
  result_h1 = load("result_h1.py")
  result_h2 = load("result_h2.py")

  if st.checkbox("Afficher les résultats du test"):
    display = st.radio('Choix du résultat:',('La production d\'énergie et la région','La consommation et le mois'))   
    if display == 'La consommation et le mois':
      if  result_h1 < 0.05:
        st.write('p.value : ', result_h1)
        st.write('Hypothèse validée : La consommation et le mois sont corrélées')
      else :
        st.write('p.value : ', result_h1)
        st.write('Hypothèse non validée : La consommation et le mois ne sont pas corrélées')
    elif display == 'La production d\'énergie et la région':
      if result_h2 < 0.05:
        st.write('p.value : ', result_h2)
        st.write('Hypothèse validée : La production d\'énergie et la région sont corrélées')
      else :
        st.write('p.value : ', result_h2)
        st.write('Hypothèse non validée : La production d\'énergie et la région ne sont pas corrélées')
    st.write('Nous pourrons nous servir de cette corrélation dans la phase de modélisation, où nous développerons des modèles prédictifs pour anticiper la consommation d\'électricité en fonction des différentes variables explicatives.')    

if page == pages[4] : 
  st.write("### Modélisation nationale")
  st.write("Après nettoyage, voici le dataframe qui va nous servir à la modélisation au niveau national :")
  et = pd.read_csv('df_new.csv', sep = ',')
  st.dataframe(et.head(5))
  et['Saison']= et['Saison'].replace(to_replace = ['Hiver', 'Printemps', 'Eté', 'Automne'], value= ['0','1','2','3']).astype(int)
  et['Consommation (MW)'] = et['Consommation (MW)'].astype(int)
  if st.checkbox("Afficher les types des variables"):
    st.dataframe(et.dtypes)
  st.write('Nos données sont au bon format avant le découpage en train/test.')

  st.write("#### Des modèles de classification")
  st.write('La variable cible est encodée en classe interquartile.')

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

  rl1 = load("reg_logistique_un.py")
  knn1 = load("knn_un.py")
  rfc1 = load("foret_aleatoire_class_un.py")
  svc1 = load("support_vector_class_un.py")

  pred_rl1 = rl1.predict(X_test_et)
  pred_knn1 = knn1.predict(X_test_et)
  pred_rfc1 = rfc1.predict(X_test_et)
  pred_svc1 = svc1.predict(X_test_et)
      
  from sklearn.metrics import accuracy_score
  choice = st.selectbox(label = "Choix du modèle de classification", options=['Régression logistique','KNN','Forêt aléatoire classifier','SVC'])
  def scoresacc(choice):
      if choice == 'Régression logistique':
          pred_et = pred_rl1
      elif choice == 'KNN':
          pred_et = pred_knn1
      elif choice == 'Forêt aléatoire classifier':
          pred_et = pred_rfc1
      elif choice == 'SVC':
          pred_et = pred_svc1
      r2 = accuracy_score(y_test_et, pred_et)
      return r2
    
  def scoresconf(choice):
      if choice == 'Régression logistique': 
          conf = pred_rl1
      elif choice == 'KNN':
          conf= pred_knn1
      elif choice == 'Forêt aléatoire classifier':
          conf = pred_rfc1
      elif choice == 'SVC':
          conf = pred_svc1
      r3 = confusion_matrix(y_test_et, conf)
      return r3
    
  display= st.radio('Choix du résultat', ('Accuracy','Confusion Matrix'))   
  if display == 'Accuracy':
      st.write(scoresacc(choice))
  elif display == 'Confusion Matrix':
      st.dataframe(scoresconf(choice))

  st.write("Les résultats des modèles de classification sont corrects, mais ce ne sont pas les modèles les plus adaptés à notre problématique. Pour cette étude, il est plus judicieux de se concentrer sur l'approche de régression ci-dessous :")

  st.write("#### Des modèles de régression")
  pb = pd.read_csv('DataFrame.csv', sep = ',')
  pb_un = pb
  y_pb = pb_un['Consommation (MW)']
  X_pb = pb_un.drop('Consommation (MW)', axis = 1)
  X_train_pb, X_test_pb, y_train_pb, y_test_pb = train_test_split(X_pb, y_pb, test_size=0.25, random_state = 42)
  cols = X_pb.columns
  sc_pb = StandardScaler()
  X_train_pb.loc[:,cols] = sc_pb.fit_transform(X_train_pb[cols])
  X_test_pb.loc[:,cols] = sc_pb.transform(X_test_pb[cols])

  lr_pb = load('model_reg_line.py')
  dtr_pb = load('model_reg_dtr.py')
  forest_pb = load('model_reg_forest.py')

  pred_l_pb = lr_pb.predict(X_test_pb)
  pred_a_pb = dtr_pb.predict(X_test_pb)
  pred_f_pb = forest_pb.predict(X_test_pb)

  pred_l_train_pb = lr_pb.predict(X_train_pb)
  pred_a_train_pb = dtr_pb.predict(X_train_pb)
  pred_f_train_pb = forest_pb.predict(X_train_pb)

  choix_modele = st.selectbox(label = 'Trois modèles de régression ont été entraînés :', options = ['Régression linéaire', 'Arbre de décision', 'Forêt aléatoire'])
  def train_modele(choix_modele):
      if choix_modele == 'Régression linéaire':
          q = lr_pb
      elif choix_modele == 'Arbre de décision':
          q = dtr_pb
      elif choix_modele == 'Forêt aléatoire':
          q = forest_pb
      s2 = q.score(X_test_pb, y_test_pb)
      return s2
  st.write('Coefficient de détermination', train_modele(choix_modele))

  st.write('##### Le choix du modèle le plus performant')
  st.write("Nous nous sommes concentrés sur le modèle le plus performant au vu de l'ensemble des métriques :")  
  # st.image(Image.open('image_recap.png'))  

  lineaire_pb = ['linéaire test', mean_absolute_error(y_test_pb, pred_l_pb), mean_squared_error(y_test_pb, pred_l_pb), np.sqrt(mean_squared_error(y_test_pb, pred_l_pb)), lr_pb.score(X_test_pb, y_test_pb)]
  arbre_pb = ['arbre test', mean_absolute_error(y_test_pb, pred_a_pb), mean_squared_error(y_test_pb, pred_a_pb), np.sqrt(mean_squared_error(y_test_pb, pred_a_pb)), dtr_pb.score(X_test_pb,y_test_pb)]
  foret_pb = ['forêt test', mean_absolute_error(y_test_pb, pred_f_pb), mean_squared_error(y_test_pb, pred_f_pb), np.sqrt(mean_squared_error(y_test_pb, pred_f_pb)), forest_pb.score(X_test_pb,y_test_pb)]
  lineaire_t_pb = ['linéaire train', mean_absolute_error(y_train_pb, pred_l_train_pb), mean_squared_error(y_train_pb, pred_l_train_pb), np.sqrt(mean_squared_error(y_train_pb, pred_l_train_pb)), lr_pb.score(X_train_pb, y_train_pb)]
  arbre_t_pb = ['arbre train', mean_absolute_error(y_train_pb, pred_a_train_pb), mean_squared_error(y_train_pb, pred_a_train_pb), np.sqrt(mean_squared_error(y_train_pb, pred_a_train_pb)), dtr_pb.score(X_train_pb, y_train_pb)]
  foret_t_pb = ['forêt train', mean_absolute_error(y_train_pb, pred_f_train_pb), mean_squared_error(y_train_pb, pred_f_train_pb), np.sqrt(mean_squared_error(y_train_pb, pred_f_train_pb)), forest_pb.score(X_train_pb, y_train_pb)]
  tableau = [lineaire_pb,lineaire_t_pb,arbre_pb,arbre_t_pb,foret_pb,foret_t_pb]
  dataF_pb = pd.DataFrame(data=tableau, columns=['modèle','mae','mse','rmse','r²'])
  st.dataframe(dataF_pb)

  coef = 0.9573220875096127

  st.write("Pour résoudre le problème de sur-apprentissage de l'arbre de décision. Nous avons testé de façon empirique différentes profondeurs d'arbre. Nous obtenons un R² de", coef ," avec un max_depth de 6, mais toujours inférieur à ceux obtenus avec la forêt aléatoire.")

  st.write('##### Le choix de la forêt aléatoire')

  st.write('C\'est le random forest regressor qui présente les meilleurs résultats :')
  st.image(Image.open('Resid_Fal.png'))  

  if st.checkbox("Voir les graphiques de la linear regression"):
      st.image(Image.open('Resid_lr.png'))

  if st.checkbox("Voir les graphiques du decision tree regressor"):
      st.image(Image.open('Resid_dtr.png'))

if page == pages[5] : 
  st.write("### Modélisation régionale")
  st.write("Après nettoyage, voici le dataframe qui va nous servir à la modélisation au niveau régional :")
  st.dataframe(dfn.head(5))

  st.write("Pour la préparation du jeu de données pour le machine learning, une standardisation a suffit, les données étant toutes numériques :")
  st.code('''
  target = df['Consommation (MW)']
  feats = df.drop('Consommation (MW)', axis = 1)
          
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)
          
  cols = feats.columns
  sc = StandardScaler()
  X_train.loc[:,cols] = sc.fit_transform(X_train[cols])
  X_test.loc[:,cols] = sc.transform(X_test[cols])
          ''', language='python')
  
  lr = load('lr.py') 
  pred_l = lr.predict(X_test)
  pred_l_train = lr.predict(X_train)

  dtr = load('dtr.py') 
  pred_a = dtr.predict(X_test)
  pred_a_train = dtr.predict(X_train)

  forest = load('forest.py') 
  pred_f = forest.predict(X_test)
  pred_f_train = forest.predict(X_train)

  st.write("#### Des modèles de régression")
  st.write("Notre variable cible étant de type quantitative, nous avons fait le choix de la régression :")

  if st.checkbox("Voir les résultats du modèle de régression linéaire") :
    st.code('''
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_l = lr.predict(X_test)
    pred_l_train = lr.predict(X_train)
            ''', language='python')
    st.image(Image.open('lr.png'))

  if st.checkbox("Voir les résultats du modèle d'arbre de décision") :
    st.code('''
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X_train, y_train)
    pred_a = dtr.predict(X_test)
    pred_a_train = dtr.predict(X_train)
            ''', language='python')
    st.image(Image.open('dtr.png'))  

  if st.checkbox("Voir les résultats du modèle de forêt aléatoire") :
    st.code('''
    forest = RandomForestRegressor(random_state=42)
    forest.fit(X_train, y_train)
    pred_f = forest.predict(X_test)
    pred_f_train = forest.predict(X_train)
            ''', language='python')
    st.image(Image.open('forest.png'))    

  st.write("#### Le choix du modèle le plus performant")
  st.write("Nous avons testé ces trois modèles sur différents nombres de colonnes, d'abord de façon aléatoire, puis en utilisant les Shap et les feature importances, mais en conservant l'ensemble de nos données, nous obtenons de meilleurs résultats.")
  if st.checkbox("Voir le Shap sur notre modèle de forêt aléatoire") :
    st.image(Image.open('shap.png')) 
  if st.checkbox("Voir la compilation des R² par modèle") :
    st.image(Image.open('detail.jpg')) 
  st.write("Nous nous sommes concentrés sur le modèle le plus performant au vu de l'ensemble des métriques :")

  lineaire = ['linéaire test', mean_absolute_error(y_test, pred_l), mean_squared_error(y_test, pred_l), np.sqrt(mean_squared_error(y_test, pred_l)), lr.score(X_test, y_test)]
  arbre = ['arbre test', mean_absolute_error(y_test, pred_a), mean_squared_error(y_test, pred_a), np.sqrt(mean_squared_error(y_test, pred_a)), dtr.score(X_test,y_test)]
  foret = ['forêt test', mean_absolute_error(y_test, pred_f), mean_squared_error(y_test, pred_f), np.sqrt(mean_squared_error(y_test, pred_f)), forest.score(X_test,y_test)]
  lineaire_t = ['linéaire train', mean_absolute_error(y_train, pred_l_train), mean_squared_error(y_train, pred_l_train), np.sqrt(mean_squared_error(y_train, pred_l_train)), lr.score(X_train, y_train)]
  arbre_t = ['arbre train', mean_absolute_error(y_train, pred_a_train), mean_squared_error(y_train, pred_a_train), np.sqrt(mean_squared_error(y_train, pred_a_train)), dtr.score(X_train, y_train)]
  foret_t = ['forêt train', mean_absolute_error(y_train, pred_f_train), mean_squared_error(y_train, pred_f_train), np.sqrt(mean_squared_error(y_train, pred_f_train)), forest.score(X_train, y_train)]
  tableau = [lineaire,lineaire_t,arbre,arbre_t,foret,foret_t]
  dataF = pd.DataFrame(data=tableau, columns=['modèle','mae','mse','rmse','r²'])
  st.dataframe(dataF)

  st.write("Pour résoudre le problème de sur-apprentissage de l'arbre de décision, nous avons testé de façon empirique différentes profondeurs d'arbre. A 15, nous obtenions un résultat intéressant mais toujours inférieur à ceux obtenus avec la forêt aléatoire.")

  st.write("#### L'optimisation de la forêt aléatoire")
  if st.checkbox("Voir les résultats de l'hyperparamétrage du random forest regressor avec la méthode Grid Search") :
    st.code('''
            'max_depth': 30,   # Profondeur maximale des arbres
            'max_features': 'log2',   # Nombre maximum de caractéristiques à considérer pour le fractionnement
            'min_samples_leaf': 1,   # Nombre minimum d'échantillons requis dans une feuille
            'min_samples_split': 2,   # Nombre minimum d'échantillons requis pour diviser un nœud
            'n_estimators': 413   # Nombre d'arbres dans la forêt
            ''', language='python')
  st.write("Voici les résultats de son entraînement :")
  st.code('''
  forest_op = RandomForestRegressor(n_estimators=413, max_depth=30, min_samples_split=2, min_samples_leaf=1, max_features='log2', random_state=42)
  forest_op.fit(X_train, y_train)
  pred_fo = forest_op.predict(X_test)
  pred_fo_train = forest_op.predict(X_train)
          ''', language='python')
  forest_op = load('forest_op.py') 
  pred_fo = forest_op.predict(X_test)
  pred_fo_train = forest_op.predict(X_train)
  st.image(Image.open('forest_op.png'))      

  st.write('Voici les métriques du modèle de forêt aléatoire optimisée :')
  fo = ['forêt optimisée test', mean_absolute_error(y_test, pred_fo), mean_squared_error(y_test, pred_fo), np.sqrt(mean_squared_error(y_test, pred_fo)), forest_op.score(X_test,y_test)]
  fo_t = ['forêt optimisée train', mean_absolute_error(y_train, pred_fo_train), mean_squared_error(y_train, pred_fo_train), np.sqrt(mean_squared_error(y_train, pred_fo_train)), forest_op.score(X_train, y_train)]
  tableau_fo = [fo,fo_t]
  dataFO = pd.DataFrame(data=tableau_fo, columns=['modèle','mae','mse','rmse','r²'])  
  st.dataframe(dataFO)

  st.write("#### Le Gradient Boosting Regressor")
  st.write("Nous avons également testé un Gradient Boosting Regressor. En manipulant les paramètres de façon assez empirique, nous obtenons rapidement d'excellents résultats. Le principal problème de ce modèle est sa lenteur.")
  st.code('''
  gbr = GradientBoostingRegressor(max_depth=9, random_state=42, n_estimators=1400, learning_rate=0.05)
  gbr.fit(X_train, y_train)
  pred_g = gbr.predict(X_test)
  pred_g_train = gbr.predict(X_train)
          ''', language='python')
  gbr = load('gbr.py') 
  pred_g = gbr.predict(X_test)
  pred_g_train = gbr.predict(X_train)
  st.image(Image.open('gbr.png'))

  st.write('Voici les métriques du Gradient Boosting Regressor :')
  gradboost = ['gradboost test', mean_absolute_error(y_test, pred_g), mean_squared_error(y_test, pred_g), np.sqrt(mean_squared_error(y_test, pred_g)), gbr.score(X_test,y_test)]
  gradboost_t = ['gradboost train', mean_absolute_error(y_train, pred_g_train), mean_squared_error(y_train, pred_g_train), np.sqrt(mean_squared_error(y_train, pred_g_train)), gbr.score(X_train, y_train)]
  tableauG = [gradboost,gradboost_t]
  dataG = pd.DataFrame(data=tableauG, columns=['modèle','mae','mse','rmse','r²'])  
  st.dataframe(dataG)

  st.write("#### Un modèle sans machine learning")
  st.write("En calculant simplement les moyennes des consommations journalières par région, il est possible d'obtenir des résultats proches du machine learning. En réalisant ces moyennes sur 10 ans, nous avons testé ce modèle sur le même jeu de test que précédemment et voici le résultat :")
  if st.checkbox("Voir les résultats du modèle sans machine learning") :
    df_nml = dfm
    dfi = df_nml.groupby(['Code INSEE région', 'Mois', 'Jour'])['Consommation (MW)'].mean().reset_index()
    df_merged = pd.merge(df_nml, dfi, on=['Code INSEE région', 'Mois', 'Jour'], suffixes=('_original', '_mean'))
    st.write('Coefficient de détermination du modèle :', r2_score(df_merged['Consommation (MW)_original'], df_merged['Consommation (MW)_mean']))
    st.write('Erreurs moyennes absolues du modèle :', mean_absolute_error(df_merged['Consommation (MW)_original'], df_merged['Consommation (MW)_mean']))

if page == pages[6] : 
  st.write("### Prédiction pour 2023")  
  st.write("Il est possible de tester les deux meilleurs modèles sur un autre jeu de données afin de voir s’ils s’en sortent avec une source de données différente. Un dataset de test a été créé avec les jeux de données éCO2mix 2023. Le dataset de train étant le jeu de données d'origine.")

  forest_pred = load('forest_pred.py') 
  pred_fp = forest_pred.predict(X_test_p)

  gbr_pred = load('gbr_pred.py') 
  pred_gbrp = gbr_pred.predict(X_test_p)

  dfi = dfm.groupby(['Code INSEE région', 'Mois', 'Jour']).agg({'Consommation (MW)' : 'mean'}).reset_index()
  df_final = pd.merge(dfp, dfi, on=['Code INSEE région', 'Mois', 'Jour'], suffixes=('', '_mean'))
  df_final['Prédiction forêt optimisée'] = pred_fp
  df_final['Prédiction forêt optimisée'] = df_final['Prédiction forêt optimisée'].round()
  df_final['Prédiction gradient boosting'] = pred_gbrp
  df_final['Prédiction gradient boosting'] = df_final['Prédiction gradient boosting'].round()

  def predictions (option_region, option_jour, option_mois) :
    region = {'Île-de-France' : 11, 'Centre-Val de Loire' : 24, 'Bourgogne-Franche-Comté' : 27, 'Normandie' : 28, 'Hauts-de-France' : 32, 'Grand Est' : 44, 'Pays de la Loire' : 52, 'Bretagne' : 53, 'Nouvelle-Aquitaine' : 75, 'Occitanie' : 76, 'Auvergne-Rhône-Alpes' : 84, 'Provence-Alpes-Côte d’Azur' : 93}
    mois = {'janvier' : 1, 'février' : 2, 'mars' : 3, 'avril' : 4, 'mai' : 5, 'juin' : 6, 'juillet' : 7, 'août' : 8, 'septembre' : 9}
    res = df_final.loc[(df_final['Code INSEE région']==region[option_region]) & (df_final['Jour']==option_jour) & (df_final['Mois']==mois[option_mois]) & (df_final['Année']==2023),['Consommation (MW)', 'Prédiction forêt optimisée', 'Prédiction gradient boosting']]
    return res

  st.write('Effectuer une prédiction 2023 (de janvier à septembre) :')
  option_jour = st.slider('Choisissez un jour', 1, 31)
  choix_mois = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre']
  option_mois = st.selectbox('Choisissez un mois', choix_mois)
  choix_region = ['Île-de-France', 'Centre-Val de Loire', 'Bourgogne-Franche-Comté', 'Normandie', 'Hauts-de-France', 'Grand Est', 'Pays de la Loire', 'Bretagne', 'Nouvelle-Aquitaine', 'Occitanie', 'Auvergne-Rhône-Alpes', 'Provence-Alpes-Côte d’Azur']
  option_region = st.selectbox('Choisissez une région', choix_region)
  st.write('Vous avez choisi :', option_jour, " ", option_mois, " 2023 - ", option_region)
  
  if st.button('Voir les prédictions 2023'):
    st.write(predictions (option_region, option_jour, option_mois))

  if st.button('Voir les coefficients de détermination'):
    st.write('R² de la forêt optimisée :', forest_pred.score(X_test_p, y_test_p))
    st.write('R² du gradient boosting :', gbr_pred.score(X_test_p, y_test_p))

if page == pages[7] : 
  st.write("### Regard critique et perspectives")  
  st.write("#### Atteinte des objectifs")
  st.write("Des modèles de prévision de la consommation énergétique ont été développés tant au niveau national que régional, avec une optimisation de certains de ces modèles. Les prévisions pour 2022 et 2023 ont été réalisées avec succès, démontrant ainsi la robustesse de la modélisation. L'objectif principal a été atteint.")
  st.write("#### Piste d'amélioration pour augmenter les performances")
  st.markdown('''
    - l’optimisation d'autres modèles
    - l'ajout de données supplémentaires : jours de la semaine, données de population…
    - la réalisation de prévisions sur l'avenir
    - la création de modèles régionaux
    - un travail plus approfondi du modèle sans machine Learning
  ''')
  st.write("#### Conclusion")
  st.write("Le projet a contribué à démontrer comment l'utilisation de données climatiques peut améliorer la précision des prévisions de consommation énergétique. Il a également montré l'impact de différentes approches de modélisation sur la qualité des prédictions. De nombreux axes d'amélioration auraient pu être exploités avec un délai supplémentaire. Ces pistes non explorées ou partiellement exploitées constituent des axes de développement futurs prometteurs pour renforcer encore davantage la qualité de nos modèles et leurs applications potentielles dans le secteur énergétique.")