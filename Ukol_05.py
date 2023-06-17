"""
Fyzická kondice
Stáhni si data ze souboru bodyPerformance.csv o fyzické kondici, která byla jihokorejskou organizací Korea Sports Promotion Foundation. Data obsahují různé vstupní hodnoty a výstupní hodnotu, kterou je zařazení jedince do jedné ze čtyř výkonnostních tříd.

age = věk (20 až 64 let),
gender = pohlaví (F, M),
height_cm = výška v cm,
weight_kg = hmotnost v kg,
body_fat_% = tělesný tuk v procentech,
distolic = diastolický krevní tlak (min),
systolic = systolický krevní tlak (min),
gripForce = síla stisku,
sit and bend forward_cm = sed a předklon v cm,
sit-ups counts = počty sedů-lehů,
broad jump_cm = skok do dálky v cm,
class = třída fyzické výkonnosti (4 třídy, A je nejlepší a D nejhorší).
Uvažuj, že chceš přijímat lidi do organizace, která vyžaduje vysokou fyzickou výkonnost. Tvou snahou je zkrátit a zefektivnit přijímací proces. Zkus tedy zjistit, nakolik přesné je zařazení jedinců do výkonnostních tříd bez nutnosti měření jejich výkoknu při vykonání jednotlivých cviků. Využij tedy všechny vstupní proměnné s výjimkou sit and bend forward_cm, sit-ups counts a broad jump_cm.

K rozřazení jedinců do skupin využij rozhodovací strom a jeden ze zbývajících dvou algoritmů probíraných na lekcích (tj. K Nearest Neighbours nebo Support Vector Machine). Rozhodovacímu stromu omez maximální počet pater na 5 a poté si zobraz graficky a vlož ho do Jupyter notebooku nebo jako obrázek ve formátu PNG jako součást řešení.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image  
import pydotplus
from pydotplus import graph_from_dot_data
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import os
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

#načtení dat
data = pd.read_csv('bodyPerformance.csv')
print(data.head())

# stanovení vysvětlované (závislé) proměnné
y = data['class']

# rozdělení dat na kategoriální a numerická
categorical_columns = ['gender']
numeric_columns = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic', 'gripForce']

# úprava kategoriálních hodnot převodem textových hodnot na číselné hodnoty pomocí binárního vektoru.
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()
print(encoded_columns)

# převedení numerických dat na pole
numeric_data = data[numeric_columns].to_numpy()

#sloučení převedených dat
X = np.concatenate([encoded_columns, numeric_data], axis=1)


#rozdělení dat na testovací a tréninková
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#vytvoření rozhodovacího stromu
clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#vygenerování obrázku stromu v png
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=["A", "B", "C", "D"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree_1.png')

# Vytvoř matici záměn pro rozhodovací strom. Kolik jedinců s nejvyšší fyzickou výkonností (tj. ze skupiny A) bylo klasifikování správně? Kolik pak bylo zařazeno do skupin B, C a D? Uveď výsledky do komentáře v programu nebo do buňky v Jupyter notebooku.

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title('Matice záměn')
plt.show()
# 605 jedinců s nejvyšší fyzickou výkonností (tj. ze slupiny A) bylo klasifikováno správně, 64 bylo chybně zařazeno do B, 322 do C a 49 do D.

#Urči metriku accuracy pro rozhodovací strom a pro jeden ze dvou vybraných algoritmů. Který algoritmus si vedl lépe? Odpověď napiš do komentáře. Níže pro strom.

print(accuracy_score(y_test, y_pred))

# Hodnota metriky accuracy je 44,89% (0.4489795918367347) pro daný strom.


#Algoritmus K Nearest Neighbors

#znovu provedeme úpravu dat, abychom pracovali s daty bez omezení úrovně

#normalizace hodnot dat numerických sloupců
scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])

# úprava kategoriálních hodnot převodem textových hodnot na číselné hodnoty pomocí binárního vektoru.
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

#sloučení dat
X = np.concatenate([encoded_columns, numeric_data], axis=1)

#rozdělení dat na testovací a tréninková
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#matice záměn
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()

# 599 jedinců s nejvyšší fyzickou výkonností (tj. ze slupiny A) bylo klasifikováno správně, 280 bylo chybně zařazeno do B, 123 do C a 38 do D.

# výpočet metriky accuracy 

print(accuracy_score(y_test, y_pred))

# Hodnota metriky accuracy je 41,04% (0.41040318566450973) pro algoritmus K Nearest Neighbors. Strom je tedy přesnější.

#POČÍTÁNO POUZE PRO JEDNU VYSVĚTLUJÍCÍ PROMĚNNOU - 'broad jump_cm'
#Nyní uvažuj, že se rozhodneš testovat jedince pomocí jednoho ze cviků. Vyber cvik, který dle tebe nejvíce vypovídá o fyzické výkonnosti jedince. Porovnej, o kolik se zvýšila hodnota metriky accuracy pro oba algoritmy.

#načtení dat
data = pd.read_csv('bodyPerformance.csv')
print(data.head())

# stanovení vysvětlované (závislé) proměnné
y = data['class']

# výběr cviku, který nejvíce vypovídá o fyzické kondici
selected_columns = 'broad jump_cm'

# stanovení nezávislé proměnné
X = data[[selected_columns]].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)


#rozdělení dat na testovací a tréninková
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vytvoření rozhodovacího stromu

clf_tree = DecisionTreeClassifier(max_depth=5)
clf_tree.fit(X_train, y_train)
y_pred_tree = clf_tree.predict(X_test)

#vygenerování obrázku stromu v png
dot_data = StringIO()
export_graphviz(clf_tree, out_file=dot_data, filled=True, feature_names=[selected_columns], class_names=["A", "B", "C", "D"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree_2.png')

#Výpočet hodnoty metriky accuracy pro strom
print(accuracy_score(y_test, y_pred_tree))

# Hodnota metriky accuracy pro vysvětlující proměnnou 'broad jump_cm' je 35,02% (0.3501742160278746) pro daný strom, tedy nižší o 9% než při použití původních vysvětlujících proměnných.

# Algoritmus KNN

clf_knn = KNeighborsClassifier()
clf_knn.fit(X_train, y_train)
y_pred_knn = clf_knn.predict(X_test)

#matice záměn
ConfusionMatrixDisplay.from_estimator(clf_knn, X_test, y_test)
plt.show()

## 427 jedinců s nejvyšší fyzickou výkonností (tj. ze slupiny A) bylo podle vysvětlující proměnné 'broad jump_cm' klasifikováno správně, 306 bylo chybně zařazeno do B, 147 do C a 160 do D.

#Výpočet hodnoty metriky accuracy pro strom
print(accuracy_score(y_test, y_pred_knn))

# Hodnota metriky accuracy pro vysvětlující proměnnou 'broad jump_cm' je 29,94% (0.2994026879044301) pro KNN, tedy nižší o 11% než při použití původních vysvětlujících proměnných.

#POČÍTÁNO PRO PŮVODNÍ ZADÁNÍ + JEDNA DALŠÍ PROMĚNNÁ 'sit-ups counts'


#načtení dat
data = pd.read_csv('bodyPerformance.csv')
print(data.head())

# stanovení vysvětlované (závislé) proměnné
y = data['class']

# rozdělení dat na kategoriální a numerická
categorical_columns = ['gender']
numeric_columns = ['age', 'height_cm', 'weight_kg', 'body fat_%', 'diastolic', 'systolic', 'gripForce', 'sit-ups counts' ]

# úprava kategoriálních hodnot převodem textových hodnot na číselné hodnoty pomocí binárního vektoru.
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()
print(encoded_columns)

# převedení numerických dat na pole
numeric_data = data[numeric_columns].to_numpy()

#sloučení převedených dat
X = np.concatenate([encoded_columns, numeric_data], axis=1)


#rozdělení dat na testovací a tréninková
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#vytvoření rozhodovacího stromu
clf = DecisionTreeClassifier(max_depth=5)
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#vygenerování obrázku stromu v png
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, feature_names=list(encoder.get_feature_names_out()) + numeric_columns, class_names=["A", "B", "C", "D"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree_3.png')

# Vytvoř matici záměn pro rozhodovací strom. Kolik jedinců s nejvyšší fyzickou výkonností (tj. ze skupiny A) bylo klasifikování správně? Kolik pak bylo zařazeno do skupin B, C a D? Uveď výsledky do komentáře v programu nebo do buňky v Jupyter notebooku.

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.title('Matice záměn')
plt.show()
# 738 jedinců s nejvyšší fyzickou výkonností (tj. ze slupiny A) bylo klasifikováno správně, 243 bylo chybně zařazeno do B, 36 do C a 23 do D.

#Urči metriku accuracy pro rozhodovací strom a pro jeden ze dvou vybraných algoritmů. Který algoritmus si vedl lépe? Odpověď napiš do komentáře. Níže pro strom.

print(accuracy_score(y_test, y_pred))

# Hodnota metriky accuracy je 50,27 (0.5027376804380289) pro daný strom, tedy o 5,38% vyšší než bez proměnné 'sit-ups counts.


#Algoritmus K Nearest Neighbors

#znovu provedeme úpravu dat, abychom pracovali s daty bez omezení úrovně

#normalizace hodnot dat numerických sloupců
scaler = StandardScaler()
numeric_data = scaler.fit_transform(data[numeric_columns])

# úprava kategoriálních hodnot převodem textových hodnot na číselné hodnoty pomocí binárního vektoru.
encoder = OneHotEncoder()
encoded_columns = encoder.fit_transform(data[categorical_columns])
encoded_columns = encoded_columns.toarray()

#sloučení dat
X = np.concatenate([encoded_columns, numeric_data], axis=1)

#rozdělení dat na testovací a tréninková
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

#matice záměn
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()

# 722 jedinců s nejvyšší fyzickou výkonností (tj. ze slupiny A) bylo klasifikováno správně, 249 bylo chybně zařazeno do B, 60 do C a 9 do D.

# výpočet metriky accuracy 

print(accuracy_score(y_test, y_pred))

# Hodnota metriky accuracy je 51,64% (0.5164260826281732) pro algoritmus K Nearest Neighbors a tedy o 10,6% přesnější než bez proměnné 'sit-ups counts. Strom je tedy v tomto případě méně přesný než KNN.