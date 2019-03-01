# -*-coding：utf-8-*-
import pandas as pd
wine = pd.read_csv('data/wine.csv', names=["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])
wine.head()
wine.describe().transpose()
print(wine.shape)
X = wine.drop('Cultivator', axis=1)
y = wine['Cultivator']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(13, 13, 13), max_iter=500)
mlp.fit(X_train, y_train)
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
len(mlp.coefs_)
len(mlp.coefs_[0])
len(mlp.intercepts_[0])
