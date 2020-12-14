import pandas as pd
import numpy as np
import csv
import random

df = pd.read_csv("datasetmot.csv")
df.head()

from sklearn.model_selection import train_test_split
df

X = df.iloc[:, :-1].values
y = df.iloc[:, 6].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

# y_pred = gnb.fit(X_train, y_train).predict(X_test)
# from sklearn.metrics import classification_report,confusion_matrix