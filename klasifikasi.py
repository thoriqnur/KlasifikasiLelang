import pandas as pd
import numpy as np
import csv
import random
import joblib
from sklearn.model_selection import train_test_split
df = pd.read_csv("datasetmotor.csv")
# print(df.Kunci)
print(df.head())
X = df.iloc[:, :-1].values
y = df.iloc[:, 6].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix

# model.fit(X_train, y_train)

# predicted = model.predict()

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))