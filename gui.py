
import PySimpleGUI as sg
import os.path

import pandas as pd
import numpy as np
import csv
import random
import joblib

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.metrics import accuracy_score

# First the window layout in 2 columns

left_column = [
    [
        sg.Text("CSV Files"),
        sg.In(size=(20, 1), enable_events=True, key="-csvfile-"),
        sg.FileBrowse(file_types=(("CSV Files", "*.csv"),))
    ],

]

# For now will only show result
right_column = [
    [sg.Text("Hasil Klasifikasi:")],
    [sg.Text(size=(80, 20), key="-hasil-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(left_column),
        sg.VSeperator(),
        sg.Column(right_column),
    ]
]

window = sg.Window("Hitung Hasil Klasifikasi", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # File name was filled in
    if event == "-csvfile-":
        csv = values["-csvfile-"]
        try:
            # Task
            df = pd.read_csv(csv)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, 6].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50)
            gnb = GaussianNB()

            y_pred = gnb.fit(X_train, y_train).predict(X_test)
            akurasi = metrics.accuracy_score(y_test, y_pred)
            conf_matriks = confusion_matrix(y_test,y_pred)
            laporan = classification_report(y_test,y_pred)

            # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
            # print(confusion_matrix(y_test,y_pred))
            # print(classification_report(y_test,y_pred))

        except:
            file_list = []

        # window["-hasil-"].update(df.head())
        window["-hasil-"].update(laporan)


window.close()
