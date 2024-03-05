#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:38:24 2024

@author: djoudi
"""

import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


titanic_data = fetch_openml(name='titanic', version=1, as_frame=True)
X = titanic_data['data']
y = titanic_data['target']
X.head()

from flask import Flask, request
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

# Mettez en place votre modèle KNN
titanic_data = fetch_openml(name='titanic', version=1, as_frame=True)
X = titanic_data['data']
y = titanic_data['target']
X = X.drop(['name', 'ticket', 'cabin', 'embarked', 'home.dest', 'boat'], axis=1)
label_encoder = LabelEncoder()
X['sex'] = label_encoder.fit_transform(X['sex'])
X = X.fillna(X.median())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

@app.route('/')
def hello_world():
    texte = 'Ceci est la page principale<br>'
    texte += '<br>'
    texte += 'Pour une prédiction de type KNN veuillez entrer sous la forme par exemple :<br>'
    texte += 'http://127.0.0.1:5000//predict_survival?pclass=1&sex=0&age=25&sibsp=1&parch=0&fare=50&embarked=1&class=2&who=1&adult_male=0&deck=3&embark_town=1&alive=1&alone=0<br>'
    texte += "Le passager de l'exemple précédent est un non survivant par exemple et le suivant (lien ci-dessous est un survivant :<br>"
    texte += "http://127.0.0.1:5000//predict_survival?pclass=1&sex=1&age=38&sibsp=1&parch=0&fare=71.2833&embarked=1&class=1&who=1&adult_male=1&deck=4&embark_town=1&alive=0&alone=1<br>"
    texte += '<br>'
    texte += 'Vous pouvez changé les paramètres qui suivent les ='
    return texte

@app.route('/predict_survival', methods=['GET'])
def predict_survival():
    pclass = int(request.args.get('pclass'))
    sex = int(request.args.get('sex'))
    age = float(request.args.get('age'))
    sibsp = int(request.args.get('sibsp'))
    parch = int(request.args.get('parch'))
    fare = float(request.args.get('fare'))
    embarked = float(request.args.get('embarked'))
    classe = float(request.args.get('class'))
    who = float(request.args.get('who'))
    adult_male = float(request.args.get('adult_male'))
    deck = float(request.args.get('deck'))
    embark_town = float(request.args.get('embark_town'))
    alive = float(request.args.get('alive'))
    alone = float(request.args.get('alone'))

    input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, classe, who, adult_male, deck, embark_town, alive, alone]])

    prediction = knn_model.predict(input_data)
    
    resultat = 'non survivant'
    
    if prediction == 1:
        resultat = 'survivant'
    
    texte = 'Le passager est un ' + resultat

    return texte

app.run(host="0.0.0.0")
