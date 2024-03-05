# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:36:44 2024

@author: Eliot



"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request 
from sklearn.datasets import fetch_openml
import numpy as np
import requests
def predict(data):
    titanic_data = fetch_openml(name='titanic', version=1, as_frame=True)
    X = titanic_data['data']
    y = titanic_data['target']
    X = X.drop(['name', 'ticket', 'cabin', 'embarked', 'home.dest',"boat"], axis=1)
    label_encoder = LabelEncoder()
    X['sex'] = label_encoder.fit_transform(X['sex'])
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    #print("Exactitude (Accuracy) : {:.2f}%".format(accuracy * 100))
    return model.predict(data)
    
    



app = Flask(__name__)

@app.route('/predict/', methods=['get'])
def hello_world():
    #http://127.0.0.1:5000//predict?pclass=1&sex=0&age=25&sibsp=1&parch=0&fare=50&body=10
    pclass = int(request.args.get('pclass'))
    sex = int(request.args.get('sex'))
    age = float(request.args.get('age'))
    sibsp = int(request.args.get('sibsp'))
    parch = int(request.args.get('parch'))
    fare = float(request.args.get('fare'))
    body = int(request.args.get('body'))
    acc = predict(np.array([[pclass, sex, age, sibsp, parch, fare, body]]))
    print(acc[0])
    data = {'predict': str(acc[0])}
    print(data)
    return jsonify(data)
    
@app.route('/consensus/', methods=['get'])
def concensus():
    response_eliot = requests.get("http://127.0.0.1:5000//predict?pclass=1&sex=0&age=25&sibsp=1&parch=0&fare=50&body=10")
    data_eliot = response_eliot.json()
    predict_eliot = data_eliot["predict"]
    
    response_cecilia = requests.get("https://af06-2a01-e34-ec63-3510-c5b0-a263-d931-75a5.ngrok-free.app/predict_survival?pclass=1&sex=0&age=25&sibsp=1&parch=0&fare=50&embarked=1&class=2&who=1&adult_male=0&deck=3&embark_town=1&alive=1&alone=0")
    data_cecilia = response_cecilia.json()
    predict_cecilia = data_cecilia["prediction"]
    
    mean = np.mean([int(predict_cecilia),int(predict_eliot)])
    data = {'mean': str(mean)}
    return jsonify(data)
    
    
    
    
app.run(host="0.0.0.0")