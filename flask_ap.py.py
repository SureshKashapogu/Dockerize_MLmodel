# -*- coding: utf-8 -*-
"""
Created on Sat May 20 15:28:31 2023

@author: sures
"""

from flask import Flask, request
import numpy as np
import pandas as pd
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open('weight_model.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return 'Welcome All'

@app.route('/predict', methods=['GET'])
def predict_obesity_classification():
    
    """Let's find out Obesity Prediction
    ---
    parameters:
        - name: Age
          in: query
          type: number
          required: true
        - name: Gender
          in: query
          type: number
          required: true
        - name: Height
          in: query
          type: number
          required: true
        - name: Weight
          in: query
          type: number
          required: true
        - name: BMI
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    """
    Age = request.args.get('Age')
    Gender = request.args.get('Gender')
    Height = request.args.get('Height')
    Weight = request.args.get('Weight')
    BMI = request.args.get('BMI')
    prediction = classifier.predict([[Age,Gender,Height,Weight,BMI]])
    return 'The predicted value is' + str(prediction)

@app.route('/predict_file',methods=['POST'])
def predict_obesity_file():
    """Let's find out Obesity Prediction
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: The output values
    """
    df_test = pd.read_csv(request.files.get('file'))
    prediction = classifier.predict(df_test)
    return 'The predicted values for the test file are'+ str(list(prediction))


if __name__=='__main__':
    app.run()