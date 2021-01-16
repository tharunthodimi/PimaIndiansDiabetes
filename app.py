# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 21:53:07 2020

@author: Tharun Tej Reddy Thodimi
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction

    return render_template('index.html', prediction_text='Class of the pima Indians dataset is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
