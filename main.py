# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:53:08 2020

@author: Shifana A
"""

from flask import Flask, render_template,url_for, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pickle

#load the model from disk
file_name = 'stock_sentiment_analysis_model.pkl'
classifier =pickle.load(open(file_name, 'rb'))
cv = pickle.load(open('transform.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
    return render_template('result.html', prediction = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
    