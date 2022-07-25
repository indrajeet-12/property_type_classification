

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("final_model.pkl")

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)


    return render_template('index1.html', prediction_text='Property Type Classified by the model is "{}"'.format(prediction[0]))


if __name__ == "__main__":
    app.run(debug=True)