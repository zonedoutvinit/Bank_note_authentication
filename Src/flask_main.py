#importing modules
from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

#setting start point for application 
app = Flask(__name__)
Swagger(app)

#loading the model
pickle_in = open("../Bank_note_aunthetication/Model/Model.pkl", 'rb')
Model = pickle.load(pickle_in)

#adding Decorators i.e '@' before each function for flask
@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict', methods = ["GET"])
def predict():
    
    """Lets Authenticate a bank note
    This is using docstring for specification.
    ---
    parameters:
     - name: variance
       in: query
       type: number
       required: true
     - name: skewness
       in: query
       type: number
       required: true  
     - name: curtosis
       in: query
       type: number
       required: true
     - name: entropy
       in: query
       type: number
       required: true
    responses:
        200:
            description: The output values.
            
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = Model.predict([[variance, skewness, curtosis, entropy]]) #prediction function in use
    print(prediction)
    return "the predicted value is :"+ str(prediction)

@app.route('/predict_file', methods = ["POST"])
def predict_file():
    """Lets Authenticate a bank note
    This is using docstring for specification.
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
    df_test = pd.read_csv(request.files.get("file"))
    print(df_test.head())
    prediction = Model.predict(df_test) #prediction for a small csv file. 
    return "the predicted value from csv are :"+ str(list(prediction))


if __name__ == '__main__':
    app.run()