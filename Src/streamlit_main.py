import numpy as np
import pickle 
import pandas as pd
import streamlit as st
from PIL import Image
 
pickle_in = open("../Model/Model.pkl", 'rb')
model = pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_note(variance, skewness, curtosis, entropy):
    prediction = model.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return prediction

def main():
    st.title("Bank Authenticator")
    html_temp =  """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    variance = st.text_input("Variance", "Type Here")
    skewness = st.text_input("Skewness", "Type Here")
    curtosis = st.text_input("Curtosis", "Type Here")
    entropy = st.text_input("Entropy", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note(variance, skewness, curtosis, entropy)
    st.success('The output is {}'.format(result))
    
if __name__ =='__main__':
    main()