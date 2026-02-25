import streamlit as st
import joblib
import numpy as np

model = joblib.load("model/diabetes_pipeline.pkl")

st.title("Diabetes Risk Predictor")

glucose = st.number_input("Glucose Level")
bmi = st.number_input("BMI")
age = st.number_input("Age")
pregnancies = st.number_input("Pregnancies")

threshold = 0.45
if st.button("Predict"):
    data = np.array([[glucose, bmi, age, pregnancies]])
    prediction = model.predict_proba(data)[:,1]
    prediction = (prediction >= threshold) 

    if prediction[0] == 1:
        st.error("High Risk")
    else:
        st.success("Low Risk")