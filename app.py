import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_path = "/mount/src/da-heart-disease-prediction/model/best_random_forest_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Streamlit app configuration
st.title("Heart Disease Risk Prediction")
st.write("This app predicts the risk of heart disease based on user input.")

# User inputs for prediction
st.header("Enter Patient Data")

# Input fields
sex = st.selectbox("Sex (1: Male, 0: Female)", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-Anginal Pain, 3: Asymptomatic)", [0, 1, 2, 3])
max_heart_rate = st.slider("Maximum Heart Rate Achieved", 60, 220, 100)
exercise_angina = st.selectbox("Exercise Induced Angina (1: Yes, 0: No)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise relative to rest)", min_value=0.0, max_value=6.0, value=0.0, step=0.1)
st_slope = st.selectbox("ST Slope (0: Upsloping, 1: Flat, 2: Downsloping)", [0, 1, 2])

# Prediction button
if st.button("Predict Heart Disease Risk"):
    # Prepare input data
    input_data = np.array([[sex, chest_pain_type, max_heart_rate, exercise_angina, oldpeak, st_slope]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Display the result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.write("The patient is at **high risk** of heart disease.")
    else:
        st.write("The patient is at **low risk** of heart disease.")

    # Display probabilities
    st.write(f"Probability of low risk: {prediction_proba[0]:.2f}")
    st.write(f"Probability of high risk: {prediction_proba[1]:.2f}")

st.markdown("---")
st.write("**Note:** This prediction is based on the input data provided and should not substitute medical advice. Please consult with a healthcare professional for accurate diagnosis and treatment.")
