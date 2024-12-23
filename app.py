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
patient_name = st.text_input("Patient Name", help="Enter the patient's full name.")

sex = st.selectbox(
    "Sex", 
    ["Male", "Female"], 
    help="Select the biological sex of the patient (Male or Female)."
)
sex_value = 1 if sex == "Male" else 0

chest_pain_type = st.selectbox(
    "Chest Pain Type", 
    ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
    help=(
        "Type of chest pain experienced by the patient:\n"
        "- Typical Angina: Pain caused by reduced blood flow to the heart.\n"
        "- Atypical Angina: Less common symptoms of angina.\n"
        "- Non-Anginal Pain: Chest pain not related to heart conditions.\n"
        "- Asymptomatic: No chest pain symptoms."
    )
)
chest_pain_type_value = ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(chest_pain_type)

max_heart_rate = st.slider(
    "Maximum Heart Rate Achieved", 
    60, 220, 100, 
    help="The highest heart rate achieved during physical activity or exercise, measured in beats per minute (bpm)."
)

exercise_angina = st.selectbox(
    "Exercise Induced Angina", 
    ["Yes", "No"], 
    help="Indicates whether the patient experienced angina (chest pain) during exercise:\n- Yes: Angina occurred.\n- No: Angina did not occur."
)
exercise_angina_value = 1 if exercise_angina == "Yes" else 0

oldpeak = st.number_input(
    "Oldpeak (ST Depression)", 
    min_value=0.0, max_value=5.0, value=0.0, step=0.1,
    help=(
        "ST depression measured during exercise relative to rest. "
        "This indicates changes in the heart's electrical activity."
    )
)

st_slope = st.selectbox(
    "ST Slope", 
    ["Upsloping", "Flat", "Downsloping"],
    help=(
        "The slope of the ST segment during exercise:\n"
        "- Upsloping: Generally indicates better heart condition.\n"
        "- Flat: Suggests a moderate risk.\n"
        "- Downsloping: Often linked to higher risk of heart issues."
    )
)
st_slope_value = ["Upsloping", "Flat", "Downsloping"].index(st_slope)

# Prediction button
if st.button("Predict Heart Disease Risk"):
    # Prepare input data
    input_data = np.array([[sex_value, chest_pain_type_value, max_heart_rate, exercise_angina_value, oldpeak, st_slope_value]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    prediction_proba = model.predict_proba(input_data)[0]

    # Display the result
    st.subheader(f"Prediction Result for {patient_name}")
    if prediction == 1:
        st.write(f"{patient_name} is at **high risk** of heart disease.")
        st.write("**Suggested Next Steps:**")
        st.write("- Schedule an appointment with a cardiologist.")
        st.write("- Undergo further diagnostic tests such as ECG, stress test, or angiography.")
        st.write("- Adopt a heart-healthy lifestyle: balanced diet, regular exercise, and stress management.")
    else:
        st.write(f"Congratulations {patient_name}, you are at **low risk** of heart disease.")
        st.write("Keep maintaining a healthy lifestyle and consult your doctor for regular check-ups.")

    # Display probabilities
    st.write(f"Probability of low risk: {prediction_proba[0]:.2f}")
    st.write(f"Probability of high risk: {prediction_proba[1]:.2f}")

st.markdown("---")
st.write("**Note:** This prediction is based on the input data provided and should not substitute medical advice. Please consult with a healthcare professional for accurate diagnosis and treatment.")
