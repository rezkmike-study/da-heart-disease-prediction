import streamlit as st
import numpy as np
import pickle
from fpdf import FPDF

# Load the trained model
model_path = "./model/best_xgboost_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Streamlit app configuration
st.title("Heart Disease Risk Prediction")
st.write("This app predicts the risk of heart disease based on user input.")

# User inputs for prediction
st.header("Enter Patient Data")

hospital_name = "Sunwei Specialist Hospital"

patient_name = st.text_input("Patient Name", help="Enter the patient's full name.")
dob = st.date_input("Date of Birth")
nurse_name = st.selectbox(
    "Registered Nurse Name",
    ["Jia Hui", "Gayathri", "Kamil", "Yong Hui"],
    help="Select the name of the nurse entering the patient information."
)

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

    # Risk Assessment
    if prediction == 1:
        risk_level = "High Risk"
        comment = "Schedule an appointment with a cardiologist. Undergo diagnostic tests like ECG or angiography. Adopt a heart-healthy lifestyle."
    else:
        risk_level = "Low Risk"
        comment = "Keep maintaining a healthy lifestyle and have regular check-ups."

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Hospital Name
    pdf.cell(200, 10, txt=hospital_name, ln=True, align="C")

    # Consent Notice
    pdf.multi_cell(0, 10, txt=(
        "Consent Notice: This document contains the medical evaluation for heart disease risk based on patient-provided data. "
        "By proceeding with this evaluation, you consent to its use for diagnostic purposes."
    ))

    # Patient Information Table
    pdf.ln(10)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(50, 10, "Patient Name", border=1)
    pdf.cell(140, 10, patient_name, border=1, ln=True)
    pdf.cell(50, 10, "Date of Birth", border=1)
    pdf.cell(140, 10, str(dob), border=1, ln=True)
    pdf.cell(50, 10, "Sex", border=1)
    pdf.cell(140, 10, sex, border=1, ln=True)
    pdf.cell(50, 10, "Chest Pain Type", border=1)
    pdf.cell(140, 10, chest_pain_type, border=1, ln=True)
    pdf.cell(50, 10, "Max Heart Rate", border=1)
    pdf.cell(140, 10, f"{max_heart_rate} bpm", border=1, ln=True)
    pdf.cell(50, 10, "Exercise Angina", border=1)
    pdf.cell(140, 10, exercise_angina, border=1, ln=True)
    pdf.cell(50, 10, "Oldpeak", border=1)
    pdf.cell(140, 10, str(oldpeak), border=1, ln=True)
    pdf.cell(50, 10, "ST Slope", border=1)
    pdf.cell(140, 10, st_slope, border=1, ln=True)

    # Risk Assessment Table
    pdf.ln(10)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Risk Assessment", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(50, 10, "Risk Level", border=1)
    pdf.cell(140, 10, risk_level, border=1, ln=True)

    # Comment Section
    pdf.ln(5)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Comment", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=comment)

    # Medical Officer Information
    pdf.ln(10)
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Medical Officer Information", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(50, 10, "Registered Nurse", border=1)
    pdf.cell(140, 10, nurse_name, border=1, ln=True)

    # Footer
    pdf.ln(10)
    pdf.set_font("Arial", "I", size=10)
    pdf.multi_cell(0, 10, txt="This result was generated by the Sunwei Specialist Hospital's Heart Disease Risk Prediction System.")

    # Save PDF in memory
    pdf_output = f"{patient_name}_heart_disease_report.pdf"
    pdf.output(pdf_output)

    # Streamlit download button
    with open(pdf_output, "rb") as f:
        st.download_button(
            label="Download Report as PDF",
            data=f,
            file_name=pdf_output,
            mime="application/pdf"
        )
