import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page config (first Streamlit command)
st.set_page_config(page_title="Heart Failure Prediction", layout="centered")

# Load model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart_failure_clinical_records_dataset.csv")

try:
    df = load_data()
except:
    df = None

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Summary", "Graphs", "Predict"])

# --- Pages ---

# Home Page
if page == "Home":
    st.title("💓 HeartCare Predictor")
    st.write("Welcome to the **HeartCare Predictor App**.")
    st.markdown("Use the sidebar to navigate between pages.")

# Dataset Page
elif page == "Dataset":
    st.title("📊 Dataset")
    if df is not None:
        st.dataframe(df)
    else:
        st.error("Dataset not found. Please place 'heart_failure_clinical_records_dataset.csv' in the app directory.")

# Summary Page
elif page == "Summary":
    st.title("📋 Data Summary")
    if df is not None:
        st.write("### Summary Statistics")
        st.write(df.describe())
        st.write("### Missing Values")
        st.write(df.isnull().sum())
    else:
        st.warning("Dataset not loaded.")

# Graphs Page
elif page == "Graphs":
    st.title("📈 Graphs")
    if df is not None:
        st.write("### Age Distribution")
        st.bar_chart(df["age"])

        st.write("### Serum Creatinine over Time")
        st.line_chart(df[["time", "serum_creatinine"]].sort_values("time"))

        st.write("### Death Event Count")
        st.bar_chart(df["DEATH_EVENT"].value_counts())
    else:
        st.warning("No dataset available to plot.")

# Predict Page
elif page == "Predict":
    st.title("💓 HeartCare Predictor")
    st.subheader("Enter patient data to predict risk of death event")

    # Input form
    age = st.number_input("Age", 18, 100, 60)
    anaemia = st.selectbox("Anaemia", [0, 1])
    creatinine_phosphokinase = st.number_input("CPK Level", 23, 7861, 250)
    diabetes = st.selectbox("Diabetes", [0, 1])
    ejection_fraction = st.slider("Ejection Fraction (%)", 10, 80, 38)
    high_blood_pressure = st.selectbox("High Blood Pressure", [0, 1])
    platelets = st.number_input("Platelets", 25000.0, 850000.0, 263358.03)
    serum_creatinine = st.number_input("Serum Creatinine", 0.5, 10.0, 1.1)
    serum_sodium = st.number_input("Serum Sodium", 110, 150, 137)
    sex = st.selectbox("Sex", [0, 1])  # 0: Female, 1: Male
    smoking = st.selectbox("Smoking", [0, 1])
    time = st.slider("Follow-up Period (Days)", 0, 300, 120)

    if st.button("Predict"):
        input_data = np.array([[age, anaemia, creatinine_phosphokinase, diabetes,
                                ejection_fraction, high_blood_pressure, platelets,
                                serum_creatinine, serum_sodium, sex, smoking, time]])
        prediction = model.predict(input_data)[0]
        st.markdown(f"### 🧾 Prediction: **{'Died 😔' if prediction == 1 else 'Survived 😊'}**")
