import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# Load trained model
# -------------------------------------------------------
model = joblib.load("models/final_mlr_model.joblib")

# -------------------------------------------------------
# Load dataset for dropdown values
# -------------------------------------------------------
df = pd.read_excel("cleaned file.xlsx")

st.set_page_config(page_title="Salary Predictor", page_icon="💰", layout="centered")

st.title("💼 HR Analytics — Salary Prediction App")
st.write("Enter employee details to predict their expected Salary using the trained MLR model.")

# Dropdown values from dataset
roles = sorted(df["Role"].dropna().unique())
countries = sorted(df["Country"].dropna().unique())

# Inputs
age = st.slider("Age", 18, 70, 30)
happiness = st.slider("Total Happiness", 1.0, 10.0, 7.0)
role = st.selectbox("Select Role", roles)
country = st.selectbox("Select Country", countries)

# Predict Button
if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Total Happines": happiness,
        "Role": role,
        "Country": country
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 **Predicted Salary: {prediction:.2f} K USD**")
