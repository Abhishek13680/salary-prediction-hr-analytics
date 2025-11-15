import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Load trained model
# ----------------------------
model = joblib.load("models/final_mlr_model.joblib")

# ----------------------------
# Load dataset to extract dropdown options
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_excel("cleaned file.xlsx")

df = load_data()

st.title("📊 HR Analytics — Salary Prediction App")
st.write("Enter employee details below to predict salary using the trained MLR model.")

# Dropdown values
roles = sorted(df["Role"].dropna().unique())
countries = sorted(df["Country"].dropna().unique())

# Inputs
age = st.slider("Age", 18, 70, 30)
happiness = st.slider("Total Happiness", 1.0, 10.0, 7.0)
role = st.selectbox("Select Role", roles)
country = st.selectbox("Select Country", countries)

# Prediction
if st.button("Predict Salary"):
    new_data = pd.DataFrame([{
        "Age": age,
        "Total Happines": happiness,
        "Role": role,
        "Country": country
    }])
    
    pred = model.predict(new_data)[0]
    st.success(f"💰 Predicted Average Salary: {pred:.2f} K USD")
