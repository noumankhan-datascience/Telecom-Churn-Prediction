import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load("churn_model.pkl")

st.title("Customer Churn Prediction App")

# User Inputs
international_plan = st.selectbox("International Plan", ["No", "Yes"])
voice_mail_plan = st.selectbox("Voice Mail Plan", ["No", "Yes"])
number_vmail_messages = st.number_input("Number of Voicemail Messages", min_value=0, max_value=100, step=1)
total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0, max_value=400.0, step=0.1)
total_day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, step=1)
total_day_charge = st.number_input("Total Day Charge", min_value=0.0, max_value=100.0, step=0.1)
total_eve_minutes = st.number_input("Total Evening Minutes", min_value=0.0, max_value=400.0, step=0.1)
total_eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, step=1)
total_eve_charge = st.number_input("Total Evening Charge", min_value=0.0, max_value=100.0, step=0.1)
total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0, max_value=400.0, step=0.1)
total_night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, step=1)
total_night_charge = st.number_input("Total Night Charge", min_value=0.0, max_value=100.0, step=0.1)
total_intl_minutes = st.number_input("Total International Minutes", min_value=0.0, max_value=50.0, step=0.1)
total_intl_calls = st.number_input("Total International Calls", min_value=0, max_value=20, step=1)
total_intl_charge = st.number_input("Total International Charge", min_value=0.0, max_value=10.0, step=0.1)
customer_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, step=1)

# Encode categorical variables
international_plan = 1 if international_plan == "Yes" else 0
voice_mail_plan = 1 if voice_mail_plan == "Yes" else 0

# Prepare input data
input_data = np.array([[international_plan, voice_mail_plan, number_vmail_messages,
                        total_day_minutes, total_day_calls, total_day_charge,
                        total_eve_minutes, total_eve_calls, total_eve_charge,
                        total_night_minutes, total_night_calls, total_night_charge,
                        total_intl_minutes, total_intl_calls, total_intl_charge,
                        customer_service_calls]])

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data)[0]
    st.write("Prediction:", "**Churn (Stop use services)**" if prediction == 1 else "**Not Churn (Stay and use services)**")
