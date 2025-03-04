import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

# Streamlit UI
st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate details to get the best crop recommendation.")

# Input fields
nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200)
phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200)
potassium = st.number_input("Potassium (K)", min_value=0, max_value=200)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0)

# Prediction Button
if st.button("Predict Crop"):
    # Prepare input for prediction
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    st.success(f"🌱 Recommended Crop: **{prediction[0]}**")
