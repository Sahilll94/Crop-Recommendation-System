import streamlit as st
import joblib
import numpy as np

st.set_page_config(
    page_title="Crop Recommendation System 🌾",  
    page_icon="🌱",  
    layout="centered",  
)

model = joblib.load("crop_recommendation_model.pkl")

st.title("🌾 Crop Recommendation System")
st.write("Enter soil and climate details to get the best crop recommendation.")

nitrogen = st.number_input("Nitrogen (N)", min_value=0, max_value=200)
phosphorus = st.number_input("Phosphorus (P)", min_value=0, max_value=200)
potassium = st.number_input("Potassium (K)", min_value=0, max_value=200)
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0)
ph = st.number_input("pH Level", min_value=0.0, max_value=14.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0)

if st.button("Predict Crop"):
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
    
    prediction = model.predict(input_data)
    
    st.success(f"🌱 Recommended Crop: **{prediction[0]}**")

st.markdown("""
---
### 👨‍💻 Made by:
- Sahil <a href="https://github.com/Sahilll94" target="_blank">
  <img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png" width="20"></a>
- Chandra Sekhar Dutta <a href="https://github.com/Chandra-Sekhar-Dutta" target="_blank">
  <img src="https://img.icons8.com/ios-glyphs/30/ffffff/github.png" width="20"></a>
""", unsafe_allow_html=True)

