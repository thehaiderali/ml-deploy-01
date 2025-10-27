import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load('linear_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit App
st.title("📘 Student Test Score Predictor")
st.write("Predict test scores based on study-related inputs using a pre-trained Linear Regression model.")

# --- Input Section ---
st.header("Enter Input Data")

# If you have only 1 feature (Hours Studied)
hours = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=1.0)

# If your model uses multiple features, uncomment and add here:
# sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.1)
# past_scores = st.number_input("Past Score", min_value=0.0, max_value=100.0, step=1.0)
# breaks = st.number_input("Study Breaks", min_value=0, max_value=10, step=1)

# Convert to array (modify accordingly if using more features)
input_data = np.array([[hours]])  # change to [[hours, sleep, past_scores, breaks]] if using 4 features

# Predict Button
if st.button("Predict Test Score"):
    try:
        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)

        st.success(f"✅ Predicted Test Score: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.write("---")
st.caption("Developed with 💙 using Streamlit & Scikit-Learn")

