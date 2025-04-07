import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("tesla_model.pkl")
scaler = joblib.load("tesla_scaler.pkl")

st.set_page_config(page_title="Tesla Stock Price Predictor", layout="wide")
st.title("📈 Tesla Stock Price Predictor")
st.markdown("---")

# Layout: 3 columns for input + 1 for spacing
col1, col2, col3 = st.columns([1, 1, 1])

# === LEFT COLUMN ===
with col1:
    st.subheader("Select Stock")
    selected_stock = st.selectbox("Choose a stock", ["TSLA"], key="stock")

# === MIDDLE COLUMN ===
with col2:
    st.subheader("Data Preprocessing")
    high = st.number_input("High", value=0.0, format="%f")
    low = st.number_input("Low", value=0.0, format="%f")
    open_price = st.number_input("Open", value=0.0, format="%f")
    volume = st.number_input("Volume", value=0.0, format="%f")

    if st.button("Preprocess Data"):
        features = np.array([[high, low, open_price, volume]])
        scaled_features = scaler.transform(features)
        st.session_state["scaled_features"] = scaled_features
        st.success("✅ Data preprocessed successfully!")

# === RIGHT COLUMN ===
with col3:
    st.subheader("Data Modelling")
    if st.button("Run Model"):
        if "scaled_features" in st.session_state:
            prediction = model.predict(st.session_state["scaled_features"])
            st.session_state["prediction"] = prediction[0]
            st.success("✅ Model prediction ready!")
        else:
            st.error("❌ Please preprocess data first.")

# === VALIDATION SECTION ===
st.markdown("---")
st.subheader("🔍 Validation")
center_col = st.columns([2, 1, 2])[1]

with center_col:
    if st.button("Validate Prediction"):
        if "prediction" in st.session_state:
            st.success(f"📊 Predicted Close Price: **${st.session_state['prediction']:.2f}**")
        else:
            st.error("❌ Please run the model first.")