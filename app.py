import streamlit as st
import joblib
import numpy as np
import pandas as pd

# --- Session state for navigation ---
if "page" not in st.session_state:
    st.session_state.page = "Home"

def navigate(page_name):
    st.session_state.page = page_name

# Load the trained model and scaler
model = joblib.load("tesla_model.pkl")
scaler = joblib.load("tesla_scaler.pkl")

# Load stock data
def load_stock_data(stock_name):
    if stock_name == "AAPL":
        return pd.read_csv(r"C:\Users\sanju\Downloads\Sanjana Duraiswamy - AAPL.csv")
    elif stock_name == "ADANIGREEN":
        return pd.read_csv(r"C:\Users\sanju\Downloads\Sanjana Duraiswamy - ADANIGREEN.NS.csv")
    elif stock_name == "RELIANCE":
        return pd.read_csv(r"C:\Users\sanju\Downloads\Sanjana Duraiswamy - RELIANCE.NS.csv")
    elif stock_name == "SBIN":
        return pd.read_csv(r"C:\Users\sanju\Downloads\Sanjana Duraiswamy - SBIN.NS (2).csv")
    elif stock_name == "TSLA":
        return pd.read_csv(r"C:\Users\sanju\Documents\ML NOTES\TSLA.csv")
    else:
        return None

# Page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
if st.sidebar.button("🏠 Home", key="nav_home"):
    navigate("Home")
if st.sidebar.button("🧹 Data Preprocessing", key="nav_preprocess"):
    navigate("Data Preprocessing")
if st.sidebar.button("🔮 Model Prediction", key="nav_model"):
    navigate("Model Prediction")
if st.sidebar.button("🔍 Validation", key="nav_validate"):
    navigate("Validation")

page = st.session_state.page

# --- Pages ---

# Home Page
if page == "Home":
    st.title("📈 Stock Price Predictor")
    st.subheader("Welcome!")
    st.write("This app predicts stock prices using a pre-trained model.")
    st.button("➡️ Go to Data Preprocessing", key="home_next", on_click=lambda: navigate("Data Preprocessing"))

# Data Preprocessing Page
elif page == "Data Preprocessing":
    st.title("📊 Data Preprocessing")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select Stock")
        selected_stock = st.selectbox("Choose a stock", ["AAPL", "ADANIGREEN", "RELIANCE", "SBIN", "TSLA"], key="select_stock")
        stock_data = load_stock_data(selected_stock)
        if stock_data is not None:
            st.write("### Raw Data")
            st.dataframe(stock_data.head())

    with col2:
        st.subheader("Input Stock Data")
        high = st.number_input("High", value=0.0, format="%f")
        low = st.number_input("Low", value=0.0, format="%f")
        open_price = st.number_input("Open", value=0.0, format="%f")
        volume = st.number_input("Volume", value=0.0, format="%f")

    if st.button("Preprocess Data", key="do_preprocess"):
        features = np.array([[high, low, open_price, volume]])
        scaled_features = scaler.transform(features)
        st.session_state["scaled_features"] = scaled_features
        st.success("✅ Data preprocessed successfully!")

        st.write("### Preprocessed Data:")
        preprocessed_df = pd.DataFrame(scaled_features, columns=["High", "Low", "Open", "Volume"])
        st.dataframe(preprocessed_df)

        st.button("➡️ Go to Model Prediction", key="to_model", on_click=lambda: navigate("Model Prediction"))

# Model Prediction Page
# Model Prediction Page
elif page == "Model Prediction":
    st.title("🔮 Model Prediction")

    if "scaled_features" in st.session_state:
        if st.button("Run Model", key="run_model"):
            prediction = model.predict(st.session_state["scaled_features"])
            st.session_state["prediction"] = prediction[0]
            st.success("✅ Prediction complete.")
            # Navigation button to Validation page
            st.button("➡️ Go to Validation", key="go_to_validation", on_click=lambda: navigate("Validation"))
    else:
        st.error("❌ Please preprocess data first.")
    
    st.button("⬅️ Back to Preprocessing", key="model_back", on_click=lambda: navigate("Data Preprocessing"))


# Validation Page
elif page == "Validation":
    st.title("🔍 Validation")

    if "prediction" in st.session_state:
        st.subheader(f"📊 Predicted Close Price: **${st.session_state['prediction']:.2f}**")
    else:
        st.error("❌ Prediction not found.")

    st.button("🏠 Back to Home", key="validate_home", on_click=lambda: navigate("Home"))