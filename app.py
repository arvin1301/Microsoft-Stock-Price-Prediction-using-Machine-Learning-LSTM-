import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import joblib
import os

# =========================================
# Streamlit App Configuration
# =========================================
st.set_page_config(page_title="📈 Microsoft Stock Price Predictor (LSTM)", layout="wide")
st.title(" LSTM Stock Price Prediction using Yahoo Finance")

# =========================================
# Load Model and Scaler
# =========================================
MODEL_PATH = "msft_lstm_model.h5"
SCALER_PATH = "scaler.pkl"

@st.cache_resource
def load_model_and_scaler():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)):
        st.error("⚠️ Model or Scaler not found in the directory!")
        st.stop()
    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"❌ Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_model_and_scaler()
st.success("✅ Model and Scaler loaded successfully!")

# =========================================
# Sidebar Configuration
# =========================================
st.sidebar.header("🔧 Configuration Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value="MSFT")
years = st.sidebar.slider("Select Number of Years of Data", 2, 10, 5)
future_days = st.sidebar.slider("Predict Future Days", 30, 120, 60)

# ✅ Add Enter button
start_prediction = st.sidebar.button("🔮 Enter & Start Prediction")

st.sidebar.write("---")

if start_prediction:
    # =========================================
    # Fetch Stock Data
    # =========================================
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years * 365)

    st.write(f"📅 Showing data from **{start_date.date()}** to **{end_date.date()}**")

    with st.spinner("📥 Fetching stock data..."):
        data = yf.download(ticker, start=start_date, end=end_date)

    if data.empty:
        st.error("❌ No data found! Please check the stock ticker symbol.")
        st.stop()

    data = data[['Close']]
    st.subheader(f"📊 {ticker} Stock Closing Price History (Prices in USD)")
    st.line_chart(data['Close'], use_container_width=True)

    # =========================================
    # Prepare Data for LSTM
    # =========================================
    scaled_data = scaler.transform(data[['Close']])
    time_step = 60

    X_test = []
    for i in range(time_step, len(scaled_data)):
        X_test.append(scaled_data[i - time_step:i, 0])
    X_test = np.array(X_test).reshape(-1, time_step, 1)

    # =========================================
    # Predict Future Prices
    # =========================================
    with st.spinner("🔮 Predicting future stock prices..."):
        last_60_days = scaled_data[-60:].reshape(1, 60, 1)
        future_scaled = []

        for _ in range(future_days):
            next_pred = model.predict(last_60_days, verbose=0)[0, 0]
            future_scaled.append(next_pred)
            new_input = np.append(last_60_days[0][1:], [[next_pred]], axis=0)
            last_60_days = np.array([new_input])

    future_scaled = np.array(future_scaled).reshape(-1, 1)
    future_predicted = scaler.inverse_transform(future_scaled)

    # =========================================
    # Prepare Future DataFrame
    # =========================================
    future_dates = pd.date_range(end_date + timedelta(days=1), periods=future_days)
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predicted.flatten()})
    future_df.set_index('Date', inplace=True)

    # =========================================
    # Combine Historical + Future Data
    # =========================================
    combined_df = pd.concat([
        data[['Close']].rename(columns={'Close': 'Actual_Close'}),
        future_df.rename(columns={'Predicted_Close': 'Actual_Close'})
    ])

    # =========================================
    # Display Future Predictions
    # =========================================
    st.subheader(f"📈 {ticker} Predicted Stock Price for Next {future_days} Days (Prices in USD)")
    st.line_chart(combined_df, use_container_width=True)

    st.dataframe(future_df.head(10), use_container_width=True)

    # =========================================
    # Download Predictions
    # =========================================
    csv_data = future_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Download Future Predictions (CSV)",
        data=csv_data,
        file_name=f"{ticker}_future_predictions.csv",
        mime="text/csv",
    )

    st.success("✅ Prediction completed successfully!")
    st.caption("Model: LSTM | Data Source: Yahoo Finance | Developed by Arvind Sharma")
else:
    st.info("👆 Adjust settings in the sidebar and click **'Enter & Start Prediction'** to begin.")
