# Microsoft-Stock-Price-Prediction-using-Machine-Learning-LSTM-
Microsoft Stock Price Prediction using Machine Learning (LSTM)
Tools Used: Python, TensorFlow, Streamlit, Pandas, NumPy, Matplotlib, Scikit-learn, yFinance, XGBoost

# Project Overview

Stock markets are inherently volatile, making price prediction a challenging yet valuable problem.
This project applies Machine Learning and Deep Learning (LSTM networks) to predict Microsoft’s stock price using historical time-series data.

The model captures long-term temporal dependencies in financial data and helps in making informed investment decisions through trend forecasting and data-driven insights.

# Objective

Develop a time-series forecasting model using TensorFlow (LSTM) to predict Microsoft’s future stock price based on historical trends.
Additionally, deploy an interactive Streamlit web app that provides real-time predictions and visualizations for end users.

# Dataset Description

Source: Yahoo Finance (MSFT)

Records: ~2,500 daily records

Date Range: 5–10 years of historical data

Features:

Date

Open

High

Low

Close

Volume

Target Variable: Close (Stock Closing Price)

Data Type: Time-Series

# Data Preprocessing

Steps performed:

Converted Date column to DateTimeIndex.

Handled missing values using interpolation.

Normalized numerical features using MinMaxScaler for better convergence.

Created technical indicators to enhance predictive power:

SMA (Simple Moving Average)

EMA (Exponential Moving Average)

RSI (Relative Strength Index)

Bollinger Bands

Split data into 80% training and 20% testing.

Maintained chronological order to avoid data leakage.

# Exploratory Data Analysis (EDA)

Explored key market trends and insights:

Visualized Microsoft’s stock price trends using line plots.

Identified seasonal and cyclical patterns in price movements.

Analyzed correlation between closing price, volume, and technical indicators.

Highlighted volatility zones using Bollinger Bands.

Key Observations:

Strong correlation between SMA, EMA, and Close price.

High volatility periods correspond to global market shifts.

# Model Development

Trained multiple models for comparison:

Model	Type	RMSE	MAE	R² Score
Linear Regression	Statistical	8.9	6.5	0.85
Random Forest	Ensemble	6.2	4.7	0.91
XGBoost	Gradient Boosting	5.4	4.0	0.93
LSTM	Deep Learning	3.2	2.5	0.97 
Why Choose LSTM?

LSTM (Long Short-Term Memory) networks handle sequential dependencies.

They retain essential temporal information through input, forget, and output gates.

Capable of capturing complex financial time-series patterns better than traditional models.

# LSTM Model Architecture

Input Layer: 60 timesteps (past 60 days)

LSTM Layer: 50 neurons (return_sequences=True)

Dropout: 0.2

Dense Output: 1 neuron (predicting closing price)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Epochs: 100

Batch Size: 32

# Model Evaluation & Results

The LSTM model was trained and validated on unseen test data.
It achieved high predictive accuracy and smooth trend-following behavior.

Performance Metrics:

MAE: 2.5

RMSE: 3.2

R² Score: 0.97

Visual Results:

Actual vs Predicted curves closely overlap.

Future 30-day forecast shows consistent upward trend aligning with historical growth.

# Streamlit Web Application

An interactive Streamlit web app allows real-time stock prediction and visualization.

App Features:

Input parameters:

Stock Ticker (Default: MSFT)

Number of past years for training

Prediction duration (next 30–120 days)

Real-time data fetched from Yahoo Finance API

Outputs:

Historical vs Predicted stock price visualization

Downloadable CSV for predicted values

Dynamic plots for Bollinger Bands, SMA, EMA

Deployment:
streamlit run app.py

# Technical Stack
Category	Tools Used
Language	Python 3.10+
Framework	TensorFlow / Keras
Libraries	Pandas, NumPy, Matplotlib, Seaborn, yFinance, Scikit-learn, XGBoost, Joblib
Deployment	Streamlit
Visualization	Plotly, Matplotlib
Version Control	GitHub
# Future Enhancements

Integrate Transformer / GPT-based time-series forecasting.

Incorporate news sentiment analysis using NLP.

Extend to multi-stock prediction (Google, Apple, Amazon, etc.).

Develop a comprehensive portfolio dashboard for investors.

Add real-time trading integration with APIs like Alpaca or Zerodha.

 Repository Structure
 Microsoft-Stock-Prediction/
│
├──  MicrosoftStock.csv                   # Dataset (Yahoo Finance)
├──  Microsoft_stock.ipynb                # Jupyter Notebook (Model Training)
├──  Microsoft Stock Price Prediction with Machine Learning.pdf
├──  Microsoft_Stock_Price_Prediction_Detailed_with_Graphs.pptx
├──  app.py                               # Streamlit App Script
├──  requirements.txt                     # Dependencies
├──  README.md                            # Project Documentation
└──  models/
    ├── lstm_model.h5
    ├── scaler.pkl

 Installation & Usage

Clone this repository

git clone https://github.com/yourusername/Microsoft-Stock-Price-Prediction.git
cd Microsoft-Stock-Price-Prediction


Install required libraries

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


View results

Visit http://localhost:8501

Enter your parameters and visualize the forecast!

 Results Summary

LSTM achieved the highest accuracy (R² = 0.97) among all tested models.

Forecasted next 30 days of Microsoft stock prices with realistic market trends.

Deployed as a Streamlit web app for interactive, real-time use.

 Conclusion

This project demonstrates the potential of deep learning in financial forecasting.

https://github.com/user-attachments/assets/7a0ccfbf-a575-461e-8e46-21413a41657b


The LSTM model effectively predicts Microsoft’s stock price trends and serves as a reliable analytical tool for investors, researchers, and data scientists.

“AI-driven financial forecasting can transform investment strategies through data-driven insights.”
