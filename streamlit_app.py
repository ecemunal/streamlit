import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import plotly.graph_objs as go

# Function to fetch historical data
def fetch_historical_data(ticker, period="2y", interval="1d"):
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period=period, interval=interval)
    return hist_data

# Function to preprocess data
def preprocess_data(hist_data, scaler, time_step=60):
    scaled_data = scaler.transform(hist_data['Close'].values.reshape(-1,1))
    X = []
    for i in range(len(scaled_data) - time_step - 1):
        a = scaled_data[i:(i + time_step), 0]
        X.append(a)
    return np.array(X)

# Load LSTM model
@st.cache(allow_output_mutation=True)
def load_lstm_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Streamlit UI
def main():
    st.title("Stock Price Prediction using LSTM")

    # User input for ticker symbol
    ticker_symbol = st.text_input("Enter Ticker Symbol (e.g., META)", "META")

    # Load model
    lstm_model = load_lstm_model('lstmm_model.h5')

    if st.button("Predict"):
        with st.spinner('Fetching data and making prediction...'):
            # Fetch historical data
            hist_data = fetch_historical_data(ticker_symbol)

            # Initialize and fit scaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(hist_data['Close'].values.reshape(-1,1))

            # Preprocess and predict
            X = preprocess_data(hist_data, scaler)
            X = X.reshape(X.shape[0], X.shape[1], 1)
            predicted = lstm_model.predict(X)
            predicted_prices = scaler.inverse_transform(predicted)

            # Plot the predicted prices alongside actual closing prices
            actual_prices = hist_data['Close'].values[-len(predicted_prices):]
            predicted_dates = hist_data.index[-len(predicted_prices):]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predicted_dates, y=actual_prices, mode='lines', name='Actual Price'))
            fig.add_trace(go.Scatter(x=predicted_dates, y=predicted_prices.flatten(), mode='lines', name='Predicted Price'))
            fig.update_layout(title='Predicted vs Actual Stock Prices', xaxis_title='Date', yaxis_title='Price')
            
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()



