import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow first to avoid conflicts
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import sentiment analysis function
from sentiment import get_avg_sentiment

st.set_page_config(page_title="Stock Market Prediction with Sentiment", layout="wide")

st.title("📈 Stock Market Prediction using LSTM + Sentiment Analysis")
st.markdown("Predict stock prices using OHLCV data combined with news sentiment analysis")

# Sidebar for user inputs
st.sidebar.header("Configuration")
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock symbol (e.g., AAPL, GOOGL, TSLA)")
lookback_days = st.sidebar.slider("Lookback Period (days)", 60, 120, 90)
prediction_days = st.sidebar.slider("Prediction Period (days)", 15, 45, 30)

# Advanced settings
st.sidebar.subheader("Model Parameters")
lstm_units = st.sidebar.slider("LSTM Units", 32, 128, 64)
epochs = st.sidebar.slider("Training Epochs", 20, 100, 50)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)

def fetch_stock_data(symbol, lookback_days, prediction_days):
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=lookback_days + prediction_days + 100)
        
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"No data found for symbol {symbol}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def add_sentiment_data(data, symbol):
    """Add sentiment scores to the stock data"""
    sentiment_scores = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, date in enumerate(data.index):
        date_str = date.strftime('%Y-%m-%d')
        
        try:
            sentiment = get_avg_sentiment(symbol, date_str)
            # If no sentiment data available, use neutral (1)
            sentiment_scores.append(sentiment if sentiment is not None else 1.0)
        except Exception as e:
            st.warning(f"Error getting sentiment for {date_str}: {e}")
            sentiment_scores.append(1.0)  # Default to neutral
        
        # Update progress
        progress_bar.progress((i + 1) / len(data))
        status_text.text(f'Fetching sentiment data... ')#{i + 1}/{len(data)}')
    
    progress_bar.empty()
    status_text.empty()
    
    data['Sentiment'] = sentiment_scores
    return data

def create_sequences(data, lookback_days):
    """Create sequences for multi-feature LSTM"""
    X, y = [], []
    for i in range(lookback_days, len(data)):
        X.append(data[i-lookback_days:i, :])  # All features for lookback period
        y.append(data[i, 3])  # Predict Close price (index 3)
    return np.array(X), np.array(y)

def prepare_data(data, lookback_days, prediction_days):
    """Prepare multi-feature data for LSTM"""
    # Select features: Open, High, Low, Close, Volume, Sentiment
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']
    features = data[feature_cols].values
    
    total_days = len(features)
    train_size = total_days - prediction_days
    
    train_data = features[:train_size]
    test_data = features[train_size - lookback_days:]
    
    # Scale features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_train = scaler.fit_transform(train_data)
    scaled_test = scaler.transform(test_data)
    
    X_train, y_train = create_sequences(scaled_train, lookback_days)
    X_test, y_test = create_sequences(scaled_test, lookback_days)
    
    return X_train, y_train, X_test, y_test, train_data, test_data, scaler

def build_lstm_model(lookback_days, num_features, lstm_units=64):
    """Build LSTM model for multi-feature input"""
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(lookback_days, num_features)),
        Dropout(0.2),
        LSTM(lstm_units, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_lstm_model(model, X_train, y_train, epochs=50, batch_size=32):
    """Train LSTM model with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    history = []
    for epoch in range(epochs):
        hist = model.fit(X_train, y_train, 
                        epochs=1, 
                        batch_size=batch_size, 
                        verbose=0)
        history.append(hist.history['loss'][0])
        
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f'Training... Epoch {epoch + 1}/{epochs}, Loss: {hist.history["loss"][0]:.6f}')
    
    progress_bar.empty()
    status_text.empty()
    
    return history

def make_predictions(model, X_test, scaler):
    """Make predictions and inverse transform only the Close price"""
    predictions_scaled = model.predict(X_test)
    
    # Create dummy array with same shape as original features for inverse transform
    dummy_features = np.zeros((predictions_scaled.shape[0], scaler.n_features_in_))
    dummy_features[:, 3] = predictions_scaled.flatten()  # Close price is at index 3
    
    # Inverse transform and extract close price predictions
    predictions_full = scaler.inverse_transform(dummy_features)
    predictions = predictions_full[:, 3].reshape(-1, 1)  # Extract Close price column
    
    return predictions

def get_actual_prices(y_test, test_data, scaler):
    """Get actual prices from scaled test data"""
    # Create dummy array for inverse transform
    dummy_features = np.zeros((len(y_test), scaler.n_features_in_))
    dummy_features[:, 3] = y_test  # Close price is at index 3
    
    # Inverse transform and extract close prices
    actual_full = scaler.inverse_transform(dummy_features)
    actual_prices = actual_full[:, 3].reshape(-1, 1)  # Extract Close price column
    
    return actual_prices

def calculate_metrics(actual, predicted):
    """Calculate various prediction metrics"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    accuracy = max(0, min(100, 100 - mape))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Accuracy': accuracy
    }

if st.sidebar.button("Run Prediction"):
    if symbol:
        # Fetch data
        with st.spinner("Fetching stock data..."):
            data = fetch_stock_data(symbol, lookback_days, prediction_days)
        
        if data is not None:
            # Add sentiment data
            with st.spinner("Fetching sentiment data..."):
                data = add_sentiment_data(data, symbol)
            
            # Display stock info
            stock = yf.Ticker(symbol)
            info = stock.info
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Current Price", f"${data['Close'][-1]:.2f}")
            with col2:
                change = data['Close'][-1] - data['Close'][-2]
                st.metric("Daily Change", f"${change:.2f}", f"{change:.2f}")
            with col3:
                st.metric("Volume", f"{data['Volume'][-1]:,.0f}")
            with col4:
                avg_sentiment = data['Sentiment'].mean()
                sentiment_label = "Positive" if avg_sentiment > 1.5 else "Negative" if avg_sentiment < 0.5 else "Neutral"
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}", sentiment_label)
            
            # Show feature summary
            st.subheader("📊 Feature Summary")
            feature_summary = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']].describe()
            st.dataframe(feature_summary.round(4))
            
            # Prepare data
            with st.spinner("Preparing data..."):
                X_train, y_train, X_test, y_test, train_data, test_data, scaler = prepare_data(
                    data, lookback_days, prediction_days
                )
            
            num_features = X_train.shape[2]  # Number of features (should be 6)
            st.info(f"Using {num_features} features: Open, High, Low, Close, Volume, Sentiment")
            
            # Build and train model
            with st.spinner("Training LSTM model..."):
                model = build_lstm_model(lookback_days, num_features, lstm_units)
                history = train_lstm_model(model, X_train, y_train, epochs, batch_size)
            
            # Make predictions
            predictions = make_predictions(model, X_test, scaler)
            actual_prices = get_actual_prices(y_test, test_data, scaler)
            
            # Calculate metrics
            metrics = calculate_metrics(actual_prices.flatten(), predictions.flatten())
            
            # Display metrics
            st.subheader("📊 Model Performance")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("RMSE", f"{metrics['RMSE']:.2f}")
            with col2:
                st.metric("MAE", f"{metrics['MAE']:.2f}")
            with col3:
                st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
            with col4:
                st.metric("Accuracy", f"{metrics['Accuracy']:.1f}%")
            with col5:
                directional_accuracy = np.mean(np.sign(actual_prices[1:] - actual_prices[:-1]) == 
                                             np.sign(predictions[1:] - predictions[:-1])) * 100
                st.metric("Direction Accuracy", f"{directional_accuracy:.1f}%")
            
            # Create visualizations
            st.subheader("📈 Predictions vs Actual Prices")
            
            # Prepare data for plotting
            dates = data.index[-len(predictions):]
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, actual_prices, label='Actual Prices', color='blue', linewidth=2)
            ax.plot(dates, predictions, label='Predicted Prices', color='red', linewidth=2, linestyle='--')
            ax.set_title(f'{symbol} Stock Price Prediction (with Sentiment)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
            # Sentiment visualization
            st.subheader("📰 Sentiment Analysis Over Time")
            fig4, ax4 = plt.subplots(figsize=(12, 4))
            ax4.plot(data.index[-50:], data['Sentiment'][-50:], color='orange', linewidth=2)
            ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Neutral')
            ax4.set_title('News Sentiment Over Time (Last 50 days)')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Sentiment Score')
            ax4.legend(['Sentiment', 'Neutral Line'])
            ax4.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig4)
            
            # Training loss plot
            st.subheader("🎯 Training Progress")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(history)
            ax2.set_title('Model Training Loss')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss')
            ax2.grid(True, alpha=0.3)
            
            st.pyplot(fig2)
            
            # Historical price chart
            st.subheader("📊 Historical Price Chart")
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(data.index, data['Close'], label='Historical Prices', color='green')
            ax3.axvline(x=data.index[-prediction_days], color='red', linestyle='--', alpha=0.7, label='Prediction Start')
            ax3.set_title(f'{symbol} Historical Prices (Last {len(data)} days)')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Price ($)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig3)
            
            # Prediction table with sentiment
            st.subheader("📋 Detailed Predictions with Sentiment")
            results_df = pd.DataFrame({
                'Date': dates,
                'Actual Price': actual_prices.flatten(),
                'Predicted Price': predictions.flatten(),
                'Difference': (predictions.flatten() - actual_prices.flatten()),
                'Error %': ((predictions.flatten() - actual_prices.flatten()) / actual_prices.flatten() * 100),
                'Sentiment': data['Sentiment'][-len(predictions):].values
            })
            
            st.dataframe(results_df.round(4))
            
            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions with Sentiment as CSV",
                data=csv,
                file_name=f'{symbol}_predictions_with_sentiment.csv',
                mime='text/csv'
            )