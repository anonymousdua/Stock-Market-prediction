import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Stock Market Prediction", layout="wide")

# Title and description
st.title("📈 Stock Market Prediction using LSTM")
st.markdown("Predict stock prices using the last 90 days to forecast the next 30 days")

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

class StockPredictor:
    def __init__(self, lookback_days=90, prediction_days=30):
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.data = None
        
    def fetch_data(self, symbol):
        """Fetch stock data from yfinance"""
        try:
            # Calculate date range - we need extra days for predictions
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=self.lookback_days + self.prediction_days + 100)
            
            stock = yf.Ticker(symbol)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                st.error(f"No data found for symbol {symbol}")
                return None
                
            return data
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return None
    
    def prepare_data(self, data):
        """Prepare data for LSTM training"""
        # Use closing prices
        prices = data['Close'].values.reshape(-1, 1)
        
        # Split data for backtesting
        total_days = len(prices)
        train_size = total_days - self.prediction_days
        
        train_data = prices[:train_size]
        test_data = prices[train_size - self.lookback_days:]  # Include lookback for prediction
        
        # Scale the data
        scaled_train = self.scaler.fit_transform(train_data)
        scaled_test = self.scaler.transform(test_data)
        
        # Create sequences
        X_train, y_train = self.create_sequences(scaled_train)
        X_test, y_test = self.create_sequences(scaled_test)
        
        return X_train, y_train, X_test, y_test, train_data, test_data
    
    def create_sequences(self, data):
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(self.lookback_days, len(data)):
            X.append(data[i-self.lookback_days:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def build_model(self, lstm_units=64):
        """Build LSTM model"""
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(self.lookback_days, 1)),
            Dropout(0.2),
            LSTM(lstm_units, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    
    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the LSTM model"""
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        self.model = self.build_model(lstm_units)
        
        # Train with progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        history = []
        for epoch in range(epochs):
            hist = self.model.fit(X_train, y_train, 
                                epochs=1, 
                                batch_size=batch_size, 
                                verbose=0)
            history.append(hist.history['loss'][0])
            
            progress_bar.progress((epoch + 1) / epochs)
            status_text.text(f'Training... Epoch {epoch + 1}/{epochs}, Loss: {hist.history["loss"][0]:.6f}')
        
        progress_bar.empty()
        status_text.empty()
        
        return history
    
    def make_predictions(self, X_test):
        """Make predictions"""
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        predictions = self.model.predict(X_test)
        return self.scaler.inverse_transform(predictions)
    
    def calculate_metrics(self, actual, predicted):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual, predicted)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        # Accuracy as 100 - MAPE, clipped between 0 and 100
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
        predictor = StockPredictor(lookback_days, prediction_days)
        
        # Fetch data
        with st.spinner("Fetching stock data..."):
            data = predictor.fetch_data(symbol)
        
        if data is not None:
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
                market_cap = info.get('marketCap', None)
                if market_cap:
                    st.metric("Market Cap", f"${market_cap:,}")
                else:
                    st.metric("Market Cap", "N/A")
            
            # Prepare data
            with st.spinner("Preparing data..."):
                X_train, y_train, X_test, y_test, train_data, test_data = predictor.prepare_data(data)
            
            # Train model
            with st.spinner("Training LSTM model..."):
                history = predictor.train_model(X_train, y_train, epochs, batch_size)
            
            # Make predictions
            predictions = predictor.make_predictions(X_test)
            actual_prices = predictor.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Calculate metrics
            metrics = predictor.calculate_metrics(actual_prices.flatten(), predictions.flatten())
            
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
            ax.set_title(f'{symbol} Stock Price Prediction')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(fig)
            
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
            
            # Prediction table
            st.subheader("📋 Detailed Predictions")
            results_df = pd.DataFrame({
                'Date': dates,
                'Actual Price': actual_prices.flatten(),
                'Predicted Price': predictions.flatten(),
                'Difference': (predictions.flatten() - actual_prices.flatten()),
                'Error %': ((predictions.flatten() - actual_prices.flatten()) / actual_prices.flatten() * 100)
            })
            
            st.dataframe(results_df.round(2))
            
            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name=f'{symbol}_predictions.csv',
                mime='text/csv'
            )