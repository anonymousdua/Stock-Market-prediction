import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
from sentimentTest import batch_get_sentiments 

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- App Title and Description ---
st.title("LSTM Stock Price Predictor with Technical Indicators")
st.markdown("This app uses an LSTM neural network, along with sentiment analysis and Simple Moving Averages (SMA), to predict stock prices.")


# --- Sidebar for User Inputs ---
st.sidebar.header("Stock Selection")
# A predefined list of top 10 popular stocks
POPULAR_STOCKS = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA",
    "Meta Platforms": "META",
    "JPMorgan Chase": "JPM",
    "Johnson & Johnson": "JNJ",
    "Visa": "V"
}
stock_name = st.sidebar.selectbox(
    "Choose a Stock",
    options=list(POPULAR_STOCKS.keys()),
    index=0,
    help="Select a stock from the list of top 10 popular companies."
)
symbol = POPULAR_STOCKS[stock_name]

# --- Model Parameters ---
st.sidebar.header("Model Parameters")
lookback_period = st.sidebar.slider(
    "Lookback Period (Days)",
    min_value=30,
    max_value=120,
    value=90,
    help="Number of past days' data to use for predicting the next day."
)
epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=20,
    max_value=100,
    value=50,
    help="Number of times the model will cycle through the training data."
)
batch_size = st.sidebar.select_slider(
    "Batch Size",
    options=[16, 32, 64],
    value=32,
    help="Number of training samples utilized in one iteration."
)

# --- Prediction Parameters ---
st.sidebar.header("Prediction Settings")
prediction_days = st.sidebar.slider(
    "Days to Predict",
    min_value=7,
    max_value=60,
    value=30,
    help="Number of days to predict and evaluate. These are historical days used for testing the model's accuracy."
)

# --- Data Parameters ---
st.sidebar.header("Data Settings")
total_days = st.sidebar.slider(
    "Total Historical Days to Fetch",
    min_value=200,
    max_value=500,
    value=300,
    help="Total number of historical trading days to fetch. More data can improve model accuracy."
)

# --- Data Fetching Function ---
@st.cache_data
def fetch_stock_data(symbol, days_to_fetch):
    try:
        end_date = pd.Timestamp.now()
        # Fetch extra days to account for weekends, holidays, and SMA calculations
        start_date = end_date - pd.Timedelta(days=int(days_to_fetch * 1.7))
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"No data found for symbol {symbol}. Please try another stock.")
            return None
        
        # Select relevant columns and take the specified number of trading days
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(days_to_fetch)

        if len(data) < days_to_fetch * 0.8:
            st.warning(f"Only found {len(data)} trading days of data (requested {days_to_fetch}). The model might be less accurate.")

        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# --- Data Preparation Function ---
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 3])  # The target is the 'Close' price, which is at index 3
    return np.array(X), np.array(y)

# --- LSTM Model Building Function ---
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# --- Main Application Logic ---
if st.sidebar.button("Run Prediction", type="primary"):
    # Validate parameters
    min_required_days = lookback_period + prediction_days + 50  # 50 extra for training
    if total_days < min_required_days:
        st.error(f"⚠️ Not enough data! With a lookback period of {lookback_period} days and {prediction_days} days to predict, you need at least {min_required_days} total days. Please increase 'Total Historical Days to Fetch' or adjust other parameters.")
        st.stop()
    
    with st.spinner(f"Fetching {total_days} days of data for {stock_name}..."):
        data = fetch_stock_data(symbol, total_days)
        
    if data is not None:
        # Display fetched data info
        st.info(f"Fetched {len(data)} trading days of data for {stock_name} ({symbol})")
        
        # 1. Feature Engineering: Add SMAs and Sentiment
        with st.spinner("Adding features (Sentiment & SMAs)..."):
            data = data.reset_index()
            
            # Add Sentiment
            dates_list = data["Date"].dt.strftime('%Y-%m-%d').tolist()
            sentiment_results = batch_get_sentiments(symbol, dates_list)
            data["Sentiment"] = data["Date"].dt.strftime('%Y-%m-%d').map(sentiment_results)
            
            # Add SMAs
            data['SMA5'] = data['Close'].rolling(window=5).mean()
            data['SMA10'] = data['Close'].rolling(window=10).mean()
            data['SMA15'] = data['Close'].rolling(window=15).mean()
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            
            # Drop rows with NaN values created by SMA calculation
            data.dropna(inplace=True)
        
        # Show recent data
        with st.expander(f"View Recent Data for {stock_name} ({symbol})", expanded=False):
            st.dataframe(data.tail(10))

        # 2. Data Splitting and Scaling
        with st.spinner("Preparing data and scaling..."):
            # Use all features for scaling
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 
                        'SMA5', 'SMA10', 'SMA15', 'SMA20']
            num_features = len(features)
            data_featured = data[features].values
            
            # Ensure we have enough data after feature engineering
            min_data_needed = lookback_period + prediction_days + 20  # 20 for minimum training
            if len(data_featured) < min_data_needed:
                st.error(f"Not enough data after adding features! Need at least {min_data_needed} days but only have {len(data_featured)}. Please increase total days to fetch or reduce lookback/prediction days.")
                st.stop()

            # Split data: Reserve prediction_days for testing
            training_data_len = len(data_featured) - prediction_days
            train_data = data_featured[:training_data_len]
            test_data = data_featured[training_data_len-lookback_period:]
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_test_data = scaler.transform(test_data)
            
        # 3. Create Training Sequences
        with st.spinner("Creating training sequences..."):
            X_train, y_train = create_sequences(scaled_train_data, lookback_period)
            
            if len(X_train) < 10:
                st.error(f"Not enough training sequences! Only {len(X_train)} sequences created. Consider reducing the lookback period or increasing total days.")
                st.stop()
            

        # 4. Build and Train the LSTM Model
        with st.spinner("Building and training the LSTM model... This may take a moment."):
            model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Training progress
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Custom training loop to show progress
            for epoch in range(epochs):
                history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                loss = history.history['loss'][0]
                status_text.text(f"Training... Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")

            progress_bar.empty()
            status_text.success(f"Model training completed! Final loss: {loss:.6f}")

        # 5. Create Test Sequences and Make Predictions
        with st.spinner(f"Making predictions for {prediction_days} days..."):
            X_test, y_test_actual_scaled = create_sequences(scaled_test_data, lookback_period)
            
            if len(X_test) == 0:
                st.error("No test sequences could be created. Please adjust your parameters.")
                st.stop()
                
            predictions_scaled = model.predict(X_test)

            # Inverse transform the predictions
            dummy_predictions = np.zeros((len(predictions_scaled), num_features))
            dummy_predictions[:, 3] = predictions_scaled.flatten()
            predictions = scaler.inverse_transform(dummy_predictions)[:, 3]

            # Get the actual prices for the test period
            actual_prices = data['Close'].values[training_data_len:]

        # 6. Calculate Metrics
        st.subheader("Model Performance Metrics")
        
        # Calculate various metrics
        rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
        mae = np.mean(np.abs(actual_prices - predictions))
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        accuracy = 100 - mape
        
        # Directional Accuracy
        actual_direction = np.diff(actual_prices) > 0
        predicted_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        # R-squared
        ss_res = np.sum((actual_prices - predictions) ** 2)
        ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("RMSE", f"${rmse:.2f}", help="Root Mean Squared Error - Lower is better")
        col2.metric("MAE", f"${mae:.2f}", help="Mean Absolute Error - Lower is better")
        col3.metric("Prediction Accuracy", f"{accuracy:.2f}%", help="100% - MAPE (Mean Absolute Percentage Error)")
        col4.metric("Directional Accuracy", f"{directional_accuracy:.2f}%", help="Percentage of correct up/down predictions")
        
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("R² Score", f"{r2_score:.4f}", help="Coefficient of determination - Closer to 1 is better")
        col6.metric("MAPE", f"{mape:.2f}%", help="Mean Absolute Percentage Error - Lower is better")
        col7.metric("Training Samples", f"{len(X_train)}", help="Number of sequences used for training")
        col8.metric("Test Samples", f"{len(X_test)}", help="Number of sequences used for testing")

        # 7. Visualize the Results
        st.subheader(f"Actual vs. Predicted Prices (Last {prediction_days} Days)")
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Main price plot
        plot_dates = data['Date'][training_data_len:].reset_index(drop=True)

        ax1.plot(plot_dates, actual_prices, label='Actual Price', color='blue', marker='o', markersize=4, linewidth=2)
        ax1.plot(plot_dates, predictions, label='Predicted Price', color='red', linestyle='--', marker='x', markersize=4, linewidth=2)

        # Use plot_dates instead of plot_dates.index
        ax1.fill_between(plot_dates, actual_prices, predictions, 
                        where=(actual_prices >= predictions), interpolate=True, alpha=0.3, color='green', label='Underestimation')
        ax1.fill_between(plot_dates, actual_prices, predictions, 
                        where=(actual_prices < predictions), interpolate=True, alpha=0.3, color='red', label='Overestimation')

        ax1.set_title(f'{stock_name} ({symbol}) - Price Prediction ({prediction_days} Days)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.legend(loc='best')
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        # Error plot
        errors = predictions - actual_prices
        ax2.bar(plot_dates, errors, color=['green' if e < 0 else 'red' for e in errors], alpha=0.6)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('Prediction Errors (Predicted - Actual)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylabel('Error (USD)', fontsize=10)
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)

        # 8. Display Prediction Data Table
        st.subheader("Detailed Prediction Data")
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'Date': data['Date'][training_data_len:].dt.strftime('%Y-%m-%d'),
            'Actual Price': actual_prices,
            'Predicted Price': predictions,
            'Difference ($)': predictions - actual_prices,
            'Difference (%)': ((predictions - actual_prices) / actual_prices) * 100,
        })
        
        # Style the dataframe
        styled_df = prediction_df.style.format({
            'Actual Price': '${:,.2f}',
            'Predicted Price': '${:,.2f}',
            'Difference ($)': '{:+,.2f}',
            'Difference (%)': '{:+,.2f}%'
        }).background_gradient(subset=['Difference (%)'], cmap='RdYlGn_r', vmin=-10, vmax=10)
        
        st.dataframe(styled_df, use_container_width=True)
