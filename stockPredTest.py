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
from sentimentTest import batch_get_sentiments  # Assuming sentiment.py is in the same directory

warnings.filterwarnings('ignore')

# --- Page Configuration ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- App Title and Description ---
st.title("LSTM Stock Price Predictor")
st.markdown("""
This application predicts the closing price of a stock for the next 30 days.
- It uses the last 120 days of stock data from Yahoo Finance.
- The first 90 days are used to **train** a Long Short-Term Memory (LSTM) model.
- The last 30 days are used to **test** the model's performance.
- Key metrics like **RMSE, Accuracy, and Directional Accuracy** are calculated on the test data.
""")

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
    max_value=80,
    value=60,
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


# --- Data Fetching Function ---
@st.cache_data
def fetch_stock_data(symbol):
    """
    Fetches the last 120 days of stock data for the given symbol.
    """
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=180) # Fetch more to ensure we get 120 trading days
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"No data found for symbol {symbol}. Please try another stock.")
            return None
        
        # Select relevant columns and take the last 120 trading days
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(120)

        if len(data) < 120:
            st.warning(f"Only found {len(data)} trading days of data. The model might be less accurate.")

        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

# --- Data Preparation Function ---
def create_sequences(data, lookback):
    """
    Creates sequences of data for LSTM model training and testing.
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 3])  # The target is the 'Close' price, which is at index 3
    return np.array(X), np.array(y)

# --- LSTM Model Building Function ---
def build_lstm_model(input_shape):
    """
    Builds and compiles the LSTM model.
    """
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
if st.sidebar.button("Run Prediction"):
    with st.spinner(f"Fetching data for {stock_name}..."):
        data = fetch_stock_data(symbol)
        data = data.reset_index()
        with st.spinner("Fetching sentiment data..."):
            dates_list = data["Date"].dt.strftime('%Y-%m-%d').tolist()
            sentiment_results = batch_get_sentiments(symbol, dates_list)
            data["Sentiment"] = data["Date"].dt.strftime('%Y-%m-%d').map(sentiment_results)

    if data is not None:
        st.subheader(f"Recent Data for {stock_name} ({symbol})")
        st.dataframe(data.tail())

        # 1. Data Splitting and Scaling
        with st.spinner("Preparing data and scaling..."):
            # Use all 5 features for scaling
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment']
            data_featured = data[features].values

            # Split data into 90 days for training and 30 for testing
            training_data_len = 90
            train_data = data_featured[:training_data_len]
            test_data = data_featured[training_data_len-lookback_period:]
            print(data)

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_train_data = scaler.fit_transform(train_data)
            scaled_test_data = scaler.transform(test_data)

        # 2. Create Training Sequences
        with st.spinner("Creating training sequences..."):
            X_train, y_train = create_sequences(scaled_train_data, lookback_period)

        # 3. Build and Train the LSTM Model
        with st.spinner("Building and training the LSTM model... This may take a moment."):
            model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            # Custom training loop to show progress
            for epoch in range(epochs):
                model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
                progress = (epoch + 1) / epochs
                progress_bar.progress(progress)
                status_text.text(f"Training... Epoch {epoch + 1}/{epochs}")

            progress_bar.empty()
            status_text.empty()

        # 4. Create Test Sequences and Make Predictions
        with st.spinner("Making predictions on the test data..."):
            X_test, y_test_actual_scaled = create_sequences(scaled_test_data, lookback_period)
            predictions_scaled = model.predict(X_test)

            # We need to inverse transform the predictions to get the actual price values
            # Create a dummy array with the same shape as the scaler expects (5 features)
            dummy_predictions = np.zeros((len(predictions_scaled), 6))
            dummy_predictions[:, 3] = predictions_scaled.flatten() # Put predictions in the 'Close' column
            
            # Inverse transform the dummy array
            predictions = scaler.inverse_transform(dummy_predictions)[:, 3] # Extract the 'Close' price

            # Get the actual prices for the test period
            actual_prices = data['Close'].values[training_data_len:]

        # 5. Calculate Metrics
        st.subheader("Model Performance on Test Data")
        rmse = np.sqrt(mean_squared_error(actual_prices, predictions))
        mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
        accuracy = 100 - mape
        
        # Directional Accuracy
        actual_direction = np.diff(actual_prices) > 0
        predicted_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == predicted_direction) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Root Mean Squared Error (RMSE)", f"${rmse:.2f}")
        col2.metric("Prediction Accuracy", f"{accuracy:.2f}%")
        col3.metric("Directional Accuracy", f"{directional_accuracy:.2f}%")
        st.caption("Prediction Accuracy is calculated as 100% - MAPE (Mean Absolute Percentage Error).")

        # 6. Visualize the Results
        st.subheader("Actual vs. Predicted Prices (Last 30 Days)")
        
        fig, ax = plt.subplots(figsize=(14, 7))
        plot_dates = data['Date'][training_data_len:]
        ax.plot(plot_dates, actual_prices, label='Actual Price', color='blue', marker='o')
        ax.plot(plot_dates, predictions, label='Predicted Price', color='red', linestyle='--', marker='x')  
        #ax.plot(data.index[training_data_len:], actual_prices, label='Actual Price', color='blue', marker='o')
        #ax.plot(data.index[training_data_len:], predictions, label='Predicted Price', color='red', linestyle='--', marker='x')
        ax.set_title(f'{stock_name} ({symbol}) - Price Prediction', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Price (USD)', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        # 7. Display Prediction Data Table
        st.subheader("Detailed Prediction Data")
        prediction_df = pd.DataFrame({
            'Date': data['Date'][training_data_len:].dt.strftime('%Y-%m-%d'),
            #'Date': data.index[training_data_len:],
            'Actual Price': actual_prices,
            'Predicted Price': predictions,
            'Difference ($)': predictions - actual_prices,
            'Difference (%)': ((predictions - actual_prices) / actual_prices) * 100
        })
        st.dataframe(prediction_df.style.format({
            'Actual Price': '${:,.2f}',
            'Predicted Price': '${:,.2f}',
            'Difference ($)': '{:,.2f}',
            'Difference (%)': '{:,.2f}%'
        }))

else:
    st.info("Click the 'Run Prediction' button in the sidebar to start.")

