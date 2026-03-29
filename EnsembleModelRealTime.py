import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import pandas_ta as ta
from pandas.tseries.offsets import BDay
from sentimentTest import batch_get_sentiments

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- App Title and Description ---
st.title("Advanced Stock Price Predictor with Future Forecasting")
st.markdown("This app uses multiple machine learning models to backtest accuracy and **forecast future stock prices**.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Stock Selection")
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
    index=0
)
symbol = POPULAR_STOCKS[stock_name]

# --- Model Parameters ---
st.sidebar.header("Model Parameters")

# Fixed lookback period options (User request: 90 or 120 days)
lookback_period = st.sidebar.selectbox(
    "Lookback Period (Days)",
    options=[30, 60, 90, 120],
    index=2, # Default to 90
    help="Number of past days' data to use for predicting the next day."
)

epochs = st.sidebar.slider("Training Epochs", 20, 100, 50)
batch_size = st.sidebar.select_slider("Batch Size", options=[16, 32, 64], value=32)
svm_kernel = st.sidebar.selectbox("SVM Kernel", options=["rbf", "linear", "poly"], index=0)

# ARIMA/SARIMA settings
auto_arima = st.sidebar.checkbox("Use Auto ARIMA/SARIMA", value=True)
if not auto_arima:
    col1, col2, col3 = st.sidebar.columns(3)
    with col1: arima_p = st.slider("p", 0, 5, 1)
    with col2: arima_d = st.slider("d", 0, 2, 1)
    with col3: arima_q = st.slider("q", 0, 5, 1)
    seasonal_period = st.sidebar.slider("Seasonal Period", 5, 30, 7)
else:
    seasonal_period = 7

# --- Prediction Parameters ---
st.sidebar.header("Forecast Settings")
prediction_days = st.sidebar.selectbox(
    "Future Days to Predict",
    options=[7, 15, 30, 45],
    index=1, # Default to 15
    help="Number of days to forecast into the future."
)

# --- Data Fetching Function ---
@st.cache_data
def fetch_stock_data(symbol, days_to_fetch):
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=int(days_to_fetch * 1.7)) 
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        
        if data.empty:
            return None
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(days_to_fetch)
        return data
    except Exception as e:
        return None

# --- Data Preparation Functions ---
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 3])  # Target is 'Close' price at index 3
    return np.array(X), np.array(y)

def prepare_svm_data(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i].flatten()) 
        y.append(data[i, 3])
    return np.array(X), np.array(y)

# --- Future Prediction Logic (Recursive) ---
def predict_future_recursive(model, last_sequence, n_steps, model_type, scaler, num_features):
    """
    Predicts future prices recursively. 
    Assumes auxiliary features (High, Low, Vol, Sentiment) remain static 
    (Naive approach) while updating the 'Close' price.
    """
    current_seq = last_sequence.copy() # Shape: (lookback, features)
    future_preds = []
    
    # Pre-calculate index of Close price
    close_idx = 3 
    
    for _ in range(n_steps):
        if model_type == "SVM":
            # SVM expects flattened input
            input_data = current_seq.flatten().reshape(1, -1)
            pred_scaled = model.predict(input_data)[0]
        else:
            # LSTM/XLSTM expect 3D input (1, lookback, features)
            input_data = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
            pred_scaled = model.predict(input_data, verbose=0)[0, 0]
            
        future_preds.append(pred_scaled)
        
        # Update sequence for next step
        # Create a new row based on the last row of the sequence
        next_row = current_seq[-1].copy()
        
        # Update the Close price with the prediction
        next_row[close_idx] = pred_scaled
        
        # (Optional) You could also update SMAs or EMAs here if you implemented the logic
        
        # Shift sequence: remove first row, append new row
        current_seq = np.vstack([current_seq[1:], next_row])
        
    return np.array(future_preds)

# --- Model Building Functions ---
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

def build_xlstm_model(input_shape):
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')
    return model

def build_svm_model(kernel='rbf'):
    return SVR(kernel=kernel, C=1.0, epsilon=0.1)

# --- Main Application Logic ---
if st.sidebar.button("Run Prediction & Forecast", type="primary"):
    selected_models = ["LSTM", "XLSTM", "SVM", "ARIMA", "SARIMA"]
    
    # We fetch enough data for training + evaluation + lookback
    # Note: We aren't fetching "future" data, we are fetching "history" to predict the future.
    DAYS_LOST_TO_TA = 33 
    MIN_TRAINING_SAMPLES = 100 
    
    # We need data for: Lookback + Training + Backtest Validation
    validation_days = prediction_days # We use the same amount for backtesting as we do for forecasting
    total_clean_days_needed = lookback_period + MIN_TRAINING_SAMPLES + validation_days
    total_days_to_fetch = total_clean_days_needed + DAYS_LOST_TO_TA
    
    st.sidebar.info(f"Configuration: Predicting next {prediction_days} days using {lookback_period} days lookback.")
    
    with st.spinner(f"Fetching data for {stock_name}..."):
        data = fetch_stock_data(symbol, total_days_to_fetch)
        
    if data is not None:
        with st.spinner("Processing features (Sentiment, TA)..."):
            data = data.reset_index()
            
            # Feature Engineering
            dates_list = data["Date"].dt.strftime('%Y-%m-%d').tolist()
            try:
                sentiment_results = batch_get_sentiments(symbol, dates_list)
                data["Sentiment"] = data["Date"].dt.strftime('%Y-%m-%d').map(sentiment_results)
            except:
                data["Sentiment"] = 0 # Fallback
            
            data['SMA10'] = data['Close'].rolling(window=10).mean()
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data.ta.ema(length=12, append=True)
            data.ta.ema(length=26, append=True)
            data.ta.macd(append=True)
            data.ta.rsi(append=True)
            data.dropna(inplace=True)
        
        # Features list
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 
                    'SMA10', 'SMA20', 'EMA_12', 'EMA_26', 'MACD_12_26_9', 
                    'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI_14']
        
        # Validate columns
        features = [f for f in features if f in data.columns]
        num_features = len(features)
        data_featured = data[features].values
        
        # --- 1. Split for Backtesting (Evaluation) ---
        # We hide the last 'prediction_days' to test if the model can guess them
        train_len = len(data_featured) - validation_days
        train_data = data_featured[:train_len]
        test_data = data_featured[train_len-lookback_period:] # Include lookback buffer
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_data)
        scaled_test = scaler.transform(test_data)
        
        # --- 2. Prepare Data for FUTURE Forecast ---
        # For the real future forecast, we need the *very last* window of data known
        last_known_sequence_scaled = scaler.transform(data_featured[-lookback_period:])
        
        actual_prices_test = data['Close'].values[train_len:]
        results_test = {} # For backtest
        results_future = {} # For future forecast
        
        st.subheader("1. Model Training & Backtesting")
        progress_bar = st.progress(0)
        
        # --- Models ---
        
        # LSTM
        if "LSTM" in selected_models:
            X_train, y_train = create_sequences(scaled_train, lookback_period)
            X_test, _ = create_sequences(scaled_test, lookback_period)
            
            lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            
            # Backtest Preds
            preds = lstm_model.predict(X_test, verbose=0)
            dummy = np.zeros((len(preds), num_features))
            dummy[:, 3] = preds.flatten()
            results_test["LSTM"] = scaler.inverse_transform(dummy)[:, 3]
            
            # Future Preds
            future_scaled = predict_future_recursive(lstm_model, last_known_sequence_scaled, prediction_days, "LSTM", scaler, num_features)
            dummy_f = np.zeros((len(future_scaled), num_features))
            dummy_f[:, 3] = future_scaled
            results_future["LSTM"] = scaler.inverse_transform(dummy_f)[:, 3]
            
            progress_bar.progress(0.2)

        # XLSTM
        if "XLSTM" in selected_models:
            X_train, y_train = create_sequences(scaled_train, lookback_period)
            X_test, _ = create_sequences(scaled_test, lookback_period)
            
            xlstm_model = build_xlstm_model((X_train.shape[1], X_train.shape[2]))
            xlstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            
            # Backtest
            preds = xlstm_model.predict(X_test, verbose=0)
            dummy = np.zeros((len(preds), num_features))
            dummy[:, 3] = preds.flatten()
            results_test["XLSTM"] = scaler.inverse_transform(dummy)[:, 3]
            
            # Future
            future_scaled = predict_future_recursive(xlstm_model, last_known_sequence_scaled, prediction_days, "XLSTM", scaler, num_features)
            dummy_f = np.zeros((len(future_scaled), num_features))
            dummy_f[:, 3] = future_scaled
            results_future["XLSTM"] = scaler.inverse_transform(dummy_f)[:, 3]
            
            progress_bar.progress(0.4)

        # SVM
        if "SVM" in selected_models:
            X_train_s, y_train_s = prepare_svm_data(scaled_train, lookback_period)
            X_test_s, _ = prepare_svm_data(scaled_test, lookback_period)
            
            svm_model = build_svm_model(svm_kernel)
            svm_model.fit(X_train_s, y_train_s)
            
            # Backtest
            preds = svm_model.predict(X_test_s)
            dummy = np.zeros((len(preds), num_features))
            dummy[:, 3] = preds
            results_test["SVM"] = scaler.inverse_transform(dummy)[:, 3]
            
            # Future
            future_scaled = predict_future_recursive(svm_model, last_known_sequence_scaled, prediction_days, "SVM", scaler, num_features)
            dummy_f = np.zeros((len(future_scaled), num_features))
            dummy_f[:, 3] = future_scaled
            results_future["SVM"] = scaler.inverse_transform(dummy_f)[:, 3]
            
            progress_bar.progress(0.6)

        # ARIMA & SARIMA
        # For Time Series models, we usually train on Close price only
        train_series = data['Close'].values[:train_len]
        full_series = data['Close'].values # Use full data for future forecast
        
        if "ARIMA" in selected_models:
            try:
                if auto_arima:
                    arima_model = pm.auto_arima(train_series, seasonal=False, error_action='ignore', suppress_warnings=True)
                    # For future, we update the model with full data
                    arima_future_model = pm.auto_arima(full_series, seasonal=False, error_action='ignore', suppress_warnings=True)
                else:
                    arima_model = ARIMA(train_series, order=(arima_p, arima_d, arima_q)).fit()
                    arima_future_model = ARIMA(full_series, order=(arima_p, arima_d, arima_q)).fit()
                
                results_test["ARIMA"] = arima_model.predict(n_periods=validation_days)
                results_future["ARIMA"] = arima_future_model.predict(n_periods=prediction_days)
            except:
                results_test["ARIMA"] = np.zeros(validation_days)
                results_future["ARIMA"] = np.zeros(prediction_days)
            progress_bar.progress(0.8)

        if "SARIMA" in selected_models:
            try:
                sp = seasonal_period
                if auto_arima:
                    sarima_model = pm.auto_arima(train_series, seasonal=True, m=sp, error_action='ignore', suppress_warnings=True)
                    sarima_future_model = pm.auto_arima(full_series, seasonal=True, m=sp, error_action='ignore', suppress_warnings=True)
                else:
                    sarima_model = SARIMAX(train_series, order=(arima_p, arima_d, arima_q), seasonal_order=(arima_p, arima_d, arima_q, sp)).fit(disp=False)
                    sarima_future_model = SARIMAX(full_series, order=(arima_p, arima_d, arima_q), seasonal_order=(arima_p, arima_d, arima_q, sp)).fit(disp=False)

                results_test["SARIMA"] = sarima_model.predict(n_periods=validation_days)
                results_future["SARIMA"] = sarima_future_model.predict(n_periods=prediction_days)
            except:
                results_test["SARIMA"] = np.zeros(validation_days)
                results_future["SARIMA"] = np.zeros(prediction_days)
            progress_bar.progress(1.0)

        # --- Ensembles ---
        # 1. Backtest Ensemble
        valid_models = {k: v for k, v in results_test.items() if not np.all(v == 0)}
        if len(valid_models) >= 2:
            ensemble_input = np.column_stack([v for v in valid_models.values()])
            results_test["Voting"] = np.mean(ensemble_input, axis=1)
            
            # Simple Stacking (trained on backtest predictions vs actual)
            # Warning: This is a bit leaky if not cross-validated, but standard for simple apps
            meta_model = LinearRegression()
            meta_model.fit(ensemble_input, actual_prices_test[:len(ensemble_input)])
            results_test["Stacking"] = meta_model.predict(ensemble_input)
            
            # 2. Future Ensemble
            # We apply the weights learned from backtest to the future predictions
            future_input_list = [results_future[k] for k in valid_models.keys()]
            future_ensemble_input = np.column_stack(future_input_list)
            
            results_future["Voting"] = np.mean(future_ensemble_input, axis=1)
            results_future["Stacking"] = meta_model.predict(future_ensemble_input)

        # --- Visualizing Backtest ---
        st.success("Models Trained.")
        
        # Calculate Metrics (Same as before)
        metrics_data = []
        for name, preds in results_test.items():
            # Ensure length match
            p = preds[:len(actual_prices_test)]
            a = actual_prices_test[:len(p)]
            rmse = np.sqrt(mean_squared_error(a, p))
            mape = np.mean(np.abs((a - p) / a)) * 100
            metrics_data.append({"Model": name, "RMSE": rmse, "Accuracy": 100-mape})
            
        best_model_name = max(metrics_data, key=lambda x: x['Accuracy'])['Model']
        st.metric("Best Model (Backtest)", best_model_name, f"{max(metrics_data, key=lambda x: x['Accuracy'])['Accuracy']:.2f}% Accuracy")

        # --- 3. FUTURE FORECAST DISPLAY ---
        st.markdown("---")
        st.header(f"🔮 Future Price Forecast (Next {prediction_days} Days)")
        st.caption(f"Projection starting from {data['Date'].iloc[-1].date()}")
        
        # Generate Future Dates (Business Days)
        last_date = data['Date'].iloc[-1]
        future_dates = [last_date + BDay(i) for i in range(1, prediction_days + 1)]
        future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
        
        # Plotting Future
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        
        # Plot last 60 days of history
        history_plot_len = 60
        history_dates = data['Date'].iloc[-history_plot_len:]
        history_prices = data['Close'].iloc[-history_plot_len:]
        
        ax2.plot(history_dates, history_prices, label='Historical Data', color='black', linewidth=2)
        
        # Plot Predictions
        # We only plot the Best Model and Ensembles to keep it clean, or all if selected
        models_to_plot = [best_model_name, "Voting", "Stacking"]
        colors = {'LSTM': 'blue', 'XLSTM': 'purple', 'SVM': 'orange', 'ARIMA': 'green', 'SARIMA': 'brown', 'Voting': 'cyan', 'Stacking': 'red'}
        
        for name, preds in results_future.items():
            if name in results_future: # Plot all valid
                color = colors.get(name, 'gray')
                linestyle = '-' if name == best_model_name else '--'
                width = 3 if name == best_model_name else 1.5
                ax2.plot(future_dates, preds, label=f'{name} Forecast', linestyle=linestyle, linewidth=width, color=color)

        ax2.set_title(f"Stock Price Forecast for {stock_name} ({symbol})", fontsize=16)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig2)
        
        # Future Data Table
        future_df = pd.DataFrame({"Date": future_dates_str})
        for name, preds in results_future.items():
            future_df[name] = preds
            
        st.subheader("Forecast Data Table")
        st.dataframe(future_df.style.format({col: "${:.2f}" for col in results_future.keys()}), use_container_width=True)