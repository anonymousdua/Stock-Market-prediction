import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pmdarima as pm
import pandas_ta as ta
from sentimentTest import batch_get_sentiments 

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# --- App Title and Description ---
st.title("Advanced Stock Price Predictor with Multiple Models")
st.markdown("This app uses multiple machine learning models including LSTM, xLSTM, SVM, ARIMA, SARIMA, and ensemble methods (Stacking & Voting) to predict stock prices.1")

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
    index=0,
    help="Select a stock from the list of top 10 popular companies."
)
symbol = POPULAR_STOCKS[stock_name]

# --- Model Parameters ---
st.sidebar.header("Model Parameters")

# Fixed lookback period options
lookback_period = st.sidebar.selectbox(
    "Lookback Period (Days)",
    options=[30, 90, 120],
    index=1,
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

# SVM Kernel selection
svm_kernel = st.sidebar.selectbox(
    "SVM Kernel",
    options=["rbf", "linear", "poly"],
    index=0,
    help="Kernel type for SVM model"
)

# ARIMA/SARIMA settings
auto_arima = st.sidebar.checkbox(
    "Use Auto ARIMA/SARIMA",
    value=True,
    help="Automatically find the best parameters for ARIMA/SARIMA"
)

if not auto_arima:
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        arima_p = st.slider("ARIMA p (AR term)", 0, 5, 1)
    with col2:
        arima_d = st.slider("ARIMA d (Differencing)", 0, 2, 1)
    with col3:
        arima_q = st.slider("ARIMA q (MA term)", 0, 5, 1)
    
    seasonal_period = st.sidebar.slider("Seasonal Period", 5, 30, 7)

# --- Prediction Parameters ---
st.sidebar.header("Prediction Settings")
prediction_days = st.sidebar.selectbox(
    "Days to Predict",
    options=[15, 30, 45],
    index=1,
    help="Number of days to predict and evaluate."
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
            st.error(f"No data found for symbol {symbol}. Please try another stock.")
            return None
        
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(days_to_fetch)

        if len(data) < days_to_fetch:
            st.warning(f"Only found {len(data)} trading days of data (requested {days_to_fetch}). Results may be inaccurate.")
            if len(data) < days_to_fetch * 0.8:
                 st.error(f"Significantly less data found ({len(data)}) than requested ({days_to_fetch}). Stopping.")
                 return None

        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
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
        X.append(data[i-lookback:i].flatten())  # Flatten for SVM
        y.append(data[i, 3])
    return np.array(X), np.array(y)

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
    """Enhanced LSTM with Bidirectional layers"""
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

def fit_arima_model(train_data, auto=True, p=1, d=1, q=1):
    if auto:
        model = pm.auto_arima(train_data, 
                              seasonal=False, 
                              stepwise=True,
                              suppress_warnings=True, 
                              error_action='ignore')
        return model
    else:
        model = ARIMA(train_data, order=(p, d, q))
        return model.fit()

def fit_sarima_model(train_data, auto=True, p=1, d=1, q=1, seasonal_period=7):
    if auto:
        model = pm.auto_arima(train_data, 
                              seasonal=True, 
                              m=seasonal_period,
                              stepwise=True,
                              suppress_warnings=True, 
                              error_action='ignore')
        return model
    else:
        model = SARIMAX(train_data, 
                        order=(p, d, q), 
                        seasonal_order=(p, d, q, seasonal_period))
        return model.fit(disp=False)

# --- Main Application Logic ---
if st.sidebar.button("Run Prediction", type="primary"):
    selected_models = ["LSTM", "XLSTM", "SVM", "ARIMA", "SARIMA"]
    
    # --- Calculate Required Data ---
    DAYS_LOST_TO_TA = 33 
    MIN_TRAINING_SAMPLES = 50
    total_clean_days_needed = (lookback_period + MIN_TRAINING_SAMPLES) + prediction_days
    total_days_to_fetch = total_clean_days_needed + DAYS_LOST_TO_TA
    
    st.sidebar.info(f"Fetching {total_days_to_fetch} trading days:\n"
                    f"- {prediction_days} (for prediction)\n"
                    f"- {lookback_period} (for lookback)\n"
                    f"- {MIN_TRAINING_SAMPLES} (min training)\n"
                    f"- {DAYS_LOST_TO_TA} (for TA warmup)")
    
    with st.spinner(f"Fetching {total_days_to_fetch} days of data for {stock_name}..."):
        data = fetch_stock_data(symbol, total_days_to_fetch)
        
    if data is not None:
        with st.spinner("Adding features (Sentiment, SMAs, TA)..."):
            data = data.reset_index()
            
            # Add Sentiment
            dates_list = data["Date"].dt.strftime('%Y-%m-%d').tolist()
            sentiment_results = batch_get_sentiments(symbol, dates_list)
            data["Sentiment"] = data["Date"].dt.strftime('%Y-%m-%d').map(sentiment_results)
            
            # Add SMAs
            data['SMA10'] = data['Close'].rolling(window=10).mean()
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            
            # Add Technical Indicators
            data.ta.ema(length=12, append=True)
            data.ta.ema(length=26, append=True)
            data.ta.macd(append=True)
            data.ta.rsi(append=True)
            
            data.dropna(inplace=True)
        
        with st.expander(f"View Recent Data for {stock_name} ({symbol})", expanded=False):
            st.dataframe(data.tail(10))

        # Feature list
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Sentiment', 
                    'SMA10', 'SMA20',
                    'EMA_12', 'EMA_26', 'MACD_12_26_9', 
                    'MACDh_12_26_9', 'MACDs_12_26_9', 'RSI_14']
        num_features = len(features)
        
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            st.error(f"Error: The following features are missing after calculation: {missing_features}")
            st.info("This might happen if 'pandas_ta' is not installed or if data is too short.")
            st.stop()
            
        data_featured = data[features].values
        
        min_data_needed_after_dropna = lookback_period + prediction_days + MIN_TRAINING_SAMPLES
        if len(data_featured) < min_data_needed_after_dropna:
            st.error(f"Not enough data after adding features! Need at least {min_data_needed_after_dropna} days but only have {len(data_featured)}.")
            st.info(f"This was calculated from: {lookback_period} (lookback) + {prediction_days} (predict) + {MIN_TRAINING_SAMPLES} (min training).")
            st.stop()

        # Split data
        training_data_len = len(data_featured) - prediction_days
        train_data = data_featured[:training_data_len]
        test_data = data_featured[training_data_len-lookback_period:]
        
        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_data = scaler.fit_transform(train_data)
        scaled_test_data = scaler.transform(test_data)
        
        # Get actual prices for evaluation
        actual_prices = data['Close'].values[training_data_len:]
        results = {}
        
        # Display model training progress
        st.subheader("Model Training Progress")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_models = len(selected_models)
        
        # --- Model Training and Prediction ---
        
        # LSTM Model
        if "LSTM" in selected_models:
            status_text.text("Training LSTM model...")
            try:
                X_train, y_train = create_sequences(scaled_train_data, lookback_period)
                X_test, _ = create_sequences(scaled_test_data, lookback_period)
                
                if len(X_train) == 0:
                    st.error(f"LSTM Error: Not enough data to create training sequences (Need {lookback_period+1} days, got {len(scaled_train_data)})")
                    raise Exception("Insufficient data for LSTM training sequences.")

                lstm_model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
                for epoch in range(epochs):
                    lstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
                
                predictions_scaled = lstm_model.predict(X_test)
                dummy_predictions = np.zeros((len(predictions_scaled), num_features))
                dummy_predictions[:, 3] = predictions_scaled.flatten()
                lstm_predictions = scaler.inverse_transform(dummy_predictions)[:, 3]
                results["LSTM"] = lstm_predictions
                progress_bar.progress(1/total_models)
            except Exception as e:
                st.error(f"Error in LSTM model: {str(e)}")
                results["LSTM"] = np.zeros(len(actual_prices))
        
        # XLSTM Model
        if "XLSTM" in selected_models:
            status_text.text("Training XLSTM model...")
            try:
                X_train, y_train = create_sequences(scaled_train_data, lookback_period)
                X_test, _ = create_sequences(scaled_test_data, lookback_period)
                
                if len(X_train) == 0:
                    st.error(f"XLSTM Error: Not enough data to create training sequences (Need {lookback_period+1} days, got {len(scaled_train_data)})")
                    raise Exception("Insufficient data for XLSTM training sequences.")

                xlstm_model = build_xlstm_model((X_train.shape[1], X_train.shape[2]))
                for epoch in range(epochs):
                    xlstm_model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0)
                
                predictions_scaled = xlstm_model.predict(X_test)
                dummy_predictions = np.zeros((len(predictions_scaled), num_features))
                dummy_predictions[:, 3] = predictions_scaled.flatten()
                xlstm_predictions = scaler.inverse_transform(dummy_predictions)[:, 3]
                results["XLSTM"] = xlstm_predictions
                progress_bar.progress(2/total_models)
            except Exception as e:
                st.error(f"Error in XLSTM model: {str(e)}")
                results["XLSTM"] = np.zeros(len(actual_prices))
        
        # SVM Model
        if "SVM" in selected_models:
            status_text.text("Training SVM model...")
            try:
                X_train_svm, y_train_svm = prepare_svm_data(scaled_train_data, lookback_period)
                X_test_svm, _ = prepare_svm_data(scaled_test_data, lookback_period)
                
                if len(X_train_svm) == 0:
                    st.error(f"SVM Error: Not enough data to create training sequences (Need {lookback_period+1} days, got {len(scaled_train_data)})")
                    raise Exception("Insufficient data for SVM training sequences.")

                svm_model = build_svm_model(svm_kernel)
                svm_model.fit(X_train_svm, y_train_svm)
                
                predictions_scaled = svm_model.predict(X_test_svm)
                dummy_predictions = np.zeros((len(predictions_scaled), num_features))
                dummy_predictions[:, 3] = predictions_scaled
                svm_predictions = scaler.inverse_transform(dummy_predictions)[:, 3]
                results["SVM"] = svm_predictions
                progress_bar.progress(3/total_models)
            except Exception as e:
                st.error(f"Error in SVM model: {str(e)}")
                results["SVM"] = np.zeros(len(actual_prices))
        
        # ARIMA Model
        if "ARIMA" in selected_models:
            status_text.text("Training ARIMA model...")
            try:
                close_prices = data['Close'].values
                train_arima = close_prices[:training_data_len]
                
                if auto_arima:
                    arima_model = fit_arima_model(train_arima, auto=True)
                    st.sidebar.info(f"Auto ARIMA selected order: {arima_model.order}")
                else:
                    arima_model = fit_arima_model(train_arima, auto=False, p=arima_p, d=arima_d, q=arima_q)
                
                arima_predictions = arima_model.predict(n_periods=prediction_days)
                results["ARIMA"] = arima_predictions
                progress_bar.progress(4/total_models)
            except Exception as e:
                st.error(f"Error in ARIMA model: {str(e)}")
                results["ARIMA"] = np.zeros(len(actual_prices))
        
        # SARIMA Model
        if "SARIMA" in selected_models:
            status_text.text("Training SARIMA model...")
            try:
                close_prices = data['Close'].values
                train_sarima = close_prices[:training_data_len]
                
                seasonal_period_val = 7
                if not auto_arima:
                    seasonal_period_val = seasonal_period
                
                if auto_arima:
                    sarima_model = fit_sarima_model(train_sarima, auto=True, seasonal_period=seasonal_period_val)
                    st.sidebar.info(f"Auto SARIMA selected order: {sarima_model.order}, seasonal order: {sarima_model.seasonal_order}")
                else:
                    sarima_model = fit_sarima_model(train_sarima, auto=False, p=arima_p, d=arima_d, q=arima_q, seasonal_period=seasonal_period_val)
                
                sarima_predictions = sarima_model.predict(n_periods=prediction_days)
                results["SARIMA"] = sarima_predictions
                progress_bar.progress(5/total_models)
            except Exception as e:
                st.error(f"Error in SARIMA model: {str(e)}")
                results["SARIMA"] = np.zeros(len(actual_prices))

        # --- ENSEMBLE MODELS: Stacking & Voting ---
        valid_models = {k: v for k, v in results.items() if len(v) == len(actual_prices) and not np.all(v == 0)}
        
        if len(valid_models) >= 2:
            ensemble_input = np.column_stack([pred for pred in valid_models.values()])
            actual_for_ensemble = actual_prices[:ensemble_input.shape[0]]

            # Stacking Ensemble
            try:
                meta_model = LinearRegression()
                meta_model.fit(ensemble_input, actual_for_ensemble)
                stacking_pred = meta_model.predict(ensemble_input)
                results["Stacking"] = stacking_pred
            except Exception as e:
                st.error(f"Stacking failed: {e}")
                results["Stacking"] = np.zeros(len(actual_prices))

            # Voting Ensemble (Averaging)
            try:
                voting_pred = np.mean(ensemble_input, axis=1)
                results["Voting"] = voting_pred
            except Exception as e:
                st.error(f"Voting failed: {e}")
                results["Voting"] = np.zeros(len(actual_prices))
        else:
            st.warning("Not enough valid models to create ensembles.")

        progress_bar.empty()
        status_text.success("All models (including ensembles) trained successfully!")

        # --- Model Comparison and Visualization ---
        st.subheader("Model Performance Comparison")
        
        # Prepare color map
        base_colors = ['blue', 'red', 'green', 'orange', 'purple']
        all_model_names = list(results.keys())
        color_map = {}
        for i, name in enumerate(all_model_names):
            if name == "Stacking":
                color_map[name] = 'gold'
            elif name == "Voting":
                color_map[name] = 'cyan'
            else:
                color_map[name] = base_colors[i % len(base_colors)]
        
        # Calculate metrics
        metrics_data = []
        for model_name, predictions in results.items():
            if len(predictions) > len(actual_prices):
                predictions = predictions[:len(actual_prices)]
                actual_trimmed = actual_prices
            elif len(predictions) < len(actual_prices):
                actual_trimmed = actual_prices[:len(predictions)]
            else:
                actual_trimmed = actual_prices
            
            if len(predictions) == 0 or np.all(predictions == 0):
                st.warning(f"Model {model_name} produced no valid predictions. Skipping.")
                continue
            
            rmse = np.sqrt(mean_squared_error(actual_trimmed, predictions))
            mae = mean_absolute_error(actual_trimmed, predictions)
            mape = np.mean(np.abs((actual_trimmed - predictions) / actual_trimmed)) * 100
            accuracy = 100 - mape
            
            if len(actual_trimmed) > 1:
                actual_direction = np.diff(actual_trimmed) > 0
                predicted_direction = np.diff(predictions) > 0
                directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            else:
                directional_accuracy = 0
            
            metrics_data.append({
                'Model': model_name,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'Accuracy': accuracy,
                'Directional Accuracy': directional_accuracy
            })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(f"{symbol}_metrics.csv", index=False)
            
            # Display metrics in columns
            cols = st.columns(len(metrics_data))
            for idx, (col, metric_row) in enumerate(zip(cols, metrics_data)):
                with col:
                    st.metric(
                        label=metric_row['Model'],
                        value=f"{metric_row['Accuracy']:.1f}%",
                        delta=f"RMSE: ${metric_row['RMSE']:.2f}",
                        help=f"{metric_row['Model']} - Accuracy: {metric_row['Accuracy']:.1f}%"
                    )
            
            # Detailed metrics table
            st.dataframe(metrics_df.style.format({
                'RMSE': '${:.2f}',
                'MAE': '${:.2f}',
                'MAPE': '{:.2f}%',
                'Accuracy': '{:.2f}%',
                'Directional Accuracy': '{:.2f}%'
            }), use_container_width=True)

            # Visualization
            st.subheader(f"Actual vs. Predicted Prices (Last {prediction_days} Days)")
            fig, ax1 = plt.subplots(figsize=(14, 8))
            plot_dates = data['Date'][training_data_len:].reset_index(drop=True)
            ax1.plot(plot_dates, actual_prices, label='Actual Price', color='black', linewidth=3, marker='o', markersize=4)
            
            for model_name, predictions in results.items():
                if model_name not in color_map or len(predictions) != len(plot_dates):
                    continue
                ax1.plot(plot_dates, predictions, label=f'{model_name} Prediction', 
                         color=color_map[model_name], linestyle='--', linewidth=2, marker='x', markersize=3)
            
            ax1.set_title(f'{stock_name} ({symbol}) - Price Prediction Comparison\n(Lookback: {lookback_period} days, Prediction: {prediction_days} days)', 
                          fontsize=16, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Price (USD)', fontsize=12)
            ax1.legend(loc='best')
            ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

            # Best model recommendation
            best_model = max(metrics_data, key=lambda x: (x['Directional Accuracy'], x['Accuracy']))
            best_name = best_model['Model']
            if best_name in ["Stacking", "Voting"]:
                st.success(f"**Best Performing Model**: **{best_name} Ensemble** "
                           f"(Directional Accuracy: {best_model['Directional Accuracy']:.2f}%, "
                           f"Accuracy: {best_model['Accuracy']:.2f}%)")
            else:
                st.success(f"**Best Performing Model**: {best_model['Model']} "
                           f"(Directional Accuracy: {best_model['Directional Accuracy']:.2f}%, "
                           f"Accuracy: {best_model['Accuracy']:.2f}%)")

            # Detailed prediction table for the best model
            st.subheader("Detailed Prediction Data (Best Model)")
            best_predictions = results[best_name]
            if len(best_predictions) > len(actual_prices):
                best_predictions = best_predictions[:len(actual_prices)]
            
            prediction_df = pd.DataFrame({
                'Date': data['Date'][training_data_len:].dt.strftime('%Y-%m-%d').values[:len(best_predictions)],
                'Actual Price': actual_prices[:len(best_predictions)],
                f'Predicted Price ({best_name})': best_predictions,
                'Difference ($)': best_predictions - actual_prices[:len(best_predictions)],
                'Difference (%)': ((best_predictions - actual_prices[:len(best_predictions)]) / actual_prices[:len(best_predictions)]) * 100,
            })
            
            styled_df = prediction_df.style.format({
                'Actual Price': '${:,.2f}',
                f'Predicted Price ({best_name})': '${:,.2f}',
                'Difference ($)': '{:+,.2f}',
                'Difference (%)': '{:+,.2f}%'
            })
            
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.error("No models produced valid predictions. Please check the parameters and try again.")