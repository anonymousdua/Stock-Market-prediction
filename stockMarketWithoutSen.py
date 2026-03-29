import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ========================= DATA FUNCTIONS =========================

@st.cache_data
def load_stock_data(symbol, period):
    """Load stock data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        info = ticker.info
        return data, info
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return None, None


def prepare_lstm_data(data, sequence_length):
    """Prepare data for LSTM training"""
    # Use closing price
    prices = data['Close'].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i-sequence_length:i, 0])
        y.append(scaled_prices[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


def split_train_test_data(X, y, split_ratio=0.8):
    """Split data into training and testing sets"""
    split_index = int(split_ratio * len(X))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


# ========================= MODEL FUNCTIONS =========================

def create_lstm_model(sequence_length, lstm_units=50, dropout_rate=0.2):
    """Create LSTM model architecture"""
    model = Sequential([
        LSTM(units=lstm_units, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """Custom callback to update Streamlit progress bar during training"""
    def __init__(self, epochs, progress_bar, status_text):
        super().__init__()
        self.epochs = epochs
        self.progress_bar = progress_bar
        self.status_text = status_text
    
    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.epochs
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Training: Epoch {epoch + 1}/{self.epochs} - "
            f"Loss: {logs['loss']:.6f} - Val Loss: {logs['val_loss']:.6f}"
        )


def train_lstm_model(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """Train the LSTM model with progress tracking"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    callback = StreamlitProgressCallback(epochs, progress_bar, status_text)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[callback],
        verbose=0
    )
    
    progress_bar.empty()
    status_text.empty()
    
    return history


def calculate_accuracy_metrics(actual, predicted):
    """Calculate various accuracy metrics in percentage"""
    actual = actual.flatten()
    predicted = predicted.flatten()
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    smape = np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted))) * 100
    
    # Direction Accuracy (percentage of correct trend predictions)
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
    
    # Accuracy based on tolerance (within 5% of actual value)
    tolerance = 0.05  # 5%
    within_tolerance = np.abs((actual - predicted) / actual) <= tolerance
    tolerance_accuracy = np.mean(within_tolerance) * 100
    
    # R-squared (coefficient of determination) as percentage
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = (1 - (ss_res / ss_tot)) * 100 if ss_tot != 0 else 0
    
    return {
        'mape': mape,
        'smape': smape,
        'direction_accuracy': direction_accuracy,
        'tolerance_accuracy': tolerance_accuracy,
        'r2_score': r2
    }


def evaluate_model_performance(model, X_train, y_train, X_test, y_test, scaler):
    """Evaluate model performance and return comprehensive metrics"""
    # Make predictions
    train_predictions = model.predict(X_train, verbose=0)
    test_predictions = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate traditional metrics
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predictions))
    train_mae = mean_absolute_error(y_train_actual, train_predictions)
    test_mae = mean_absolute_error(y_test_actual, test_predictions)
    
    # Calculate accuracy metrics
    train_accuracy = calculate_accuracy_metrics(y_train_actual, train_predictions)
    test_accuracy = calculate_accuracy_metrics(y_test_actual, test_predictions)
    
    metrics = {
        # Traditional metrics
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        
        # Accuracy metrics (percentages)
        'train_mape': train_accuracy['mape'],
        'test_mape': test_accuracy['mape'],
        'train_smape': train_accuracy['smape'],
        'test_smape': test_accuracy['smape'],
        'train_direction_accuracy': train_accuracy['direction_accuracy'],
        'test_direction_accuracy': test_accuracy['direction_accuracy'],
        'train_tolerance_accuracy': train_accuracy['tolerance_accuracy'],
        'test_tolerance_accuracy': test_accuracy['tolerance_accuracy'],
        'train_r2': train_accuracy['r2_score'],
        'test_r2': test_accuracy['r2_score'],
        
        # Store predictions for further analysis
        'train_predictions': train_predictions,
        'test_predictions': test_predictions,
        'y_train_actual': y_train_actual,
        'y_test_actual': y_test_actual
    }
    
    return metrics


# ========================= PREDICTION FUNCTIONS =========================

def make_future_predictions(model, data, scaler, sequence_length, prediction_days):
    """Generate future stock price predictions"""
    # Get the last sequence from training data
    last_sequence = data[-sequence_length:].reshape(1, sequence_length, 1)
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(prediction_days):
        # Predict next day
        next_pred = model.predict(current_sequence, verbose=0)
        predictions.append(next_pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[:, 1:, :], 
                                   next_pred.reshape(1, 1, 1), axis=1)
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()


def generate_future_dates(last_date, prediction_days):
    """Generate future date range for predictions"""
    return pd.date_range(
        start=last_date + timedelta(days=1),
        periods=prediction_days,
        freq='D'
    )


def create_prediction_dataframe(future_dates, predictions):
    """Create DataFrame with predictions for download"""
    return pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions
    })


# ========================= VISUALIZATION FUNCTIONS =========================

def plot_historical_data(data, stock_symbol):
    """Create historical stock price chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue')
    ))
    fig.update_layout(
        title=f"{stock_symbol} Historical Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    return fig


def plot_training_history(history):
    """Create training history chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history['loss'],
        mode='lines',
        name='Training Loss',
        line=dict(color='red')
    ))
    fig.add_trace(go.Scatter(
        y=history.history['val_loss'],
        mode='lines',
        name='Validation Loss',
        line=dict(color='orange')
    ))
    fig.update_layout(
        title="Model Loss During Training",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        height=400
    )
    return fig


def plot_predictions(data, stock_symbol, future_dates, predictions, prediction_years):
    """Create prediction visualization chart"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Historical Prices',
        line=dict(color='blue')
    ))
    
    # Predictions
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predictions,
        mode='lines',
        name=f'{prediction_years}-Year Predictions',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{stock_symbol} Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500
    )
    return fig


# ========================= UI FUNCTIONS =========================

def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="Stock Market LSTM Predictor",
        page_icon="📈",
        layout="wide"
    )


def create_sidebar_inputs():
    """Create sidebar input controls and return their values"""
    st.sidebar.header("Configuration")
    
    # Stock selection
    stock_symbol = st.sidebar.text_input(
        "Enter Stock Symbol (e.g., AAPL, GOOGL, TSLA)", 
        value="AAPL"
    ).upper()
    
    # Prediction range selection
    prediction_years = st.sidebar.selectbox(
        "Select Prediction Range",
        [1, 2, 3, 4],
        index=0
    )
    
    # Historical data period
    data_period = st.sidebar.selectbox(
        "Historical Data Period",
        ["2y", "3y", "5y", "10y"],
        index=2
    )
    
    # LSTM parameters
    st.sidebar.subheader("LSTM Parameters")
    sequence_length = st.sidebar.slider("Sequence Length (days)", 10, 100, 60)
    epochs = st.sidebar.slider("Training Epochs", 10, 100, 10)
    batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64], index=1)
    
    return {
        'stock_symbol': stock_symbol,
        'prediction_years': prediction_years,
        'data_period': data_period,
        'sequence_length': sequence_length,
        'epochs': epochs,
        'batch_size': batch_size
    }


def display_stock_info(info, stock_symbol):
    """Display basic stock information"""
    if info:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Company", info.get('longName', stock_symbol))
        with col2:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
        with col3:
            market_cap = info.get('marketCap')
            if market_cap:
                st.metric("Market Cap", f"${market_cap:,}")
            else:
                st.metric("Market Cap", "N/A")


def display_model_metrics(metrics):
    """Display comprehensive model performance metrics"""
    st.subheader("Model Performance")
    
    # Create tabs for different metric categories
    tab1, tab2, tab3 = st.tabs(["📊 Error Metrics", "🎯 Accuracy Metrics", "📈 Advanced Metrics"])
    
    with tab1:
        st.markdown("**Traditional Error Metrics**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train RMSE", f"${metrics['train_rmse']:.2f}")
        with col2:
            st.metric("Test RMSE", f"${metrics['test_rmse']:.2f}")
        with col3:
            st.metric("Train MAE", f"${metrics['train_mae']:.2f}")
        with col4:
            st.metric("Test MAE", f"${metrics['test_mae']:.2f}")
    
    with tab2:
        st.markdown("**Percentage-based Accuracy Metrics**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Training Accuracy**")
            st.metric("MAPE (Lower is Better)", f"{metrics['train_mape']:.2f}%")
            st.metric("SMAPE (Lower is Better)", f"{metrics['train_smape']:.2f}%")
            st.metric("Direction Accuracy", f"{metrics['train_direction_accuracy']:.2f}%")
            st.metric("Tolerance Accuracy (±5%)", f"{metrics['train_tolerance_accuracy']:.2f}%")
        
        with col2:
            st.markdown("**Testing Accuracy**")
            st.metric("MAPE (Lower is Better)", f"{metrics['test_mape']:.2f}%")
            st.metric("SMAPE (Lower is Better)", f"{metrics['test_smape']:.2f}%")
            st.metric("Direction Accuracy", f"{metrics['test_direction_accuracy']:.2f}%")
            st.metric("Tolerance Accuracy (±5%)", f"{metrics['test_tolerance_accuracy']:.2f}%")
    
    with tab3:
        st.markdown("**R-squared and Model Quality**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train R²", f"{metrics['train_r2']:.2f}%")
        with col2:
            st.metric("Test R²", f"{metrics['test_r2']:.2f}%")
        with col3:
            # Overall model quality assessment
            avg_direction_accuracy = (metrics['train_direction_accuracy'] + metrics['test_direction_accuracy']) / 2
            st.metric("Avg Direction Accuracy", f"{avg_direction_accuracy:.2f}%")
        with col4:
            # Overall tolerance accuracy
            avg_tolerance_accuracy = (metrics['train_tolerance_accuracy'] + metrics['test_tolerance_accuracy']) / 2
            st.metric("Avg Tolerance Accuracy", f"{avg_tolerance_accuracy:.2f}%")
    
    # Model quality interpretation
    st.markdown("---")
    st.markdown("**📋 Model Quality Interpretation:**")
    
    # Determine model quality based on test metrics
    test_direction_acc = metrics['test_direction_accuracy']
    test_tolerance_acc = metrics['test_tolerance_accuracy']
    test_r2 = metrics['test_r2']
    
    quality_score = (test_direction_acc + test_tolerance_acc + max(0, test_r2)) / 3
    
    if quality_score >= 80:
        quality_color = "🟢"
        quality_text = "Excellent"
    elif quality_score >= 70:
        quality_color = "🟡"
        quality_text = "Good"
    elif quality_score >= 60:
        quality_color = "🟠"
        quality_text = "Fair"
    else:
        quality_color = "🔴"
        quality_text = "Needs Improvement"
    
    st.markdown(f"{quality_color} **Overall Model Quality: {quality_text} ({quality_score:.1f}%)**")
    
    # Detailed explanations
    with st.expander("📖 Metric Explanations"):
        st.markdown("""
        **Error Metrics:**
        - **RMSE**: Root Mean Square Error - Average prediction error in dollars
        - **MAE**: Mean Absolute Error - Average absolute prediction error in dollars
        
        **Accuracy Metrics:**
        - **MAPE**: Mean Absolute Percentage Error - Average percentage error (lower is better)
        - **SMAPE**: Symmetric Mean Absolute Percentage Error - Balanced percentage error (lower is better)
        - **Direction Accuracy**: Percentage of correct trend predictions (up/down movements)
        - **Tolerance Accuracy**: Percentage of predictions within ±5% of actual values
        
        **Advanced Metrics:**
        - **R²**: Coefficient of determination - How well the model explains variance (higher is better)
        - **Quality Score**: Combined metric considering direction accuracy, tolerance accuracy, and R²
        
        **Quality Ratings:**
        - 🟢 Excellent (≥80%): Highly reliable predictions
        - 🟡 Good (≥70%): Reliable with some limitations
        - 🟠 Fair (≥60%): Moderately reliable, use with caution
        - 🔴 Needs Improvement (<60%): Consider adjusting parameters or more data
        """)


def plot_prediction_accuracy(metrics, data, sequence_length):
    """Create accuracy visualization charts"""
    
    # Get predictions and actual values
    train_pred = metrics['train_predictions'].flatten()
    test_pred = metrics['test_predictions'].flatten()
    train_actual = metrics['y_train_actual'].flatten()
    test_actual = metrics['y_test_actual'].flatten()
    
    # Create dates for plotting
    total_len = len(train_actual) + len(test_actual)
    start_date = data.index[sequence_length]  # Start after sequence length
    dates = pd.date_range(start=start_date, periods=total_len, freq='D')
    
    train_dates = dates[:len(train_actual)]
    test_dates = dates[len(train_actual):]
    
    # Create subplot
    fig = go.Figure()
    
    # Training data
    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_actual,
        mode='lines',
        name='Training Actual',
        line=dict(color='blue', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=train_dates,
        y=train_pred,
        mode='lines',
        name='Training Predicted',
        line=dict(color='lightblue', width=1, dash='dot')
    ))
    
    # Testing data
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_actual,
        mode='lines',
        name='Testing Actual',
        line=dict(color='red', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=test_pred,
        mode='lines',
        name='Testing Predicted',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title="Model Predictions vs Actual Prices",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        showlegend=True
    )
    
    return fig


def display_prediction_summary(data, predictions, prediction_years):
    """Display prediction summary metrics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Price", 
            f"${data['Close'].iloc[-1]:.2f}"
        )
    with col2:
        st.metric(
            f"Predicted Price ({prediction_years}Y)", 
            f"${predictions[-1]:.2f}"
        )
    with col3:
        change_pct = ((predictions[-1] - data['Close'].iloc[-1]) / data['Close'].iloc[-1]) * 100
        st.metric(
            f"Expected Change ({prediction_years}Y)", 
            f"{change_pct:+.1f}%"
        )


def create_download_button(prediction_df, stock_symbol, prediction_years):
    """Create download button for predictions"""
    csv = prediction_df.to_csv(index=False)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name=f"{stock_symbol}_predictions_{prediction_years}y.csv",
        mime="text/csv"
    )


def display_instructions():
    """Display usage instructions"""
    with st.expander("How to Use This App"):
        st.markdown("""
        1. **Enter Stock Symbol**: Type the ticker symbol (e.g., AAPL for Apple, GOOGL for Google)
        2. **Select Prediction Range**: Choose how many years into the future you want to predict
        3. **Adjust Parameters**: Modify LSTM parameters if needed (default values usually work well)
        4. **Click 'Run Prediction'**: The app will download data, train the model, and show predictions
        5. **Analyze Results**: Review the model performance metrics and prediction charts
        6. **Download Results**: Save predictions as a CSV file for further analysis
        
        **Model Parameters:**
        - **Sequence Length**: Number of previous days used to predict the next day
        - **Epochs**: Number of training iterations
        - **Batch Size**: Number of samples processed before updating the model
        """)


def display_disclaimer():
    """Display investment disclaimer"""
    pass  # Removed disclaimer for project purposes


# ========================= MAIN APPLICATION FUNCTION =========================

def run_prediction_pipeline(params):
    """Main prediction pipeline that orchestrates all functions"""
    
    # Load data
    with st.spinner(f"Loading data for {params['stock_symbol']}..."):
        data, info = load_stock_data(params['stock_symbol'], params['data_period'])
        
        if data is None or len(data) <= params['sequence_length']:
            st.error("Unable to load sufficient data. Please check the stock symbol and try again.")
            return
    
    # Display stock information
    display_stock_info(info, params['stock_symbol'])
    
    # Display historical data
    st.subheader("Historical Stock Data")
    fig_hist = plot_historical_data(data, params['stock_symbol'])
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Prepare data for LSTM
    with st.spinner("Preparing data and training LSTM model..."):
        X, y, scaler = prepare_lstm_data(data, params['sequence_length'])
        X_train, X_test, y_train, y_test = split_train_test_data(X, y)
        
        # Create and train model
        model = create_lstm_model(params['sequence_length'])
        history = train_lstm_model(
            model, X_train, y_train, X_test, y_test, 
            params['epochs'], params['batch_size']
        )
        
        # Evaluate model performance
        metrics = evaluate_model_performance(model, X_train, y_train, X_test, y_test, scaler)
    
    # Display model metrics with accuracy percentages
    display_model_metrics(metrics)
    
    # Display accuracy visualization
    st.subheader("Prediction Accuracy Visualization")
    fig_accuracy = plot_prediction_accuracy(metrics, data, params['sequence_length'])
    st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Display training history
    st.subheader("Training History")
    fig_loss = plot_training_history(history)
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Generate future predictions
    with st.spinner(f"Generating {params['prediction_years']}-year predictions..."):
        prediction_days = params['prediction_years'] * 365
        scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))
        future_predictions = make_future_predictions(
            model, scaled_data, scaler, params['sequence_length'], prediction_days
        )
        
        # Create future dates
        future_dates = generate_future_dates(data.index[-1], prediction_days)
    
    # Display predictions
    st.subheader(f"Stock Price Predictions ({params['prediction_years']} Year{'s' if params['prediction_years'] > 1 else ''})")
    
    # Plot predictions
    fig_pred = plot_predictions(data, params['stock_symbol'], future_dates, future_predictions, params['prediction_years'])
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Display prediction summary
    display_prediction_summary(data, future_predictions, params['prediction_years'])
    
    # Create download option
    prediction_df = create_prediction_dataframe(future_dates, future_predictions)
    create_download_button(prediction_df, params['stock_symbol'], params['prediction_years'])
    
    # Display disclaimer
    display_disclaimer()


# ========================= MAIN APPLICATION =========================

def main():
    """Main application function"""
    # Setup page
    setup_page_config()
    
    # Title and description
    st.title("📈 Stock Market LSTM Predictor")
    st.markdown("Predict stock prices using Long Short-Term Memory (LSTM) neural networks")
    
    # Get user inputs
    params = create_sidebar_inputs()
    
    # Run prediction button
    if st.sidebar.button("Run Prediction"):
        run_prediction_pipeline(params)
    
    # Display instructions
    display_instructions()
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, TensorFlow, and yfinance")


# Run the application
if __name__ == "__main__":
    main()