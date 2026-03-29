import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, Concatenate, Flatten
)
from tensorflow.keras.optimizers import Adam
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
import pandas_ta as ta
from sentimentTest import batch_get_sentiments

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Price Predictor", layout="wide")

# ---------------------------------------------------------------------------
# App Title and Description
# ---------------------------------------------------------------------------
st.title("Advanced Stock Price Predictor with Multiple Models")
st.markdown(
    "This app uses multiple machine learning models including LSTM, xLSTM, SVM, ARIMA, SARIMA, "
    "and ensemble methods (Stacking & Voting) to predict stock prices."
)

# ---------------------------------------------------------------------------
# Sidebar – Stock Selection
# ---------------------------------------------------------------------------
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
    "Visa": "V",
}
stock_name = st.sidebar.selectbox(
    "Choose a Stock",
    options=list(POPULAR_STOCKS.keys()),
    index=0,
    help="Select a stock from the list of top 10 popular companies.",
)
symbol = POPULAR_STOCKS[stock_name]

# ---------------------------------------------------------------------------
# Sidebar – Model Parameters
# ---------------------------------------------------------------------------
st.sidebar.header("Model Parameters")

lookback_period = st.sidebar.selectbox(
    "Lookback Period (Days)",
    options=[30, 90, 120],
    index=1,
    help="Number of past days used by Branch A (xLSTM) for sequence modelling.",
)

epochs = st.sidebar.slider(
    "Training Epochs",
    min_value=20,
    max_value=100,
    value=50,
    help="Number of training cycles for the neural network branches.",
)

batch_size = st.sidebar.select_slider(
    "Batch Size",
    options=[16, 32, 64],
    value=32,
    help="Number of training samples per gradient update.",
)

svm_kernel = st.sidebar.selectbox(
    "SVM Kernel",
    options=["rbf", "linear", "poly"],
    index=0,
    help="Kernel type for the SVM decision head.",
)

auto_arima = st.sidebar.checkbox(
    "Use Auto ARIMA/SARIMA",
    value=True,
    help="Automatically select the best ARIMA/SARIMA parameters.",
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

# ---------------------------------------------------------------------------
# Sidebar – Prediction Settings
# ---------------------------------------------------------------------------
st.sidebar.header("Prediction Settings")
prediction_days = st.sidebar.selectbox(
    "Days to Predict",
    options=[15, 30, 45],
    index=1,
    help="Number of days to predict and evaluate.",
)

# ---------------------------------------------------------------------------
# Helper – Data Fetching
# ---------------------------------------------------------------------------
@st.cache_data
def fetch_stock_data(symbol, days_to_fetch):
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=int(days_to_fetch * 1.7))
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            st.error(f"No data found for symbol {symbol}.")
            return None
        data = data[["Open", "High", "Low", "Close", "Volume"]].tail(days_to_fetch)
        if len(data) < days_to_fetch:
            st.warning(
                f"Only found {len(data)} trading days (requested {days_to_fetch}). "
                "Results may be less accurate."
            )
            if len(data) < days_to_fetch * 0.8:
                st.error("Significantly less data than requested. Stopping.")
                return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None


# ---------------------------------------------------------------------------
# Helper – Sequence builder for Branch A (xLSTM)
# ---------------------------------------------------------------------------
def create_sequences(ohlcv_scaled, lookback):
    """
    Returns:
        X  – shape (N, lookback, num_ohlcv_features)
        y  – shape (N,)   Close price index = 3
    """
    X, y = [], []
    for i in range(lookback, len(ohlcv_scaled)):
        X.append(ohlcv_scaled[i - lookback : i])
        y.append(ohlcv_scaled[i, 3])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# Helper – ARIMA / SARIMA rolling one-step forecasts
#           Returns an array of length (len(close_prices) - start_index)
# ---------------------------------------------------------------------------
def rolling_arima_forecasts(close_prices, start_index, auto=True,
                             p=1, d=1, q=1, seasonal=False, seasonal_period=7):
    """
    For each position i >= start_index, fit ARIMA/SARIMA on prices[:i]
    and produce a one-step-ahead forecast for position i.
    This gives us a forecast column aligned with every row we need.
    """
    forecasts = []
    for i in range(start_index, len(close_prices)):
        train_slice = close_prices[:i]
        try:
            if auto:
                model = pm.auto_arima(
                    train_slice,
                    seasonal=seasonal,
                    m=seasonal_period if seasonal else 1,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action="ignore",
                    maxiter=50,
                )
                fc = model.predict(n_periods=1)[0]
            else:
                if seasonal:
                    from statsmodels.tsa.statespace.sarimax import SARIMAX
                    m = SARIMAX(
                        train_slice,
                        order=(p, d, q),
                        seasonal_order=(p, d, q, seasonal_period),
                    ).fit(disp=False)
                    fc = m.forecast(steps=1)[0]
                else:
                    from statsmodels.tsa.arima.model import ARIMA
                    m = ARIMA(train_slice, order=(p, d, q)).fit()
                    fc = m.forecast(steps=1)[0]
        except Exception:
            # Fall back to last known price if model fails
            fc = train_slice[-1]
        forecasts.append(fc)
    return np.array(forecasts)


# ---------------------------------------------------------------------------
# Model – Branch A  (xLSTM: Bidirectional LSTM)
# ---------------------------------------------------------------------------
def build_branch_a(lookback, n_ohlcv_features, latent_dim=64):
    """
    Returns a Keras Model:
        input  -> (batch, lookback, n_ohlcv_features)
        output -> (batch, latent_dim)   latent representation
    """
    inp = Input(shape=(lookback, n_ohlcv_features), name="branch_a_input")
    x = Bidirectional(LSTM(128, return_sequences=True), name="xLSTM_bidir_1")(inp)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, return_sequences=True), name="xLSTM_bidir_2")(x)
    x = Dropout(0.3)(x)
    x = LSTM(32, return_sequences=False, name="xLSTM_3")(x)
    x = Dropout(0.2)(x)
    out = Dense(latent_dim, activation="relu", name="branch_a_latent")(x)
    return Model(inputs=inp, outputs=out, name="BranchA_xLSTM")


# ---------------------------------------------------------------------------
# Model – Branch B  (Dense snapshot reader)
# ---------------------------------------------------------------------------
def build_branch_b(n_snapshot_features, latent_dim=64):
    """
    Returns a Keras Model:
        input  -> (batch, n_snapshot_features)
        output -> (batch, latent_dim)   latent representation
    """
    inp = Input(shape=(n_snapshot_features,), name="branch_b_input")
    x = Dense(128, activation="relu", name="branch_b_dense_1")(inp)
    x = Dropout(0.2)(x)
    x = Dense(64, activation="relu", name="branch_b_dense_2")(x)
    x = Dropout(0.2)(x)
    out = Dense(latent_dim, activation="relu", name="branch_b_latent")(x)
    return Model(inputs=inp, outputs=out, name="BranchB_Dense")


# ---------------------------------------------------------------------------
# Model – Full two-branch feature extractor  (no SVM head yet)
# ---------------------------------------------------------------------------
def build_feature_extractor(lookback, n_ohlcv_features, n_snapshot_features, latent_dim=64):
    """
    Combines Branch A and Branch B via Concatenate.
    Output is the merged latent vector fed into the SVM.
    """
    branch_a = build_branch_a(lookback, n_ohlcv_features, latent_dim)
    branch_b = build_branch_b(n_snapshot_features, latent_dim)

    inp_a = Input(shape=(lookback, n_ohlcv_features), name="seq_input")
    inp_b = Input(shape=(n_snapshot_features,), name="snap_input")

    lat_a = branch_a(inp_a)
    lat_b = branch_b(inp_b)

    merged = Concatenate(name="merge_node")([lat_a, lat_b])
    # One extra dense to let the merged representation breathe before SVM
    out = Dense(latent_dim, activation="relu", name="pre_svm_head")(merged)

    return Model(inputs=[inp_a, inp_b], outputs=out, name="FeatureExtractor")


# ---------------------------------------------------------------------------
# Model – Thin Keras regression head used ONLY for pre-training the extractor
# ---------------------------------------------------------------------------
def build_pretrain_model(feature_extractor):
    """
    Wraps the feature extractor with a single Dense(1) so we can
    pre-train it end-to-end on price regression before handing
    features over to the SVM.
    """
    inp_a = feature_extractor.input[0]
    inp_b = feature_extractor.input[1]
    features = feature_extractor.output
    price_out = Dense(1, name="pretrain_price")(features)
    model = Model(inputs=[inp_a, inp_b], outputs=price_out, name="PretrainModel")
    model.compile(optimizer=Adam(learning_rate=0.0005), loss="mean_squared_error")
    return model


# ===========================================================================
# MAIN APPLICATION LOGIC
# ===========================================================================
if st.sidebar.button("Run Prediction", type="primary"):

    # -----------------------------------------------------------------------
    # 1. Determine how much data to fetch
    # -----------------------------------------------------------------------
    DAYS_LOST_TO_TA = 33
    MIN_TRAINING_SAMPLES = 50
    total_clean_days_needed = (lookback_period + MIN_TRAINING_SAMPLES) + prediction_days
    total_days_to_fetch = total_clean_days_needed + DAYS_LOST_TO_TA

    st.sidebar.info(
        f"Fetching {total_days_to_fetch} trading days:\n"
        f"- {prediction_days} (prediction window)\n"
        f"- {lookback_period} (xLSTM lookback)\n"
        f"- {MIN_TRAINING_SAMPLES} (min training)\n"
        f"- {DAYS_LOST_TO_TA} (TA warm-up)"
    )

    with st.spinner(f"Fetching {total_days_to_fetch} days of data for {stock_name}..."):
        data = fetch_stock_data(symbol, total_days_to_fetch)

    if data is None:
        st.stop()

    # -----------------------------------------------------------------------
    # 2. Feature Engineering
    # -----------------------------------------------------------------------
    with st.spinner("Adding features (Sentiment, SMAs, TA)..."):
        data = data.reset_index()

        # Sentiment
        dates_list = data["Date"].dt.strftime("%Y-%m-%d").tolist()
        sentiment_results = batch_get_sentiments(symbol, dates_list)
        data["Sentiment"] = data["Date"].dt.strftime("%Y-%m-%d").map(sentiment_results)

        # SMAs
        data["SMA10"] = data["Close"].rolling(window=10).mean()
        data["SMA20"] = data["Close"].rolling(window=20).mean()

        # Technical indicators via pandas_ta
        data.ta.ema(length=12, append=True)
        data.ta.ema(length=26, append=True)
        data.ta.macd(append=True)
        data.ta.rsi(append=True)

        data.dropna(inplace=True)
        data = data.reset_index(drop=True)

    with st.expander(f"View Recent Data for {stock_name} ({symbol})", expanded=False):
        st.dataframe(data.tail(10))

    # -----------------------------------------------------------------------
    # 3. Define feature groups
    # -----------------------------------------------------------------------
    # Branch A – OHLCV sequence (fed to xLSTM)
    OHLCV_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

    # Branch B snapshot features (TA + Sentiment) – ARIMA/SARIMA cols added later
    SNAPSHOT_BASE = [
        "Sentiment", "SMA10", "SMA20",
        "EMA_12", "EMA_26",
        "MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9",
        "RSI_14",
    ]

    missing = [f for f in OHLCV_FEATURES + SNAPSHOT_BASE if f not in data.columns]
    if missing:
        st.error(f"Missing features after calculation: {missing}")
        st.stop()

    training_data_len = len(data) - prediction_days

    # Validate data volume
    min_needed = lookback_period + prediction_days + MIN_TRAINING_SAMPLES
    if len(data) < min_needed:
        st.error(
            f"Not enough data after feature engineering! "
            f"Need {min_needed} rows, have {len(data)}."
        )
        st.stop()

    # -----------------------------------------------------------------------
    # 4. Rolling ARIMA / SARIMA forecasts  (used as Branch B features)
    # -----------------------------------------------------------------------
    st.subheader("Model Training Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    close_prices = data["Close"].values
    arima_start = lookback_period   # need at least lookback_period prices to fit

    arima_kwargs = dict(
        auto=auto_arima,
        p=1 if auto_arima else arima_p,
        d=1 if auto_arima else arima_d,
        q=1 if auto_arima else arima_q,
    )
    seasonal_period_val = 7 if auto_arima else seasonal_period

    status_text.text("Computing rolling ARIMA forecasts (Branch B feature)...")
    arima_forecasts_full = rolling_arima_forecasts(
        close_prices, arima_start, seasonal=False, **arima_kwargs
    )
    # Pad the first `arima_start` rows with the first forecast value
    arima_col = np.concatenate([
        np.full(arima_start, arima_forecasts_full[0]),
        arima_forecasts_full,
    ])
    data["ARIMA_Forecast"] = arima_col
    progress_bar.progress(0.15)

    status_text.text("Computing rolling SARIMA forecasts (Branch B feature)...")
    sarima_forecasts_full = rolling_arima_forecasts(
        close_prices, arima_start, seasonal=True,
        seasonal_period=seasonal_period_val, **arima_kwargs
    )
    sarima_col = np.concatenate([
        np.full(arima_start, sarima_forecasts_full[0]),
        sarima_forecasts_full,
    ])
    data["SARIMA_Forecast"] = sarima_col
    progress_bar.progress(0.30)

    # Final snapshot feature list (now includes ARIMA + SARIMA as indicators)
    SNAPSHOT_FEATURES = SNAPSHOT_BASE + ["ARIMA_Forecast", "SARIMA_Forecast"]

    # -----------------------------------------------------------------------
    # 5. Scale data
    # -----------------------------------------------------------------------
    scaler_ohlcv = MinMaxScaler(feature_range=(0, 1))
    scaler_snap  = MinMaxScaler(feature_range=(0, 1))
    scaler_close = MinMaxScaler(feature_range=(0, 1))

    ohlcv_values   = data[OHLCV_FEATURES].values
    snapshot_values = data[SNAPSHOT_FEATURES].values
    close_values   = data[["Close"]].values

    scaled_ohlcv   = scaler_ohlcv.fit_transform(ohlcv_values)
    scaled_snap    = scaler_snap.fit_transform(snapshot_values)
    scaled_close   = scaler_close.fit_transform(close_values).flatten()

    # -----------------------------------------------------------------------
    # 6. Build training / test windows
    # -----------------------------------------------------------------------
    # For Branch A we need sequences of length lookback_period.
    # Row i in X_a corresponds to rows [i .. i+lookback_period-1] of ohlcv,
    # and its target is the Close at row i+lookback_period.
    def make_windows(ohlcv_s, snap_s, close_s, lookback):
        Xa, Xb, y = [], [], []
        for i in range(lookback, len(ohlcv_s)):
            Xa.append(ohlcv_s[i - lookback : i])   # sequence
            Xb.append(snap_s[i])                    # snapshot for the target day
            y.append(close_s[i])
        return np.array(Xa), np.array(Xb), np.array(y)

    Xa_all, Xb_all, y_all = make_windows(scaled_ohlcv, scaled_snap, scaled_close, lookback_period)

    # The last `prediction_days` windows are the test set
    n_test = prediction_days
    Xa_train, Xb_train, y_train = Xa_all[:-n_test], Xb_all[:-n_test], y_all[:-n_test]
    Xa_test,  Xb_test,  y_test  = Xa_all[-n_test:],  Xb_all[-n_test:],  y_all[-n_test:]

    actual_prices = scaler_close.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # -----------------------------------------------------------------------
    # 7. Build and pre-train the feature extractor
    # -----------------------------------------------------------------------
    status_text.text("Building hybrid xLSTM + Dense feature extractor...")
    n_ohlcv   = len(OHLCV_FEATURES)
    n_snap    = len(SNAPSHOT_FEATURES)
    LATENT_DIM = 64

    feature_extractor = build_feature_extractor(lookback_period, n_ohlcv, n_snap, LATENT_DIM)
    pretrain_model    = build_pretrain_model(feature_extractor)

    status_text.text("Pre-training feature extractor (xLSTM + Dense branches)...")
    pretrain_model.fit(
        [Xa_train, Xb_train],
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,
        validation_split=0.1,
    )
    progress_bar.progress(0.65)

    # -----------------------------------------------------------------------
    # 8. Extract merged features  →  train SVM decision head
    # -----------------------------------------------------------------------
    status_text.text("Extracting merged latent features for SVM head...")
    features_train = feature_extractor.predict([Xa_train, Xb_train], verbose=0)
    features_test  = feature_extractor.predict([Xa_test,  Xb_test],  verbose=0)

    # Inverse-transform y_train back to real prices for SVM target
    y_train_real = scaler_close.inverse_transform(y_train.reshape(-1, 1)).flatten()

    status_text.text("Training SVM decision head on merged features...")
    svm_head = SVR(kernel=svm_kernel, C=100, epsilon=0.01)
    svm_head.fit(features_train, y_train_real)

    svm_predictions = svm_head.predict(features_test)
    progress_bar.progress(0.85)

    # -----------------------------------------------------------------------
    # 9. Fill results dict  (keeps original UI intact)
    # -----------------------------------------------------------------------
    # We surface the final hybrid model under names that match the old UI.
    # The neural network's own pre-training predictions are also shown for
    # comparison, mapped to the names the UI expects.

    # Hybrid SVM (the primary result – shown as "SVM" in the old metrics table)
    results = {}
    results["SVM (Hybrid Head)"] = svm_predictions

    # Pre-training model predictions (shown as "xLSTM" in the old UI slot)
    pretrain_preds_scaled = pretrain_model.predict([Xa_test, Xb_test], verbose=0).flatten()
    pretrain_preds = scaler_close.inverse_transform(
        pretrain_preds_scaled.reshape(-1, 1)
    ).flatten()
    results["xLSTM+Dense"] = pretrain_preds

    # ARIMA standalone test-window forecast (from the rolling array)
    # These are already in price space, just slice the test window
    arima_test = arima_col[training_data_len : training_data_len + prediction_days]
    sarima_test = sarima_col[training_data_len : training_data_len + prediction_days]
    results["ARIMA"] = arima_test[: len(actual_prices)]
    results["SARIMA"] = sarima_test[: len(actual_prices)]

    # Ensemble: Stacking (linear meta-model over all four)
    valid_for_ensemble = {k: v for k, v in results.items()
                          if len(v) == len(actual_prices) and not np.all(v == 0)}
    if len(valid_for_ensemble) >= 2:
        ens_input = np.column_stack(list(valid_for_ensemble.values()))
        meta = LinearRegression()
        meta.fit(ens_input, actual_prices)
        results["Stacking"] = meta.predict(ens_input)
        results["Voting"]   = np.mean(ens_input, axis=1)
    else:
        st.warning("Not enough valid models to create ensembles.")

    progress_bar.empty()
    status_text.success("All models (including ensembles) trained successfully!")

    # -----------------------------------------------------------------------
    # 10. Metrics + Visualisation  (original UI code, unchanged)
    # -----------------------------------------------------------------------
    st.subheader("Model Performance Comparison")

    base_colors = ["blue", "red", "green", "orange", "purple"]
    color_map = {}
    for i, name in enumerate(results.keys()):
        if name == "Stacking":
            color_map[name] = "gold"
        elif name == "Voting":
            color_map[name] = "cyan"
        else:
            color_map[name] = base_colors[i % len(base_colors)]

    metrics_data = []
    for model_name, predictions in results.items():
        if len(predictions) > len(actual_prices):
            predictions = predictions[: len(actual_prices)]
            actual_trimmed = actual_prices
        elif len(predictions) < len(actual_prices):
            actual_trimmed = actual_prices[: len(predictions)]
        else:
            actual_trimmed = actual_prices

        if len(predictions) == 0 or np.all(predictions == 0):
            st.warning(f"Model {model_name} produced no valid predictions. Skipping.")
            continue

        rmse = np.sqrt(mean_squared_error(actual_trimmed, predictions))
        mae  = mean_absolute_error(actual_trimmed, predictions)
        mape = np.mean(np.abs((actual_trimmed - predictions) / actual_trimmed)) * 100
        accuracy = 100 - mape

        if len(actual_trimmed) > 1:
            actual_dir    = np.diff(actual_trimmed) > 0
            pred_dir      = np.diff(predictions) > 0
            dir_accuracy  = np.mean(actual_dir == pred_dir) * 100
        else:
            dir_accuracy = 0

        metrics_data.append({
            "Model": model_name,
            "RMSE": rmse,
            "MAE": mae,
            "MAPE": mape,
            "Accuracy": accuracy,
            "Directional Accuracy": dir_accuracy,
        })

    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(f"{symbol}_metrics.csv", index=False)

        cols = st.columns(len(metrics_data))
        for col, row in zip(cols, metrics_data):
            with col:
                st.metric(
                    label=row["Model"],
                    value=f"{row['Accuracy']:.1f}%",
                    delta=f"RMSE: ${row['RMSE']:.2f}",
                    help=f"{row['Model']} – Accuracy: {row['Accuracy']:.1f}%",
                )

        st.dataframe(
            metrics_df.style.format({
                "RMSE": "${:.2f}",
                "MAE": "${:.2f}",
                "MAPE": "{:.2f}%",
                "Accuracy": "{:.2f}%",
                "Directional Accuracy": "{:.2f}%",
            }),
            use_container_width=True,
        )

        st.subheader(f"Actual vs. Predicted Prices (Last {prediction_days} Days)")
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_dates = data["Date"][training_data_len :].reset_index(drop=True)

        ax.plot(
            plot_dates, actual_prices,
            label="Actual Price", color="black",
            linewidth=3, marker="o", markersize=4,
        )
        for model_name, predictions in results.items():
            if model_name not in color_map:
                continue
            pred_plot = predictions[: len(plot_dates)]
            if len(pred_plot) != len(plot_dates):
                continue
            ax.plot(
                plot_dates, pred_plot,
                label=f"{model_name} Prediction",
                color=color_map[model_name],
                linestyle="--", linewidth=2, marker="x", markersize=3,
            )

        ax.set_title(
            f"{stock_name} ({symbol}) – Price Prediction Comparison\n"
            f"(Lookback: {lookback_period} days, Prediction: {prediction_days} days)",
            fontsize=16, fontweight="bold",
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Price (USD)", fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        best_model = max(
            metrics_data,
            key=lambda x: (x["Directional Accuracy"], x["Accuracy"]),
        )
        best_name = best_model["Model"]
        label = f"**{best_name} Ensemble**" if best_name in ("Stacking", "Voting") else best_name
        st.success(
            f"**Best Performing Model**: {label} "
            f"(Directional Accuracy: {best_model['Directional Accuracy']:.2f}%, "
            f"Accuracy: {best_model['Accuracy']:.2f}%)"
        )

        st.subheader("Detailed Prediction Data (Best Model)")
        best_predictions = results[best_name]
        if len(best_predictions) > len(actual_prices):
            best_predictions = best_predictions[: len(actual_prices)]

        prediction_df = pd.DataFrame({
            "Date": data["Date"][training_data_len :]
                        .dt.strftime("%Y-%m-%d")
                        .values[: len(best_predictions)],
            "Actual Price": actual_prices[: len(best_predictions)],
            f"Predicted Price ({best_name})": best_predictions,
            "Difference ($)": best_predictions - actual_prices[: len(best_predictions)],
            "Difference (%)": (
                (best_predictions - actual_prices[: len(best_predictions)])
                / actual_prices[: len(best_predictions)]
            ) * 100,
        })

        st.dataframe(
            prediction_df.style.format({
                "Actual Price": "${:,.2f}",
                f"Predicted Price ({best_name})": "${:,.2f}",
                "Difference ($)": "{:+,.2f}",
                "Difference (%)": "{:+,.2f}%",
            }),
            use_container_width=True,
        )
    else:
        st.error(
            "No models produced valid predictions. "
            "Please check the parameters and try again."
        )