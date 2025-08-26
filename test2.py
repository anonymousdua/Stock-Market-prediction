import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Fetch Apple stock data (AAPL)
ticker = "AAPL"
apple = yf.Ticker(ticker)

# Get historical market data (last 6 months)
data = apple.history(period="6mo")

# Display the first few rows
print("Apple Stock Data (Last 6 Months):")
print(data.head())

# Plot Closing Price
plt.figure(figsize=(10,5))
plt.plot(data.index, data['Close'], label="Close Price")
plt.title("Apple Stock Closing Price (Last 6 Months)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()
