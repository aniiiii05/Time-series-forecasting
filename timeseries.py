

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Load the stock prices data
# The CSV should have columns: Date and Close (closing price)
df = pd.read_csv('stock_prices.csv', parse_dates=['Date'], index_col='Date')
df = df.sort_index()

print("Stock Prices Data Preview:")
print(df.head())

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Closing Price')
plt.title('Stock Closing Prices Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# Check for stationarity using Augmented Dickey-Fuller test
adf_result = adfuller(df['Close'])
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")

# If p-value > 0.05, the series is non-stationary and differencing may be required.
# For this example, we perform first-order differencing.
df['Close_diff'] = df['Close'].diff().dropna()

# Remove NA values after differencing
df_diff = df.dropna()

# Plot differenced series
plt.figure(figsize=(10, 6))
plt.plot(df_diff['Close_diff'], label='Differenced Closing Price', color='orange')
plt.title('Differenced Stock Closing Prices')
plt.xlabel('Date')
plt.ylabel('Differenced Price')
plt.legend()
plt.tight_layout()
plt.show()

# Fit an ARIMA model
# For simplicity, we choose ARIMA(1,1,1) model; parameters may be optimized with further analysis.
model = ARIMA(df['Close'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Forecasting for the next 30 days
forecast_steps = 30
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1)[1:]


# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df['Close'], label='Historical Closing Price')
plt.plot(forecast_index, forecast, label='Forecasted Closing Price', color='red', linestyle='--')
plt.title('Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.tight_layout()
plt.show()

# Save the forecast data
forecast_df = pd.DataFrame({'Forecast': forecast}, index=forecast_index)
forecast_df.to_csv('stock_price_forecast.csv')
print("Forecast data saved to 'stock_price_forecast.csv'")
