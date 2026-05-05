```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(42)

dates = pd.date_range(start="2021-01-01", end="2023-12-31", freq='D')
trend = np.linspace(100, 300, len(dates))
seasonality = 50 * np.sin(2 * np.pi * dates.dayofyear / 365)
noise = np.random.normal(0, 10, len(dates))
sales = trend + seasonality + noise
df = pd.DataFrame({"Date": dates, "Sales": sales})
df.to_csv("sales_data.csv", index=False)
print(df.head())

df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
print(df.isnull().sum())
df['Sales'].fillna(method='ffill', inplace=True)
print(df.head())

plt.figure()
plt.plot(df['Sales'])
plt.title("Daily Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid()
plt.show()

decomposition = seasonal_decompose(df['Sales'], model='additive', period=365)
decomposition.plot()
plt.show()

result = adfuller(df['Sales'])
print("ADF Statistic:", result[0])
print("p-value:", result[1])

df_diff = df['Sales'].diff().dropna()
result = adfuller(df_diff)
print("After differencing p-value:", result[1])

model = ARIMA(df['Sales'], order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=30)
plt.figure()
plt.plot(df['Sales'], label='Actual')
plt.plot(forecast, label='Forecast', linestyle='dashed')
plt.legend()
plt.title("Sales Forecast (30 Days)")
plt.show()

train = df.iloc[:-30]
test = df.iloc[-30:]
model = ARIMA(train['Sales'], order=(1,1,1))
model_fit = model.fit()
pred = model_fit.forecast(steps=30)
mae = mean_absolute_error(test['Sales'], pred)
rmse = np.sqrt(mean_squared_error(test['Sales'], pred))
print("MAE:", mae)
print("RMSE:", rmse)
```

