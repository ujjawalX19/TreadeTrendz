import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import plotly.express as px
from datetime import date
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# App title
st.set_page_config(page_title="TradeTrendz", layout="wide")
st.title("📈 TradeTrendz")
st.subheader("Predict Tomorrow’s Market Today")

# Sidebar
st.sidebar.header('📅 Choose Parameters')
start_date = st.sidebar.date_input("Start Date", date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", date(2025, 12, 31))
ticker_list = ["AAPL", "MSFT", "GOOGL", "PYPL", "TSLA", "NFLX", "NVDA", "META"]
ticker = st.sidebar.selectbox("Select Company", ticker_list)
show_sma = st.sidebar.checkbox("Show 20-Day SMA")
forecast_period = st.sidebar.slider("Forecast Days", 7, 90, 30)

# Download data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    return data

data = load_data(ticker)
st.success(f"Data Loaded for {ticker} from {start_date} to {end_date}")

# Tabs for layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Chart", "📉 Stationarity", "🔮 Forecast", "📥 Download"])

with tab1:
    st.subheader("Stock Closing Price")
    fig = px.line(data, x='Date', y='Close', title=f"{ticker} Closing Price", width=1000, height=500)

    if show_sma:
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        fig.add_scatter(x=data['Date'], y=data['SMA_20'], mode='lines', name='20-Day SMA')

    st.plotly_chart(fig)

with tab2:
    st.subheader("ADF Test (Check Stationarity)")

    def adf_test(series):
        result = adfuller(series.dropna())
        labels = ['ADF Statistic', 'p-value', '# Lags Used', '# Observations']
        output = pd.Series(result[0:4], index=labels)
        for key, val in result[4].items():
            output[f'Critical Value ({key})'] = val
        return output

    adf_result = adf_test(data['Close'])
    st.write(adf_result)
    if adf_result['p-value'] < 0.05:
        st.success("✅ Series is Stationary (Good for forecasting)")
    else:
        st.warning("⚠ Series is Not Stationary (Consider Differencing)")

with tab3:
    st.subheader(f"{forecast_period}-Day Stock Price Forecast with Prophet")

    df_train = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet()
    model.fit(df_train)

    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    st.plotly_chart(px.line(forecast, x='ds', y='yhat', title="Forecasted Closing Price"))

    actual = df_train['y'].iloc[-forecast_period:]
    predicted = forecast['yhat'].iloc[-forecast_period:]

    if len(actual) == len(predicted):
        mae = mean_absolute_error(actual, predicted)
        st.metric("MAE (Last Days)", f"{mae:.2f}")
    else:
        st.info("⚠ Not enough actual data to compare prediction")

with tab4:
    st.subheader("Download Data")
    csv = data.to_csv(index=False).encode()
    st.download_button("📥 Download Raw Data (CSV)", csv, file_name=f"{ticker}_data.csv", mime='text/csv')