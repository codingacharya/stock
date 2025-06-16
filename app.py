import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# ----------------------------
# Streamlit UI Configuration
# ----------------------------
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
st.title("📈 Stock Price Prediction App")

# ----------------------------
# Sidebar Data Source Option
# ----------------------------
data_source = st.sidebar.radio("Choose Data Source:", ["Yahoo Finance", "Upload CSV"])

forecast_days = st.sidebar.slider("Days to Forecast", min_value=1, max_value=30, value=7)

# ----------------------------
# Load Data (Yahoo or CSV)
# ----------------------------
data = None

if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL)", value="AAPL").upper()
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

    if start_date >= end_date:
        st.error("❌ Start date must be before end date.")
        st.stop()

    @st.cache_data
    def load_yahoo_data(ticker, start, end):
        try:
            df = yf.download(ticker, start=start, end=end)
            if df.empty:
                return None
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            return None

    data = load_yahoo_data(ticker, start_date, end_date)

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV with 'Date' and 'Close' columns", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, parse_dates=["Date"])
            if "Date" not in df.columns or "Close" not in df.columns:
                st.error("CSV must contain 'Date' and 'Close' columns.")
                st.stop()
            df = df.sort_values("Date")
            data = df[["Date", "Close"]].copy()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()

# ----------------------------
# Display & Validate Data
# ----------------------------
if data is None or data.empty:
    st.warning("⚠️ No data available. Please check your input.")
    st.stop()

st.subheader("📊 Historical Close Price")
st.line_chart(data.set_index("Date")["Close"])

# ----------------------------
# Feature Engineering
# ----------------------------
df = data.copy()
df["Target"] = df["Close"].shift(-forecast_days)
df.dropna(inplace=True)

X = df[["Close"]]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Model Definitions
# ----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

# ----------------------------
# Train & Predict
# ----------------------------
st.subheader("🧠 Model Forecasts & Evaluation")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    # Forecast
    future_input = df[["Close"]].tail(forecast_days)
    future_pred = model.predict(future_input)
    future_dates = pd.date_range(df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")
    forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_pred})

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Historical"))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], name="Forecast"))
    fig.update_layout(title=f"{name} - MSE: {mse:.2f}", xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig, use_container_width=True)
