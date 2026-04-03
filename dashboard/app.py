import streamlit as st
import pandas as pd
import requests
import os
import sys
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(page_title="AeroFlow Crypto Dashboard", layout="wide")
st.title("📈 AeroFlow: Bitcoin Price Forecaster")
st.markdown("Automated MLOps Pipeline with DVC & MLflow")

@st.cache_data
def load_data():
    df = pd.read_csv("data/btc-usd_historical.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

data = load_data()
st.line_chart(data.set_index('Date')['Close'].tail(365))

st.sidebar.header("Model Controls")
if st.sidebar.button("Get Next Day Forecast"):
    last_30_prices = data['Close'].tail(30).tolist()
    try:
        response = requests.post("http://localhost:8000/predict", json={"prices": last_30_prices})
        prediction = response.json().get("prediction")
        st.metric("Predicted Next Day Close", f"${prediction:,.2f}")
        last_date = data['Date'].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'].tail(7), y=data['Close'].tail(7), name="Actual (Last 7 Days)"))
        fig.add_trace(go.Scatter(x=[last_date, next_date], y=[data['Close'].iloc[-1], prediction], line=dict(dash='dash', color='red'), name="Forecast"))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e: st.error(f"Error connecting to Prediction API: {e}")
