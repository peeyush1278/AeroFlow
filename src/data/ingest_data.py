import pandas as pd
import numpy as np
import os
import argparse
import yfinance as yf

def ingest_data(symbol="BTC-USD", start_date="2020-01-01", end_date=None):
    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No data found for {symbol}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.name == 'Date' or 'Date' not in df.columns:
        df = df.reset_index()
    os.makedirs("data", exist_ok=True)
    save_path = f"data/{symbol.lower()}_historical.csv"
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTC-USD")
    args = parser.parse_args()
    ingest_data(symbol=args.symbol)
