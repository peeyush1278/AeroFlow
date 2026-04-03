import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import mlflow
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.mlflow_utils import setup_mlflow

def create_features(df):
    df = df.copy()
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.sort_values('ds')
    for i in [1, 7, 30]:
        df[f'lag_{i}'] = df['y'].shift(i)
    for i in [7, 30]:
        df[f'rolling_mean_{i}'] = df['y'].shift(1).rolling(window=i).mean()
    return df.dropna()

def train_forecaster(data_path="data/btc-usd_historical.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}.")
    df = pd.read_csv(data_path).rename(columns={"Date": "ds", "Close": "y"})
    df_feat = create_features(df)
    feature_cols = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
    X, y = df_feat[feature_cols], df_feat['y']
    X_train, X_test = X.iloc[:-30], X.iloc[-30:]
    y_train, y_test = y.iloc[:-30], y.iloc[-30:]
    setup_mlflow()
    with mlflow.start_run(run_name="BTC_XGBoost_Forecaster"):
        params = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05, "objective": "reg:squarederror", "eval_metric": "mae"}
        mlflow.log_params(params)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        os.makedirs("src/model/artifacts", exist_ok=True)
        model.save_model("src/model/artifacts/xgb_model.json")
        mlflow.log_artifact("src/model/artifacts/xgb_model.json")

if __name__ == "__main__":
    train_forecaster()
