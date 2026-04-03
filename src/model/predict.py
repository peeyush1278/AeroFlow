import pandas as pd
import xgboost as xgb
import os

class Predictor:
    def __init__(self, model_path="src/model/artifacts/xgb_model.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

    def predict(self, last_30_days_prices):
        df = pd.DataFrame(last_30_days_prices, columns=['y'])
        lag_1, lag_7, lag_30 = df['y'].iloc[-1], df['y'].iloc[-7], df['y'].iloc[-30]
        rolling_mean_7, rolling_mean_30 = df['y'].tail(7).mean(), df['y'].tail(30).mean()
        feature_cols = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_mean_30']
        features = pd.DataFrame([{ 'lag_1': lag_1, 'lag_7': lag_7, 'lag_30': lag_30, 'rolling_mean_7': rolling_mean_7, 'rolling_mean_30': rolling_mean_30 }])
        features = features[feature_cols]
        return float(self.model.predict(features)[0])
