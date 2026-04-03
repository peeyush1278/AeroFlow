from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.model.predict import Predictor

app = FastAPI(title="AeroFlow Crypto API")
predictor = Predictor()

class PriceHistory(BaseModel):
    prices: list[float]

@app.get("/")
def read_root(): return {"message": "AeroFlow Crypto API is Live"}

@app.post("/predict")
def predict_price(history: PriceHistory):
    if len(history.prices) < 30: raise HTTPException(status_code=400, detail="Require at least 30 historical prices.")
    try:
        prediction = predictor.predict(history.prices)
        return {"prediction": round(prediction, 2)}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
