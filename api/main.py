from fastapi import FastAPI
import os, joblib, json, pandas as pd
from typing import List
from pydantic import BaseModel

app = FastAPI(title="Sales Forecast API")

class SalesRequest(BaseModel):
    data: List[dict]

model = None
features = None

def load_model():
    global model, features
    if model is None:
        VERSION = os.getenv("VERSION", "dev")
        MODEL_PATH = f"models/model_{VERSION}.pkl"
        FEATURE_PATH = "models/features.json"
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_PATH, "r") as f:
            features = json.load(f)

@app.post("/predict")
def predict(request: SalesRequest):
    try:
        load_model()
        df = pd.DataFrame(request.data)
        X = df[features]
        preds = model.predict(X)
        return {"predictions": preds.tolist()}
    except KeyError as e:
        return {"error": f"Missing column: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}