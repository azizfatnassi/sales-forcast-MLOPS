import os 
import joblib
from fastapi import FastAPI
import pandas as pd
from typing import List
import json
from pydantic import BaseModel

VERSION= os.getenv("VERSION","local")
MODEL_PATH=f"models/model_{VERSION}.pkl"
FEATUTRE_PATH="models/features.json"

model=joblib.load(MODEL_PATH)

with open(FEATUTRE_PATH,"r") as f :
    features=json.load(f)

app=FastAPI(title="Sales Forcast API")

class SalesRequest(BaseModel):
    data:List[dict]

@app.post("/predict")   
def predict(request: SalesRequest):
   
   try:
    df=pd.DataFrame(request.data) 
    X=df[features]
    preds = model.predict(X)

    return{"predictions":preds.tolist()}
   except KeyError as e:
        return {"error": f"Missing column in input: {str(e)}"}
   except Exception as e:
        return {"error": str(e)}
    