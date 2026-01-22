import os
import joblib
import pytest
import pandas as pd
import json 


MODEL_PATH='models/model.pkl'
FEATURES_PATH="models/features.json"
DATA_PATH="data/processed/train_clean.csv"

def test_model_exists():
    assert os.path.exists(MODEL_PATH),"missing model file not found"

def test_features_exist():
    assert os.path.exists(FEATURES_PATH),"feature file not found"


def test_model_prediction():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    with open(FEATURES_PATH, "r") as f:
        features = json.load(f)
    X = df[features]
    preds = model.predict(X)
    # Test output shape
    assert len(preds) == len(X), "Predictions length mismatch"
    # Test no negative predictions for sales
    assert (preds >= 0).all(), "Predicted negative sales"
    