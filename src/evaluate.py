import json
import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error 

MODEL_PATH="models/model.pkl"
DATA_PATH="data/processed/train_clean.csv"
METRICS_PATH="models/metrics.json"
PREVIOUS_METRICS_PATH = "models/metrics_prev.json"

def load_data():
    return pd.read_csv(DATA_PATH)

def load_model():
     if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(" model file not found")
     return joblib.load(MODEL_PATH)

def evaluate_model(model,x_val,y_val):
    preds=model.predict(x_val)
    mse= mean_squared_error(y_val,preds)
    rmse= mse**0.5
    mae=mean_absolute_error(y_val,preds)

    return rmse,mae

def run_evaluation():
    df=load_data()

    features = ["Store", "DayOfWeek", "Open", "Promo"]
    target = "Sales"

    X = df[features]
    Y = df[target]

    x_train,x_val,y_train,y_val = train_test_split(X,Y,test_size=0.2,random_state=42)

    model= load_model()

    rmse,mae=evaluate_model(model,x_val,y_val)

    metrics={
        "rmse": rmse,
        "mae":mae
    }

    if os.path.exists(PREVIOUS_METRICS_PATH):
        with open(PREVIOUS_METRICS_PATH, "r") as f:
            prev_metrics = json.load(f)

        if rmse > prev_metrics["rmse"]:
            raise ValueError(
                f"Model performance degraded: RMSE increased "
                f"from {prev_metrics['rmse']} to {rmse}"
            )

    # Save current metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Update previous metrics
    with open(PREVIOUS_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation passed. Model performance acceptable.")
if __name__ == "__main__":
    run_evaluation()   
    

