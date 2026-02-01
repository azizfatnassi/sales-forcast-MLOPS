import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

PROCESSED_DATA_PATH = "data/processed/train_clean.csv"
MODELS_DIR = "models"

def get_version():
    return os.getenv("VERSION", "dev")

def load_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError("Processed data not found")
    return pd.read_csv(PROCESSED_DATA_PATH)

def prepare_features(df):
    target = "Sales"
    features = ["Store", "DayOfWeek", "Open", "Promo"]

    X = df[features]
    y = df[target]
    return X, y, features

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

def save_artifacts(model, features, version):
    version_dir = os.path.join(MODELS_DIR, version)
    os.makedirs(version_dir, exist_ok=True)

    joblib.dump(model, os.path.join(version_dir, "model.pkl"))

    with open(os.path.join(version_dir, "features.json"), "w") as f:
        json.dump(features, f)

    print(f"Artifacts saved with version: {version}")

def main():
    version = get_version()
    df = load_data()
    X, y, features = prepare_features(df)
    model = train_model(X, y)
    save_artifacts(model, features, version)

    print(f"Training completed with version: {version}")

if __name__ == "__main__":
    main()
