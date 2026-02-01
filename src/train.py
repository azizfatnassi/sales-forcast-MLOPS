import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

# -------------------------
# BASE PATHS
# -------------------------
# Get repo root (two levels up from this script)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROCESSED_DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# -------------------------
# FUNCTIONS
# -------------------------
def get_version():
    """
    Get the model version from environment variable 'VERSION'
    Defaults to 'dev' if not set
    """
    return os.getenv("VERSION", "dev")

def load_data():
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {PROCESSED_DATA_PATH}")
    print(f"Loading data from: {PROCESSED_DATA_PATH}")
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

    model_path = os.path.join(version_dir, "model.pkl")
    features_path = os.path.join(version_dir, "features.json")

    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump(features, f)

    print(f"Artifacts saved with version: {version}")
    print(f"Model path: {model_path}")
    print(f"Features path: {features_path}")

# -------------------------
# MAIN
# -------------------------
def main():
    version = get_version()
    df = load_data()
    X, y, features = prepare_features(df)
    model = train_model(X, y)
    save_artifacts(model, features, version)

    print(f"Training completed with version: {version}")

if __name__ == "__main__":
    main()
