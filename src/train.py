import os
import json
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


PROCESSED_DATA_PATH="data/processed/train_clean.csv"
MODEL_DIR="models"
MODEL_PATH=os.path.join(MODEL_DIR,"model.pkl")
FEATURES_PATH=os.path.join(MODEL_DIR,"features.json")


def load_data():
   if not os.path.exists(PROCESSED_DATA_PATH):
      raise FileNotFoundError("processed data not found")
   return pd.read_csv(PROCESSED_DATA_PATH)

def prepare_features(df):
   target= "Sales"

   features=[
        "Store",
        "DayOfWeek",
        "Open",
        "Promo"
    ]
   
   X=df[features]
   Y=df[target]

   return X,Y, features

def train_model(X, Y):
   model=LinearRegression()
   model.fit(X, Y)
   return model

def save_artifacts(model,features):
   os.makedirs(MODEL_DIR,exist_ok=True)
   joblib.dump(model,MODEL_PATH)

   with open(FEATURES_PATH,"w") as f:
      json.dump(features,f)

def run_training_pipeline():
   df=load_data()
   X,Y, features=prepare_features(df)

   model=train_model(X, Y) 

   save_artifacts(model, features)
   print("Training completed")

if __name__ =="__main__":
   run_training_pipeline()  
