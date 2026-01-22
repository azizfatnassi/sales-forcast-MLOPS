import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "..", "data", "raw")
PROCESSED_PATH = os.path.join(BASE_DIR, "..", "data", "processed")

REQUIRED_COLUMNS = ["Date","Store","Sales","Open"]

def load_raw_data():

    train_path= os.path.join(RAW_PATH,"train.csv")
    store_path= os.path.join(RAW_PATH,"store.csv")

    if not os.path.exists(train_path):
      raise FileNotFoundError("train.csv not found in data\raw")
    if not os.path.exists(store_path):
     raise FileNotFoundError('store.csv not found in data\raw')
     
    train=pd.read_csv(train_path,parse_dates=["Date"])
    store = pd.read_csv(store_path)

    if train.empty:
      raise ValueError("train data is empty")
    
    if store.empty:
      raise ValueError("store is empty")
    
    df=train.merge(store,on='Store',how="left")

    return df

def validate_schema(df):
  
  missing_cols=[col for col in REQUIRED_COLUMNS if col not in df.columns]
  if missing_cols:
    raise ValueError(f"missing required columns:{missing_cols}")
  

def validate_types(df):
  
  if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
    raise TypeError("Date column must be datetime")
  if not pd.api.types.is_numeric_dtype(df["Sales"]):
    raise TypeError("sales must be numeric")
  
def validate_missing_values(df):

   missing_ratio=df.isnull().mean()
   critical_missing= missing_ratio[missing_ratio>0.6]
   if not critical_missing.empty:
     raise ValueError(f"too many values in columns: {list(critical_missing.index)}")  
   

def validate_business_rules(df):
  if (df["Sales"]<0).any() :
    raise ValueError("negative Sale detected") 
  if ((df["Sales"]>0) & (df["Open"]==0) ).any():
    raise ValueError("cant have sales wen stores are closed")
  
def preprocess_data(df):
  df= df[df["Open"]==1].copy()

  num_cols=[
    "CompetitionDistance",
        "CompetitionOpenSinceMonth",
        "CompetitionOpenSinceYear",
        "Promo2SinceWeek",
        "Promo2SinceYear",
  ]

  df[num_cols]=df[num_cols].fillna(0)

  cat_cols= ["StoreType", "Assortment"]
  df[cat_cols] = df[cat_cols].fillna("Unknown")

    # Time features
  df["Year"] = df["Date"].dt.year
  df["Month"] = df["Date"].dt.month
  df["WeekOfYear"] = df["Date"].dt.isocalendar().week

  return df

def save_processed_data(df):
  
  os.makedirs(PROCESSED_PATH,exist_ok=True)
  output_path= os.path.join(PROCESSED_PATH,"train_clean.csv")
  df.to_csv(output_path,index=False)
  print(f"Processed data saved to {output_path}")


def run_data_pipeline():
    df = load_raw_data()
    validate_schema(df)
    validate_types(df)
    validate_missing_values(df)
    validate_business_rules(df)
    df = preprocess_data(df)
    save_processed_data(df)


if __name__ == "__main__":
    run_data_pipeline()
   

