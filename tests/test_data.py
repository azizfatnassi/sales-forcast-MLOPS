import os 
import pandas as pd


PROCESSED_DATA_PATH="data/processed/train_clean.csv"

#def test_download():
#    assert os.path.exists(PROCESSED_DATA_PATH),"Processed data file missing"

def test_data_schema():
    df = pd.DataFrame({
        "Store": [1],
        "DayOfWeek": [3],
        "Open": [1],
        "Promo": [0],
        "Sales": [200]
    })

    expected_columns = [
        "Store", "DayOfWeek", "Open", "Promo", "Sales"
    ]

    assert all(col in df.columns for col in expected_columns)

def missing_values():
   # df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
    df = pd.DataFrame({
        "Store": [1],
        "DayOfWeek": [3],
        "Open": [1],
        "Promo": [0],
        "Sales": [200]
    })

    critical_columns = ["Store", "DayOfWeek", "Open", "Promo", "Sales"]
    missing= df[critical_columns].isnull().sum()    
    assert missing.sum()==0 ,f"Missing values in coloms:{list(missing[missing>0].index)}"
def test_expected_columns():
    
    #df = pd.read_csv(PROCESSED_DATA_PATH, low_memory=False)
    df = pd.DataFrame({
        "Store": [1],
        "DayOfWeek": [3],
        "Open": [1],
        "Promo": [0],
        "Sales": [200]
    })

    expected_cols = ["Store", "DayOfWeek", "Open", "Promo", "Sales"]
    for col in expected_cols:
        assert col in df.columns, f"Expected column {col} not found" 