import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from datetime import datetime

def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file and do an initial data quality check."""
    df = pd.read_csv(path)
    
    if df.empty:
        raise ValueError("The dataset is empty!")
    
    generate_data_quality_report(df, "initial")
    return df

def generate_data_quality_report(df: pd.DataFrame, stage: str):
    """Print a quick data quality summary."""
    report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }
    print(f"\nData Quality Report ({stage}):")
    for key, value in report.items():
        print(f"{key}: {value}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numbers, mode for categorical."""
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"\nRemoved duplicates: {initial_shape[0] - df.shape[0]} rows")
    print(f"New dataset shape: {df.shape}")
    return df

def save_processed_data(df: pd.DataFrame, filename: str):
    """Save cleaned data to data/processed/ folder."""
    save_path = os.path.join("data/processed", filename)
    df.to_csv(save_path, index=False)
    print(f"\nProcessed data saved to: {save_path}")
    return save_path

def display_cleaned_data_summary(df: pd.DataFrame):
    """Show basic info, descriptive stats, and top categorical values."""
    print("\n=== Cleaned Dataset Info ===")
    print(df.info())
    
    print("\n=== Cleaned Dataset Summary Statistics ===")
    print(df.describe(include='all'))
    
    # Top 5 most common values for categorical columns
    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        print(f"\nTop 5 values in '{col}':")
        print(df[col].value_counts().head())

def check_cleaned_data(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Check the cleaned dataset for:
      - Missing values
      - Duplicate rows
      - Column data types
    Returns a dictionary with keys 'missing_values', 'duplicates', 'dtypes'.
    """
    results = {
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.to_dict()
    }

    if verbose:
        print("\n=== Cleaned Data Final Checks ===")
        if results["missing_values"] == 0:
            print("No missing values ✅")
        else:
            print(f"Warning: {results['missing_values']} missing values ❌")
        
        if results["duplicates"] == 0:
            print("No duplicate rows ✅")
        else:
            print(f"Warning: {results['duplicates']} duplicate rows ❌")
        
        print("\nColumn Data Types:")
        for col, dtype in results["dtypes"].items():
            print(f"{col}: {dtype}")

    return results

DATA_PATH = "data/raw/DataScience_salaries_2025.csv"

df = load_data(DATA_PATH)
df = handle_missing_values(df)
df = remove_duplicates(df)
processed_path = save_processed_data(df, "DataScience_salaries_2025_cleaned.csv")
display_cleaned_data_summary(df)
cleaned_data_check = check_cleaned_data(df)
