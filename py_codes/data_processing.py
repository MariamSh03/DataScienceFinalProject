import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from datetime import datetime

def load_data(path: str) -> pd.DataFrame:
    """
    Load a CSV file and perform an initial data quality check.
    
    Parameters:
    -----------
    path : str
        The file path to the CSV file to be loaded. Can be a relative or absolute path.
        The file must exist and be readable.
    
    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the loaded data. The DataFrame will have
        the same structure as the CSV file with rows and columns preserved.
    
    Raises:
    -------
    FileNotFoundError
        If the specified file path does not exist.
    pd.errors.EmptyDataError
        If the CSV file is empty (no data rows).
    ValueError
        If the loaded DataFrame is empty (has no rows after loading).
    PermissionError
        If the file exists but cannot be read due to permission restrictions.
    UnicodeDecodeError
        If the file encoding is not compatible (e.g., not UTF-8).
    """
    df = pd.read_csv(path)
    
    if df.empty:
        raise ValueError("The dataset is empty!")
    
    generate_data_quality_report(df, "initial")
    return df

def generate_data_quality_report(df: pd.DataFrame, stage: str):
    """
    Generate and print a data quality summary report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to analyze. Must be a valid DataFrame object.
    stage : str
        A string identifier for the processing stage (e.g., "initial", "cleaned", 
        "transformed"). Used in the report header to indicate when the check was performed.
    
    Returns:
    --------
    None
        This function does not return a value. It prints the report to the console.
    
    Raises:
    -------
    AttributeError
        If df is not a pandas DataFrame or doesn't have the required methods
        (shape, isnull, duplicated).
    TypeError
        If df is None or not a DataFrame object.
    """
    report = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum())
    }
    print(f"\nData Quality Report ({stage}):")
    for key, value in report.items():
        print(f"{key}: {value}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame by filling them with appropriate statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame containing missing values to be handled. The original
        DataFrame is not modified; a copy is created and returned.
    
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with missing values filled:
        - Numeric columns (int64, float64): Missing values are filled with the median
        - Categorical/object columns: Missing values are filled with the mode (most frequent value)
        The original DataFrame remains unchanged.
    
    Raises:
    -------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If a categorical column has all missing values (no mode can be calculated).
    AttributeError
        If df doesn't have the required pandas DataFrame methods.
    
    Note:
    -----
    If a column has all NaN values, median() will return NaN and mode() will raise
    an IndexError if the column is empty. The function will attempt to fill with
    the first mode value, which may fail if no mode exists.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["int64", "float64"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame from which duplicate rows will be removed.
        Duplicates are identified by comparing all column values across rows.
    
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with duplicate rows removed. Only the first occurrence
        of each duplicate row is kept. The original DataFrame is not modified.
        The function also prints the number of duplicates removed and the new shape.
    
    Raises:
    -------
    TypeError
        If df is not a pandas DataFrame.
    AttributeError
        If df doesn't have the drop_duplicates method or shape attribute.
    """
    initial_shape = df.shape
    df = df.drop_duplicates()
    print(f"\nRemoved duplicates: {initial_shape[0] - df.shape[0]} rows")
    print(f"New dataset shape: {df.shape}")
    return df

def ensure_directories_exist():
    """
    Create data/raw and data/processed directories if they don't exist.
    
    Parameters:
    -----------
    None
        This function takes no parameters. It creates directories relative to
        the current working directory.
    
    Returns:
    --------
    None
        This function does not return a value. It creates directories and prints
        confirmation messages to the console.
    
    Raises:
    -------
    PermissionError
        If the current user doesn't have permission to create directories in
        the current working directory.
    OSError
        If there's a system-level error preventing directory creation (e.g., 
        disk full, invalid path characters, or path too long on Windows).
    """
    directories = ["data/raw", "data/processed"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Ensured directory exists: {directory}")

def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Save a processed DataFrame to a CSV file in the data/processed directory.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to save. Must be a valid DataFrame object.
        The DataFrame will be saved as a CSV file without the index column.
    filename : str
        The name of the output CSV file (e.g., "cleaned_data.csv"). 
        Should include the .csv extension. The file will be saved in the
        data/processed/ directory.
    
    Returns:
    --------
    str
        The full file path where the CSV file was saved (e.g., 
        "data/processed/cleaned_data.csv").
    
    Raises:
    -------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If filename is empty or None.
    PermissionError
        If the file cannot be written due to insufficient permissions.
    OSError
        If the directory cannot be created or the file cannot be written
        (e.g., disk full, invalid filename characters, or path too long).
    AttributeError
        If df doesn't have the to_csv method.
    """
    ensure_directories_exist()  # Ensure directory exists before saving
    save_path = os.path.join("data/processed", filename)
    df.to_csv(save_path, index=False)
    print(f"\nProcessed data saved to: {save_path}")
    return save_path

def display_cleaned_data_summary(df: pd.DataFrame):
    """
    Display a comprehensive summary of the cleaned dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to summarize. Should be a cleaned/processed DataFrame.
        The function displays information about all columns and data types.
    
    Returns:
    --------
    None
        This function does not return a value. It prints summary information
        to the console including:
        - Dataset info (shape, columns, data types, memory usage)
        - Descriptive statistics for all columns (numeric and categorical)
        - Top 5 most frequent values for each categorical column
    
    Raises:
    -------
    TypeError
        If df is not a pandas DataFrame.
    AttributeError
        If df doesn't have the required methods (info, describe, select_dtypes,
        value_counts).
    """
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
    Perform final quality checks on a cleaned dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The pandas DataFrame to check. Should be a cleaned/processed dataset
        that has undergone data cleaning operations.
    verbose : bool, optional
        If True (default), prints detailed check results to the console.
        If False, only returns the results dictionary without printing.
    
    Returns:
    --------
    dict
        A dictionary containing the quality check results with the following keys:
        - 'missing_values': int
            Total count of missing values across the entire DataFrame
        - 'duplicates': int
            Number of duplicate rows in the DataFrame
        - 'dtypes': dict
            Dictionary mapping column names to their data types (pandas dtype objects)
    
    Raises:
    -------
    TypeError
        If df is not a pandas DataFrame.
    AttributeError
        If df doesn't have the required methods (isnull, duplicated, dtypes).

    """
    results = {
        "missing_values": df.isnull().sum().sum(),
        "duplicates": df.duplicated().sum(),
        "dtypes": df.dtypes.to_dict()
    }

    if verbose:
        print("\n=== Cleaned Data Final Checks ===")
        if results["missing_values"] == 0:
            print("No missing values ")
        else:
            print(f"Warning: {results['missing_values']} missing values ")
        
        if results["duplicates"] == 0:
            print("No duplicate rows ")
        else:
            print(f"Warning: {results['duplicates']} duplicate rows ")
        
        print("\nColumn Data Types:")
        for col, dtype in results["dtypes"].items():
            print(f"{col}: {dtype}")

    return results

# Ensure directories exist before processing
ensure_directories_exist()

DATA_PATH = "data/raw/DataScience_salaries_2025.csv"

df = load_data(DATA_PATH)
df = handle_missing_values(df)
df = remove_duplicates(df)
processed_path = save_processed_data(df, "DataScience_salaries_2025_cleaned.csv")
display_cleaned_data_summary(df)
cleaned_data_check = check_cleaned_data(df)
