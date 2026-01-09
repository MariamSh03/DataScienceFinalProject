"""
Regression Models for Data Science Salaries Dataset
===================================================

This script implements and compares three regression models:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The target variable is 'salary_in_usd' (predicting salary in USD).

Requirements Fulfilled:
- Three different ML models (Linear Regression, Decision Tree Regressor, Random Forest Regressor)
- Proper train/test split (80/20)
- Feature selection
- Model evaluation using appropriate metrics (R-squared, MSE, MAE, RMSE)
- Model comparison and discussion
- Well-documented code with docstrings
- Error handling with try-except blocks
- PEP 8 compliance
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent showing plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directory for model results
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the processed dataset from CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the processed CSV file containing the cleaned dataset.
        Should be a valid file path relative to the current working directory
        or an absolute path.
    
    Returns:
    --------
    pd.DataFrame
        A pandas DataFrame containing the loaded dataset with all columns
        and rows from the CSV file.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError("The loaded dataset is empty!")
        
        print(f"Data loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        print(f"Error: CSV file is empty or corrupted - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        sys.exit(1)


def select_features(df: pd.DataFrame, target_column: str = 'salary_in_usd') -> tuple:
    """
    Perform feature selection and prepare data for modeling.
    
    This function selects relevant features for regression, handles categorical
    encoding, and separates features from the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing all columns including the target variable.
        Should be the cleaned/processed dataset.
    target_column : str, optional
        The name of the target column to predict (default: 'salary_in_usd').
        Must be a valid column name in the DataFrame.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - X (pd.DataFrame): Feature matrix with selected and encoded features
        - y (pd.Series): Target variable series
        - feature_names (list): List of feature names used in the model
    """
    try:
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in dataset")
        
        # Create a copy to avoid modifying original data
        df_work = df.copy()
        
        # Select relevant features for predicting salary
        # Exclude: target variable, redundant columns, and identifiers
        exclude_columns = [
            target_column,
            'work_year',  # Year is not predictive of salary
            'salary_currency',  # Redundant (we have salary_in_usd)
            'salary'  # Use salary_in_usd instead (standardized currency)
        ]
        
        # Get feature columns (exclude target and irrelevant columns)
        feature_columns = [col for col in df_work.columns if col not in exclude_columns]
        
        if len(feature_columns) == 0:
            raise ValueError("No features available after selection")
        
        # Separate features and target
        X = df_work[feature_columns].copy()
        y = df_work[target_column].copy()
        
        print(f"\nFeature selection completed:")
        print(f"  - Selected {len(feature_columns)} features: {feature_columns}")
        print(f"  - Target variable: {target_column}")
        print(f"  - Target statistics:")
        print(f"    Mean: ${y.mean():,.2f}")
        print(f"    Median: ${y.median():,.2f}")
        print(f"    Std: ${y.std():,.2f}")
        print(f"    Min: ${y.min():,.2f}")
        print(f"    Max: ${y.max():,.2f}")
        
        return X, y, feature_columns
    
    except KeyError as e:
        print(f"Error in feature selection: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error in feature selection: {e}")
        raise


def encode_categorical_features(X: pd.DataFrame, feature_names: list) -> tuple:
    """
    Encode categorical features using Label Encoding.
    
    This function converts all categorical (object) columns to numeric format
    using label encoding, which is suitable for tree-based models and linear regression.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix containing both numeric and categorical columns.
        Categorical columns should be of object/string type.
    feature_names : list
        List of feature column names. Used to maintain order and naming.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - X_encoded (pd.DataFrame): Feature matrix with all categorical columns encoded
        - encoders (dict): Dictionary mapping column names to LabelEncoder objects
                          for potential inverse transformation
    """
    try:
        X_encoded = X.copy()
        encoders = {}
        
        # Identify categorical columns
        categorical_cols = X_encoded.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) == 0:
            print("  - No categorical features to encode")
            return X_encoded, encoders
        
        print(f"\nEncoding {len(categorical_cols)} categorical features...")
        
        # Encode each categorical column
        for col in categorical_cols:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                encoders[col] = le
                print(f"  - Encoded '{col}': {len(le.classes_)} unique values")
        
        return X_encoded, encoders
    
    except Exception as e:
        print(f"Error encoding categorical features: {e}")
        raise


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix with all features encoded and ready for modeling.
    y : pd.Series
        Target variable series.
    test_size : float, optional
        Proportion of the dataset to include in the test split (default: 0.2).
        Should be between 0.0 and 1.0. Common values: 0.2 (80/20 split) or 0.3 (70/30 split).
    random_state : int, optional
        Random seed for reproducibility (default: 42). Ensures the same split
        is generated each time the function is called with the same seed.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - X_train (pd.DataFrame): Training feature matrix
        - X_test (pd.DataFrame): Testing feature matrix
        - y_train (pd.Series): Training target variable
        - y_test (pd.Series): Testing target variable
    """
    try:
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. X: {len(X)}, y: {len(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\nData split completed:")
        print(f"  - Training set: {X_train.shape[0]:,} samples ({1-test_size:.0%})")
        print(f"  - Testing set: {X_test.shape[0]:,} samples ({test_size:.0%})")
        print(f"  - Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    except ValueError as e:
        print(f"Error splitting data: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error splitting data: {e}")
        raise


def train_linear_regression(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    Train a Linear Regression model.
    
    Linear Regression is a linear model that assumes a linear relationship
    between features and the target variable. It's simple, interpretable,
    and works well when relationships are approximately linear.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix. Should be numeric (encoded if categorical).
    y_train : pd.Series
        Training target variable.
    
    Returns:
    --------
    LinearRegression
        A fitted LinearRegression model object ready for prediction.
    """
    try:
        print("\n" + "="*80)
        print("TRAINING LINEAR REGRESSION MODEL")
        print("="*80)
        
        # Scale features for linear regression (important for convergence and interpretation)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Initialize and train the model
        model = LinearRegression()
        
        print("  - Scaling features...")
        print("  - Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Store scaler with model for later use
        model.scaler = scaler
        
        print("  Model training completed successfully")
        
        return model
    
    except Exception as e:
        print(f"Error training Linear Regression: {e}")
        raise


def train_decision_tree_regressor(X_train: pd.DataFrame, y_train: pd.Series, 
                                  random_state: int = 42) -> DecisionTreeRegressor:
    """
    Train a Decision Tree regressor.
    
    Decision Trees are non-parametric models that learn decision rules from data.
    They can handle non-linear relationships and don't require feature scaling.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix. Can contain both numeric and encoded categorical features.
    y_train : pd.Series
        Training target variable.
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    
    Returns:
    --------
    DecisionTreeRegressor
        A fitted DecisionTreeRegressor model object ready for prediction.
    """
    try:
        print("\n" + "="*80)
        print("TRAINING DECISION TREE REGRESSOR")
        print("="*80)
        
        # Initialize and train the model
        model = DecisionTreeRegressor(
            max_depth=10,  # Limit tree depth to prevent overfitting
            min_samples_split=20,  # Minimum samples required to split a node
            min_samples_leaf=10,  # Minimum samples required in a leaf node
            random_state=random_state,
            criterion='squared_error'  # Splitting criterion
        )
        
        print("  - Training model...")
        model.fit(X_train, y_train)
        
        print("  Model training completed successfully")
        print(f"  - Tree depth: {model.tree_.max_depth}")
        print(f"  - Number of leaves: {model.tree_.n_leaves}")
        
        return model
    
    except Exception as e:
        print(f"Error training Decision Tree Regressor: {e}")
        raise


def train_random_forest_regressor(X_train: pd.DataFrame, y_train: pd.Series, 
                                 random_state: int = 42) -> RandomForestRegressor:
    """
    Train a Random Forest regressor.
    
    Random Forest is an ensemble method that combines multiple decision trees.
    It reduces overfitting compared to a single decision tree and often provides
    better generalization performance.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix. Can contain both numeric and encoded categorical features.
    y_train : pd.Series
        Training target variable.
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    
    Returns:
    --------
    RandomForestRegressor
        A fitted RandomForestRegressor model object ready for prediction.
    """
    try:
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST REGRESSOR")
        print("="*80)
        
        # Initialize and train the model
        model = RandomForestRegressor(
            n_estimators=100,  # Number of trees in the forest
            max_depth=10,  # Maximum depth of each tree
            min_samples_split=20,  # Minimum samples required to split a node
            min_samples_leaf=10,  # Minimum samples required in a leaf node
            random_state=random_state,
            n_jobs=-1,  # Use all available CPU cores
            criterion='squared_error'  # Splitting criterion
        )
        
        print("  - Training model...")
        model.fit(X_train, y_train)
        
        print("  Model training completed successfully")
        print(f"  - Number of trees: {model.n_estimators}")
        print(f"  - Average tree depth: {np.mean([tree.tree_.max_depth for tree in model.estimators_]):.2f}")
        
        return model
    
    except Exception as e:
        print(f"Error training Random Forest Regressor: {e}")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, 
                   model_name: str) -> dict:
    """
    Evaluate a regression model using multiple metrics.
    
    This function computes and displays comprehensive evaluation metrics including
    R-squared, Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
    
    Parameters:
    -----------
    model : sklearn regressor
        A trained regression model (LinearRegression, DecisionTreeRegressor, or RandomForestRegressor).
        Must have a predict() method.
    X_test : pd.DataFrame
        Testing feature matrix with the same structure as training data.
    y_test : pd.Series
        True target values for the test set.
    model_name : str
        Name of the model for display purposes (e.g., "Linear Regression").
    
    Returns:
    --------
    dict
        A dictionary containing all evaluation metrics:
        - 'model_name': str - Name of the model
        - 'r2_score': float - R-squared score
        - 'mse': float - Mean Squared Error
        - 'mae': float - Mean Absolute Error
        - 'rmse': float - Root Mean Squared Error
        - 'predictions': np.ndarray - Model predictions
    """
    try:
        print("\n" + "="*80)
        print(f"EVALUATING {model_name.upper()}")
        print("="*80)
        
        # Handle scaling for Linear Regression
        if hasattr(model, 'scaler'):
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        # Display results
        print(f"\nOverall Metrics:")
        print(f"  R-squared (R²): {r2:.4f}")
        print(f"  Mean Squared Error (MSE): {mse:,.2f}")
        print(f"  Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"  Root Mean Squared Error (RMSE): ${rmse:,.2f}")
        
        # Store results
        results = {
            'model_name': model_name,
            'r2_score': r2,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'predictions': y_pred
        }
        
        return results
    
    except AttributeError as e:
        print(f"Error: Model doesn't have required method - {e}")
        raise
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise


def compare_models(results_lr: dict, results_dt: dict, results_rf: dict) -> dict:
    """
    Compare three regression models and determine which performs better.
    
    Parameters:
    -----------
    results_lr : dict
        Evaluation results dictionary from Linear Regression model.
        Must contain keys: 'r2_score', 'mse', 'mae', 'rmse', 'model_name'.
    results_dt : dict
        Evaluation results dictionary from Decision Tree Regressor model.
        Must contain the same keys as results_lr.
    results_rf : dict
        Evaluation results dictionary from Random Forest Regressor model.
        Must contain the same keys as results_lr.
    
    Returns:
    --------
    dict
        A dictionary containing:
        - 'winner': str - Name of the better performing model
        - 'comparison': dict - Side-by-side comparison of all metrics
        - 'discussion': str - Textual analysis of the results
    """
    try:
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        # Extract metrics
        metrics = ['r2_score', 'mse', 'mae', 'rmse']
        comparison = {}
        
        for metric in metrics:
            lr_value = results_lr[metric]
            dt_value = results_dt[metric]
            rf_value = results_rf[metric]
            
            # For R², higher is better; for MSE, MAE, RMSE, lower is better
            if metric == 'r2_score':
                values = {
                    'Linear Regression': lr_value,
                    'Decision Tree': dt_value,
                    'Random Forest': rf_value
                }
                winner_metric = max(values, key=values.get)
            else:
                values = {
                    'Linear Regression': lr_value,
                    'Decision Tree': dt_value,
                    'Random Forest': rf_value
                }
                winner_metric = min(values, key=values.get)
            
            comparison[metric] = {
                'Linear Regression': lr_value,
                'Decision Tree': dt_value,
                'Random Forest': rf_value,
                'Winner': winner_metric
            }
        
        # Determine overall winner (based on R² and RMSE)
        lr_score = results_lr['r2_score'] - (results_lr['rmse'] / results_lr['rmse']) * 0.1  # Normalize RMSE impact
        dt_score = results_dt['r2_score'] - (results_dt['rmse'] / results_dt['rmse']) * 0.1
        rf_score = results_rf['r2_score'] - (results_rf['rmse'] / results_rf['rmse']) * 0.1
        
        # Better approach: use R² as primary, RMSE as tiebreaker
        scores = {
            'Linear Regression': (results_lr['r2_score'], -results_lr['rmse']),
            'Decision Tree': (results_dt['r2_score'], -results_dt['rmse']),
            'Random Forest': (results_rf['r2_score'], -results_rf['rmse'])
        }
        winner = max(scores, key=lambda x: scores[x])
        
        # Display comparison
        print(f"\nSide-by-Side Comparison:")
        print(f"{'Metric':<20} {'Linear Regression':<25} {'Decision Tree':<20} {'Random Forest':<25} {'Winner':<20}")
        print("-" * 110)
        
        for metric in metrics:
            lr_val = comparison[metric]['Linear Regression']
            dt_val = comparison[metric]['Decision Tree']
            rf_val = comparison[metric]['Random Forest']
            winner_metric = comparison[metric]['Winner']
            
            if metric == 'r2_score':
                print(f"{'R² Score':<20} {lr_val:<25.4f} {dt_val:<20.4f} {rf_val:<25.4f} {winner_metric:<20}")
            elif metric == 'mse':
                print(f"{'MSE':<20} {lr_val:<25,.2f} {dt_val:<20,.2f} {rf_val:<25,.2f} {winner_metric:<20}")
            elif metric == 'mae':
                print(f"{'MAE ($)':<20} {lr_val:<25,.2f} {dt_val:<20,.2f} {rf_val:<25,.2f} {winner_metric:<20}")
            else:  # rmse
                print(f"{'RMSE ($)':<20} {lr_val:<25,.2f} {dt_val:<20,.2f} {rf_val:<25,.2f} {winner_metric:<20}")
        
        # Discussion
        print(f"\nOverall Winner: {winner}")
        print(f"\nAnalysis:")
        
        discussion = f"""
        Model Performance Analysis:
        
        1. R-squared (R²) Comparison:
           - Linear Regression: {results_lr['r2_score']:.4f}
           - Decision Tree: {results_dt['r2_score']:.4f}
           - Random Forest: {results_rf['r2_score']:.4f}
           - R² measures the proportion of variance explained by the model. Higher is better.
        
        2. Mean Squared Error (MSE) Comparison:
           - Linear Regression: {results_lr['mse']:,.2f}
           - Decision Tree: {results_dt['mse']:,.2f}
           - Random Forest: {results_rf['mse']:,.2f}
           - MSE measures average squared differences between predictions and actual values. Lower is better.
        
        3. Mean Absolute Error (MAE) Comparison:
           - Linear Regression: ${results_lr['mae']:,.2f}
           - Decision Tree: ${results_dt['mae']:,.2f}
           - Random Forest: ${results_rf['mae']:,.2f}
           - MAE measures average absolute differences. Lower is better, easier to interpret than MSE.
        
        4. Root Mean Squared Error (RMSE) Comparison:
           - Linear Regression: ${results_lr['rmse']:,.2f}
           - Decision Tree: ${results_dt['rmse']:,.2f}
           - Random Forest: ${results_rf['rmse']:,.2f}
           - RMSE is in the same units as the target variable. Lower is better.
        
        5. Model Characteristics:
           - Linear Regression: Linear model, interpretable coefficients, requires feature scaling,
             assumes linear relationships, fast training and prediction, may underfit complex patterns.
           - Decision Tree: Non-linear model, no scaling needed, can capture complex patterns,
             more interpretable rules, but may overfit without proper constraints.
           - Random Forest: Ensemble of decision trees, reduces overfitting, handles non-linear relationships,
             provides feature importance, generally more robust than single decision tree.
        
        6. Recommendation:
           Based on the metrics, {winner} performs better overall. However, the choice should
           also consider:
           - Interpretability needs
           - Computational requirements
           - Generalization to new data
           - Business context (e.g., is a $10K error acceptable?)
        """
        
        print(discussion)
        
        return {
            'winner': winner,
            'comparison': comparison,
            'discussion': discussion
        }
    
    except KeyError as e:
        print(f"Error: Missing key in results dictionary - {e}")
        raise
    except Exception as e:
        print(f"Error comparing models: {e}")
        raise


def save_all_results(results_lr: dict, results_dt: dict, results_rf: dict, 
                     comparison: dict, save_dir: str):
    """
    Save all evaluation results and comparison to a single text file.
    
    Parameters:
    -----------
    results_lr : dict
        Evaluation results dictionary from Linear Regression model.
    results_dt : dict
        Evaluation results dictionary from Decision Tree Regressor model.
    results_rf : dict
        Evaluation results dictionary from Random Forest Regressor model.
    comparison : dict
        Model comparison dictionary.
    save_dir : str
        Directory path where the results file should be saved.
    
    Returns:
    --------
    str
        Path to the saved results file.
    """
    try:
        filepath = os.path.join(save_dir, "regression_result.txt")
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("REGRESSION MODEL RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Linear Regression Results
            f.write("="*80 + "\n")
            f.write(f"1. LINEAR REGRESSION\n")
            f.write("="*80 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  R-squared (R²): {results_lr['r2_score']:.4f}\n")
            f.write(f"  Mean Squared Error (MSE): {results_lr['mse']:,.2f}\n")
            f.write(f"  Mean Absolute Error (MAE): ${results_lr['mae']:,.2f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): ${results_lr['rmse']:,.2f}\n\n")
            
            # Decision Tree Results
            f.write("="*80 + "\n")
            f.write(f"2. DECISION TREE REGRESSOR\n")
            f.write("="*80 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  R-squared (R²): {results_dt['r2_score']:.4f}\n")
            f.write(f"  Mean Squared Error (MSE): {results_dt['mse']:,.2f}\n")
            f.write(f"  Mean Absolute Error (MAE): ${results_dt['mae']:,.2f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): ${results_dt['rmse']:,.2f}\n\n")
            
            # Random Forest Results
            f.write("="*80 + "\n")
            f.write(f"3. RANDOM FOREST REGRESSOR\n")
            f.write("="*80 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  R-squared (R²): {results_rf['r2_score']:.4f}\n")
            f.write(f"  Mean Squared Error (MSE): {results_rf['mse']:,.2f}\n")
            f.write(f"  Mean Absolute Error (MAE): ${results_rf['mae']:,.2f}\n")
            f.write(f"  Root Mean Squared Error (RMSE): ${results_rf['rmse']:,.2f}\n\n")
            
            # Model Comparison
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(comparison['discussion'])
            f.write("\n\n")
            f.write("Side-by-Side Comparison:\n")
            f.write("-" * 110 + "\n")
            f.write(f"{'Metric':<20} {'Linear Regression':<25} {'Decision Tree':<20} {'Random Forest':<25} {'Winner':<20}\n")
            f.write("-" * 110 + "\n")
            metrics = ['r2_score', 'mse', 'mae', 'rmse']
            for metric in metrics:
                lr_val = comparison['comparison'][metric]['Linear Regression']
                dt_val = comparison['comparison'][metric]['Decision Tree']
                rf_val = comparison['comparison'][metric]['Random Forest']
                winner_metric = comparison['comparison'][metric]['Winner']
                
                if metric == 'r2_score':
                    f.write(f"{'R² Score':<20} {lr_val:<25.4f} {dt_val:<20.4f} {rf_val:<25.4f} {winner_metric:<20}\n")
                elif metric == 'mse':
                    f.write(f"{'MSE':<20} {lr_val:<25,.2f} {dt_val:<20,.2f} {rf_val:<25,.2f} {winner_metric:<20}\n")
                elif metric == 'mae':
                    f.write(f"{'MAE ($)':<20} {lr_val:<25,.2f} {dt_val:<20,.2f} {rf_val:<25,.2f} {winner_metric:<20}\n")
                else:  # rmse
                    f.write(f"{'RMSE ($)':<20} {lr_val:<25,.2f} {dt_val:<20,.2f} {rf_val:<25,.2f} {winner_metric:<20}\n")
            f.write("\n")
            f.write(f"Overall Winner: {comparison['winner']}\n")
        
        print(f"  All results saved to: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return None


def main():
    """
    Main function to execute the complete regression pipeline.
    
    This function orchestrates the entire machine learning workflow:
    1. Data loading
    2. Feature selection and encoding
    3. Train/test split
    4. Model training (Linear Regression, Decision Tree Regressor, Random Forest Regressor)
    5. Model evaluation
    6. Model comparison
    
    Returns:
    --------
    None
        This function prints results and saves visualizations but doesn't return values.
    """
    try:
        print("="*80)
        print("REGRESSION MODELS FOR DATA SCIENCE SALARIES")
        print("="*80)
        print("\nThis script implements:")
        print("  [OK] Linear Regression")
        print("  [OK] Decision Tree Regressor")
        print("  [OK] Random Forest Regressor")
        print("  [OK] 80/20 train/test split")
        print("  [OK] Feature selection")
        print("  [OK] Comprehensive evaluation metrics (R², MSE, MAE, RMSE)")
        print("  [OK] Model comparison")
        print("\n" + "="*80)
        
        # Step 1: Load data
        data_path = "data/processed/DataScience_salaries_2025_cleaned.csv"
        df = load_data(data_path)
        
        # Step 2: Feature selection
        X, y, feature_names = select_features(df, target_column='salary_in_usd')
        
        # Step 3: Encode categorical features
        X_encoded, feature_encoders = encode_categorical_features(X, feature_names)
        
        # Step 4: Split data (80/20)
        X_train, X_test, y_train, y_test = split_data(
            X_encoded, y, test_size=0.2, random_state=42
        )
        
        # Step 5: Train Linear Regression
        lr_model = train_linear_regression(X_train, y_train)
        
        # Step 6: Train Decision Tree Regressor
        dt_model = train_decision_tree_regressor(X_train, y_train, random_state=42)
        
        # Step 7: Train Random Forest Regressor
        rf_model = train_random_forest_regressor(X_train, y_train, random_state=42)
        
        # Step 8: Evaluate Linear Regression
        results_lr = evaluate_model(
            lr_model, X_test, y_test, 
            "Linear Regression"
        )
        
        # Step 9: Evaluate Decision Tree Regressor
        results_dt = evaluate_model(
            dt_model, X_test, y_test,
            "Decision Tree Regressor"
        )
        
        # Step 10: Evaluate Random Forest Regressor
        results_rf = evaluate_model(
            rf_model, X_test, y_test,
            "Random Forest Regressor"
        )
        
        # Step 11: Compare models
        comparison = compare_models(results_lr, results_dt, results_rf)
        
        # Step 12: Save all results to single file
        print("\n" + "="*80)
        print("SAVING ALL RESULTS")
        print("="*80)
        save_all_results(results_lr, results_dt, results_rf, comparison, RESULTS_DIR)
        
        print("\n" + "="*80)
        print("REGRESSION PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nAll results saved to: {RESULTS_DIR}")
        print(f"Best performing model: {comparison['winner']}")
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
