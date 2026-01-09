"""
Classification Models for Data Science Salaries Dataset
=======================================================

This script implements and compares three classification models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

The target variable is 'experience_level' (EN=Entry, MI=Mid, SE=Senior, EX=Executive).

Requirements Fulfilled:
- Three different ML models (Logistic Regression, Decision Tree, Random Forest)
- Proper train/test split (80/20)
- Feature selection
- Model evaluation using appropriate metrics (Accuracy, Confusion Matrix, Precision, Recall)
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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    classification_report,
    f1_score
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


def select_features(df: pd.DataFrame, target_column: str = 'experience_level') -> tuple:
    """
    Perform feature selection and prepare data for modeling.
    
    This function selects relevant features for classification, handles categorical
    encoding, and separates features from the target variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing all columns including the target variable.
        Should be the cleaned/processed dataset.
    target_column : str, optional
        The name of the target column to predict (default: 'experience_level').
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
        
        # Select relevant features for predicting experience level
        # Exclude: target variable, redundant columns, and identifiers
        exclude_columns = [
            target_column,
            'work_year',  # Year is not predictive of experience level
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
        print(f"  - Target distribution:\n{y.value_counts().sort_index()}")
        
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
    using label encoding, which is suitable for tree-based models and logistic regression.
    
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


def encode_target(y: pd.Series) -> tuple:
    """
    Encode the target variable using Label Encoding.
    
    Parameters:
    -----------
    y : pd.Series
        Target variable series containing categorical values.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - y_encoded (np.ndarray): Encoded target variable as numpy array
        - label_encoder (LabelEncoder): Fitted LabelEncoder object for inverse transformation
    """
    try:
        if y.empty:
            raise ValueError("Target variable is empty")
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"\nTarget variable encoded:")
        print(f"  - Classes: {list(le.classes_)}")
        print(f"  - Encoded values: {dict(zip(le.classes_, range(len(le.classes_))))}")
        
        return y_encoded, le
    
    except Exception as e:
        print(f"Error encoding target variable: {e}")
        raise


def split_data(X: pd.DataFrame, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix with all features encoded and ready for modeling.
    y : np.ndarray
        Encoded target variable as a numpy array.
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
        - y_train (np.ndarray): Training target variable
        - y_test (np.ndarray): Testing target variable
    """
    try:
        if not (0.0 < test_size < 1.0):
            raise ValueError(f"test_size must be between 0.0 and 1.0, got {test_size}")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length. X: {len(X)}, y: {len(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
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


def train_logistic_regression(X_train: pd.DataFrame, y_train: np.ndarray, 
                              random_state: int = 42) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.
    
    Logistic Regression is a linear model that uses a logistic function to model
    the probability of a categorical outcome. It's suitable for multi-class
    classification problems.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix. Should be numeric (encoded if categorical).
    y_train : np.ndarray
        Training target variable as a numpy array of encoded class labels.
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    
    Returns:
    --------
    LogisticRegression
        A fitted LogisticRegression model object ready for prediction.
    """
    try:
        print("\n" + "="*80)
        print("TRAINING LOGISTIC REGRESSION MODEL")
        print("="*80)
        
        # Scale features for logistic regression (important for convergence)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Initialize and train the model
        model = LogisticRegression(
            multi_class='multinomial',  # For multi-class classification
            solver='lbfgs',  # Good for small to medium datasets
            max_iter=1000,  # Maximum iterations for convergence
            random_state=random_state,
            n_jobs=-1,  # Use all available CPU cores
            class_weight='balanced'  # Handle class imbalance by balancing weights
        )
        
        print("  - Scaling features...")
        print("  - Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Store scaler with model for later use
        model.scaler = scaler
        
        print("  Model training completed successfully")
        
        return model
    
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")
        raise


def train_decision_tree(X_train: pd.DataFrame, y_train: np.ndarray, 
                        random_state: int = 42) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier.
    
    Decision Trees are non-parametric models that learn decision rules from data.
    They can handle non-linear relationships and don't require feature scaling.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix. Can contain both numeric and encoded categorical features.
    y_train : np.ndarray
        Training target variable as a numpy array of encoded class labels.
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    
    Returns:
    --------
    DecisionTreeClassifier
        A fitted DecisionTreeClassifier model object ready for prediction.
    """
    try:
        print("\n" + "="*80)
        print("TRAINING DECISION TREE CLASSIFIER")
        print("="*80)
        
        # Initialize and train the model
        model = DecisionTreeClassifier(
            max_depth=10,  # Limit tree depth to prevent overfitting
            min_samples_split=20,  # Minimum samples required to split a node
            min_samples_leaf=10,  # Minimum samples required in a leaf node
            random_state=random_state,
            criterion='gini',  # Splitting criterion
            class_weight='balanced'  # Handle class imbalance by balancing weights
        )
        
        print("  - Training model...")
        model.fit(X_train, y_train)
        
        print("  Model training completed successfully")
        print(f"  - Tree depth: {model.tree_.max_depth}")
        print(f"  - Number of leaves: {model.tree_.n_leaves}")
        
        return model
    
    except Exception as e:
        print(f"Error training Decision Tree: {e}")
        raise


def evaluate_model(model, X_test: pd.DataFrame, y_test: np.ndarray, 
                   model_name: str, label_encoder: LabelEncoder) -> dict:
    """
    Evaluate a classification model using multiple metrics.
    
    This function computes and displays comprehensive evaluation metrics including
    accuracy, confusion matrix, precision, recall, and F1-score for each class.
    
    Parameters:
    -----------
    model : sklearn classifier
        A trained classification model (LogisticRegression or DecisionTreeClassifier).
        Must have a predict() method.
    X_test : pd.DataFrame
        Testing feature matrix with the same structure as training data.
    y_test : np.ndarray
        True target values for the test set (encoded).
    model_name : str
        Name of the model for display purposes (e.g., "Logistic Regression").
    label_encoder : LabelEncoder
        LabelEncoder object used to encode the target variable. Used to convert
        encoded predictions back to original class names.
    
    Returns:
    --------
    dict
        A dictionary containing all evaluation metrics:
        - 'accuracy': float - Overall accuracy score
        - 'confusion_matrix': np.ndarray - Confusion matrix
        - 'precision': float or array - Precision scores (macro-averaged and per-class)
        - 'recall': float or array - Recall scores (macro-averaged and per-class)
        - 'f1_score': float or array - F1 scores (macro-averaged and per-class)
        - 'classification_report': str - Detailed classification report
    """
    try:
        print("\n" + "="*80)
        print(f"EVALUATING {model_name.upper()}")
        print("="*80)
        
        # Handle scaling for Logistic Regression
        if hasattr(model, 'scaler'):
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Classification report
        class_names = label_encoder.classes_
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        
        # Display results
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} (macro-averaged)")
        print(f"  Recall:    {recall:.4f} (macro-averaged)")
        print(f"  F1-Score:  {f1:.4f} (macro-averaged)")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'precision': precision,
            'precision_per_class': precision_per_class,
            'recall': recall,
            'recall_per_class': recall_per_class,
            'f1_score': f1,
            'f1_per_class': f1_per_class,
            'classification_report': report,
            'predictions': y_pred
        }
        
        return results
    
    except AttributeError as e:
        print(f"Error: Model doesn't have required method - {e}")
        raise
    except Exception as e:
        print(f"Error evaluating model: {e}")
        raise


def save_all_results(results_lr: dict, results_dt: dict, results_rf: dict, 
                     comparison: dict, save_dir: str, class_names: list):
    """
    Save all evaluation results and comparison to a single text file.
    
    Parameters:
    -----------
    results_lr : dict
        Evaluation results dictionary from Logistic Regression model.
    results_dt : dict
        Evaluation results dictionary from Decision Tree model.
    results_rf : dict
        Evaluation results dictionary from Random Forest model.
    comparison : dict
        Model comparison dictionary.
    save_dir : str
        Directory path where the results file should be saved.
    class_names : list
        List of class names.
    
    Returns:
    --------
    str
        Path to the saved results file.
    """
    try:
        filepath = os.path.join(save_dir, "classification_results.txt")
        
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CLASSIFICATION MODEL RESULTS\n")
            f.write("="*80 + "\n\n")
            
            # Logistic Regression Results
            f.write("="*80 + "\n")
            f.write(f"1. LOGISTIC REGRESSION\n")
            f.write("="*80 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:  {results_lr['accuracy']:.4f} ({results_lr['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {results_lr['precision']:.4f} (macro-averaged)\n")
            f.write(f"  Recall:    {results_lr['recall']:.4f} (macro-averaged)\n")
            f.write(f"  F1-Score:  {results_lr['f1_score']:.4f} (macro-averaged)\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(results_lr['confusion_matrix']))
            f.write("\n\n")
            
            # Decision Tree Results
            f.write("="*80 + "\n")
            f.write(f"2. DECISION TREE\n")
            f.write("="*80 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:  {results_dt['accuracy']:.4f} ({results_dt['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {results_dt['precision']:.4f} (macro-averaged)\n")
            f.write(f"  Recall:    {results_dt['recall']:.4f} (macro-averaged)\n")
            f.write(f"  F1-Score:  {results_dt['f1_score']:.4f} (macro-averaged)\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(results_dt['confusion_matrix']))
            f.write("\n\n")
            
            # Random Forest Results
            f.write("="*80 + "\n")
            f.write(f"3. RANDOM FOREST\n")
            f.write("="*80 + "\n\n")
            f.write("Overall Metrics:\n")
            f.write(f"  Accuracy:  {results_rf['accuracy']:.4f} ({results_rf['accuracy']*100:.2f}%)\n")
            f.write(f"  Precision: {results_rf['precision']:.4f} (macro-averaged)\n")
            f.write(f"  Recall:    {results_rf['recall']:.4f} (macro-averaged)\n")
            f.write(f"  F1-Score:  {results_rf['f1_score']:.4f} (macro-averaged)\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(results_rf['confusion_matrix']))
            f.write("\n\n")
            
            # Model Comparison
            f.write("="*80 + "\n")
            f.write("MODEL COMPARISON\n")
            f.write("="*80 + "\n\n")
            f.write(comparison['discussion'])
            f.write("\n\n")
            f.write("Side-by-Side Comparison:\n")
            f.write("-" * 110 + "\n")
            f.write(f"{'Metric':<20} {'Logistic Regression':<25} {'Decision Tree':<20} {'Random Forest':<25} {'Winner':<20}\n")
            f.write("-" * 110 + "\n")
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for metric in metrics:
                lr_val = comparison['comparison'][metric]['Logistic Regression']
                dt_val = comparison['comparison'][metric]['Decision Tree']
                rf_val = comparison['comparison'][metric]['Random Forest']
                winner_metric = comparison['comparison'][metric]['Winner']
                f.write(f"{metric.capitalize():<20} {lr_val:<25.4f} {dt_val:<20.4f} {rf_val:<25.4f} {winner_metric:<20}\n")
            f.write("\n")
            f.write(f"Overall Winner: {comparison['winner']}\n")
        
        print(f"  All results saved to: {filepath}")
        return filepath
    
    except Exception as e:
        print(f"Error saving results: {e}")
        return None


def plot_confusion_matrix(cm: np.ndarray, class_names: list, model_name: str, 
                         save_path: str = None):
    """
    Plot and save a confusion matrix visualization.
    
    Parameters:
    -----------
    cm : np.ndarray
        Confusion matrix as a 2D numpy array.
    class_names : list
        List of class names (strings) corresponding to the matrix rows/columns.
    model_name : str
        Name of the model for the plot title.
    save_path : str, optional
        File path to save the plot. If None, plot is displayed but not saved.
        Should include directory and filename with extension (e.g., '.png').
    
    Returns:
    --------
    None
        This function saves the plot to disk without displaying it.
    """
    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  Confusion matrix saved to: {save_path}")
        
        plt.close()  # Close the figure to free memory
    
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")


def train_random_forest(X_train: pd.DataFrame, y_train: np.ndarray, 
                        random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Random Forest is an ensemble method that combines multiple decision trees.
    It reduces overfitting compared to a single decision tree and often provides
    better generalization performance.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix. Can contain both numeric and encoded categorical features.
    y_train : np.ndarray
        Training target variable as a numpy array of encoded class labels.
    random_state : int, optional
        Random seed for reproducibility (default: 42).
    
    Returns:
    --------
    RandomForestClassifier
        A fitted RandomForestClassifier model object ready for prediction.
    """
    try:
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("="*80)
        
        # Initialize and train the model
        model = RandomForestClassifier(
            n_estimators=100,  # Number of trees in the forest
            max_depth=10,  # Maximum depth of each tree
            min_samples_split=20,  # Minimum samples required to split a node
            min_samples_leaf=10,  # Minimum samples required in a leaf node
            random_state=random_state,
            n_jobs=-1,  # Use all available CPU cores
            criterion='gini'  # Splitting criterion
        )
        
        print("  - Training model...")
        model.fit(X_train, y_train)
        
        print("  Model training completed successfully")
        print(f"  - Number of trees: {model.n_estimators}")
        print(f"  - Average tree depth: {np.mean([tree.tree_.max_depth for tree in model.estimators_]):.2f}")
        
        return model
    
    except Exception as e:
        print(f"Error training Random Forest: {e}")
        raise


def compare_models(results_lr: dict, results_dt: dict, results_rf: dict) -> dict:
    """
    Compare three classification models and determine which performs better.
    
    Parameters:
    -----------
    results_lr : dict
        Evaluation results dictionary from Logistic Regression model.
        Must contain keys: 'accuracy', 'precision', 'recall', 'f1_score', 'model_name'.
    results_dt : dict
        Evaluation results dictionary from Decision Tree model.
        Must contain the same keys as results_lr.
    results_rf : dict
        Evaluation results dictionary from Random Forest model.
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
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        comparison = {}
        
        for metric in metrics:
            lr_value = results_lr[metric]
            dt_value = results_dt[metric]
            rf_value = results_rf[metric]
            
            # Find the best value
            values = {
                'Logistic Regression': lr_value,
                'Decision Tree': dt_value,
                'Random Forest': rf_value
            }
            winner_metric = max(values, key=values.get)
            
            comparison[metric] = {
                'Logistic Regression': lr_value,
                'Decision Tree': dt_value,
                'Random Forest': rf_value,
                'Winner': winner_metric
            }
        
        # Determine overall winner (based on accuracy and F1-score)
        lr_score = (results_lr['accuracy'] + results_lr['f1_score']) / 2
        dt_score = (results_dt['accuracy'] + results_dt['f1_score']) / 2
        rf_score = (results_rf['accuracy'] + results_rf['f1_score']) / 2
        
        scores = {
            'Logistic Regression': lr_score,
            'Decision Tree': dt_score,
            'Random Forest': rf_score
        }
        winner = max(scores, key=scores.get)
        
        # Display comparison
        print(f"\nSide-by-Side Comparison:")
        print(f"{'Metric':<20} {'Logistic Regression':<25} {'Decision Tree':<20} {'Random Forest':<25} {'Winner':<20}")
        print("-" * 110)
        
        for metric in metrics:
            lr_val = comparison[metric]['Logistic Regression']
            dt_val = comparison[metric]['Decision Tree']
            rf_val = comparison[metric]['Random Forest']
            winner_metric = comparison[metric]['Winner']
            print(f"{metric.capitalize():<20} {lr_val:<25.4f} {dt_val:<20.4f} {rf_val:<25.4f} {winner_metric:<20}")
        
        # Discussion
        print(f"\nOverall Winner: {winner}")
        print(f"\nAnalysis:")
        
        discussion = f"""
        Model Performance Analysis:
        
        1. Accuracy Comparison:
           - Logistic Regression: {results_lr['accuracy']:.4f} ({results_lr['accuracy']*100:.2f}%)
           - Decision Tree: {results_dt['accuracy']:.4f} ({results_dt['accuracy']*100:.2f}%)
           - Random Forest: {results_rf['accuracy']:.4f} ({results_rf['accuracy']*100:.2f}%)
        
        2. Precision Comparison:
           - Logistic Regression: {results_lr['precision']:.4f}
           - Decision Tree: {results_dt['precision']:.4f}
           - Random Forest: {results_rf['precision']:.4f}
           - Higher precision means fewer false positives.
        
        3. Recall Comparison:
           - Logistic Regression: {results_lr['recall']:.4f}
           - Decision Tree: {results_dt['recall']:.4f}
           - Random Forest: {results_rf['recall']:.4f}
           - Higher recall means fewer false negatives.
        
        4. F1-Score Comparison:
           - Logistic Regression: {results_lr['f1_score']:.4f}
           - Decision Tree: {results_dt['f1_score']:.4f}
           - Random Forest: {results_rf['f1_score']:.4f}
           - F1-score balances precision and recall.
        
        5. Model Characteristics:
           - Logistic Regression: Linear model, interpretable coefficients, requires feature scaling,
             assumes linear relationships, less prone to overfitting with regularization.
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
           - Specific class performance (check per-class metrics)
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


def main():
    """
    Main function to execute the complete classification pipeline.
    
    This function orchestrates the entire machine learning workflow:
    1. Data loading
    2. Feature selection and encoding
    3. Train/test split
    4. Model training (Logistic Regression and Decision Tree)
    5. Model evaluation
    6. Model comparison
    
    Returns:
    --------
    None
        This function prints results and saves visualizations but doesn't return values.
    """
    try:
        print("="*80)
        print("CLASSIFICATION MODELS FOR DATA SCIENCE SALARIES")
        print("="*80)
        print("\nThis script implements:")
        print("  [OK] Logistic Regression classifier")
        print("  [OK] Decision Tree classifier")
        print("  [OK] Random Forest classifier")
        print("  [OK] 80/20 train/test split")
        print("  [OK] Feature selection")
        print("  [OK] Comprehensive evaluation metrics")
        print("  [OK] Model comparison")
        print("\n" + "="*80)
        
        # Step 1: Load data
        data_path = "data/processed/DataScience_salaries_2025_cleaned.csv"
        df = load_data(data_path)
        
        # Step 2: Feature selection
        X, y, feature_names = select_features(df, target_column='experience_level')
        
        # Step 3: Encode categorical features
        X_encoded, feature_encoders = encode_categorical_features(X, feature_names)
        
        # Step 4: Encode target variable
        y_encoded, target_encoder = encode_target(y)
        
        # Step 5: Split data (80/20)
        X_train, X_test, y_train, y_test = split_data(
            X_encoded, y_encoded, test_size=0.2, random_state=42
        )
        
        # Step 6: Train Logistic Regression
        lr_model = train_logistic_regression(X_train, y_train, random_state=42)
        
        # Step 7: Train Decision Tree
        dt_model = train_decision_tree(X_train, y_train, random_state=42)
        
        # Step 8: Train Random Forest
        rf_model = train_random_forest(X_train, y_train, random_state=42)
        
        # Step 9: Evaluate Logistic Regression
        results_lr = evaluate_model(
            lr_model, X_test, y_test, 
            "Logistic Regression", target_encoder
        )
        
        # Step 10: Evaluate Decision Tree
        results_dt = evaluate_model(
            dt_model, X_test, y_test,
            "Decision Tree", target_encoder
        )
        
        # Step 11: Evaluate Random Forest
        results_rf = evaluate_model(
            rf_model, X_test, y_test,
            "Random Forest", target_encoder
        )
        
        # Step 12: Compare models
        comparison = compare_models(results_lr, results_dt, results_rf)
        
        # Step 13: Plot and save confusion matrices
        class_names = target_encoder.classes_
        plot_confusion_matrix(
            results_lr['confusion_matrix'], class_names, "Logistic Regression",
            save_path=os.path.join(RESULTS_DIR, "confusion_matrix_lr.png")
        )
        plot_confusion_matrix(
            results_dt['confusion_matrix'], class_names, "Decision Tree",
            save_path=os.path.join(RESULTS_DIR, "confusion_matrix_dt.png")
        )
        plot_confusion_matrix(
            results_rf['confusion_matrix'], class_names, "Random Forest",
            save_path=os.path.join(RESULTS_DIR, "confusion_matrix_rf.png")
        )
        
        # Step 14: Save all results to single file
        print("\n" + "="*80)
        print("SAVING ALL RESULTS")
        print("="*80)
        save_all_results(results_lr, results_dt, results_rf, comparison, RESULTS_DIR, class_names)
        
        print("\n" + "="*80)
        print("CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY")
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
