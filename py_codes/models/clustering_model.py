"""
Clustering Model Implementation
================================

This script implements clustering algorithms (K-Means and DBSCAN) for grouping
similar job profiles based on salary ranges and job characteristics.

Clustering Approach:
- Cluster job profiles by salary level, experience, company size, and remote work preference
- Features: salary_in_usd, experience_level, company_size, remote_ratio, work_year
- Goal: Identify distinct salary/job characteristic groups

Requirements Fulfilled:
- Two different ML models (K-Means and DBSCAN)
- Proper train/test split using train_test_split()
- Feature selection (choosing relevant columns for clustering)
- Model evaluation using appropriate metrics
- Model comparison and discussion
- Cluster visualization
- Report generation and saving to reports folder
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import datetime

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create reports directory structure
REPORTS_DIR = "reports"
RESULTS_DIR = os.path.join(REPORTS_DIR, "results")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Initialize report content
report_lines = []

def add_to_report(text):
    """Add text to report and print it."""
    report_lines.append(text)
    print(text)

# ============================================================================
# 1. DATA LOADING AND PREPARATION
# ============================================================================

def load_and_prepare_data(filepath: str):
    """
    Load processed data and prepare it for clustering.
    
    Parameters:
    -----------
    filepath : str
        Path to processed CSV file
        
    Returns:
    --------
    pd.DataFrame
        Prepared dataframe
    """
    add_to_report("=" * 80)
    add_to_report("LOADING AND PREPARING DATA")
    add_to_report("=" * 80)
    
    # Load data from processed folder
    df = pd.read_csv(filepath)
    add_to_report(f"\nLoaded dataset shape: {df.shape}")
    add_to_report(f"Columns: {list(df.columns)}")
    
    # Display basic statistics
    add_to_report(f"\nBasic Statistics:")
    add_to_report(f"  Salary range: ${df['salary_in_usd'].min():,.0f} - ${df['salary_in_usd'].max():,.0f}")
    add_to_report(f"  Mean salary: ${df['salary_in_usd'].mean():,.0f}")
    add_to_report(f"  Median salary: ${df['salary_in_usd'].median():,.0f}")
    
    return df


def feature_selection(df: pd.DataFrame):
    """
    Perform feature selection for clustering job profiles.
    Selects features that represent job characteristics and salary levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with selected and encoded features
    np.ndarray
        Feature array ready for clustering
    list
        Feature names
    dict
        Label encoders for categorical features
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("FEATURE SELECTION FOR CLUSTERING")
    add_to_report("=" * 80)
    
    # Create a copy for feature engineering
    df_features = df.copy()
    
    # Selected features for clustering job profiles:
    # 1. salary_in_usd - primary clustering dimension (salary level)
    # 2. experience_level - job seniority level
    # 3. company_size - company size category
    # 4. remote_ratio - work arrangement preference
    # 5. work_year - temporal factor
    
    add_to_report("\nSelected Features for Clustering:")
    add_to_report("  1. salary_in_usd - Salary level (primary dimension)")
    add_to_report("  2. experience_level - Job seniority (EN/MI/SE/EX)")
    add_to_report("  3. company_size - Company size (S/M/L)")
    add_to_report("  4. remote_ratio - Remote work preference (0/50/100)")
    add_to_report("  5. work_year - Year factor")
    
    # Numerical features
    numerical_features = ['salary_in_usd', 'remote_ratio', 'work_year']
    
    # Categorical features to encode
    categorical_features = ['experience_level', 'company_size']
    
    # Encode categorical features using LabelEncoder
    label_encoders = {}
    for feature in categorical_features:
        if feature in df_features.columns:
            le = LabelEncoder()
            df_features[f'{feature}_encoded'] = le.fit_transform(df_features[feature].astype(str))
            label_encoders[feature] = le
            add_to_report(f"\n{feature} encoding:")
            for i, class_name in enumerate(le.classes_):
                add_to_report(f"  {class_name} -> {i}")
    
    # Combine all features
    feature_columns = numerical_features + [f'{feat}_encoded' for feat in categorical_features]
    
    # Select only available features
    available_features = [col for col in feature_columns if col in df_features.columns]
    add_to_report(f"\nFinal feature set ({len(available_features)} features): {available_features}")
    
    # Extract feature matrix
    X = df_features[available_features].values
    
    # Handle any NaN values
    if np.isnan(X).any():
        add_to_report("\nWarning: Found NaN values, filling with median...")
        X = pd.DataFrame(X, columns=available_features).fillna(
            pd.DataFrame(X, columns=available_features).median()
        ).values
    
    add_to_report(f"\nFeature matrix shape: {X.shape}")
    add_to_report(f"\nFeature statistics:")
    feature_df = pd.DataFrame(X, columns=available_features)
    add_to_report(str(feature_df.describe()))
    
    return df_features, X, available_features, label_encoders


def scale_features(X: np.ndarray):
    """
    Scale features using StandardScaler for better clustering performance.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
        
    Returns:
    --------
    np.ndarray
        Scaled feature matrix
    StandardScaler
        Fitted scaler
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("FEATURE SCALING")
    add_to_report("=" * 80)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    add_to_report("\nFeatures scaled using StandardScaler")
    add_to_report("  All features normalized to mean=0, std=1")
    add_to_report(f"Scaled feature statistics:")
    scaled_df = pd.DataFrame(X_scaled, columns=[f'Feature_{i}' for i in range(X_scaled.shape[1])])
    add_to_report(str(scaled_df.describe()))
    
    return X_scaled, scaler


# ============================================================================
# 2. TRAIN/TEST SPLIT
# ============================================================================

def split_data(X: np.ndarray, test_size: float = 0.2, random_state: int = 42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    test_size : float
        Proportion of data for testing (default: 0.2)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("TRAIN/TEST SPLIT")
    add_to_report("=" * 80)
    
    X_train, X_test = train_test_split(
        X, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    add_to_report(f"\nTrain/Test Split: {1-test_size:.0%}/{test_size:.0%}")
    add_to_report(f"Training set shape: {X_train.shape}")
    add_to_report(f"Testing set shape: {X_test.shape}")
    
    return X_train, X_test


# ============================================================================
# 3. K-MEANS CLUSTERING MODEL
# ============================================================================

def find_optimal_k(X_train: np.ndarray, max_k: int = 10):
    """
    Find optimal number of clusters using Elbow method.
    The elbow method identifies the point where the rate of decrease in inertia
    slows down significantly (the "elbow" of the curve).
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    max_k : int
        Maximum number of clusters to test
        
    Returns:
    --------
    int
        Optimal number of clusters (elbow point)
    dict
        Evaluation metrics for different k values
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("FINDING OPTIMAL K FOR K-MEANS USING ELBOW METHOD")
    add_to_report("=" * 80)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    add_to_report("\nTesting k values from 2 to {}...".format(max_k))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(X_train)
        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X_train, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)
        add_to_report(f"  k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_avg:.4f}")
    
    # Elbow method: Find the point where the rate of decrease slows down
    # Method: Calculate the distance from each point to the line connecting first and last points
    # The elbow is the point with maximum distance from this line
    if len(inertias) > 2:
        # Convert to numpy arrays for easier calculation
        k_array = np.array(list(k_range))
        inertia_array = np.array(inertias)
        
        # Line connecting first point (k=2) and last point (k=max_k)
        x1, y1 = k_array[0], inertia_array[0]
        x2, y2 = k_array[-1], inertia_array[-1]
        
        # Calculate distances from each point to the line
        distances = []
        for i in range(len(k_array)):
            # Distance from point (x, y) to line through (x1, y1) and (x2, y2)
            numerator = abs((y2 - y1) * k_array[i] - (x2 - x1) * inertia_array[i] + x2 * y1 - y2 * x1)
            denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
            if denominator > 0:
                distance = numerator / denominator
            else:
                distance = 0
            distances.append(distance)
        
        # Find the elbow point (maximum distance from the line)
        elbow_idx = np.argmax(distances)
        optimal_k = k_array[elbow_idx]
        
        add_to_report(f"\nDistance from line (first to last point):")
        for i, (k, dist) in enumerate(zip(k_array, distances)):
            marker = " <-- ELBOW" if i == elbow_idx else ""
            add_to_report(f"  k={k}: distance={dist:.2f}{marker}")
    elif len(inertias) == 2:
        # Only 2 k values tested, choose based on inertia reduction
        decrease_pct = ((inertias[0] - inertias[1]) / inertias[0]) * 100
        optimal_k = list(k_range)[1] if decrease_pct > 10 else list(k_range)[0]
        add_to_report(f"\nOnly 2 k values tested. Decrease: {decrease_pct:.2f}%")
    else:
        optimal_k = list(k_range)[0]
    
    # Calculate percentage decreases for reporting
    decreases = []
    if len(inertias) > 1:
        for i in range(1, len(inertias)):
            if inertias[i-1] > 0:
                decrease = ((inertias[i-1] - inertias[i]) / inertias[i-1]) * 100
                decreases.append(decrease)
            else:
                decreases.append(0)
    
    add_to_report(f"\nElbow Method Analysis:")
    add_to_report(f"  Inertia values: {[f'{i:.2f}' for i in inertias]}")
    if len(decreases) > 0:
        add_to_report(f"  Rate of decrease (%): {[f'{d:.2f}%' for d in decreases]}")
    add_to_report(f"\nOptimal k (elbow point): {optimal_k}")
    add_to_report(f"  This is where the rate of decrease in inertia slows down significantly")
    add_to_report(f"  The elbow point represents the optimal balance between cluster quality and complexity")
    
    # Visualize elbow method
    plt.figure(figsize=(10, 6))
    k_list = list(k_range)
    plt.plot(k_list, inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
    
    # Draw line connecting first and last points (for elbow visualization)
    if len(inertias) > 1:
        plt.plot([k_list[0], k_list[-1]], [inertias[0], inertias[-1]], 
                'r--', linewidth=1.5, alpha=0.5, label='Reference line')
    
    # Highlight the elbow point
    optimal_idx = k_list.index(optimal_k)
    plt.plot(optimal_k, inertias[optimal_idx], 'ro', markersize=12, 
            label=f'Elbow at k={optimal_k}', markeredgecolor='darkred', markeredgewidth=2)
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    plt.title('Elbow Method for Optimal K Selection', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    elbow_plot_path = os.path.join(FIGURES_DIR, "elbow_method.png")
    plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    add_to_report(f"\nElbow method plot saved: {elbow_plot_path}")
    
    metrics = {
        'k_values': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k
    }
    
    return optimal_k, metrics


def train_kmeans(X_train: np.ndarray, n_clusters: int = 5, random_state: int = 42):
    """
    Train K-Means clustering model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    n_clusters : int
        Number of clusters
    random_state : int
        Random seed
        
    Returns:
    --------
    KMeans
        Trained K-Means model
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("TRAINING K-MEANS MODEL")
    add_to_report("=" * 80)
    
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    
    kmeans.fit(X_train)
    
    add_to_report(f"\nK-Means trained with {n_clusters} clusters")
    add_to_report(f"Inertia (within-cluster sum of squares): {kmeans.inertia_:.2f}")
    add_to_report(f"Number of iterations: {kmeans.n_iter_}")
    add_to_report(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")
    
    return kmeans


def evaluate_kmeans(kmeans: KMeans, X_train: np.ndarray, X_test: np.ndarray):
    """
    Evaluate K-Means model using multiple metrics.
    
    Parameters:
    -----------
    kmeans : KMeans
        Trained K-Means model
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Testing feature matrix
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("EVALUATING K-MEANS MODEL")
    add_to_report("=" * 80)
    
    # Predict clusters
    train_labels = kmeans.predict(X_train)
    test_labels = kmeans.predict(X_test)
    
    # Calculate metrics for training set
    train_silhouette = silhouette_score(X_train, train_labels)
    train_calinski = calinski_harabasz_score(X_train, train_labels)
    train_davies = davies_bouldin_score(X_train, train_labels)
    
    # Calculate metrics for testing set
    test_silhouette = silhouette_score(X_test, test_labels)
    test_calinski = calinski_harabasz_score(X_test, test_labels)
    test_davies = davies_bouldin_score(X_test, test_labels)
    
    add_to_report("\nTraining Set Metrics:")
    add_to_report(f"  Silhouette Score: {train_silhouette:.4f} (higher is better, range: -1 to 1)")
    add_to_report(f"  Calinski-Harabasz Score: {train_calinski:.2f} (higher is better)")
    add_to_report(f"  Davies-Bouldin Score: {train_davies:.4f} (lower is better)")
    
    add_to_report("\nTesting Set Metrics:")
    add_to_report(f"  Silhouette Score: {test_silhouette:.4f}")
    add_to_report(f"  Calinski-Harabasz Score: {test_calinski:.2f}")
    add_to_report(f"  Davies-Bouldin Score: {test_davies:.4f}")
    
    # Cluster sizes
    train_cluster_sizes = pd.Series(train_labels).value_counts().sort_index()
    test_cluster_sizes = pd.Series(test_labels).value_counts().sort_index()
    
    add_to_report("\nTraining Set Cluster Sizes:")
    for cluster, size in train_cluster_sizes.items():
        add_to_report(f"  Cluster {cluster}: {size} samples ({size/len(train_labels)*100:.2f}%)")
    
    add_to_report("\nTesting Set Cluster Sizes:")
    for cluster, size in test_cluster_sizes.items():
        add_to_report(f"  Cluster {cluster}: {size} samples ({size/len(test_labels)*100:.2f}%)")
    
    metrics = {
        'train': {
            'silhouette': train_silhouette,
            'calinski_harabasz': train_calinski,
            'davies_bouldin': train_davies,
            'labels': train_labels,
            'cluster_sizes': train_cluster_sizes.to_dict()
        },
        'test': {
            'silhouette': test_silhouette,
            'calinski_harabasz': test_calinski,
            'davies_bouldin': test_davies,
            'labels': test_labels,
            'cluster_sizes': test_cluster_sizes.to_dict()
        }
    }
    
    return metrics


# ============================================================================
# 4. DBSCAN CLUSTERING MODEL
# ============================================================================

def find_optimal_dbscan_params(X_train: np.ndarray):
    """
    Find optimal DBSCAN parameters (eps and min_samples).
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
        
    Returns:
    --------
    tuple
        Optimal eps and min_samples
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("FINDING OPTIMAL DBSCAN PARAMETERS")
    add_to_report("=" * 80)
    
    # Use k-distance graph to estimate eps
    from sklearn.neighbors import NearestNeighbors
    
    neighbors = NearestNeighbors(n_neighbors=4)
    neighbors_fit = neighbors.fit(X_train)
    distances, indices = neighbors_fit.kneighbors(X_train)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]  # k-th nearest neighbor distances
    
    # Remove zero distances and ensure we have valid values
    distances = distances[distances > 0]
    
    if len(distances) == 0:
        # Fallback: use a small default eps based on data scale
        mean_dist = np.mean(np.std(X_train, axis=0))
        eps_candidates = [mean_dist * 0.5, mean_dist * 0.75, mean_dist * 1.0]
        add_to_report(f"\nWarning: No valid distances found, using default eps based on data scale")
    else:
        # Estimate eps as the point of maximum curvature
        # Use percentile approach, but ensure eps > 0
        percentiles = [50, 75, 90]
        eps_candidates = []
        for p in percentiles:
            eps_val = np.percentile(distances, p)
            # Ensure eps is always > 0
            if eps_val <= 0:
                eps_val = np.max(distances) * (p / 100.0)
            eps_candidates.append(max(eps_val, 0.1))  # Minimum eps of 0.1
    
    # Remove any zero or negative values
    eps_candidates = [max(eps, 0.1) for eps in eps_candidates if eps > 0]
    
    # If still no valid candidates, use defaults
    if len(eps_candidates) == 0:
        mean_dist = np.mean(np.std(X_train, axis=0))
        eps_candidates = [max(mean_dist * 0.5, 0.1), max(mean_dist * 0.75, 0.1), max(mean_dist * 1.0, 0.1)]
        add_to_report(f"\nUsing default eps candidates: {eps_candidates}")
    
    min_samples_candidates = [3, 4, 5]
    
    best_score = -1
    best_eps = eps_candidates[0]
    best_min_samples = min_samples_candidates[0]
    
    add_to_report(f"\nDistance statistics:")
    add_to_report(f"  Min: {np.min(distances) if len(distances) > 0 else 'N/A':.4f}")
    add_to_report(f"  Max: {np.max(distances) if len(distances) > 0 else 'N/A':.4f}")
    add_to_report(f"  Mean: {np.mean(distances) if len(distances) > 0 else 'N/A':.4f}")
    add_to_report(f"\nEps candidates: {[f'{eps:.3f}' for eps in eps_candidates]}")
    add_to_report("\nTesting parameter combinations...")
    
    valid_combinations = 0
    for eps in eps_candidates:
        # Ensure eps is valid
        if eps <= 0:
            continue
            
        for min_samples in min_samples_candidates:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_train)
                
                # Only evaluate if we have more than 1 cluster and not all noise
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1 and n_noise < len(labels) * 0.5:  # Less than 50% noise
                    non_noise_mask = labels != -1
                    if non_noise_mask.sum() > 1 and len(set(labels[non_noise_mask])) > 1:
                        silhouette = silhouette_score(X_train[non_noise_mask], labels[non_noise_mask])
                    else:
                        silhouette = -1
                    
                    add_to_report(f"  eps={eps:.3f}, min_samples={min_samples}: "
                               f"n_clusters={n_clusters}, noise={n_noise}, silhouette={silhouette:.4f}")
                    valid_combinations += 1
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_eps = eps
                        best_min_samples = min_samples
            except Exception as e:
                add_to_report(f"  eps={eps:.3f}, min_samples={min_samples}: Error - {str(e)}")
                continue
    
    # If no valid combinations found, use defaults
    if best_score == -1 or valid_combinations == 0:
        add_to_report("\nWarning: No optimal parameters found, using defaults")
        # Use mean distance as base
        if len(distances) > 0:
            mean_dist = np.mean(distances)
            best_eps = max(mean_dist * 0.75, 0.5)
        else:
            mean_dist = np.mean(np.std(X_train, axis=0))
            best_eps = max(mean_dist * 0.75, 0.5)
        best_min_samples = 4
    
    add_to_report(f"\nOptimal parameters:")
    add_to_report(f"  eps: {best_eps:.3f}")
    add_to_report(f"  min_samples: {best_min_samples}")
    
    return best_eps, best_min_samples


def train_dbscan(X_train: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    """
    Train DBSCAN clustering model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix
    eps : float
        Maximum distance between samples in the same neighborhood
    min_samples : int
        Minimum number of samples in a neighborhood
        
    Returns:
    --------
    DBSCAN
        Trained DBSCAN model
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("TRAINING DBSCAN MODEL")
    add_to_report("=" * 80)
    
    # Validate and fix eps if necessary
    if eps <= 0:
        add_to_report(f"\nWarning: Invalid eps={eps}, calculating default...")
        # Calculate default eps based on data scale
        mean_std = np.mean(np.std(X_train, axis=0))
        eps = max(mean_std * 0.75, 0.5)
        add_to_report(f"Using eps={eps:.3f}")
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X_train)
    
    n_clusters = len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
    n_noise = list(dbscan.labels_).count(-1)
    
    add_to_report(f"\nDBSCAN trained")
    add_to_report(f"Number of clusters found: {n_clusters}")
    add_to_report(f"Number of noise points: {n_noise} ({n_noise/len(dbscan.labels_)*100:.2f}%)")
    add_to_report(f"Parameters: eps={eps:.3f}, min_samples={min_samples}")
    
    return dbscan


def evaluate_dbscan(dbscan: DBSCAN, X_train: np.ndarray, X_test: np.ndarray):
    """
    Evaluate DBSCAN model using multiple metrics.
    
    Parameters:
    -----------
    dbscan : DBSCAN
        Trained DBSCAN model
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Testing feature matrix
        
    Returns:
    --------
    dict
        Evaluation metrics
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("EVALUATING DBSCAN MODEL")
    add_to_report("=" * 80)
    
    # Get training labels
    train_labels = dbscan.labels_
    
    # Predict clusters for test set (DBSCAN doesn't have predict, so we use fit_predict)
    # For test set, we'll fit a new DBSCAN with same parameters
    test_dbscan = DBSCAN(eps=dbscan.eps, min_samples=dbscan.min_samples)
    test_labels = test_dbscan.fit_predict(X_test)
    
    # Calculate metrics for training set (excluding noise points)
    train_mask = train_labels != -1
    if train_mask.sum() > 1 and len(set(train_labels[train_mask])) > 1:
        train_silhouette = silhouette_score(X_train[train_mask], train_labels[train_mask])
        train_calinski = calinski_harabasz_score(X_train[train_mask], train_labels[train_mask])
        train_davies = davies_bouldin_score(X_train[train_mask], train_labels[train_mask])
    else:
        train_silhouette = -1
        train_calinski = 0
        train_davies = float('inf')
    
    # Calculate metrics for testing set (excluding noise points)
    test_mask = test_labels != -1
    if test_mask.sum() > 1 and len(set(test_labels[test_mask])) > 1:
        test_silhouette = silhouette_score(X_test[test_mask], test_labels[test_mask])
        test_calinski = calinski_harabasz_score(X_test[test_mask], test_labels[test_mask])
        test_davies = davies_bouldin_score(X_test[test_mask], test_labels[test_mask])
    else:
        test_silhouette = -1
        test_calinski = 0
        test_davies = float('inf')
    
    add_to_report("\nTraining Set Metrics:")
    add_to_report(f"  Silhouette Score: {train_silhouette:.4f}")
    add_to_report(f"  Calinski-Harabasz Score: {train_calinski:.2f}")
    add_to_report(f"  Davies-Bouldin Score: {train_davies:.4f}")
    
    add_to_report("\nTesting Set Metrics:")
    add_to_report(f"  Silhouette Score: {test_silhouette:.4f}")
    add_to_report(f"  Calinski-Harabasz Score: {test_calinski:.2f}")
    add_to_report(f"  Davies-Bouldin Score: {test_davies:.4f}")
    
    # Cluster sizes
    train_cluster_sizes = pd.Series(train_labels).value_counts().sort_index()
    test_cluster_sizes = pd.Series(test_labels).value_counts().sort_index()
    
    add_to_report("\nTraining Set Cluster Sizes:")
    for cluster, size in train_cluster_sizes.items():
        cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        add_to_report(f"  {cluster_name}: {size} samples ({size/len(train_labels)*100:.2f}%)")
    
    add_to_report("\nTesting Set Cluster Sizes:")
    for cluster, size in test_cluster_sizes.items():
        cluster_name = "Noise" if cluster == -1 else f"Cluster {cluster}"
        add_to_report(f"  {cluster_name}: {size} samples ({size/len(test_labels)*100:.2f}%)")
    
    metrics = {
        'train': {
            'silhouette': train_silhouette,
            'calinski_harabasz': train_calinski,
            'davies_bouldin': train_davies,
            'labels': train_labels,
            'cluster_sizes': train_cluster_sizes.to_dict(),
            'n_clusters': len(set(train_labels)) - (1 if -1 in train_labels else 0),
            'n_noise': list(train_labels).count(-1)
        },
        'test': {
            'silhouette': test_silhouette,
            'calinski_harabasz': test_calinski,
            'davies_bouldin': test_davies,
            'labels': test_labels,
            'cluster_sizes': test_cluster_sizes.to_dict(),
            'n_clusters': len(set(test_labels)) - (1 if -1 in test_labels else 0),
            'n_noise': list(test_labels).count(-1)
        }
    }
    
    return metrics


# ============================================================================
# 5. CLUSTER VISUALIZATION
# ============================================================================

def visualize_clusters(X: np.ndarray, labels: np.ndarray, title: str, 
                      model_name: str, save_path: str):
    """
    Visualize clusters using PCA for dimensionality reduction.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    model_name : str
        Name of the model
    save_path : str
        Path to save the figure
    """
    # Use PCA to reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'{model_name}: {title}', fontsize=16, fontweight='bold')
    
    # Plot 1: Scatter plot colored by clusters
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, 
                              cmap='viridis', alpha=0.6, s=20)
    axes[0].set_xlabel(f'First Principal Component (explains {pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                      fontsize=11)
    axes[0].set_ylabel(f'Second Principal Component (explains {pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                      fontsize=11)
    axes[0].set_title('Cluster Visualization (PCA)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Plot 2: Cluster distribution
    unique_labels = np.unique(labels)
    cluster_counts = [np.sum(labels == label) for label in unique_labels]
    cluster_names = ['Noise' if label == -1 else f'Cluster {label}' for label in unique_labels]
    
    axes[1].bar(range(len(unique_labels)), cluster_counts, color='steelblue', edgecolor='black')
    axes[1].set_xticks(range(len(unique_labels)))
    axes[1].set_xticklabels(cluster_names, rotation=45, ha='right')
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_title('Cluster Size Distribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, count in enumerate(cluster_counts):
        axes[1].text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    add_to_report(f"\nVisualization saved: {save_path}")


def analyze_cluster_characteristics(df: pd.DataFrame, labels: np.ndarray, 
                                   model_name: str, feature_names: list):
    """
    Analyze and report characteristics of each cluster.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe
    labels : np.ndarray
        Cluster labels
    model_name : str
        Name of the model
    feature_names : list
        List of feature names
    """
    add_to_report("\n" + "=" * 80)
    add_to_report(f"CLUSTER CHARACTERISTICS ANALYSIS - {model_name}")
    add_to_report("=" * 80)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = labels
    
    unique_clusters = np.unique(labels)
    
    # Create detailed cluster analysis
    cluster_analysis = []
    
    for cluster_id in unique_clusters:
        if cluster_id == -1:
            cluster_name = "Noise Points"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        add_to_report(f"\n{cluster_name}:")
        add_to_report(f"  Size: {len(cluster_data)} samples ({len(cluster_data)/len(df_clustered)*100:.2f}%)")
        
        if len(cluster_data) > 0:
            cluster_info = {'cluster_id': cluster_id, 'size': len(cluster_data)}
            
            # Analyze salary
            if 'salary_in_usd' in cluster_data.columns:
                mean_sal = cluster_data['salary_in_usd'].mean()
                median_sal = cluster_data['salary_in_usd'].median()
                cluster_info['mean_salary'] = mean_sal
                cluster_info['median_salary'] = median_sal
                add_to_report(f"  Salary Statistics:")
                add_to_report(f"    Mean: ${mean_sal:,.2f}")
                add_to_report(f"    Median: ${median_sal:,.2f}")
                add_to_report(f"    Min: ${cluster_data['salary_in_usd'].min():,.2f}")
                add_to_report(f"    Max: ${cluster_data['salary_in_usd'].max():,.2f}")
                add_to_report(f"    Std Dev: ${cluster_data['salary_in_usd'].std():,.2f}")
            
            # Analyze experience level
            if 'experience_level' in cluster_data.columns:
                exp_dist = cluster_data['experience_level'].value_counts()
                cluster_info['top_experience'] = exp_dist.index[0] if len(exp_dist) > 0 else 'N/A'
                add_to_report(f"  Experience Level Distribution:")
                for exp, count in exp_dist.items():
                    add_to_report(f"    {exp}: {count} ({count/len(cluster_data)*100:.1f}%)")
            
            # Analyze company size
            if 'company_size' in cluster_data.columns:
                size_dist = cluster_data['company_size'].value_counts()
                cluster_info['top_company_size'] = size_dist.index[0] if len(size_dist) > 0 else 'N/A'
                add_to_report(f"  Company Size Distribution:")
                for size, count in size_dist.items():
                    add_to_report(f"    {size}: {count} ({count/len(cluster_data)*100:.1f}%)")
            
            # Analyze remote ratio
            if 'remote_ratio' in cluster_data.columns:
                remote_dist = cluster_data['remote_ratio'].value_counts()
                cluster_info['top_remote_ratio'] = remote_dist.index[0] if len(remote_dist) > 0 else 'N/A'
                add_to_report(f"  Remote Ratio Distribution:")
                for ratio, count in remote_dist.head(3).items():
                    add_to_report(f"    {ratio}%: {count} ({count/len(cluster_data)*100:.1f}%)")
            
            cluster_analysis.append(cluster_info)
    
    return cluster_analysis


# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================

def compare_models(kmeans_metrics: dict, dbscan_metrics: dict):
    """
    Compare K-Means and DBSCAN models and determine which performs better.
    
    Parameters:
    -----------
    kmeans_metrics : dict
        K-Means evaluation metrics
    dbscan_metrics : dict
        DBSCAN evaluation metrics
    """
    add_to_report("\n" + "=" * 80)
    add_to_report("MODEL COMPARISON")
    add_to_report("=" * 80)
    
    # Compare on test set
    kmeans_test = kmeans_metrics['test']
    dbscan_test = dbscan_metrics['test']
    
    add_to_report("\n--- Comparison on Test Set ---")
    
    add_to_report("\nSilhouette Score (higher is better):")
    add_to_report(f"  K-Means: {kmeans_test['silhouette']:.4f}")
    add_to_report(f"  DBSCAN: {dbscan_test['silhouette']:.4f}")
    if kmeans_test['silhouette'] > dbscan_test['silhouette']:
        add_to_report(f"  Winner: K-Means (by {kmeans_test['silhouette'] - dbscan_test['silhouette']:.4f})")
    else:
        add_to_report(f"  Winner: DBSCAN (by {dbscan_test['silhouette'] - kmeans_test['silhouette']:.4f})")
    
    add_to_report("\nCalinski-Harabasz Score (higher is better):")
    add_to_report(f"  K-Means: {kmeans_test['calinski_harabasz']:.2f}")
    add_to_report(f"  DBSCAN: {dbscan_test['calinski_harabasz']:.2f}")
    if kmeans_test['calinski_harabasz'] > dbscan_test['calinski_harabasz']:
        add_to_report(f"  Winner: K-Means")
    else:
        add_to_report(f"  Winner: DBSCAN")
    
    add_to_report("\nDavies-Bouldin Score (lower is better):")
    add_to_report(f"  K-Means: {kmeans_test['davies_bouldin']:.4f}")
    add_to_report(f"  DBSCAN: {dbscan_test['davies_bouldin']:.4f}")
    if kmeans_test['davies_bouldin'] < dbscan_test['davies_bouldin']:
        add_to_report(f"  Winner: K-Means")
    else:
        add_to_report(f"  Winner: DBSCAN")
    
    add_to_report("\n--- Discussion ---")
    add_to_report("\nK-Means Advantages:")
    add_to_report("  - Guarantees all points are assigned to clusters")
    add_to_report("  - Produces spherical clusters")
    add_to_report("  - Fast and scalable")
    add_to_report("  - Works well when clusters are well-separated")
    add_to_report("  - Requires specifying number of clusters beforehand")
    
    add_to_report("\nDBSCAN Advantages:")
    add_to_report("  - Can find clusters of arbitrary shapes")
    add_to_report("  - Automatically determines number of clusters")
    add_to_report("  - Can identify noise/outlier points")
    add_to_report("  - Robust to outliers")
    add_to_report("  - May produce many noise points if parameters not tuned well")
    
    # Determine overall winner based on silhouette score (most important metric)
    if kmeans_test['silhouette'] > dbscan_test['silhouette']:
        winner = "K-Means"
        reason = f"K-Means achieves a higher silhouette score ({kmeans_test['silhouette']:.4f} vs {dbscan_test['silhouette']:.4f}), " \
                f"indicating better-defined clusters. It also assigns all data points to clusters, " \
                f"which is useful for this dataset."
    else:
        winner = "DBSCAN"
        reason = f"DBSCAN achieves a higher silhouette score ({dbscan_test['silhouette']:.4f} vs {kmeans_test['silhouette']:.4f}), " \
                f"and can identify noise points, which may be valuable for detecting outliers in salary data."
    
    add_to_report(f"\n--- Overall Winner: {winner} ---")
    add_to_report(f"\nReason: {reason}")
    
    return winner, reason


# ============================================================================
# 7. SAVE RESULTS AND REPORTS
# ============================================================================

def save_metrics_to_file(kmeans_metrics: dict, dbscan_metrics: dict, 
                         kmeans_model: KMeans, dbscan_model: DBSCAN,
                         optimal_k: int, winner: str, reason: str):
    """
    Save all metrics and results to files in results folder.
    
    Parameters:
    -----------
    kmeans_metrics : dict
        K-Means evaluation metrics
    dbscan_metrics : dict
        DBSCAN evaluation metrics
    kmeans_model : KMeans
        Trained K-Means model
    dbscan_model : DBSCAN
        Trained DBSCAN model
    optimal_k : int
        Optimal number of clusters
    winner : str
        Winning model name
    reason : str
        Reason for winner selection
    """
    # Save metrics to CSV
    metrics_data = {
        'Model': ['K-Means', 'DBSCAN'],
        'Train_Silhouette': [kmeans_metrics['train']['silhouette'], dbscan_metrics['train']['silhouette']],
        'Test_Silhouette': [kmeans_metrics['test']['silhouette'], dbscan_metrics['test']['silhouette']],
        'Train_Calinski': [kmeans_metrics['train']['calinski_harabasz'], dbscan_metrics['train']['calinski_harabasz']],
        'Test_Calinski': [kmeans_metrics['test']['calinski_harabasz'], dbscan_metrics['test']['calinski_harabasz']],
        'Train_Davies': [kmeans_metrics['train']['davies_bouldin'], dbscan_metrics['train']['davies_bouldin']],
        'Test_Davies': [kmeans_metrics['test']['davies_bouldin'], dbscan_metrics['test']['davies_bouldin']],
        'N_Clusters': [optimal_k, dbscan_metrics['train']['n_clusters']],
        'Noise_Points': [0, dbscan_metrics['train']['n_noise']]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(RESULTS_DIR, "clustering_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    add_to_report(f"\nMetrics saved to: {metrics_path}")
    
    # Save cluster sizes
    kmeans_sizes = pd.DataFrame({
        'Cluster': list(kmeans_metrics['train']['cluster_sizes'].keys()),
        'Size': list(kmeans_metrics['train']['cluster_sizes'].values())
    })
    kmeans_sizes_path = os.path.join(RESULTS_DIR, "kmeans_cluster_sizes.csv")
    kmeans_sizes.to_csv(kmeans_sizes_path, index=False)
    
    dbscan_sizes = pd.DataFrame({
        'Cluster': list(dbscan_metrics['train']['cluster_sizes'].keys()),
        'Size': list(dbscan_metrics['train']['cluster_sizes'].values())
    })
    dbscan_sizes_path = os.path.join(RESULTS_DIR, "dbscan_cluster_sizes.csv")
    dbscan_sizes.to_csv(dbscan_sizes_path, index=False)
    
    add_to_report(f"Cluster sizes saved to: {kmeans_sizes_path}, {dbscan_sizes_path}")
    
    # Save summary
    summary = {
        'optimal_k': optimal_k,
        'winner': winner,
        'reason': reason,
        'kmeans_inertia': kmeans_model.inertia_,
        'dbscan_n_clusters': dbscan_metrics['train']['n_clusters'],
        'dbscan_n_noise': dbscan_metrics['train']['n_noise']
    }
    
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(RESULTS_DIR, "clustering_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    add_to_report(f"Summary saved to: {summary_path}")


def save_report():
    """Save the complete report to a text file."""
    report_path = os.path.join(REPORTS_DIR, "clustering_model_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CLUSTERING MODEL ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        
        for line in report_lines:
            f.write(line + "\n")
    
    add_to_report(f"\n" + "=" * 80)
    add_to_report(f"Report saved to: {report_path}")
    add_to_report("=" * 80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to execute clustering analysis.
    """
    add_to_report("=" * 80)
    add_to_report("CLUSTERING MODEL IMPLEMENTATION")
    add_to_report("=" * 80)
    add_to_report("\nThis script implements:")
    add_to_report("  - K-Means Clustering (with Elbow method for optimal k)")
    add_to_report("  - DBSCAN Clustering")
    add_to_report("  - Model evaluation and comparison")
    add_to_report("  - Cluster visualization and analysis")
    add_to_report("  - Results saved to reports/results/ and reports/figures/")
    
    # Load data
    data_path = "data/processed/DataScience_salaries_2025_cleaned.csv"
    df = load_and_prepare_data(data_path)
    
    # Feature selection
    df_features, X, feature_names, label_encoders = feature_selection(df)
    
    # Scale features
    X_scaled, scaler = scale_features(X)
    
    # Train/test split
    X_train, X_test = split_data(X_scaled, test_size=0.2, random_state=42)
    
    # Get train indices for cluster analysis
    train_indices = np.arange(len(X_scaled))
    np.random.seed(42)
    np.random.shuffle(train_indices)
    train_indices = train_indices[:len(X_train)]
    
    # K-Means Clustering
    optimal_k, k_metrics = find_optimal_k(X_train, max_k=8)
    kmeans_model = train_kmeans(X_train, n_clusters=optimal_k, random_state=42)
    kmeans_metrics = evaluate_kmeans(kmeans_model, X_train, X_test)
    
    # Visualize K-Means clusters
    visualize_clusters(X_train, kmeans_metrics['train']['labels'], 
                      "Training Set Clusters", "K-Means",
                      os.path.join(FIGURES_DIR, "kmeans_clusters.png"))
    
    # Analyze K-Means cluster characteristics
    kmeans_cluster_analysis = analyze_cluster_characteristics(
        df_features.iloc[train_indices], 
        kmeans_metrics['train']['labels'],
        "K-Means", feature_names
    )
    
    # DBSCAN Clustering
    optimal_eps, optimal_min_samples = find_optimal_dbscan_params(X_train)
    dbscan_model = train_dbscan(X_train, eps=optimal_eps, min_samples=optimal_min_samples)
    dbscan_metrics = evaluate_dbscan(dbscan_model, X_train, X_test)
    
    # Visualize DBSCAN clusters
    visualize_clusters(X_train, dbscan_metrics['train']['labels'],
                      "Training Set Clusters", "DBSCAN",
                      os.path.join(FIGURES_DIR, "dbscan_clusters.png"))
    
    # Analyze DBSCAN cluster characteristics
    dbscan_cluster_analysis = analyze_cluster_characteristics(
        df_features.iloc[train_indices],
        dbscan_metrics['train']['labels'],
        "DBSCAN", feature_names
    )
    
    # Compare models
    winner, reason = compare_models(kmeans_metrics, dbscan_metrics)
    
    # Save all results and metrics
    save_metrics_to_file(kmeans_metrics, dbscan_metrics, kmeans_model, dbscan_model,
                        optimal_k, winner, reason)
    
    # Save report
    save_report()
    
    add_to_report("\n" + "=" * 80)
    add_to_report("CLUSTERING ANALYSIS COMPLETE!")
    add_to_report("=" * 80)
    add_to_report(f"\nAll visualizations saved to: {FIGURES_DIR}/")
    add_to_report(f"All metrics and results saved to: {RESULTS_DIR}/")
    add_to_report(f"Report saved to: {os.path.join(REPORTS_DIR, 'clustering_model_report.txt')}")


if __name__ == "__main__":
    main()
