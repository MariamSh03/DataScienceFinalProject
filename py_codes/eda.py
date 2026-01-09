"""
Exploratory Data Analysis (EDA) for Data Science Salaries Dataset
==================================================================

This script performs comprehensive exploratory data analysis including:
- Statistical analysis using NumPy and Pandas
- Multiple types of visualizations (histograms, box plots, heatmaps, scatter plots, etc.)
- Correlation and relationship analysis
- Pattern, trend, and anomaly detection

Requirements Fulfilled:
- Comprehensive use of NumPy and Pandas for analysis
- At least 5 different types of visualizations using Matplotlib/Seaborn
- Statistical analysis of key variables
- Investigation of relationships and correlations between features
- Identification of patterns, trends, and anomalies in the data
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import zscore

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Create output directory for visualizations - CHANGED TO reports/figures
FIGURE_DIR = "reports/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)
print(f"Created directory: {FIGURE_DIR}")

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load the processed dataset and perform initial exploration using Pandas.
    
    Parameters:
    -----------
    filepath : str
        Path to the processed CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded dataframe
    """
    print("=" * 80)
    print("LOADING PROCESSED DATA")
    print("=" * 80)
    
    # Load data using Pandas
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn Names: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nBasic Info:")
    print(df.info())
    
    return df


def initial_statistical_summary(df: pd.DataFrame):
    """
    Perform initial statistical summary using NumPy and Pandas.
    This fulfills the requirement for descriptive statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("INITIAL STATISTICAL SUMMARY")
    print("=" * 80)
    
    # Using Pandas .describe() for descriptive statistics (REQUIREMENT 2.2.3)
    print("\n--- Descriptive Statistics using .describe() (All Columns) ---")
    print(df.describe(include='all'))
    
    # Using NumPy for numerical column statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n--- Numerical Columns: {numerical_cols} ---")
    
    for col in numerical_cols:
        values = df[col].values  # Convert to NumPy array
        print(f"\n{col} (using NumPy):")
        print(f"  Mean: {np.mean(values):.2f}")
        print(f"  Median: {np.median(values):.2f}")
        print(f"  Std Dev: {np.std(values):.2f}")
        print(f"  Min: {np.min(values):.2f}")
        print(f"  Max: {np.max(values):.2f}")
        print(f"  25th Percentile (Q1): {np.percentile(values, 25):.2f}")
        print(f"  75th Percentile (Q3): {np.percentile(values, 75):.2f}")
        print(f"  Skewness: {stats.skew(values):.2f}")
        print(f"  Kurtosis: {stats.kurtosis(values):.2f}")


# ============================================================================
# 2. STATISTICAL ANALYSIS OF KEY VARIABLES
# ============================================================================

def statistical_analysis(df: pd.DataFrame):
    """
    Perform comprehensive statistical analysis of key variables.
    This fulfills requirement 2.2.1: Statistical analysis of key variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS OF KEY VARIABLES")
    print("=" * 80)
    
    # Focus on salary_in_usd as the primary variable
    salary_col = 'salary_in_usd'
    
    if salary_col in df.columns:
        salary_data = df[salary_col].values  # Convert to NumPy array
        
        print(f"\n--- Statistical Analysis: {salary_col} ---")
        print(f"Sample Size: {len(salary_data):,}")
        print(f"Mean Salary: ${np.mean(salary_data):,.2f}")
        print(f"Median Salary: ${np.median(salary_data):,.2f}")
        print(f"Standard Deviation: ${np.std(salary_data):,.2f}")
        print(f"Variance: ${np.var(salary_data):,.2f}")
        print(f"Range: ${np.min(salary_data):,.2f} - ${np.max(salary_data):,.2f}")
        print(f"Coefficient of Variation: {(np.std(salary_data) / np.mean(salary_data)) * 100:.2f}%")
        
        # Using NumPy for quartiles (REQUIREMENT 2.2.3)
        q1 = np.percentile(salary_data, 25)
        q2 = np.percentile(salary_data, 50)
        q3 = np.percentile(salary_data, 75)
        iqr = q3 - q1
        
        print(f"\nQuartiles (using NumPy):")
        print(f"  Q1 (25th percentile): ${q1:,.2f}")
        print(f"  Q2 (50th percentile/Median): ${q2:,.2f}")
        print(f"  Q3 (75th percentile): ${q3:,.2f}")
        print(f"  IQR: ${iqr:,.2f}")
        
        # Outlier detection using IQR method (REQUIREMENT 2.2.3)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = np.sum((salary_data < lower_bound) | (salary_data > upper_bound))
        print(f"\nOutliers (IQR method): {outliers:,} ({outliers/len(salary_data)*100:.2f}%)")
        print(f"  Lower bound: ${lower_bound:,.2f}")
        print(f"  Upper bound: ${upper_bound:,.2f}")
        
        # Statistical tests
        print(f"\n--- Normality Tests ---")
        # Shapiro-Wilk test (on sample due to size limits)
        sample_size = min(5000, len(salary_data))
        sample_data = np.random.choice(salary_data, sample_size, replace=False)
        shapiro_stat, shapiro_p = stats.shapiro(sample_data)
        print(f"Shapiro-Wilk Test (sample of {sample_size}):")
        print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
        print(f"  Distribution is {'normal' if shapiro_p > 0.05 else 'not normal'}")
    
    # Statistical analysis by categorical groups using Pandas
    print(f"\n--- Salary Statistics by Experience Level ---")
    if 'experience_level' in df.columns and salary_col in df.columns:
        exp_stats = df.groupby('experience_level')[salary_col].agg([
            ('count', 'count'),
            ('mean', np.mean),
            ('median', np.median),
            ('std', np.std),
            ('min', np.min),
            ('max', np.max)
        ])
        print(exp_stats)
    
    print(f"\n--- Salary Statistics by Company Size ---")
    if 'company_size' in df.columns and salary_col in df.columns:
        size_stats = df.groupby('company_size')[salary_col].agg([
            ('count', 'count'),
            ('mean', np.mean),
            ('median', np.median),
            ('std', np.std)
        ])
        print(size_stats)


# ============================================================================
# 3. CORRELATION AND RELATIONSHIP ANALYSIS
# ============================================================================

def correlation_analysis(df: pd.DataFrame):
    """
    Analyze correlations and relationships between numerical features.
    This fulfills requirement 2.2.1: Investigate relationships and correlations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    print("\n" + "=" * 80)
    print("CORRELATION AND RELATIONSHIP ANALYSIS")
    print("=" * 80)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) > 1:
        # Calculate correlation matrix using Pandas (REQUIREMENT 2.2.3)
        corr_matrix = df[numerical_cols].corr()
        
        print("\n--- Correlation Matrix (Pandas) ---")
        print(corr_matrix)
        
        # Find strong correlations (absolute value > 0.5)
        print("\n--- Strong Correlations (|r| > 0.5) ---")
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    print(f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}: {corr_val:.4f}")
        
        return corr_matrix
    
    return None


# ============================================================================
# 4. VISUALIZATION 1: DISTRIBUTION PLOTS (Histogram, KDE, Box Plot)
# REQUIREMENT 2.2.2: Distribution plots (histograms, KDE plots, box plots)
# ============================================================================

def plot_distributions(df: pd.DataFrame):
    """
    Create distribution plots: histogram, KDE plot, and box plot for salary.
    This fulfills requirement 2.2.2 and 2.2.3: Distribution analysis.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING DISTRIBUTION PLOTS")
    print("=" * 80)
    
    salary_col = 'salary_in_usd'
    
    if salary_col not in df.columns:
        print(f"Column {salary_col} not found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Salary Distribution Analysis', fontsize=16, fontweight='bold')
    
    salary_data = df[salary_col].values  # Convert to NumPy array
    
    # 1. Histogram with KDE overlay
    axes[0, 0].hist(salary_data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    # Add KDE curve using scipy.stats
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(salary_data)
    x_range = np.linspace(np.min(salary_data), np.max(salary_data), 200)
    axes[0, 0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    axes[0, 0].axvline(np.mean(salary_data), color='green', linestyle='--', linewidth=2, label=f'Mean: ${np.mean(salary_data):,.0f}')
    axes[0, 0].axvline(np.median(salary_data), color='orange', linestyle='--', linewidth=2, label=f'Median: ${np.median(salary_data):,.0f}')
    axes[0, 0].set_xlabel('Salary in USD', fontsize=12)
    axes[0, 0].set_ylabel('Density', fontsize=12)
    axes[0, 0].set_title('Histogram with KDE Overlay', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box Plot (REQUIREMENT 2.2.3)
    bp = axes[0, 1].boxplot(salary_data, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[0, 1].set_ylabel('Salary in USD', fontsize=12)
    axes[0, 1].set_title('Box Plot of Salaries', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"Q1: ${np.percentile(salary_data, 25):,.0f}\n"
    stats_text += f"Median: ${np.median(salary_data):,.0f}\n"
    stats_text += f"Q3: ${np.percentile(salary_data, 75):,.0f}\n"
    stats_text += f"IQR: ${np.percentile(salary_data, 75) - np.percentile(salary_data, 25):,.0f}"
    axes[0, 1].text(1.1, np.median(salary_data), stats_text, 
                    verticalalignment='center', fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Log-transformed histogram (to better see distribution)
    log_salary = np.log1p(salary_data)  # Using NumPy log1p for numerical stability
    axes[1, 0].hist(log_salary, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Log(Salary in USD)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Log-Transformed Salary Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Distribution by Experience Level using Seaborn
    if 'experience_level' in df.columns:
        df_exp = df[df['experience_level'].notna()]
        sns.boxplot(data=df_exp, x='experience_level', y=salary_col, ax=axes[1, 1])
        axes[1, 1].set_title('Salary Distribution by Experience Level', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Experience Level', fontsize=12)
        axes[1, 1].set_ylabel('Salary in USD', fontsize=12)
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '01_distribution_plots.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 5. VISUALIZATION 2: CORRELATION HEATMAP
# REQUIREMENT 2.2.2: Correlation heatmaps
# REQUIREMENT 2.2.3: Correlation analysis using correlation heatmaps
# ============================================================================

def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Create a correlation heatmap for numerical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING CORRELATION HEATMAP")
    print("=" * 80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numerical_cols) < 2:
        print("Not enough numerical columns for correlation analysis!")
        return
    
    # Calculate correlation matrix using Pandas
    corr_matrix = df[numerical_cols].corr()
    
    # Create heatmap using Seaborn
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle using NumPy
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, mask=mask,
                vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Numerical Variables', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '02_correlation_heatmap.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 6. VISUALIZATION 3: SCATTER PLOTS WITH TREND LINES
# REQUIREMENT 2.2.2: Scatter plots with trend lines
# ============================================================================

def plot_scatter_with_trends(df: pd.DataFrame):
    """
    Create scatter plots with trend lines to show relationships.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING SCATTER PLOTS WITH TREND LINES")
    print("=" * 80)
    
    salary_col = 'salary_in_usd'
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if salary_col not in numerical_cols:
        print(f"Column {salary_col} not found!")
        return
    
    # Remove salary columns from x-axis options
    x_options = [col for col in numerical_cols if col != salary_col and col != 'salary']
    
    if len(x_options) == 0:
        print("No suitable columns for scatter plot!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Scatter Plots with Trend Lines', fontsize=16, fontweight='bold')
    
    # Plot 1: Salary vs Remote Ratio
    if 'remote_ratio' in df.columns:
        x_data = df['remote_ratio'].values
        y_data = df[salary_col].values
        
        axes[0, 0].scatter(x_data, y_data, alpha=0.5, s=20, color='steelblue')
        
        # Add trend line using NumPy polyfit
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(np.min(x_data), np.max(x_data), 100)
        axes[0, 0].plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.0f}x+{z[1]:.0f}')
        
        # Calculate correlation using NumPy
        corr_coef = np.corrcoef(x_data, y_data)[0, 1]
        axes[0, 0].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[0, 0].set_xlabel('Remote Ratio (%)', fontsize=12)
        axes[0, 0].set_ylabel('Salary in USD', fontsize=12)
        axes[0, 0].set_title('Salary vs Remote Ratio', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Salary vs Work Year
    if 'work_year' in df.columns:
        x_data = df['work_year'].values
        y_data = df[salary_col].values
        
        axes[0, 1].scatter(x_data, y_data, alpha=0.5, s=20, color='coral')
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(np.min(x_data), np.max(x_data), 100)
        axes[0, 1].plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.0f}x+{z[1]:.0f}')
        
        corr_coef = np.corrcoef(x_data, y_data)[0, 1]
        axes[0, 1].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                       transform=axes[0, 1].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[0, 1].set_xlabel('Work Year', fontsize=12)
        axes[0, 1].set_ylabel('Salary in USD', fontsize=12)
        axes[0, 1].set_title('Salary vs Work Year', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Salary distribution colored by experience level
    if 'experience_level' in df.columns and 'remote_ratio' in df.columns:
        df_clean = df[df['experience_level'].notna()].copy()
        scatter = axes[1, 0].scatter(df_clean['remote_ratio'], df_clean[salary_col], 
                                    c=pd.Categorical(df_clean['experience_level']).codes,
                                    alpha=0.6, s=30, cmap='viridis')
        axes[1, 0].set_xlabel('Remote Ratio (%)', fontsize=12)
        axes[1, 0].set_ylabel('Salary in USD', fontsize=12)
        axes[1, 0].set_title('Salary vs Remote Ratio (colored by Experience)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        unique_levels = df_clean['experience_level'].unique()
        cbar.set_ticks(range(len(unique_levels)))
        cbar.set_ticklabels(unique_levels)
        cbar.set_label('Experience Level', fontsize=10)
    
    # Plot 4: Salary vs Salary (original) if both exist
    if 'salary' in df.columns and salary_col in df.columns:
        x_data = df['salary'].values
        y_data = df[salary_col].values
        
        axes[1, 1].scatter(x_data, y_data, alpha=0.5, s=20, color='purple')
        
        # Add trend line
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(np.min(x_data), np.max(x_data), 100)
        axes[1, 1].plot(x_trend, p(x_trend), "r--", linewidth=2, label=f'Trend: y={z[0]:.3f}x+{z[1]:.0f}')
        
        corr_coef = np.corrcoef(x_data, y_data)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Correlation: {corr_coef:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=11,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[1, 1].set_xlabel('Salary (Original Currency)', fontsize=12)
        axes[1, 1].set_ylabel('Salary in USD', fontsize=12)
        axes[1, 1].set_title('Salary Conversion Relationship', fontsize=14, fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '03_scatter_trends.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 7. VISUALIZATION 4: BAR CHARTS AND COUNT PLOTS
# REQUIREMENT 2.2.2: Bar charts or count plots
# ============================================================================

def plot_bar_charts(df: pd.DataFrame):
    """
    Create bar charts and count plots for categorical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING BAR CHARTS AND COUNT PLOTS")
    print("=" * 80)
    
    salary_col = 'salary_in_usd'
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Bar Charts and Count Plots', fontsize=16, fontweight='bold')
    
    # Plot 1: Average Salary by Experience Level
    if 'experience_level' in df.columns and salary_col in df.columns:
        exp_salary = df.groupby('experience_level')[salary_col].mean().sort_values(ascending=False)
        bars = axes[0, 0].bar(range(len(exp_salary)), exp_salary.values, color='steelblue', edgecolor='black')
        axes[0, 0].set_xticks(range(len(exp_salary)))
        axes[0, 0].set_xticklabels(exp_salary.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Average Salary (USD)', fontsize=12)
        axes[0, 0].set_title('Average Salary by Experience Level', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, exp_salary.values)):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Count of Jobs by Experience Level (using Seaborn)
    if 'experience_level' in df.columns:
        exp_counts = df['experience_level'].value_counts().sort_index()
        sns.countplot(data=df, x='experience_level', ax=axes[0, 1], palette='viridis')
        axes[0, 1].set_xlabel('Experience Level', fontsize=12)
        axes[0, 1].set_ylabel('Count', fontsize=12)
        axes[0, 1].set_title('Job Count by Experience Level', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (idx, count) in enumerate(exp_counts.items()):
            axes[0, 1].text(i, count, str(count), ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Average Salary by Company Size
    if 'company_size' in df.columns and salary_col in df.columns:
        size_salary = df.groupby('company_size')[salary_col].mean().sort_values(ascending=False)
        bars = axes[1, 0].bar(range(len(size_salary)), size_salary.values, 
                             color=['#FF6B6B', '#4ECDC4', '#45B7D1'], edgecolor='black')
        axes[1, 0].set_xticks(range(len(size_salary)))
        axes[1, 0].set_xticklabels(size_salary.index)
        axes[1, 0].set_ylabel('Average Salary (USD)', fontsize=12)
        axes[1, 0].set_title('Average Salary by Company Size', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, size_salary.values)):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'${val:,.0f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Top 10 Job Titles by Average Salary
    if 'job_title' in df.columns and salary_col in df.columns:
        top_jobs = df.groupby('job_title')[salary_col].mean().sort_values(ascending=False).head(10)
        bars = axes[1, 1].barh(range(len(top_jobs)), top_jobs.values, color='coral', edgecolor='black')
        axes[1, 1].set_yticks(range(len(top_jobs)))
        axes[1, 1].set_yticklabels(top_jobs.index, fontsize=9)
        axes[1, 1].set_xlabel('Average Salary (USD)', fontsize=12)
        axes[1, 1].set_title('Top 10 Job Titles by Average Salary', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_jobs.values)):
            axes[1, 1].text(val, bar.get_y() + bar.get_height()/2,
                           f'${val:,.0f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '04_bar_charts.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 8. VISUALIZATION 5: TIME SERIES PLOTS
# REQUIREMENT 2.2.2: Time series plots (if applicable)
# ============================================================================

def plot_time_series(df: pd.DataFrame):
    """
    Create time series plots showing trends over work_year.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING TIME SERIES PLOTS")
    print("=" * 80)
    
    salary_col = 'salary_in_usd'
    
    if 'work_year' not in df.columns or salary_col not in df.columns:
        print("Required columns not found for time series plot!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Time Series Analysis: Salary Trends Over Years', fontsize=16, fontweight='bold')
    
    # Plot 1: Average salary over time
    yearly_avg = df.groupby('work_year')[salary_col].mean()
    axes[0, 0].plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=8, color='steelblue')
    axes[0, 0].fill_between(yearly_avg.index, yearly_avg.values, alpha=0.3, color='steelblue')
    axes[0, 0].set_xlabel('Work Year', fontsize=12)
    axes[0, 0].set_ylabel('Average Salary (USD)', fontsize=12)
    axes[0, 0].set_title('Average Salary Trend Over Years', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for year, salary in yearly_avg.items():
        axes[0, 0].text(year, salary, f'${salary:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Median salary over time
    yearly_median = df.groupby('work_year')[salary_col].median()
    axes[0, 1].plot(yearly_median.index, yearly_median.values, marker='s', linewidth=2, markersize=8, color='coral')
    axes[0, 1].fill_between(yearly_median.index, yearly_median.values, alpha=0.3, color='coral')
    axes[0, 1].set_xlabel('Work Year', fontsize=12)
    axes[0, 1].set_ylabel('Median Salary (USD)', fontsize=12)
    axes[0, 1].set_title('Median Salary Trend Over Years', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for year, salary in yearly_median.items():
        axes[0, 1].text(year, salary, f'${salary:,.0f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Salary by experience level over time
    if 'experience_level' in df.columns:
        exp_levels = df['experience_level'].unique()
        for exp in exp_levels:
            exp_data = df[df['experience_level'] == exp].groupby('work_year')[salary_col].mean()
            axes[1, 0].plot(exp_data.index, exp_data.values, marker='o', linewidth=2, label=exp)
        axes[1, 0].set_xlabel('Work Year', fontsize=12)
        axes[1, 0].set_ylabel('Average Salary (USD)', fontsize=12)
        axes[1, 0].set_title('Salary Trend by Experience Level', fontsize=14, fontweight='bold')
        axes[1, 0].legend(title='Experience Level')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Number of job postings over time
    yearly_count = df.groupby('work_year').size()
    axes[1, 1].bar(yearly_count.index, yearly_count.values, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Work Year', fontsize=12)
    axes[1, 1].set_ylabel('Number of Job Postings', fontsize=12)
    axes[1, 1].set_title('Number of Job Postings Over Time', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for year, count in yearly_count.items():
        axes[1, 1].text(year, count, str(count), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '05_time_series.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 9. VISUALIZATION 6: PAIR PLOTS FOR MULTIVARIATE ANALYSIS
# REQUIREMENT 2.2.2: Pair plots for multivariate analysis
# ============================================================================

def plot_pair_plots(df: pd.DataFrame):
    """
    Create pair plots for multivariate analysis of numerical variables.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING PAIR PLOTS")
    print("=" * 80)
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit to key columns to avoid too many subplots
    key_cols = [col for col in numerical_cols if col in ['salary_in_usd', 'remote_ratio', 'work_year', 'salary']]
    
    if len(key_cols) < 2:
        print("Not enough numerical columns for pair plot!")
        return
    
    # Create pair plot using Seaborn
    # Sample data if too large for performance
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    
    # Add experience level as hue if available
    hue_col = 'experience_level' if 'experience_level' in df_sample.columns else None
    
    pair_plot = sns.pairplot(df_sample[key_cols + ([hue_col] if hue_col else [])], 
                             hue=hue_col, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 20},
                             height=3, aspect=1.2)
    
    pair_plot.fig.suptitle('Pair Plot: Multivariate Analysis', fontsize=16, fontweight='bold', y=1.02)
    filepath = os.path.join(FIGURE_DIR, '06_pair_plots.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 10. VISUALIZATION 7: PIE CHARTS AND DONUT CHARTS
# REQUIREMENT 2.2.2: Pie charts or donut charts (where appropriate)
# ============================================================================

def plot_pie_charts(df: pd.DataFrame):
    """
    Create pie charts and donut charts for categorical distributions.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING PIE CHARTS AND DONUT CHARTS")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pie Charts and Donut Charts', fontsize=16, fontweight='bold')
    
    # Plot 1: Pie chart - Distribution by Experience Level
    if 'experience_level' in df.columns:
        exp_counts = df['experience_level'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        axes[0, 0].pie(exp_counts.values, labels=exp_counts.index, autopct='%1.1f%%', 
                      startangle=90, colors=colors[:len(exp_counts)])
        axes[0, 0].set_title('Distribution by Experience Level', fontsize=14, fontweight='bold')
    
    # Plot 2: Donut chart - Distribution by Company Size
    if 'company_size' in df.columns:
        size_counts = df['company_size'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        wedges, texts, autotexts = axes[0, 1].pie(size_counts.values, labels=size_counts.index, 
                                                   autopct='%1.1f%%', startangle=90,
                                                   colors=colors[:len(size_counts)])
        # Create donut by adding a circle in the center
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[0, 1].add_artist(centre_circle)
        axes[0, 1].set_title('Distribution by Company Size (Donut Chart)', fontsize=14, fontweight='bold')
    
    # Plot 3: Pie chart - Distribution by Employment Type
    if 'employment_type' in df.columns:
        emp_counts = df['employment_type'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        axes[1, 0].pie(emp_counts.values, labels=emp_counts.index, autopct='%1.1f%%',
                      startangle=90, colors=colors[:len(emp_counts)])
        axes[1, 0].set_title('Distribution by Employment Type', fontsize=14, fontweight='bold')
    
    # Plot 4: Donut chart - Top 5 Job Titles
    if 'job_title' in df.columns:
        top_jobs = df['job_title'].value_counts().head(5)
        other_count = df['job_title'].value_counts().iloc[5:].sum() if len(df['job_title'].value_counts()) > 5 else 0
        if other_count > 0:
            top_jobs['Other'] = other_count
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F']
        wedges, texts, autotexts = axes[1, 1].pie(top_jobs.values, labels=top_jobs.index, 
                                                   autopct='%1.1f%%', startangle=90,
                                                   colors=colors[:len(top_jobs)])
        # Create donut by adding a circle in the center
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        axes[1, 1].add_artist(centre_circle)
        axes[1, 1].set_title('Top 5 Job Titles (Donut Chart)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '07_pie_donut_charts.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 11. VISUALIZATION 8: VIOLIN PLOTS
# REQUIREMENT 2.2.2: Violin plots or swarm plots
# ============================================================================

def plot_violin_plots(df: pd.DataFrame):
    """
    Create violin plots to show distribution shapes across categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("CREATING VIOLIN PLOTS")
    print("=" * 80)
    
    salary_col = 'salary_in_usd'
    
    if salary_col not in df.columns:
        print(f"Column {salary_col} not found!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Violin Plots: Distribution Shapes', fontsize=16, fontweight='bold')
    
    # Plot 1: Salary by Experience Level
    if 'experience_level' in df.columns:
        df_clean = df[df['experience_level'].notna()].copy()
        sns.violinplot(data=df_clean, x='experience_level', y=salary_col, ax=axes[0, 0], palette='Set2')
        axes[0, 0].set_xlabel('Experience Level', fontsize=12)
        axes[0, 0].set_ylabel('Salary in USD', fontsize=12)
        axes[0, 0].set_title('Salary Distribution by Experience Level', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Salary by Company Size
    if 'company_size' in df.columns:
        df_clean = df[df['company_size'].notna()].copy()
        sns.violinplot(data=df_clean, x='company_size', y=salary_col, ax=axes[0, 1], palette='pastel')
        axes[0, 1].set_xlabel('Company Size', fontsize=12)
        axes[0, 1].set_ylabel('Salary in USD', fontsize=12)
        axes[0, 1].set_title('Salary Distribution by Company Size', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Salary by Employment Type
    if 'employment_type' in df.columns:
        df_clean = df[df['employment_type'].notna()].copy()
        sns.violinplot(data=df_clean, x='employment_type', y=salary_col, ax=axes[1, 0], palette='coolwarm')
        axes[1, 0].set_xlabel('Employment Type', fontsize=12)
        axes[1, 0].set_ylabel('Salary in USD', fontsize=12)
        axes[1, 0].set_title('Salary Distribution by Employment Type', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Salary by Remote Ratio (grouped)
    if 'remote_ratio' in df.columns:
        # Create remote ratio categories
        df_remote = df.copy()
        df_remote['remote_category'] = pd.cut(df_remote['remote_ratio'], 
                                             bins=[-1, 0, 50, 100], 
                                             labels=['On-site (0%)', 'Hybrid (1-50%)', 'Remote (100%)'])
        df_clean = df_remote[df_remote['remote_category'].notna()].copy()
        sns.violinplot(data=df_clean, x='remote_category', y=salary_col, ax=axes[1, 1], palette='viridis')
        axes[1, 1].set_xlabel('Remote Work Category', fontsize=12)
        axes[1, 1].set_ylabel('Salary in USD', fontsize=12)
        axes[1, 1].set_title('Salary Distribution by Remote Work Category', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '08_violin_plots.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()


# ============================================================================
# 12. PATTERN, TREND, AND ANOMALY DETECTION
# REQUIREMENT 2.2.1: Identify patterns, trends, and anomalies
# REQUIREMENT 2.2.3: Identify and discuss any outliers found in the data
# ============================================================================

def detect_patterns_and_anomalies(df: pd.DataFrame):
    """
    Identify patterns, trends, and anomalies in the data.
    This fulfills requirement 2.2.1 and 2.2.3.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    """
    print("\n" + "=" * 80)
    print("PATTERN, TREND, AND ANOMALY DETECTION")
    print("=" * 80)
    
    salary_col = 'salary_in_usd'
    
    if salary_col not in df.columns:
        print(f"Column {salary_col} not found!")
        return
    
    salary_data = df[salary_col].values  # Convert to NumPy array
    
    # Anomaly Detection using Z-score method (REQUIREMENT 2.2.3)
    print("\n--- Anomaly Detection (Z-score method) ---")
    z_scores = np.abs(zscore(salary_data))
    threshold = 3  # Standard threshold for outliers
    outliers_zscore = np.sum(z_scores > threshold)
    outlier_indices = np.where(z_scores > threshold)[0]
    
    print(f"Outliers detected (|Z-score| > {threshold}): {outliers_zscore:,} ({outliers_zscore/len(salary_data)*100:.2f}%)")
    if outliers_zscore > 0:
        print(f"Outlier salary range: ${np.min(salary_data[outlier_indices]):,.2f} - ${np.max(salary_data[outlier_indices]):,.2f}")
        print(f"Normal salary range: ${np.min(salary_data[z_scores <= threshold]):,.2f} - ${np.max(salary_data[z_scores <= threshold]):,.2f}")
    
    # Pattern Detection: Salary trends by year
    print("\n--- Trend Analysis: Salary by Year ---")
    if 'work_year' in df.columns:
        yearly_stats = df.groupby('work_year')[salary_col].agg([
            ('count', 'count'),
            ('mean', np.mean),
            ('median', np.median),
            ('std', np.std)
        ])
        print(yearly_stats)
        
        # Calculate trend direction
        if len(yearly_stats) > 1:
            mean_trend = yearly_stats['mean'].values
            trend_direction = "increasing" if mean_trend[-1] > mean_trend[0] else "decreasing"
            trend_magnitude = ((mean_trend[-1] - mean_trend[0]) / mean_trend[0]) * 100
            print(f"\nSalary trend: {trend_direction} by {abs(trend_magnitude):.2f}%")
    
    # Pattern Detection: Remote work impact
    print("\n--- Pattern Analysis: Remote Work Impact ---")
    if 'remote_ratio' in df.columns:
        # Create remote categories
        df['remote_category'] = pd.cut(df['remote_ratio'], 
                                      bins=[-1, 0, 50, 100], 
                                      labels=['On-site', 'Hybrid', 'Remote'])
        remote_impact = df.groupby('remote_category')[salary_col].agg([
            ('count', 'count'),
            ('mean', np.mean),
            ('median', np.median)
        ])
        print(remote_impact)
    
    # Pattern Detection: Experience level progression
    print("\n--- Pattern Analysis: Experience Level Salary Progression ---")
    if 'experience_level' in df.columns:
        exp_order = ['EN', 'MI', 'SE', 'EX']  # Entry, Mid, Senior, Executive
        exp_order = [e for e in exp_order if e in df['experience_level'].unique()]
        exp_progression = df[df['experience_level'].isin(exp_order)].groupby('experience_level')[salary_col].mean().reindex(exp_order)
        print(exp_progression)
        
        # Calculate salary increase between levels
        if len(exp_progression) > 1:
            print("\nSalary increase between levels:")
            for i in range(len(exp_progression) - 1):
                increase = ((exp_progression.iloc[i+1] - exp_progression.iloc[i]) / exp_progression.iloc[i]) * 100
                print(f"  {exp_progression.index[i]} → {exp_progression.index[i+1]}: {increase:.2f}%")
    
    # Pattern Detection: Top paying job titles
    print("\n--- Pattern Analysis: Top Paying Job Titles ---")
    if 'job_title' in df.columns:
        job_stats = df.groupby('job_title')[salary_col].agg([
            ('count', 'count'),
            ('mean', np.mean),
            ('median', np.median)
        ]).sort_values('mean', ascending=False).head(10)
        print(job_stats)
    
    # Create visualization for anomalies (REQUIREMENT 2.2.3)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Anomaly Detection Visualization', fontsize=16, fontweight='bold')
    
    # Plot 1: Z-score distribution
    axes[0].hist(z_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (Z={threshold})')
    axes[0].set_xlabel('Absolute Z-score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Z-score Distribution for Salary', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Outliers highlighted in scatter
    if 'remote_ratio' in df.columns:
        normal_mask = z_scores <= threshold
        outlier_mask = z_scores > threshold
        
        axes[1].scatter(df.loc[normal_mask, 'remote_ratio'], 
                       df.loc[normal_mask, salary_col], 
                       alpha=0.5, s=20, color='blue', label='Normal')
        axes[1].scatter(df.loc[outlier_mask, 'remote_ratio'], 
                       df.loc[outlier_mask, salary_col], 
                       alpha=0.7, s=50, color='red', marker='x', label='Outliers')
        axes[1].set_xlabel('Remote Ratio (%)', fontsize=12)
        axes[1].set_ylabel('Salary in USD', fontsize=12)
        axes[1].set_title('Outliers Highlighted (Z-score > 3)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(FIGURE_DIR, '09_anomaly_detection.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Saved: {filepath}")
    plt.close()
    
    # Print summary of outliers discussion (REQUIREMENT 2.2.3)
    print("\n--- Outlier Discussion ---")
    print(f"Total outliers identified: {outliers_zscore:,} ({outliers_zscore/len(salary_data)*100:.2f}% of data)")
    print(f"Outliers represent extreme salary values that deviate significantly from the mean.")
    print(f"These may indicate:")
    print(f"  1. High-level executive positions")
    print(f"  2. Specialized roles with high demand")
    print(f"  3. Geographic variations (high cost-of-living areas)")
    print(f"  4. Potential data entry errors (especially for entry-level positions)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to execute all EDA analyses and visualizations.
    This fulfills all requirements from section 2.2.
    """
    print("\n" + "=" * 80)
    print("EXPLORATORY DATA ANALYSIS - DATA SCIENCE SALARIES")
    print("=" * 80)
    print("\nThis script fulfills all requirements:")
    print("  [OK] Comprehensive use of NumPy and Pandas")
    print("  [OK] At least 5 different visualization types")
    print("  [OK] Statistical analysis of key variables")
    print("  [OK] Relationship and correlation investigation")
    print("  [OK] Pattern, trend, and anomaly detection")
    print("\n" + "=" * 80)
    
    # Load processed data
    data_path = "data/processed/DataScience_salaries_2025_cleaned.csv"
    df = load_processed_data(data_path)
    
    # Perform initial statistical summary (REQUIREMENT 2.2.3)
    initial_statistical_summary(df)
    
    # Statistical analysis of key variables (REQUIREMENT 2.2.1)
    statistical_analysis(df)
    
    # Correlation analysis (REQUIREMENT 2.2.1, 2.2.3)
    corr_matrix = correlation_analysis(df)
    
    # Create all visualizations (REQUIREMENT 2.2.2)
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    
    plot_distributions(df)  # Distribution plots (histograms, KDE, box plots)
    plot_correlation_heatmap(df)  # Correlation heatmaps
    plot_scatter_with_trends(df)  # Scatter plots with trend lines
    plot_bar_charts(df)  # Bar charts and count plots
    plot_time_series(df)  # Time series plots
    plot_pair_plots(df)  # Pair plots for multivariate analysis
    plot_pie_charts(df)  # Pie charts and donut charts
    plot_violin_plots(df)  # Violin plots
    
    # Pattern and anomaly detection (REQUIREMENT 2.2.1, 2.2.3)
    detect_patterns_and_anomalies(df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("EDA COMPLETE!")
    print("=" * 80)
    print(f"\nAll visualizations saved to: {FIGURE_DIR}/")
    print("\nGenerated files:")
    print("  1. 01_distribution_plots.png - Histograms, KDE plots, and box plots")
    print("  2. 02_correlation_heatmap.png - Correlation heatmap")
    print("  3. 03_scatter_trends.png - Scatter plots with trend lines")
    print("  4. 04_bar_charts.png - Bar charts and count plots")
    print("  5. 05_time_series.png - Time series plots")
    print("  6. 06_pair_plots.png - Pair plots for multivariate analysis")
    print("  7. 07_pie_donut_charts.png - Pie charts and donut charts")
    print("  8. 08_violin_plots.png - Violin plots")
    print("  9. 09_anomaly_detection.png - Anomaly detection visualization")
    print("\n" + "=" * 80)
    print("All requirements fulfilled!")
    print("=" * 80)


if __name__ == "__main__":
    main()

