# Data Science Final Project

A comprehensive data science project analyzing Data Science salaries dataset using exploratory data analysis, machine learning models, and clustering algorithms.

## Project Overview

This project performs end-to-end data analysis on a Data Science salaries dataset, including:
- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Machine Learning Models (Classification, Regression, Clustering)
- Model evaluation and comparison
- Result visualization and reporting

## Data Source

The dataset used in this project is from Kaggle:
- **Dataset**: AI and Data Science Job Salaries (2020-2025)
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/pratyushmishradev/ai-and-data-science-job-salaries-20202025)
- **Description**: Real-world data containing information about data science job salaries, including job titles, experience levels, company sizes, remote work ratios, and salary information in USD.

## Project Structure

```
data_science_final_proj/
│
├── data/                          # Data storage directory
│   ├── raw/                       # Raw, unprocessed data
│   │   └── DataScience_salaries_2025.csv
│   └── processed/                 # Cleaned and processed data
│       └── DataScience_salaries_2025_cleaned.csv
│
├── py_codes/                      # Python source code
│   ├── data_processing.py         # Data cleaning and preprocessing
│   ├── eda.py                     # Exploratory Data Analysis
│   └── models/                    # Machine Learning models
│       ├── classification.py      # Classification models
│       ├── regression.py          # Regression models
│       └── clustering_model.py   # Clustering models (K-Means, DBSCAN)
│
├── reports/                       # Analysis reports and visualizations
│   ├── figures/                   # Generated plots and visualizations
│   │   ├── 01_distribution_plots.png
│   │   ├── 02_correlation_heatmap.png
│   │   ├── 03_scatter_trends.png
│   │   ├── 04_bar_charts.png
│   │   ├── 05_time_series.png
│   │   ├── 06_pair_plots.png
│   │   ├── 07_pie_donut_charts.png
│   │   ├── 08_violin_plots.png
│   │   ├── 09_anomaly_detection.png
│   │   ├── elbow_method.png
│   │   └── kmeans_clusters.png
│   └── results/                   # Model results and metrics
│       └── clustering_model_report.txt
│
├── results/                       # Additional model results
│   ├── classification_results.txt
│   ├── regression_result.txt
│   ├── confusion_matrix_dt.png
│   ├── confusion_matrix_lr.png
│   └── confusion_matrix_rf.png
│
├── requirements.txt               # Python package dependencies
└── README.md                      # This file
```

## Directory Explanations

### `/data/`
- **`raw/`**: Contains the original, unprocessed dataset
- **`processed/`**: Contains cleaned and preprocessed data ready for analysis

### `/py_codes/`
- **`data_processing.py`**: Handles data loading, cleaning, missing value imputation, duplicate removal, and saves processed data
- **`eda.py`**: Performs comprehensive exploratory data analysis including:
  - Statistical summaries
  - Distribution plots (histograms, KDE, box plots)
  - Correlation heatmaps
  - Scatter plots with trend lines
  - Bar charts and count plots
  - Time series plots
  - Pair plots
  - Pie/donut charts
  - Violin plots
  - Anomaly detection
- **`models/classification.py`**: Implements classification models (Logistic Regression, Decision Tree, Random Forest)
- **`models/regression.py`**: Implements regression models (Linear Regression, Decision Tree, Random Forest)
- **`models/clustering_model.py`**: Implements clustering algorithms (K-Means with Elbow method, DBSCAN)

### `/reports/`
- **`figures/`**: Stores all generated visualizations from EDA and model analysis
  - EDA visualizations (01-09)
  - Clustering visualizations (elbow method, cluster plots)
- **`results/`**: Stores detailed analysis reports and metrics
  - Clustering model reports with metrics and cluster characteristics

### `/results/`
- Stores additional model outputs:
  - Classification and regression results
  - Confusion matrices for classification models

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd data_science_final_proj
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Processing
```bash
python py_codes/data_processing.py
```
- Cleans and preprocesses raw data
- Output: `data/processed/DataScience_salaries_2025_cleaned.csv`

### 2. Exploratory Data Analysis
```bash
python py_codes/eda.py
```
- Generates comprehensive EDA visualizations
- Output: All figures saved to `reports/figures/`

### 3. Machine Learning Models

#### Classification
```bash
python py_codes/models/classification.py
```
- Trains classification models
- Output: Results saved to `results/classification_results.txt` and confusion matrices

#### Regression
```bash
python py_codes/models/regression.py
```
- Trains regression models
- Output: Results saved to `results/regression_result.txt`

#### Clustering
```bash
python py_codes/models/clustering_model.py
```
- Implements K-Means and DBSCAN clustering
- Output: 
  - Visualizations: `reports/figures/elbow_method.png`, `kmeans_clusters.png`
  - Metrics: `reports/results/clustering_model_report.txt`
  - CSV files: `reports/results/clustering_metrics.csv`, `cluster_sizes.csv`

## Result Storage

### Visualizations
All plots and figures are stored in:
- **`reports/figures/`**: EDA visualizations and clustering plots

### Model Results
- **`reports/results/`**: Detailed clustering analysis reports
- **`results/`**: Classification and regression results

### Metrics and Data
- Clustering metrics: `reports/results/clustering_metrics.csv`
- Cluster sizes: `reports/results/kmeans_cluster_sizes.csv`, `dbscan_cluster_sizes.csv`
- Summary: `reports/results/clustering_summary.csv`

## Requirements

- Python 3.8+
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- seaborn >= 0.12.0
- scikit-learn >= 1.3.0
- scipy >= 1.10.0

## Project Workflow

1. **Data Processing** → Clean and prepare data
2. **EDA** → Explore and visualize data patterns
3. **Modeling** → Train and evaluate ML models
4. **Results** → Generate reports and visualizations

## Notes

- All scripts use the processed data from `data/processed/`
- Results are automatically saved to appropriate directories
- Visualizations are generated in high resolution (300 DPI)
- Reports include detailed metrics and analysis
