# EU Chronic-Illness ML Study

Machine learning preprocessing and exploratory data analysis pipeline for ESTAT chronic illness health data.

## Status

**Preprocessing**: Fully implemented with modular, reusable components
- Parse ESTAT TSV → long CSV format with proper headers
- Detect and impute missing values (multiple strategies: mean, most_frequent, custom)
- Categorical encoding (one-hot, label encoding with drop_first=True to avoid multicollinearity)
- Numerical feature scaling (StandardScaler z-score, MinMaxScaler with configurable ranges)
- Feature engineering utilities (datetime extraction, binning, polynomial/interaction/ratio features)

**EDA Modules**: Auto-detecting target variables and data types
- Target distribution analysis (regression: histogram + stats; classification: bar chart + counts)
- Correlation heatmap with annotated values for all features
- Histograms with KDE curves for all numerical columns

**Model Training**: Full regression model pipeline with evaluation
- Data splitting (80/20 train/test split)
- Cross-validation (5-fold cross-validation with R² scoring)
- Multiple regression models: Linear Regression, Random Forest
- Model persistence (save/load trained models via pickle)

**Comprehensive Metrics**: Detailed model evaluation and comparison
- Regression metrics: MAE, MSE, RMSE, R², MAPE, Median Absolute Error
- Test set evaluation for all models
- Cross-validation statistics
- Metrics reporting (CSV export and console summary)

**End-to-End Pipeline**: main.py orchestrates preprocessing → EDA → model training → evaluation

## Project Structure

```
eu-chronic-illness-ml/
├── main.py                          # Entry point: runs complete pipeline
├── requirements.txt                 # Python dependencies with version pinning
├── README.md                        # This file
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                # Main orchestrator (5-step preprocessing)
│   ├── check_balance.py             # Target variable distribution analysis
│   ├── plot_correlation.py          # Feature correlation heatmap
│   ├── plot_histograms.py           # Feature distribution histograms
│   ├── train.py                     # Model training with cross-validation
│   ├── metrics.py                   # Metrics calculation and reporting
│   │
│   └── preprocessing/               # Reusable preprocessing components
│       ├── __init__.py              # Exports all functions
│       ├── missing_values.py        # Detect, impute, drop missing data
│       ├── encoders.py              # Label & one-hot encoding with sklearn
│       ├── scaling.py               # StandardScaler & MinMaxScaler
│       └── feature_engineering.py   # Datetime, binning, polynomial features
│
└── data/
    ├── raw/
    │   └── estat_hlth_silc_11.tsv   # Input: ESTAT chronic illness data
    ├── models/
    │   ├── linear_regression.pkl    # Trained Linear Regression model
    │   ├── random_forest.pkl        # Trained Random Forest model
    └── processed/
        ├── estat_hlth_silc_11.csv   # Preprocessed output (307,152 rows × 91 columns)
        ├── plots/
        │   ├── correlation_heatmap.png          # Feature correlation matrix visualization
        │   ├── correlation_matrix.csv           # Numeric correlation values
        │   ├── target_distribution_value.csv    # Target variable statistics
        │   ├── histogram_value.png              # Target variable distribution
        │   └── histogram_*.png                  # 91 histograms (one per feature)
        └── reports/
            └── metrics_report.csv               # Model evaluation metrics and CV stats
```

## How to Run

### 1. Install Dependencies

`
pip install -r requirements.txt
`

### 2. Run the Complete Pipeline

`
python main.py
`

This will:
1. Parse ESTAT TSV into long format (307,152 samples × 91 features)
2. Detect missing values (33.2% of target 'value' column)
3. Impute missing numericals with mean, categoricals with most frequent
4. One-hot encode categorical features (quantile, age, sex, geo, year)
5. StandardScale all numerical features to mean=0, std=1
6. Auto-detect and visualize target variable distribution
7. Generate correlation heatmap with all features
8. Create histograms for all numerical columns
9. Split data into train/test sets (80/20)
10. Perform 5-fold cross-validation on multiple models
11. Train regression models: Linear Regression, Random Forest
12. Save trained models to data/models/
13. Evaluate models on test set with comprehensive metrics
14. Generate metrics report and console summary

**Output Files**:
- Preprocessed data: data/processed/estat_hlth_silc_11.csv
- Visualizations: data/processed/plots/ (heatmaps, histograms, distributions)
- Trained models: data/models/ (*.pkl files)
- Metrics report: data/processed/reports/metrics_report.csv

# Load the data
Put the data in .tsv format in data/raw/ directory


# Preprocessing Pipeline Details

## Step 1: Parse TSV to Long Format
- Input: ESTAT TSV with multi-line headers (units, measures, dimensions)
- Output: Long format DataFrame with columns: value, unit_*, quantile_*, age_*, sex_*, geo_*, year_*
- Extracts numerical values from complex ESTAT format using regex

## Step 2: Detect Missing Values
- Scans for NaN, None, empty strings
- Reports: total missing count, columns affected, percentage of missing
- Target column 'value' has 101,851 missing values (33.2%)

## Step 3: Impute Missing Values
- Numerical columns: SimpleImputer with strategy='mean'
- Categorical columns: SimpleImputer with strategy='most_frequent'
- Default behavior: keep NaN values; can drop rows above threshold

## Step 4: Encode Categorical Features
- One-hot encoding with drop_first=True to avoid multicollinearity
- Categorical columns detected: quantile, age, sex, geo, year
- Result: Binary indicators for each category value

## Step 5: Scale Numerical Features
- StandardScaler (z-score normalization): mean=0, std=1
- All numerical columns scaled: value, unit_PC_POP, and all one-hot encoded features
- Scaler is fitted on training data and reusable for test sets

## Output Files

**Preprocessed Data** (data/processed/):
- estat_hlth_silc_11.csv - Cleaned, encoded, scaled feature matrix (307,152 × 91)

**Visualizations and Reports** (data/processed/plots/):
- correlation_heatmap.png - Annotated heatmap of feature correlations
- correlation_matrix.csv - Numeric correlation coefficients (91 × 91)
- 	arget_distribution_value.csv - Descriptive statistics for target variable
- histogram_value.png - Target variable distribution histogram
- histogram_*.png - 91 histograms for all features (age, geo, year, quantile, sex, unit, value)

## Dependencies

`
pandas>=1.0
numpy>=1.19
matplotlib>=3.1
seaborn>=0.11
scikit-learn>=0.24
`

All dependencies pinned in 
requirements.txt.

## Model Training and Evaluation

### train.py Module
Reusable functions for model training and cross-validation:
- `split_data()`: Split preprocessed data into train/test sets (80/20 default)
- `create_model()`: Create model instances (Linear Regression, Random Forest)
- `train_model()`: Train a single model on training data
- `cross_validate_models()`: Perform k-fold cross-validation with R² scoring
- `train_all_models()`: Train all models on the same training set
- `save_model()`, `load_model()`: Persist/restore trained models to disk
- `save_all_models()`: Batch save all trained models

### metrics.py Module
Comprehensive metrics calculation and reporting:
- `calculate_regression_metrics()`: Compute MAE, MSE, RMSE, R², MAPE, Median AE
- `evaluate_models()`: Evaluate all trained models on test set
- `create_metrics_dataframe()`: Format results for easy comparison
- `generate_metrics_report()`: Create CSV report with all statistics
- `print_metrics_summary()`: Display formatted console output with rankings
- `get_model_comparison()`: Rank models by specific metric

### Independent Usage Examples

**Train a single model:**
```python
from src.train import split_data, create_model, train_model
import pandas as pd

data = pd.read_csv('data/processed/estat_hlth_silc_11.csv')
X_train, X_test, y_train, y_test = split_data(data)
model = create_model('random_forest')
trained_model = train_model(X_train, y_train, model)
```

**Cross-validate models:**
```python
from src.train import cross_validate_models

cv_results = cross_validate_models(X_train, y_train, cv_folds=5)
```

**Evaluate and report metrics:**
```python
from src.metrics import evaluate_models, generate_metrics_report

results = evaluate_models(trained_models, X_test, y_test)
metrics_df = generate_metrics_report(results, cv_results)
```

## Code Design

### Modularity
- Each preprocessing step in its own file under src/preprocessing/
- Training and metrics in separate modules for independent usage
- All functions are pure (no side effects) and reusable
- Main orchestrator main.py combines all steps into a complete pipeline

### Documentation
- All functions have comprehensive docstrings with purpose, arguments, returns, and exceptions
- Type hints for parameters and return values
- Clear error handling with informative messages

### Auto-Detection & Features
- check_balance.py auto-detects target column ('value' for regression)
- EDA modules auto-detect numerical vs categorical columns
- Training pipeline uses configurable parameters (train/test split, cv folds, etc.)
- Model persistence for production deployment

## Notes

- **Data Exclusion**: data/ folder is git-ignored; only code is version controlled
- **Reusability**: All preprocessing, training, and metrics functions can be imported and used independently
- **Scalability**: Pipeline designed to handle large datasets efficiently
- **Regression Focus**: This dataset predicts continuous health values (not classification)

## Next Steps

- Feature selection and dimensionality reduction
- Hyperparameter tuning with GridSearchCV or RandomizedSearchCV
- Ensemble methods and model stacking
- Time series analysis if temporal patterns are important
- Model deployment and API creation

