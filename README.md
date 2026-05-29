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

**End-to-End Pipeline**: main.py orchestrates preprocessing → target analysis → correlation → histograms

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
    └── processed/
        ├── estat_hlth_silc_11.csv   # Preprocessed output (307,152 rows × 91 columns)
        └── plots/
            ├── correlation_heatmap.png          # Feature correlation matrix visualization
            ├── correlation_matrix.csv           # Numeric correlation values
            ├── target_distribution_value.csv    # Target variable statistics
            ├── histogram_value.png              # Target variable distribution
            └── histogram_*.png                  # 91 histograms (one per feature)
```

## How to Run

### 1. Install Dependencies

`bash
pip install -r requirements.txt
`

### 2. Run the Complete Pipeline

`bash
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

**Output**: All visualizations and data files written to data/processed/plots/

### 3. Use Individual Preprocessing Components

`python
from src.preprocessing import (
    detect_missing_values,
    impute_missing_values,
    encode_with_one_hot,
    scale_features_standard,
    extract_datetime_features
)

# Load your data
import pandas as pd
df = pd.read_csv('your_data.csv')

# Use any preprocessing function independently
missing_stats = detect_missing_values(df)
df_imputed = impute_missing_values(df)
df_scaled, scaler = scale_features_standard(df_imputed)
`

## Preprocessing Pipeline Details

### Step 1: Parse TSV to Long Format
- Input: ESTAT TSV with multi-line headers (units, measures, dimensions)
- Output: Long format DataFrame with columns: value, unit_*, quantile_*, age_*, sex_*, geo_*, year_*
- Extracts numerical values from complex ESTAT format using regex

### Step 2: Detect Missing Values
- Scans for NaN, None, empty strings
- Reports: total missing count, columns affected, percentage of missing
- Target column 'value' has 101,851 missing values (33.2%)

### Step 3: Impute Missing Values
- Numerical columns: SimpleImputer with strategy='mean'
- Categorical columns: SimpleImputer with strategy='most_frequent'
- Default behavior: keep NaN values; can drop rows above threshold

### Step 4: Encode Categorical Features
- One-hot encoding with drop_first=True to avoid multicollinearity
- Categorical columns detected: quantile, age, sex, geo, year
- Result: Binary indicators for each category value

### Step 5: Scale Numerical Features
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
equirements.txt.

## Code Design

### Modularity
- Each preprocessing step in its own file under src/preprocessing/
- All functions are pure (no side effects) and reusable
- Main orchestrator preprocess.py combines steps into a pipeline

### Documentation
- All functions have docstrings describing purpose, arguments, returns, and exceptions
- No inline comments; code is self-documenting through clear naming
- Type hints for parameters and return values

### Auto-Detection
- check_balance.py auto-detects target column ('value' for regression, categorical for classification)
- EDA modules auto-detect numerical vs categorical columns
- Graceful fallbacks with informative error messages

## Notes

- **Data Exclusion**: data/ folder is git-ignored; only code is version controlled
- **Reusability**: All preprocessing functions can be imported and used independently
- **Scalability**: Pipeline designed to handle large datasets efficiently
- **Regression Focus**: This dataset predicts continuous health values (not classification)

## Next Steps

- Implement 	rain.py for model training (regression and classification baselines)
- Feature selection and dimensionality reduction
- Hyperparameter tuning and cross-validation
- Model evaluation and comparison with baseline metrics
