"""
Handling missing values in the dataset.
"""
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def detect_missing_values(df: pd.DataFrame) -> dict:
    """
    Detect and report missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with missing value statistics
    """
    missing_count = df.isnull().sum()
    missing_percent = (df.isnull().sum() / len(df)) * 100
    
    missing_stats = {
        'total_missing': df.isnull().sum().sum(),
        'columns_with_missing': missing_count[missing_count > 0].to_dict(),
        'missing_percent': missing_percent[missing_percent > 0].to_dict()
    }
    return missing_stats


def impute_missing_values(df: pd.DataFrame, 
                         strategy_numeric: str = 'mean',
                         strategy_categorical: str = 'most_frequent',
                         fill_value_categorical: str = 'missing') -> pd.DataFrame:
    """
    Impute missing values using specified strategies.
    
    Args:
        df: Input DataFrame
        strategy_numeric: Strategy for numeric columns ('mean', 'median')
        strategy_categorical: Strategy for categorical columns ('most_frequent', 'constant')
        fill_value_categorical: Value to fill categorical columns if strategy is 'constant'
        
    Returns:
        DataFrame with imputed values
    """
    df_imputed = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(numeric_cols) > 0 and df_imputed[numeric_cols].isnull().sum().sum() > 0:
        imputer_numeric = SimpleImputer(strategy=strategy_numeric)
        df_imputed[numeric_cols] = imputer_numeric.fit_transform(df_imputed[numeric_cols])
    
    if len(categorical_cols) > 0 and df_imputed[categorical_cols].isnull().sum().sum() > 0:
        imputer_categorical = SimpleImputer(strategy=strategy_categorical, 
                                           fill_value=fill_value_categorical if strategy_categorical == 'constant' else None)
        df_imputed[categorical_cols] = imputer_categorical.fit_transform(df_imputed[categorical_cols])
    
    return df_imputed


def drop_rows_with_missing(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Drop rows with missing values above a threshold.
    
    Args:
        df: Input DataFrame
        threshold: Drop rows with missing percentage above this threshold (0-1)
        
    Returns:
        DataFrame with rows removed
    """
    if threshold > 0:
        missing_percent = df.isnull().sum(axis=1) / len(df.columns)
        return df[missing_percent <= threshold]
    else:
        return df.dropna()
