"""
Feature engineering: extracting and creating new features from existing data.
Includes date/time feature extraction, binning, and custom transformations.
"""
import pandas as pd
import numpy as np


def extract_datetime_features(df: pd.DataFrame, 
                              datetime_column: str,
                              components: list = ['year', 'month', 'day', 'dayofweek']) -> pd.DataFrame:
    """
    Extract temporal features from a datetime column.
    
    Args:
        df: Input DataFrame
        datetime_column: Name of the datetime column
        components: List of components to extract ('year', 'month', 'day', 'dayofweek', 'quarter', 'week')
        
    Returns:
        DataFrame with extracted datetime features
    """
    df_new = df.copy()
    
    if datetime_column not in df_new.columns:
        raise ValueError(f"Column '{datetime_column}' not found in DataFrame")
    
    if not pd.api.types.is_datetime64_any_dtype(df_new[datetime_column]):
        df_new[datetime_column] = pd.to_datetime(df_new[datetime_column], errors='coerce')
    
    if 'year' in components:
        df_new[f'{datetime_column}_year'] = df_new[datetime_column].dt.year
    if 'month' in components:
        df_new[f'{datetime_column}_month'] = df_new[datetime_column].dt.month
    if 'day' in components:
        df_new[f'{datetime_column}_day'] = df_new[datetime_column].dt.day
    if 'dayofweek' in components:
        df_new[f'{datetime_column}_dayofweek'] = df_new[datetime_column].dt.dayofweek
    if 'quarter' in components:
        df_new[f'{datetime_column}_quarter'] = df_new[datetime_column].dt.quarter
    if 'week' in components:
        df_new[f'{datetime_column}_week'] = df_new[datetime_column].dt.isocalendar().week
    
    return df_new


def bin_continuous_feature(df: pd.DataFrame,
                          column: str,
                          n_bins: int = None,
                          bins: list = None,
                          labels: list = None,
                          strategy: str = 'quantile') -> pd.DataFrame:
    """
    Discretize a continuous feature into categorical bins.
    
    Args:
        df: Input DataFrame
        column: Column name to bin
        n_bins: Number of bins (if bins not specified)
        bins: Custom bin edges
        labels: Custom labels for bins
        strategy: 'quantile' (equal frequency) or 'uniform' (equal width)
        
    Returns:
        DataFrame with new binned column
    """
    df_new = df.copy()
    
    if column not in df_new.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    if bins is not None:
        df_new[f'{column}_binned'] = pd.cut(df_new[column], bins=bins, labels=labels, include_lowest=True)
    elif n_bins is not None:
        if strategy == 'quantile':
            df_new[f'{column}_binned'] = pd.qcut(df_new[column], q=n_bins, labels=labels, duplicates='drop')
        else:
            df_new[f'{column}_binned'] = pd.cut(df_new[column], bins=n_bins, labels=labels)
    else:
        raise ValueError("Either 'bins' or 'n_bins' must be specified")
    
    return df_new


def create_polynomial_features(df: pd.DataFrame,
                              column: str,
                              degree: int = 2,
                              include_bias: bool = False) -> pd.DataFrame:
    """
    Create polynomial features from a numerical column.
    
    Args:
        df: Input DataFrame
        column: Column name for polynomial transformation
        degree: Polynomial degree
        include_bias: Whether to include intercept/bias term
        
    Returns:
        DataFrame with new polynomial features
    """
    df_new = df.copy()
    
    if column not in df_new.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")
    
    for d in range(2, degree + 1):
        df_new[f'{column}_pow{d}'] = df_new[column] ** d
    
    return df_new


def create_interaction_features(df: pd.DataFrame,
                               column1: str,
                               column2: str) -> pd.DataFrame:
    """
    Create interaction feature between two numerical columns.
    
    Args:
        df: Input DataFrame
        column1: First column name
        column2: Second column name
        
    Returns:
        DataFrame with new interaction feature
    """
    df_new = df.copy()
    
    if column1 not in df_new.columns or column2 not in df_new.columns:
        raise ValueError(f"Columns '{column1}' or '{column2}' not found")
    
    df_new[f'{column1}_x_{column2}'] = df_new[column1] * df_new[column2]
    
    return df_new


def create_ratio_features(df: pd.DataFrame,
                         numerator_col: str,
                         denominator_col: str,
                         avoid_division_by_zero: bool = True) -> pd.DataFrame:
    """
    Create ratio feature between two numerical columns.
    
    Args:
        df: Input DataFrame
        numerator_col: Numerator column name
        denominator_col: Denominator column name
        avoid_division_by_zero: Replace inf/NaN with 0 if True
        
    Returns:
        DataFrame with new ratio feature
    """
    df_new = df.copy()
    
    if numerator_col not in df_new.columns or denominator_col not in df_new.columns:
        raise ValueError(f"Columns '{numerator_col}' or '{denominator_col}' not found")
    
    ratio = df_new[numerator_col] / df_new[denominator_col]
    
    if avoid_division_by_zero:
        ratio = ratio.replace([np.inf, -np.inf], 0).fillna(0)
    
    df_new[f'{numerator_col}_div_{denominator_col}'] = ratio
    
    return df_new
