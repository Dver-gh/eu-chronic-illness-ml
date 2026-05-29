"""
Feature scaling and normalization for numerical features.
Uses StandardScaler from sklearn.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_numerical_columns(df: pd.DataFrame) -> list:
    """
    Identify numerical columns in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of numerical column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()


def scale_features_standard(df: pd.DataFrame, 
                           columns: list = None,
                           fit_scaler=None) -> tuple:
    """
    Scale numerical features using StandardScaler (z-score normalization).
    Formula: (x - mean) / std
    
    Args:
        df: Input DataFrame
        columns: List of columns to scale (if None, scales all numeric)
        fit_scaler: Optional pre-fitted scaler (for test set scaling)
        
    Returns:
        Tuple of (scaled_dataframe, scaler_object)
    """
    df_scaled = df.copy()
    
    if columns is None:
        columns = get_numerical_columns(df_scaled)
    
    if len(columns) == 0:
        return df_scaled, None
    
    if fit_scaler is None:
        scaler = StandardScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        return df_scaled, scaler
    else:
        df_scaled[columns] = fit_scaler.transform(df_scaled[columns])
        return df_scaled, fit_scaler


def scale_features_minmax(df: pd.DataFrame,
                         columns: list = None,
                         feature_range: tuple = (0, 1),
                         fit_scaler=None) -> tuple:
    """
    Scale numerical features using MinMaxScaler (normalization to range).
    Formula: (x - min) / (max - min) * (range_max - range_min) + range_min
    
    Args:
        df: Input DataFrame
        columns: List of columns to scale (if None, scales all numeric)
        feature_range: Target range for scaled features (default: 0-1)
        fit_scaler: Optional pre-fitted scaler (for test set scaling)
        
    Returns:
        Tuple of (scaled_dataframe, scaler_object)
    """
    df_scaled = df.copy()
    
    if columns is None:
        columns = get_numerical_columns(df_scaled)
    
    if len(columns) == 0:
        return df_scaled, None
    
    if fit_scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        return df_scaled, scaler
    else:
        df_scaled[columns] = fit_scaler.transform(df_scaled[columns])
        return df_scaled, fit_scaler


def get_scaling_statistics(scaler) -> dict:
    """
    Extract scaling statistics from a fitted scaler.
    
    Args:
        scaler: Fitted StandardScaler or MinMaxScaler
        
    Returns:
        Dictionary with scaling statistics
    """
    if isinstance(scaler, StandardScaler):
        return {
            'type': 'StandardScaler',
            'mean': scaler.mean_.tolist(),
            'std': scaler.scale_.tolist()
        }
    elif isinstance(scaler, MinMaxScaler):
        return {
            'type': 'MinMaxScaler',
            'min': scaler.data_min_.tolist(),
            'max': scaler.data_max_.tolist(),
            'range': scaler.feature_range
        }
    return {'type': 'Unknown scaler'}
