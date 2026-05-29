"""
Categorical encoding for machine learning models.
Supports LabelEncoder for target variables and OneHotEncoder/get_dummies for features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def get_categorical_columns(df: pd.DataFrame) -> list:
    """
    Identify categorical columns in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def encode_with_label_encoder(series: pd.Series, fit_encoder=None) -> tuple:
    """
    Encode a categorical series using LabelEncoder (typically for target variable).
    
    Args:
        series: Pandas Series to encode
        fit_encoder: Optional pre-fitted encoder (for test set encoding)
        
    Returns:
        Tuple of (encoded_array, encoder_object)
    """
    if fit_encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(series.astype(str))
        return encoded, encoder
    else:
        encoded = fit_encoder.transform(series.astype(str))
        return encoded, fit_encoder


def encode_with_one_hot(df: pd.DataFrame, 
                       columns: list = None,
                       drop_first: bool = True,
                       prefix: str = None) -> pd.DataFrame:
    """
    Encode categorical columns using one-hot encoding (via pandas get_dummies).
    
    Args:
        df: Input DataFrame
        columns: List of columns to encode (if None, encodes all categorical)
        drop_first: Whether to drop the first category (avoid multicollinearity)
        prefix: String prefix for encoded columns
        
    Returns:
        DataFrame with one-hot encoded categorical features
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = get_categorical_columns(df_encoded)
    
    if len(columns) > 0:
        prefix_dict = {col: prefix if prefix else col for col in columns} if prefix else None
        
        df_encoded = pd.get_dummies(
            df_encoded,
            columns=columns,
            drop_first=drop_first,
            prefix=prefix_dict,
            dtype=int
        )
    
    return df_encoded


def encode_with_sklearn_onehot(df: pd.DataFrame,
                              columns: list = None,
                              drop_first: bool = True) -> tuple:
    """
    Encode categorical columns using sklearn's OneHotEncoder.
    Useful for transforming test data with fitted encoder.
    
    Args:
        df: Input DataFrame
        columns: List of columns to encode (if None, encodes all categorical)
        drop_first: Whether to drop the first category
        
    Returns:
        Tuple of (encoded_dataframe, encoder_object, column_names)
    """
    df_encoded = df.copy()
    
    if columns is None:
        columns = get_categorical_columns(df_encoded)
    
    if len(columns) > 0:
        encoder = OneHotEncoder(drop='first' if drop_first else None, 
                               sparse_output=False,
                               handle_unknown='ignore')
        
        encoded_array = encoder.fit_transform(df_encoded[columns])
        encoded_col_names = encoder.get_feature_names_out(columns)
        
        df_encoded_ohe = pd.DataFrame(encoded_array, columns=encoded_col_names, index=df_encoded.index)
        
        non_encoded_cols = [col for col in df_encoded.columns if col not in columns]
        df_encoded = pd.concat([df_encoded[non_encoded_cols].reset_index(drop=True), 
                               df_encoded_ohe.reset_index(drop=True)], axis=1)
        
        return df_encoded, encoder, encoded_col_names
    else:
        return df_encoded, None, []


def inverse_transform_labels(encoded_array: np.ndarray, encoder: LabelEncoder) -> np.ndarray:
    """
    Inverse transform encoded labels back to original values.
    
    Args:
        encoded_array: Encoded array
        encoder: Fitted LabelEncoder
        
    Returns:
        Decoded array with original labels
    """
    return encoder.inverse_transform(encoded_array)
