"""
Model training module with reusable functions for data splitting,
cross-validation, and model training/evaluation.
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict, Any, List


def split_data(
    data: pd.DataFrame,
    target_column: str = 'value',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        data: Input DataFrame with features and target
        target_column: Name of target column
        test_size: Proportion of data for testing (default 0.2 for 80/20 split)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        ValueError: If target_column not in data or data is empty
    """
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    if data.empty:
        raise ValueError("Data is empty")
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )
    
    print(f"Data split: {len(X_train)} train samples, {len(X_test)} test samples")
    
    return X_train, X_test, y_train, y_test


def create_model(model_name: str) -> Any:
    """
    Create and return a model instance based on name.
    
    Args:
        model_name: Name of model ('linear_regression', 'random_forest')
        
    Returns:
        Initialized model instance
        
    Raises:
        ValueError: If model_name is not recognized
    """
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model: Any
) -> Any:
    """
    Train a single model on training data.
    
    Args:
        X_train: Training features
        y_train: Training target values
        model: Model instance to train
        
    Returns:
        Trained model instance
    """
    model.fit(X_train, y_train)
    return model


def cross_validate_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_names: List[str] = None,
    cv_folds: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Perform k-fold cross-validation on multiple models.
    
    Args:
        X_train: Training features
        y_train: Training target values
        model_names: List of model names to test (default: all available models)
        cv_folds: Number of cross-validation folds (default 5)
        
    Returns:
        Dictionary with cv scores for each model:
        {
            'model_name': {
                'r2_scores': array of scores,
                'r2_mean': float,
                'r2_std': float
            }
        }
    """
    if model_names is None:
        model_names = ['linear_regression', 'random_forest']
    
    cv_results = {}
    
    for model_name in model_names:
        model = create_model(model_name)
        
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring='r2',
            n_jobs=-1
        )
        
        cv_results[model_name] = {
            'r2_scores': scores,
            'r2_mean': scores.mean(),
            'r2_std': scores.std()
        }
        
        print(f"{model_name:20s} - CV R² Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return cv_results


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_names: List[str] = None
) -> Dict[str, Any]:
    """
    Train all specified models on training data.
    
    Args:
        X_train: Training features
        y_train: Training target values
        model_names: List of model names (default: all available models)
        
    Returns:
        Dictionary with trained models: {'model_name': trained_model_instance}
    """
    if model_names is None:
        model_names = ['linear_regression', 'random_forest']
    
    trained_models = {}
    
    for model_name in model_names:
        print(f"Training {model_name}...", end=' ', flush=True)
        model = create_model(model_name)
        trained_models[model_name] = train_model(X_train, y_train, model)
        print("Done")
    
    return trained_models


def save_model(model: Any, filepath: str) -> None:
    """
    Save a trained model to disk using pickle.
    
    Args:
        model: Trained model instance
        filepath: Path where to save the model
        
    Raises:
        IOError: If file cannot be written
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load a trained model from disk.
    
    Args:
        filepath: Path to the saved model file
        
    Returns:
        Loaded model instance
        
    Raises:
        FileNotFoundError: If model file not found
        EOFError: If model file is corrupted
    """
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    return model


def save_all_models(
    trained_models: Dict[str, Any],
    save_dir: str = None
) -> Dict[str, str]:
    """
    Save all trained models to disk.
    
    Args:
        trained_models: Dictionary of trained models
        save_dir: Directory to save models (default: data/models)
        
    Returns:
        Dictionary mapping model names to save paths
    """
    if save_dir is None:
        save_dir = os.path.join('data', 'models')
    
    os.makedirs(save_dir, exist_ok=True)
    
    saved_paths = {}
    for model_name, model in trained_models.items():
        filepath = os.path.join(save_dir, f'{model_name}.pkl')
        save_model(model, filepath)
        saved_paths[model_name] = filepath
    
    return saved_paths
