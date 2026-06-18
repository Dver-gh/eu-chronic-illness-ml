"""
Metrics calculation and evaluation module for regression models.
Provides functions to compute and report comprehensive regression metrics.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error
)


def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = None
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics for predicted vs actual values.
    
    Computes: MAE, MSE, RMSE, R², MAPE, Median Absolute Error
    
    Args:
        y_true: Actual/true target values
        y_pred: Predicted target values
        model_name: Optional model name for labeling
        
    Returns:
        Dictionary with calculated metrics:
        {
            'MAE': float,
            'MSE': float,
            'RMSE': float,
            'R2': float,
            'MAPE': float,
            'Median_AE': float,
            'model_name': str (if provided)
        }
        
    Raises:
        ValueError: If arrays have different lengths or contain NaN/inf
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred have different lengths: {len(y_true)} vs {len(y_pred)}")
    
    if np.any(np.isnan(y_true)) or np.any(np.isinf(y_true)):
        raise ValueError("y_true contains NaN or inf values")
    
    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
        raise ValueError("y_pred contains NaN or inf values")
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    median_ae = median_absolute_error(y_true, y_pred)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape,
        'Median_AE': median_ae
    }
    
    if model_name:
        metrics['model_name'] = model_name
    
    return metrics


def evaluate_models(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all trained models on test set and compute metrics.
    
    Args:
        trained_models: Dictionary of trained model instances
        X_test: Test features
        y_test: Test target values
        
    Returns:
        Dictionary with metrics for each model:
        {
            'model_name': {metrics_dict}
        }
    """
    evaluation_results = {}
    
    for model_name, model in trained_models.items():
        print(f"Evaluating {model_name}...", end=' ', flush=True)
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test.values, y_pred, model_name)
        evaluation_results[model_name] = metrics
        print("Done")
    
    return evaluation_results


def create_metrics_dataframe(
    evaluation_results: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Convert evaluation results to a formatted DataFrame for easy comparison.
    
    Args:
        evaluation_results: Dictionary of metrics for each model
        
    Returns:
        DataFrame with models as rows and metrics as columns
    """
    df = pd.DataFrame(evaluation_results).T
    
    metric_order = ['R2', 'MAE', 'MSE', 'RMSE', 'MAPE', 'Median_AE']
    existing_cols = [col for col in metric_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in metric_order]
    
    df = df[existing_cols + other_cols]
    
    return df


def generate_metrics_report(
    evaluation_results: Dict[str, Dict[str, float]],
    cv_results: Dict[str, Dict[str, float]] = None,
    output_dir: str = None
) -> pd.DataFrame:
    """
    Generate comprehensive metrics report and optionally save to CSV.
    
    Args:
        evaluation_results: Dictionary of test set metrics
        cv_results: Optional dictionary of cross-validation results
        output_dir: Directory to save report CSV (default: data/processed/reports)
        
    Returns:
        DataFrame with all metrics and statistics
    """
    metrics_df = create_metrics_dataframe(evaluation_results)
    
    if cv_results:
        cv_r2_means = {model: results['r2_mean'] for model, results in cv_results.items()}
        cv_r2_stds = {model: results['r2_std'] for model, results in cv_results.items()}
        
        metrics_df['CV_R2_Mean'] = pd.Series(cv_r2_means)
        metrics_df['CV_R2_Std'] = pd.Series(cv_r2_stds)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'metrics_report.csv')
        metrics_df.to_csv(filepath)
        print(f"Metrics report saved to {filepath}")
    
    return metrics_df


def print_metrics_summary(
    metrics_df: pd.DataFrame,
    cv_results: Dict[str, Dict[str, float]] = None
) -> None:
    """
    Print formatted summary of metrics to console.
    
    Args:
        metrics_df: DataFrame with model metrics
        cv_results: Optional cross-validation results for additional context
    """
    print("\n" + "="*100)
    print("TEST SET METRICS SUMMARY")
    print("="*100)
    print(metrics_df.to_string())
    print("="*100)
    
    if cv_results:
        print("\nCROSS-VALIDATION SUMMARY")
        print("-"*100)
        for model_name, cv_info in cv_results.items():
            print(f"{model_name:20s} - CV R² Mean: {cv_info['r2_mean']:.4f} (+/- {cv_info['r2_std']:.4f})")
        print("-"*100)
    
    best_model_r2 = metrics_df['R2'].idxmax()
    best_model_mae = metrics_df['MAE'].idxmin()
    
    print(f"\nBest model by R² Score:        {best_model_r2} ({metrics_df.loc[best_model_r2, 'R2']:.4f})")
    print(f"Best model by MAE (lower=better): {best_model_mae} ({metrics_df.loc[best_model_mae, 'MAE']:.4f})")


def get_model_comparison(
    evaluation_results: Dict[str, Dict[str, float]],
    metric: str = 'R2'
) -> pd.Series:
    """
    Get models ranked by specific metric.
    
    Args:
        evaluation_results: Dictionary of metrics for each model
        metric: Metric to rank by (default 'R2')
        
    Returns:
        Sorted Series with model names and metric values
    """
    metrics_df = create_metrics_dataframe(evaluation_results)
    
    if metric not in metrics_df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available: {list(metrics_df.columns)}")
    
    lower_is_better = metric in ['MAE', 'MSE', 'RMSE', 'MAPE', 'Median_AE']
    
    if lower_is_better:
        return metrics_df[metric].sort_values(ascending=True)
    else:
        return metrics_df[metric].sort_values(ascending=False)
