"""
Check class distribution and balance in the dataset.
Generates CSV report and visualization.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def find_target_column(df: pd.DataFrame) -> str:
    """
    Automatically detect target/class column.
    For classification: looks for categorical columns.
    For regression: uses the 'value' column (main outcome variable).
    
    Args:
        df: Input DataFrame
        
    Returns:
        Name of detected target column
        
    Raises:
        ValueError: If no suitable target column found
    """
    if 'value' in df.columns:
        return 'value'
    
    target_names = ['target', 'class', 'label', 'quantile', 'outcome', 'result', 'bought']
    
    for name in target_names:
        if name in df.columns:
            return name
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        return categorical_cols[0]
    
    raise ValueError("Could not find target column. Please specify column name explicitly.")


def check_balance(csv_path: str, column: str = None, out_dir: str = 'data/processed/plots'):
    """
    Analyze distribution of target variable and create summary report.
    For continuous targets: generates histogram. For categorical: bar chart.
    
    Args:
        csv_path: Path to processed CSV file
        column: Target column name (auto-detected if None)
        out_dir: Output directory for plots and reports
    """
    df = pd.read_csv(csv_path)
    
    if column is None:
        try:
            column = find_target_column(df)
            print(f"Auto-detected target column: '{column}'")
        except ValueError as e:
            print(f"Error: {e}")
            print(f"Available columns: {list(df.columns)}")
            return
    
    if column not in df.columns:
        print(f'Column "{column}" not found in {csv_path}')
        print(f'Available columns: {list(df.columns)}')
        return

    os.makedirs(out_dir, exist_ok=True)
    
    data = df[column].dropna()
    
    if df[column].dtype in ['object', 'category'] or df[column].nunique() < 50:
        counts = df[column].fillna('NaN').value_counts()
        
        csv_path_out = os.path.join(out_dir, f'class_balance_{column}.csv')
        counts.to_csv(csv_path_out)
        print(f'Saved class balance CSV to: {csv_path_out}')

        plt.figure(figsize=(10, 6))
        sns.barplot(x=counts.index.astype(str), y=counts.values, palette='viridis')
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title(f'Class Distribution: {column}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'class_balance_{column}.png'), dpi=300)
        plt.close()
        
        print(f'Class balance report:')
        print(counts)
    else:
        stats = data.describe()
        
        csv_path_out = os.path.join(out_dir, f'target_distribution_{column}.csv')
        stats.to_csv(csv_path_out)
        print(f'Saved target distribution stats to: {csv_path_out}')
        
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=50, kde=True, color='steelblue')
        plt.xlabel(column, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Target Distribution: {column}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'target_distribution_{column}.png'), dpi=300)
        plt.close()
        
        print(f'Target distribution statistics:')
        print(stats)
    
    print(f'\nWrote target analysis to: {out_dir}')
