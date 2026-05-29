"""
Generate correlation heatmap for numerical features.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(csv_path: str, out_dir: str = 'data/processed/plots', figsize: tuple = (12, 10)):
    """
    Generate correlation heatmap for all numerical columns.
    
    Args:
        csv_path: Path to processed CSV file
        out_dir: Output directory for plot
        figsize: Figure size (width, height)
    """
    df = pd.read_csv(csv_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        print(f'Need at least 2 numerical columns for correlation. Found: {numeric_cols}')
        return
    
    df_numeric = df[numeric_cols].dropna()
    
    if df_numeric.empty:
        print('No valid numerical data found after removing NaN values.')
        return
    
    corr = df_numeric.corr()
    
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'correlation_heatmap.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    corr.to_csv(os.path.join(out_dir, 'correlation_matrix.csv'))
    
    print(f'Correlation matrix:')
    print(corr)
    print(f'\nWrote correlation heatmap and matrix to: {out_dir}')
