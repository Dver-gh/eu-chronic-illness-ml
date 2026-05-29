"""
Generate histograms for numerical features.
Optionally group by categorical features.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(csv_path: str, groupby: str = None, out_dir: str = 'data/processed/plots', bins: int = 50):
    """
    Generate histograms for numerical features, optionally grouped by categorical column.
    
    Args:
        csv_path: Path to processed CSV file
        groupby: Optional categorical column to group by
        out_dir: Output directory for plots
        bins: Number of histogram bins
    """
    df = pd.read_csv(csv_path)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print('No numerical columns found in the dataset.')
        return
    
    os.makedirs(out_dir, exist_ok=True)
    
    for col in numeric_cols:
        valid_data = df[col].dropna()
        
        if valid_data.empty:
            continue
        
        plt.figure(figsize=(10, 6))
        sns.histplot(valid_data, bins=bins, kde=True, color='steelblue')
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Distribution: {col}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        safe_col = str(col).replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(out_dir, f'histogram_{safe_col}.png'), dpi=300)
        plt.close()
    
    if groupby:
        if groupby not in df.columns:
            print(f'Column "{groupby}" not found. Available columns: {list(df.columns)}')
            return
        
        groups = df[groupby].fillna('NaN').unique()
        
        for col in numeric_cols:
            for g in groups:
                subset = df[df[groupby].fillna('NaN') == g]
                valid_data = subset[col].dropna()
                
                if valid_data.empty:
                    continue
                
                plt.figure(figsize=(10, 6))
                sns.histplot(valid_data, bins=bins, kde=True, color='steelblue')
                plt.xlabel(col, fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.title(f'Distribution: {col} | {groupby}={g}', fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                safe_col = str(col).replace('/', '_').replace(' ', '_')
                safe_g = str(g).replace('/', '_').replace(' ', '_')
                plt.savefig(os.path.join(out_dir, f'histogram_{safe_col}_{groupby}_{safe_g}.png'), dpi=300)
                plt.close()
    
    print(f'Wrote {len(numeric_cols)} histogram(s) to: {out_dir}')
