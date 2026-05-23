import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(csv_path: str, out_dir: str = 'data/processed/plots'):
    df = pd.read_csv(csv_path)
    if 'year' not in df.columns or 'value' not in df.columns:
        print('Required columns (year,value) not found for correlation heatmap.')
        return

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')

    if 'geo' in df.columns:
        pivot = df.pivot_table(index='geo', columns='year', values='value', aggfunc='mean')
    else:
        pivot = df.pivot_table(index=df.index, columns='year', values='value', aggfunc='mean')

    corr = pivot.corr()
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title('Correlation heatmap (years)')
    plt.tight_layout()
    out_path = os.path.join(out_dir, 'correlation_heatmap.png')
    plt.savefig(out_path)
    plt.close()
    print(f'Wrote correlation heatmap to: {out_path}')
