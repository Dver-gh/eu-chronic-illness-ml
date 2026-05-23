import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histograms(csv_path: str, groupby: str = None, out_dir: str = 'data/processed/plots'):
    df = pd.read_csv(csv_path)
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    os.makedirs(out_dir, exist_ok=True)

    # global histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df['value'].dropna(), bins=50)
    plt.title('Histogram of value (all)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'histogram_value_all.png'))
    plt.close()

    if groupby and groupby in df.columns:
        groups = df[groupby].fillna('NaN').unique()
        for g in groups:
            subset = df[df[groupby].fillna('NaN') == g]
            if subset['value'].dropna().empty:
                continue
            plt.figure(figsize=(8, 5))
            sns.histplot(subset['value'].dropna(), bins=40)
            plt.title(f'Histogram of value — {groupby}={g}')
            plt.tight_layout()
            safe_g = str(g).replace('/', '_')
            plt.savefig(os.path.join(out_dir, f'hist_value_{groupby}_{safe_g}.png'))
            plt.close()

    print(f'Wrote histograms to: {out_dir}')
