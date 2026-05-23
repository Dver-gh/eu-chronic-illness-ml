import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_balance(csv_path: str, column: str = 'quantile', out_dir: str = 'data/processed/plots'):
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        print(f'Column "{column}" not found in {csv_path}. Available columns: {list(df.columns)}')
        return

    counts = df[column].fillna('NaN').value_counts()
    os.makedirs(out_dir, exist_ok=True)
    counts.to_csv(os.path.join(out_dir, f'class_balance_{column}.csv'))

    plt.figure(figsize=(8, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values)
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Class distribution: {column}')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'class_balance_{column}.png'))
    plt.close()
    print(f'Wrote class balance plot and counts to: {out_dir}')
