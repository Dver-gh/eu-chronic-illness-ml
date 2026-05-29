import os
import re
import argparse
import pandas as pd
import numpy as np
from .preprocessing import (
    detect_missing_values,
    impute_missing_values,
    encode_with_one_hot,
    scale_features_standard,
    get_categorical_columns,
    get_numerical_columns
)


def _extract_number(s: str):
    """Extract numerical value from string, handling missing values."""
    if s is None:
        return np.nan
    s = str(s).strip()
    if s == '' or s.startswith(':'):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return np.nan
    return np.nan


def parse_tsv_to_long_format(input_path: str) -> pd.DataFrame:
    """
    Parse ESTAT TSV file and convert to long format.
    
    Args:
        input_path: Path to input TSV file
        
    Returns:
        DataFrame in long format
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    if not lines:
        raise RuntimeError('Input file is empty')

    header_line = lines[0].replace('\\TIME_PERIOD', '\tTIME_PERIOD')
    parts = [p.strip() for p in header_line.split('\t') if p.strip() != '']
    first_part = parts[0]
    first_cols = [c.strip() for c in first_part.split(',')]
    year_cols = [c.strip() for c in parts[1:]]
    if year_cols and year_cols[0].upper().startswith('TIME'):
        year_cols = year_cols[1:]
    year_cols = [re.sub(r'[^0-9]', '', yc) for yc in year_cols]

    header = first_cols + year_cols

    rows = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split('\t')
        first = parts[0]
        first_fields = [c.strip() for c in first.split(',')]
        rest = [p.strip() for p in parts[1:]]
        row = first_fields + rest
        if len(row) < len(header):
            row += [''] * (len(header) - len(row))
        elif len(row) > len(header):
            row = row[:len(header)]
        rows.append(row)

    df = pd.DataFrame(rows, columns=header)

    for yc in year_cols:
        if yc in df.columns:
            df[yc] = df[yc].apply(_extract_number)

    id_vars = first_cols
    df_long = df.melt(id_vars=id_vars, value_vars=year_cols, var_name='year', value_name='value')
    
    return df_long


def preprocess(input_path: str, output_path: str, 
               handle_missing: bool = True,
               encode_categoricals: bool = True,
               scale_numericals: bool = True,
               verbose: bool = True):
    """
    Complete preprocessing pipeline: parse TSV, handle missing values, 
    encode categoricals, and scale numerical features.
    
    Args:
        input_path: Path to input TSV file
        output_path: Path to save processed CSV
        handle_missing: Whether to impute missing values
        encode_categoricals: Whether to encode categorical features
        scale_numericals: Whether to scale numerical features
        verbose: Whether to print progress messages
    """
    if verbose:
        print('Step 1/5 — Parsing TSV to long format')
    df = parse_tsv_to_long_format(input_path)
    
    if verbose:
        print('Step 2/5 — Detecting missing values')
    missing_stats = detect_missing_values(df)
    if verbose and missing_stats['total_missing'] > 0:
        print(f"  Found {missing_stats['total_missing']} missing values")
        for col, count in missing_stats['columns_with_missing'].items():
            pct = missing_stats['missing_percent'].get(col, 0)
            print(f"    - {col}: {count} ({pct:.1f}%)")
    
    if handle_missing and missing_stats['total_missing'] > 0:
        if verbose:
            print('Step 3/5 — Imputing missing values')
        df = impute_missing_values(df, strategy_numeric='mean', strategy_categorical='most_frequent')
    else:
        if verbose:
            print('Step 3/5 — Skipping missing value imputation')
    
    if encode_categoricals:
        if verbose:
            print('Step 4/5 — Encoding categorical features')
        cat_cols = get_categorical_columns(df)
        if cat_cols:
            df = encode_with_one_hot(df, columns=cat_cols, drop_first=True)
            if verbose:
                print(f"  Encoded {len(cat_cols)} categorical columns")
        else:
            if verbose:
                print('  No categorical columns found')
    else:
        if verbose:
            print('Step 4/5 — Skipping categorical encoding')
    
    if scale_numericals:
        if verbose:
            print('Step 5/5 — Scaling numerical features')
        num_cols = get_numerical_columns(df)
        if num_cols:
            df, scaler = scale_features_standard(df, columns=num_cols)
            if verbose:
                print(f"  Scaled {len(num_cols)} numerical columns")
        else:
            if verbose:
                print('  No numerical columns found')
    else:
        if verbose:
            print('Step 5/5 — Skipping numerical scaling')
    
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_path, index=False)
    if verbose:
        print(f'\nPreprocessing complete!')
        print(f'Output shape: {df.shape}')
        print(f'Wrote processed CSV to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Preprocess ESTAT TSV with full ML pipeline')
    parser.add_argument('--input', '-i', default=os.path.join('data', 'raw', 'estat_hlth_silc_11.tsv'),
                       help='Input TSV file path')
    parser.add_argument('--output', '-o', default=os.path.join('data', 'processed', 'estat_hlth_silc_11.csv'),
                       help='Output CSV file path')
    parser.add_argument('--no-missing', action='store_true', help='Skip missing value handling')
    parser.add_argument('--no-encoding', action='store_true', help='Skip categorical encoding')
    parser.add_argument('--no-scaling', action='store_true', help='Skip numerical scaling')
    
    args = parser.parse_args()
    
    preprocess(
        args.input, 
        args.output,
        handle_missing=not args.no_missing,
        encode_categoricals=not args.no_encoding,
        scale_numericals=not args.no_scaling,
        verbose=True
    )
