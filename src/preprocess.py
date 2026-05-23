import os
import re
import argparse
import pandas as pd
import numpy as np


def _extract_number(s: str):
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


def preprocess(input_path: str, output_path: str):
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

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df_long.to_csv(output_path, index=False)
    print(f'Wrote processed CSV to: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Preprocess ESTAT TSV to long CSV')
    parser.add_argument('--input', '-i', default=os.path.join('data', 'raw', 'estat_hlth_silc_11.tsv'))
    parser.add_argument('--output', '-o', default=os.path.join('data', 'processed', 'estat_hlth_silc_11.csv'))
    args = parser.parse_args()
    preprocess(args.input, args.output)
