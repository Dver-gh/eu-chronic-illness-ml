import os
from src import preprocess
from src import check_balance, plot_correlation, plot_histograms


def main():
    raw = os.path.join('data', 'raw', 'estat_hlth_silc_11.tsv')
    processed = os.path.join('data', 'processed', 'estat_hlth_silc_11.csv')

    print('1/4 — preprocessing')
    preprocess.preprocess(raw, processed)

    print('2/4 — checking class balance')
    os.makedirs(os.path.join('data', 'processed', 'plots'), exist_ok=True)
    check_balance.check_balance(processed, column='quantile', out_dir=os.path.join('data', 'processed', 'plots'))

    print('3/4 — correlation heatmap')
    plot_correlation.plot_heatmap(processed, out_dir=os.path.join('data', 'processed', 'plots'))

    print('4/4 — histograms')
    plot_histograms.plot_histograms(processed, groupby='age', out_dir=os.path.join('data', 'processed', 'plots'))


if __name__ == '__main__':
    main()
