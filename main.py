import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
from src import preprocess, train, metrics
from src import check_balance, plot_correlation, plot_histograms


def main():
    raw = os.path.join('data', 'raw', 'estat_hlth_silc_11.tsv')
    processed = os.path.join('data', 'processed', 'estat_hlth_silc_11.csv')

    print('1/5 – preprocessing')
    preprocess.preprocess(raw, processed)

    print('2/5 – checking class balance')
    os.makedirs(os.path.join('data', 'processed', 'plots'), exist_ok=True)
    check_balance.check_balance(processed, out_dir=os.path.join('data', 'processed', 'plots'))

    print('3/5 – correlation heatmap')
    plot_correlation.plot_heatmap(processed, out_dir=os.path.join('data', 'processed', 'plots'))

    print('4/5 – histograms')
    plot_histograms.plot_histograms(processed, out_dir=os.path.join('data', 'processed', 'plots'))

    print('5/5 – model training and evaluation')
    data = pd.read_csv(processed)
    X_train, X_test, y_train, y_test = train.split_data(data, target_column='value')
    
    print('\nPerforming 5-fold cross-validation...')
    cv_results = train.cross_validate_models(X_train, y_train, cv_folds=5)
    
    print('\nTraining models on full training set...')
    trained_models = train.train_all_models(X_train, y_train)
    
    print('\nSaving trained models...')
    train.save_all_models(trained_models, save_dir=os.path.join('data', 'models'))
    
    print('\nEvaluating models on test set...')
    evaluation_results = metrics.evaluate_models(trained_models, X_test, y_test)
    
    print('\nGenerating metrics report...')
    metrics_df = metrics.generate_metrics_report(
        evaluation_results,
        cv_results=cv_results,
        output_dir=os.path.join('data', 'processed', 'reports')
    )
    
    metrics.print_metrics_summary(metrics_df, cv_results=cv_results)


if __name__ == '__main__':
    main()
