Project: EU chronic-illness ML study

Status
- Preprocessing: implemented (parses provided ESTAT TSV into long CSV).
- Basic EDA: class balance plot, correlation heatmap, feature histograms (generated to data/processed/plots).
- Entry point: `main.py` runs the pipeline (preprocess → balance → correlation → histograms).

How to run
1. Create a Python environment and install requirements:

```bash
pip install -r requirements.txt
```

2. Run the pipeline:

```bash
python main.py
```

Notes for git
- The `data/` folder is intentionally excluded from git via `.gitignore`.
- Code is modular under `src/`; modules are safe to import and do not execute on import. The single entrypoint is `main.py`.

Next steps
- Implement feature-engineering (imputation, `OneHotEncoder`/`LabelEncoder`, scaling).
- Implement `train.py` to run baseline models and compare metrics/time.
