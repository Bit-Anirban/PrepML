# Cleaner

Cleaner is a Streamlit-based data cleaning and analysis workspace designed to help users inspect, clean, scale, and prepare tabular datasets for machine learning. It combines data diagnostics, interactive cleaning tools, feature scaling guidance, target dependency analysis, and exportable code generation.

## Key Features

- Interactive CSV upload or URL loading
- Dataset profiling with missing values, duplicates, data types, and column diagnostics
- Smart pruning recommendations for columns with high missingness
- Data cleaning tools for missing value handling, duplicate removal, column normalization, and string-to-numeric conversion
- Scaling recommendations and in-place scaling support using scikit-learn transformers
- Automatic target type detection and adaptive target dependency analysis
- ML quickstart code generation for classification and regression tasks
- Exportable audit log / generated code from the cleaning session

## Project Structure

- `app.py` / `app2.py` / `frontend2.py`: Streamlit frontends for the data cleaning app
- `engine/core2.py`: Core data cleaning engine, diagnostics, pruning, and target analysis
- `engine/scaler2.py`: Scaling recommendation and scaling application logic
- `engine/ml_templates.py`: Machine learning template generation and model param specifications
- `components/ui_analysis.py`: Streamlit UI components for analysis and modeling workflows
- `components/ui_cleaning.py`: Streamlit UI components for cleaning and scaling workflows
- `test/test_core.py`: Pytest coverage for engine behavior and audit log generation

## Installation

1. Create and activate a Python virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install project dependencies

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit app

```bash
streamlit run app.py
```

## Usage

1. Open the app in your browser after Streamlit starts.
2. Upload a CSV file or paste a direct CSV URL.
3. Explore the dataset in the dashboard tabs.
4. Use cleaning tools to drop columns, fill missing values, convert strings to numeric, and normalize data.
5. Review scaling recommendations and apply scaling where appropriate.
6. Inspect target dependency analysis to identify strong predictors.
7. Export generated code from the session audit log for reuse.

## Testing

Run tests with:

```bash
pytest
```

## Dependencies

The main dependencies are:

- `pandas`
- `numpy`
- `scipy`
- `streamlit`
- `scikit-learn`
- `pytest`

## Notes

- `xgboost` and `lightgbm` are referenced in model generation templates; install them if you plan to use the ML quickstart feature.
- The app currently uses a file-based Streamlit frontend and Python engine modules for core cleaning logic.
- The `requirements.txt` file includes version bounds for compatibility with Python 3.11+ and modern pandas/scikit-learn releases.
