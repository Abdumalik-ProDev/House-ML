# House-ML

End-to-end machine learning pipeline for predicting US housing prices using XGBoost. Features automated feature engineering, hyperparameter tuning with Optuna, experiment tracking with MLflow, a FastAPI inference API, and a Streamlit dashboard for interactive exploration.

## Project Structure

```
House-ML/
├── src/
│   ├── feature_pipeline/    # Data loading, preprocessing, feature engineering
│   ├── training_pipeline/   # Model training, evaluation, hyperparameter tuning
│   ├── inference_pipeline/  # Inference with preprocessing & feature alignment
│   ├── api/                 # FastAPI REST API for predictions
│   ├── batch/               # Monthly batch prediction runner
│   └── utils.py             # Shared utilities
├── data/
│   ├── raw/                 # Raw dataset (HouseTS.csv, usmetros.csv, splits)
│   └── processed/           # Cleaned & feature-engineered CSVs
├── models/                  # Serialized models & encoders (.pkl)
├── notebooks/               # Jupyter notebooks for EDA & development
├── tests/                   # Pytest suite (unit + integration)
├── app.py                   # Streamlit dashboard
├── Dockerfile               # FastAPI container
├── Dockerfile.streamlit     # Streamlit container
└── pyproject.toml           # Project metadata & dependencies
```

## Pipeline Overview

```
Raw Data → Load & Split → Preprocess → Feature Engineering → Train/Tune → Evaluate → Deploy API
                                                                               ↓
                                                                         Streamlit Dashboard
```

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)

### Setup

```bash
# Clone and enter the repo
git clone <repo-url> && cd House-ML

# Create virtual env and sync dependencies
uv sync

# Activate the environment
source .venv/bin/activate
```

### Run the Pipeline

```bash
# 1. Load and split raw data
python -c "from src.feature_pipeline.load import load_and_split_data; load_and_split_data()"

# 2. Preprocess (clean, normalize, deduplicate)
python src/feature_pipeline/preprocess.py

# 3. Feature engineering (encodings, date features)
python src/feature_pipeline/feature_engineering.py

# 4. Train baseline XGBoost
python src/training_pipeline/train.py

# 5. Hyperparameter tuning with Optuna + MLflow
python src/training_pipeline/tune.py
```

### Run Inference

```bash
# CLI inference on raw CSV
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv
```

### Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

```bash
# Test the API
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '[{"median_list_price": 500000, "zipcode": 90210, "city_full": "los angeles-long beach-glendale", "date": "2023-06-15", ...}]'
```

### Launch Dashboard

```bash
streamlit run app.py
```

### Monthly Batch Predictions

```bash
python -c "from src.batch.run_monthly import run_monthly_predictions; run_monthly_predictions()"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Root health check |
| `GET` | `/health` | Model & feature schema status |
| `POST` | `/predict` | Run predictions on batch of records |
| `POST` | `/run_batch` | Trigger monthly batch prediction |
| `GET` | `/latest_predictions` | Latest batch predictions preview |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_features.py -v
pytest tests/test_training.py -v
pytest tests/test_inference.py -v

# Run data quality checks
python tests/data_quality.py
```

## Docker

```bash
# Build and run FastAPI
docker build -t house-ml-api -f Dockerfile .
docker run -p 8000:8000 house-ml-api

# Build and run Streamlit (point API_URL to your API)
docker build -t house-ml-ui -f Dockerfile.streamlit .
docker run -p 8501:8501 -e API_URL=http://localhost:8000/predict house-ml-ui
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.14 |
| Package manager | uv |
| ML model | XGBoost |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow |
| Feature encoding | category-encoders |
| API framework | FastAPI |
| Dashboard | Streamlit |
| Testing | pytest |
| Data quality | Great Expectations |
| Containerization | Docker |

## Model Performance

Tuned XGBoost on eval split (2020–2021):

| Metric | Value |
|--------|-------|
| R² | 0.96 |
| MAE | ~$32,000 |
| RMSE | ~$71,000 |
