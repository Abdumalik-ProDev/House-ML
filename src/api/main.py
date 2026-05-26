from fastapi import FastAPI
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from src.inference_pipeline.inference import predict

MODEL_PATH = Path("models/xgb_best_model.pkl")
TRAIN_FE_PATH = Path("data/processed/feature_engineered_train.csv")

if TRAIN_FE_PATH.exists():
    _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
    TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
else:
    TRAIN_FEATURE_COLUMNS = None

app = FastAPI(title="Housing Regression API")


@app.get("/")
def root():
    return {"message": "Housing Regression API is running"}


@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    if not MODEL_PATH.exists():
        status["status"] = "unhealthy"
        status["error"] = "Model not found"
    else:
        status["status"] = "healthy"
        if TRAIN_FEATURE_COLUMNS:
            status["n_features_expected"] = len(TRAIN_FEATURE_COLUMNS)
    return status


@app.post("/predict")
def predict_batch(data: List[dict]):
    if not MODEL_PATH.exists():
        return {"error": f"Model not found at {str(MODEL_PATH)}"}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    preds_df = predict(df, model_path=MODEL_PATH)

    resp = {"predictions": preds_df["predicted_price"].astype(float).tolist()}
    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp


from src.batch.run_monthly import run_monthly_predictions


@app.post("/run_batch")
def run_batch():
    preds = run_monthly_predictions()
    return {
        "status": "success",
        "rows_predicted": int(len(preds)),
        "output_dir": "data/predictions/"
    }


@app.get("/latest_predictions")
def latest_predictions(limit: int = 5):
    pred_dir = Path("data/predictions")
    files = sorted(pred_dir.glob("preds_*.csv"))
    if not files:
        return {"error": "No predictions found"}

    latest_file = files[-1]
    df = pd.read_csv(latest_file)
    return {
        "file": latest_file.name,
        "rows": int(len(df)),
        "preview": df.head(limit).to_dict(orient="records")
    }
