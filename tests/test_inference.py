from pathlib import Path

import pandas as pd
import pytest

from src.inference_pipeline.inference import predict

ROOT = Path(__file__).resolve().parents[1]
CLEANING_EVAL_PATH = ROOT / "data/processed/cleaning_eval.csv"
MODEL_PATH = ROOT / "models/xgb_best_model.pkl"

_REQUIRE_DATA = pytest.mark.skipif(
    not CLEANING_EVAL_PATH.exists() or not MODEL_PATH.exists(),
    reason="cleaning_eval.csv or model not found. Run the full pipeline first.",
)


@pytest.fixture(scope="session")
def sample_df():
    df = pd.read_csv(CLEANING_EVAL_PATH).sample(5, random_state=42).reset_index(drop=True)
    return df


@_REQUIRE_DATA
def test_inference_runs_and_returns_predictions(sample_df):
    preds_df = predict(sample_df)
    assert not preds_df.empty
    assert "predicted_price" in preds_df.columns
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_price"])
