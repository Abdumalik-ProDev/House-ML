"""
Evaluate a saved XGBoost model on the eval split.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils import maybe_sample as _maybe_sample

DEFAULT_EVAL = Path("data/processed/feature_engineered_eval.csv")
DEFAULT_MODEL = Path("models/xgb_best_model.pkl")


def evaluate_model(
    model_path: Path | str = DEFAULT_MODEL,
    eval_path: Path | str = DEFAULT_EVAL,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, float]:
    eval_df = pd.read_csv(eval_path)
    eval_df = _maybe_sample(eval_df, sample_frac, random_state)

    target = "price"
    X_eval, y_eval = eval_df.drop(columns=[target]), eval_df[target]

    model = load(model_path)
    y_pred = model.predict(X_eval)

    mae = float(mean_absolute_error(y_eval, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    r2 = float(r2_score(y_eval, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    print("📊 Evaluation:")
    print(f"   MAE={mae:.2f}  RMSE={rmse:.2f}  R²={r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()
