"""
Feature engineering: date parts, frequency encoding, target encoding, drop leakage.

- Reads cleaned train/eval CSVs
- Applies feature engineering
- Saves feature-engineered CSVs
- ALSO saves fitted encoders for inference
"""

from pathlib import Path
import pandas as pd
from category_encoders import TargetEncoder
from joblib import dump #joblib.dump saves encoders/mappings to disk (important for reusing at inference).

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- feature functions ----------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    # place after date for readability (optional)
    df.insert(1, "year", df.pop("year"))
    df.insert(2, "quarter", df.pop("quarter"))
    df.insert(3, "month", df.pop("month"))
    return df


#Creates a frequency encoding (how often a value appears).
#Fit only on train, then applied to eval.
def frequency_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str):
    freq_map = train[col].value_counts()
    train[f"{col}_freq"] = train[col].map(freq_map)
    eval[f"{col}_freq"] = eval[col].map(freq_map).fillna(0)
    return train, eval, freq_map


#Uses target encoding (replace category with average of target variable).
#Fitted only on train (prevents leakage).
def target_encode(train: pd.DataFrame, eval: pd.DataFrame, col: str, target: str):
    """
    Use TargetEncoder on `col`, consistently name as <col>_encoded.
    For city_full → city_full_encoded (keeps schema aligned with inference).
    """
    te = TargetEncoder(cols=[col])
    encoded_col = f"{col}_encoded" if col != "city_full" else "city_full_encoded"
    train[encoded_col] = te.fit_transform(train[col], train[target])
    eval[encoded_col] = te.transform(eval[col])
    return train, eval, te



def drop_unused_columns(df: pd.DataFrame, also: pd.DataFrame | None = None) -> pd.DataFrame:
    drop_cols = ["date", "city_full", "city", "zipcode", "median_sale_price", "lat", "lng"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    if also is not None:
        also = also.drop(columns=[c for c in drop_cols if c in also.columns], errors="ignore")
        return df, also
    return df


# ---------- pipeline ----------

#Handles full pipeline: 
#reads cleaned CSVs → applies feature engineering → saves engineered data + encoders.
def run_feature_engineering(
    in_train_path: Path | str | None = None,
    in_eval_path: Path | str | None = None,
    in_holdout_path: Path | str | None = None,
    output_dir: Path | str = PROCESSED_DIR,
):
    """
    Run feature engineering and write outputs + encoders to disk.
    Applies the same transformations to train, eval, and holdout.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defaults for inputs
    if in_train_path is None:
        in_train_path = PROCESSED_DIR / "cleaning_train.csv"
    if in_eval_path is None:
        in_eval_path = PROCESSED_DIR / "cleaning_eval.csv"
    if in_holdout_path is None:
        in_holdout_path = PROCESSED_DIR / "cleaning_holdout.csv"

    train_df = pd.read_csv(in_train_path)
    eval_df = pd.read_csv(in_eval_path)

    has_holdout = Path(in_holdout_path).exists() if isinstance(in_holdout_path, (str, Path)) else False
    if has_holdout:
        holdout_df = pd.read_csv(in_holdout_path)
    else:
        holdout_df = pd.DataFrame()
        print("Holdout file not found; skipping holdout processing.")

    print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
    print("Eval date range:", eval_df["date"].min(), "to", eval_df["date"].max())

    train_df = add_date_features(train_df)
    eval_df = add_date_features(eval_df)
    if has_holdout:
        holdout_df = add_date_features(holdout_df)

    freq_map = None
    if "zipcode" in train_df.columns:
        train_df, eval_df, freq_map = frequency_encode(train_df, eval_df, "zipcode")
        if has_holdout:
            holdout_df["zipcode_freq"] = holdout_df["zipcode"].map(freq_map).fillna(0)
        dump(freq_map, MODELS_DIR / "freq_encoder.pkl")

    target_encoder = None
    if "city_full" in train_df.columns:
        train_df, eval_df, target_encoder = target_encode(train_df, eval_df, "city_full", "price")
        if has_holdout:
            holdout_df["city_full_encoded"] = target_encoder.transform(holdout_df["city_full"])
        dump(target_encoder, MODELS_DIR / "target_encoder.pkl")

    train_df, eval_df = drop_unused_columns(train_df, eval_df)
    if has_holdout:
        holdout_df = drop_unused_columns(holdout_df)

    out_train_path = output_dir / "feature_engineered_train.csv"
    out_eval_path = output_dir / "feature_engineered_eval.csv"
    train_df.to_csv(out_train_path, index=False)
    eval_df.to_csv(out_eval_path, index=False)

    out_holdout_path = output_dir / "feature_engineered_holdout.csv"
    if has_holdout:
        holdout_df.to_csv(out_holdout_path, index=False)

    print("Feature engineering complete.")
    print("   Train shape:", train_df.shape)
    print("   Eval  shape:", eval_df.shape)
    if has_holdout:
        print("   Holdout shape:", holdout_df.shape)
    print("   Encoders saved to models/")

    return train_df, eval_df, holdout_df, freq_map, target_encoder


if __name__ == "__main__":
    run_feature_engineering()
