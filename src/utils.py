from __future__ import annotations
from typing import Optional
import pandas as pd


def maybe_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int = 42) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
