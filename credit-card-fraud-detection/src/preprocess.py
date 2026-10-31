"""Feature engineering, scaling and time-ordered splitting."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from . import config


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw transactions -> model features (V1..V28, log_amount, hour_sin, hour_cos)."""
    vcols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    out = df[vcols].copy()
    out["log_amount"] = np.log1p(df["Amount"].clip(lower=0))
    hour = (df["Time"] % 86400) / 3600.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out


def time_split(df: pd.DataFrame):
    """Chronological train / validation / test split."""
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    a, b = int(n * config.TRAIN_FRAC), int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    return df.iloc[:a], df.iloc[a:b], df.iloc[b:]


class Preprocessor:
    """Fits a RobustScaler on *normal* training transactions only."""

    def __init__(self):
        self.scaler = RobustScaler()
        self.columns_ = None

    def fit(self, df_train_normal: pd.DataFrame):
        feats = engineer(df_train_normal)
        self.columns_ = list(feats.columns)
        self.scaler.fit(feats.values)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        feats = engineer(df)[self.columns_]
        return self.scaler.transform(feats.values)
