"""Synthetic credit card transaction generator.

Mirrors the schema of the public ULB/Kaggle dataset
(Time, V1..V28, Amount, Class) so the pipeline runs unchanged on the real file.

Realism choices:
  * V1..V28 are correlated (latent-factor model), like PCA-derived card features.
  * Legit activity follows a day/night cycle; fraud skews to the night.
  * Three fraud patterns with different difficulty:
      - card_testing     : many small amounts, moderate feature shift
      - account_takeover : large amounts, strong shift
      - subtle           : looks almost normal (the hard cases that cap recall)
"""
import numpy as np
import pandas as pd

from . import config

N_FEATS = 28
FEATURE_COLS = [f"V{i}" for i in range(1, N_FEATS + 1)]

# feature indices (0-based) each fraud pattern distorts, and the shift in std units
_PATTERNS = {
    "card_testing":     dict(share=0.35, idx=[0, 2, 3, 9, 10, 13], shift=1.6),
    "account_takeover": dict(share=0.35, idx=[2, 3, 6, 10, 11, 13, 15, 16], shift=2.4),
    "subtle":           dict(share=0.30, idx=[3, 10, 13, 16], shift=0.9),
}


def generate(n=config.N_SYNTH_ROWS, fraud_rate=config.SYNTH_FRAUD_RATE, seed=config.SEED):
    rng = np.random.default_rng(seed)
    n_fraud = int(round(n * fraud_rate))
    n_legit = n - n_fraud

    # correlated legit features from 8 latent factors + idiosyncratic noise
    loadings = rng.normal(0, 0.6, size=(8, N_FEATS))
    sd = rng.uniform(0.6, 1.8, size=N_FEATS)

    def legit_block(m):
        z = rng.normal(size=(m, 8))
        return (z @ loadings + rng.normal(0, 0.8, size=(m, N_FEATS))) * sd / 2.0

    X_legit = legit_block(n_legit)
    base_sd = X_legit.std(axis=0)

    # fraud: same base distribution + pattern-specific shifts
    X_fraud = legit_block(n_fraud)
    kinds = rng.choice(list(_PATTERNS), size=n_fraud, p=[p["share"] for p in _PATTERNS.values()])
    for k, spec in _PATTERNS.items():
        rows = np.where(kinds == k)[0]
        signs = rng.choice([-1, 1], size=len(spec["idx"]))
        for j, s in zip(spec["idx"], signs):
            mag = rng.normal(spec["shift"], 0.35, size=len(rows))
            X_fraud[rows, j] += s * mag * base_sd[j]

    # amounts
    amt_legit = np.clip(rng.lognormal(3.4, 1.15, n_legit), 0.5, 5000)
    amt_fraud = np.empty(n_fraud)
    for k, lo in [("card_testing", (1.2, 0.7)), ("account_takeover", (5.2, 0.9)), ("subtle", (3.8, 1.0))]:
        rows = kinds == k
        amt_fraud[rows] = np.clip(rng.lognormal(*lo, rows.sum()), 0.5, 8000)

    # time of day (seconds over two days): legit peaks daytime, fraud skews to night
    hour_p_legit = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 3, 2, 1], float)
    hour_p_fraud = np.array([6, 6, 6, 5, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6], float)

    def draw_time(m, p):
        day = rng.integers(0, 2, m)
        hour = rng.choice(24, size=m, p=p / p.sum())
        return day * 86400 + hour * 3600 + rng.uniform(0, 3600, m)

    t = np.concatenate([draw_time(n_legit, hour_p_legit), draw_time(n_fraud, hour_p_fraud)])

    df = pd.DataFrame(np.vstack([X_legit, X_fraud]), columns=FEATURE_COLS)
    df.insert(0, "Time", t)
    df["Amount"] = np.concatenate([amt_legit, amt_fraud]).round(2)
    df["Class"] = np.r_[np.zeros(n_legit, int), np.ones(n_fraud, int)]
    return df.sort_values("Time").reset_index(drop=True)


def load_dataset():
    """Real Kaggle file if present, otherwise (cached) synthetic data."""
    if config.REAL_DATA_PATH.exists():
        return pd.read_csv(config.REAL_DATA_PATH), "kaggle-creditcard"
    config.DATA_DIR.mkdir(exist_ok=True)
    if config.SYNTH_DATA_PATH.exists():
        return pd.read_csv(config.SYNTH_DATA_PATH), "synthetic"
    df = generate()
    df.to_csv(config.SYNTH_DATA_PATH, index=False)
    return df, "synthetic"
