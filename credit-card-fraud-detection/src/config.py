"""Central configuration for the fraud anomaly detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

# If you download the public Kaggle "Credit Card Fraud Detection" dataset
# (ULB, 284,807 rows) and save it here, the pipeline uses it automatically.
REAL_DATA_PATH = DATA_DIR / "creditcard.csv"
SYNTH_DATA_PATH = DATA_DIR / "transactions_synthetic.csv"

SEED = 42
N_SYNTH_ROWS = 120_000
SYNTH_FRAUD_RATE = 0.005

# Time-ordered split (no shuffling: we never train on the future)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # remaining 15% is the untouched test set

# Business cost model used for the savings analysis
REVIEW_COST = 5.0          # cost of an analyst reviewing one alert (any alert)
# A missed fraud costs the full transaction amount (chargeback).

# Sub-sampling caps for the O(n^2) detectors
LOF_TRAIN_CAP = 20_000
OCSVM_TRAIN_CAP = 8_000
