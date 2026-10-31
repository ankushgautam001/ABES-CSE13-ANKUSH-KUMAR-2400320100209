# Credit Card Fraud Detection using Anomaly Detection

An end-to-end, production-style pipeline that flags fraudulent card transactions by learning what
*normal* behaviour looks like and scoring how far each new transaction deviates from it.

**Why anomaly detection?** Fraud is ~0.1-0.5% of transactions, labels arrive late (chargebacks take weeks),
and fraudsters constantly change tactics. Detectors trained only on normal data can catch new patterns that a
supervised classifier has never seen.

## Pipeline

```
raw transactions -> feature engineering -> RobustScaler (fit on normal train only)
   -> 5 detectors trained on NORMAL data only -> validation: pick thresholds
   -> test: metrics + business cost -> saved model -> batch scoring with reasons
```

| Detector | Idea |
|---|---|
| Mahalanobis (Gaussian) | statistical distance from the normal centroid |
| Isolation Forest | anomalies are isolated in few random splits |
| Local Outlier Factor | lower local density than neighbours |
| One-Class SVM | boundary around normal data in kernel space |
| Autoencoder | high reconstruction error = unfamiliar pattern |
| Ensemble | average of ECDF-calibrated percentile scores |
| *Random Forest (reference)* | supervised upper-bound for context only |

### Design decisions that matter
- **Chronological split** (70/15/15): never train on the future.
- **Scaler and detectors fit on normal training rows only**; fraud labels are used only to choose thresholds (validation) and to report results (test).
- **Metrics for imbalance**: PR-AUC, precision/recall/F1, precision@1%, not accuracy.
- **Business cost**: a missed fraud costs its amount, every alert costs a review fee (`config.REVIEW_COST`).
- **Triage output**: risk score 0-100 plus the top features that made a transaction look unusual.

## Quick start
```bash
pip install -r requirements.txt
python -m src.train                  # trains, evaluates, writes reports/ and models/
python -m unittest -q                # unit tests
python -m src.predict --input data/sample_new_transactions.csv
```

## Frontend + backend (web app)
```bash
python -m backend.server          # then open http://localhost:8000
python -m backend.server 9000     # custom port
```
- **Backend** (`backend/server.py`, Python standard library only): `GET /api/health`, `/api/metrics`, `/api/sample`,
  `/api/sample_fraud`, `/api/sample_batch`; `POST /api/score` (JSON object or list), `POST /api/score_csv` (raw CSV body).
  Validates input, returns risk score, flag and top reasons.
- **Frontend** (`frontend/`, plain HTML/CSS/JS, no build step, works offline): model comparison dashboard with
  figures, single-transaction scoring (load a random or known-fraud sample), and CSV / demo batch scoring with a
  ranked alert table. Risk bands: *Flagged* (above model threshold), *Elevated* (risk >= 95), *Normal*.

Example API call:
```bash
curl -X POST localhost:8000/api/score_csv --data-binary @data/sample_new_transactions.csv
```

## Using the real dataset
Download the public *Credit Card Fraud Detection* dataset (ULB, on Kaggle), save it as
`data/creditcard.csv`, delete `models/`, and rerun `python -m src.train`. The schema
(`Time, V1..V28, Amount, Class`) is identical, so no code changes are needed.
**Do this before your interview and quote the real-data numbers.**

## Repository layout
```
src/config.py      settings        src/data_gen.py   synthetic data + loader
src/preprocess.py  features/split  src/models.py     detectors + ensemble
src/evaluate.py    metrics/cost    src/train.py      full pipeline + plots
src/predict.py     batch scoring   tests/            unit tests
backend/server.py  REST API        frontend/         web dashboard (index.html, app.js, style.css)
reports/           metrics, figures, PROJECT_REPORT.md, INTERVIEW_QA.md
```

## Results (synthetic data, 120k transactions, 0.5% fraud, 99 frauds in test)
See `reports/PROJECT_REPORT.md`. Headline: the best unsupervised detector reaches ROC-AUC 0.92 / PR-AUC 0.64
and cuts simulated fraud loss by about 65%, approaching the supervised reference (PR-AUC 0.70).

## Limitations / next steps
Concept drift monitoring, SHAP-style explanations, streaming scoring API, feedback loop from analyst decisions,
semi-supervised fine-tuning once labels accumulate.
