# Project Report: Credit Card Fraud Detection using Anomaly Detection

## 1. Problem and objective
Detect fraudulent card transactions with extreme class imbalance and few trusted labels. Objective: rank
transactions by abnormality so analysts review a small share of traffic while catching most fraud loss.

## 2. Data
Synthetic dataset with the same schema as the public ULB/Kaggle dataset: `Time`, `V1..V28`, `Amount`, `Class`.
120,000 transactions over two days, 0.5% fraud (600). Legit features are correlated; fraud is generated in three
patterns of different difficulty (card testing, account takeover, subtle). Fraud skews to night hours.
The pipeline reads `data/creditcard.csv` automatically when present.

| Split | Rows | Fraud |
|---|---|---|
| Train (earliest 70%) | 84,000 | 466 (unused for training detectors) |
| Validation (next 15%) | 18,000 | 35 |
| Test (latest 15%) | 18,000 | 99 |

## 3. Features
`V1..V28`, `log(1+Amount)` (tames skew), cyclic hour-of-day (`sin`, `cos`). RobustScaler (median/IQR) is used
because it is not distorted by the extreme amounts typical of this domain.

## 4. Methods
Five detectors trained on normal training data only (see README), an ensemble that averages ECDF-calibrated
percentile scores, and a supervised Random Forest as a reference. Thresholds maximise F1 on validation; the test
set is touched once.

## 5. Results (test set)
| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 | Loss reduction |
|---|---|---|---|---|---|---|
| Random Forest (supervised ref) | 0.935 | 0.697 | 0.645 | 0.697 | 0.670 | 64% |
| Mahalanobis | 0.920 | 0.637 | 0.964 | 0.535 | 0.688 | 65% |
| Autoencoder | 0.903 | 0.510 | 0.635 | 0.404 | 0.494 | 64% |
| Local Outlier Factor | 0.907 | 0.499 | 0.679 | 0.384 | 0.490 | 64% |
| Ensemble | 0.904 | 0.404 | 0.521 | 0.374 | 0.435 | 62% |
| One-Class SVM | 0.880 | 0.367 | 0.920 | 0.232 | 0.371 | 59% |
| Isolation Forest | 0.823 | 0.041 | 0.182 | 0.020 | 0.036 | 2% |

Figures are in `reports/figures/`. Base fraud rate in test is 0.55%, so PR-AUC of 0.64 is more than 100x random.

## 6. Interpretation (read this before presenting)
- **Mahalanobis wins here because the synthetic data is Gaussian.** That is a property of the generator, not proof that it is the best method in general. Real card data is non-Gaussian and multimodal, where Isolation Forest, the autoencoder and ensembles usually do better. Rerun on the real dataset and report those numbers.
- **Isolation Forest is weak** because the injected fraud shifts *correlations among several features* rather than pushing a single feature to an extreme, which axis-aligned random splits isolate poorly.
- **The ensemble underperformed its best member** since it equally weights weaker detectors. A weighted ensemble tuned on validation, or a stacked model, is a clear next step.
- **Test set has only 99 frauds**, so metrics have noticeable variance; one run, one seed.
- The dollar savings are small in absolute terms because the synthetic amounts are small; the percentage is the meaningful figure.

## 7. Deployment sketch
`src.predict` scores batches and returns anomaly score, flag, 0-100 risk score and top reasons. In production:
real-time scoring service, thresholds set from the analyst capacity (alert budget curve, figure 06), weekly drift
checks on score distribution, and analyst feedback stored as labels to retrain or move to a hybrid model.

## 8. Limitations
Synthetic data; single split; no hyper-parameter search beyond defaults; no calibration of probabilities;
explanations are feature deviations, not causal.
