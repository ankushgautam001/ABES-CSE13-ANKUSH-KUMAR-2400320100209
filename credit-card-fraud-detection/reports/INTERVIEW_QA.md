# Interview prep and resume lines

## Resume bullets
- Built an end-to-end credit card fraud detection system using anomaly detection (Mahalanobis, Isolation Forest, LOF, One-Class SVM, Autoencoder) on 120k transactions at 0.5% fraud, with a chronological train/validation/test protocol.
- Reached ROC-AUC 0.92 and PR-AUC 0.64 (>100x random) with an unsupervised detector, within 0.06 PR-AUC of a supervised Random Forest, cutting simulated fraud loss ~65% under a cost model (missed fraud amount + review cost).
- Delivered a reusable pipeline: feature engineering, leakage-safe scaling, threshold selection, unit tests, batch scoring CLI with risk score and per-transaction reasons for analyst triage.

## Questions you will be asked
**Why not just use a supervised classifier?** Labels are scarce, delayed and biased toward past fraud. Anomaly detection learns normal behaviour and can flag new tactics. I included a Random Forest as a reference to show the gap.

**Why not accuracy?** Predicting "all legit" gives 99.5% accuracy. I use PR-AUC, precision, recall, F1, precision@1% and a cost model.

**How did you avoid leakage?** Time-ordered split, scaler and detectors fit on normal training rows only, thresholds chosen on validation, test used once.

**How do you pick the threshold?** Max F1 on validation here; in production it follows analyst capacity and the cost of a miss versus a review (alert budget curve).

**How does Isolation Forest work?** Random axis splits; anomalies need fewer splits to isolate, so shorter average path length means higher score.

**LOF vs Isolation Forest?** LOF compares local density to neighbours (good for clusters of different density) but is O(n^2) and needs subsampling; IF is fast and scales.

**What does the autoencoder do?** Compresses to a 6-unit bottleneck and reconstructs; trained on normal data, so unfamiliar patterns reconstruct badly. Score is reconstruction MSE.

**Why did Mahalanobis win, and is that trustworthy?** My data is synthetic and Gaussian, which suits it. I say so openly and rerun on the real dataset.

**Why did the ensemble not help?** Equal weights let weak models dilute strong ones; I would weight or stack using validation data.

**What about concept drift?** Monitor score distribution and alert rate, retrain on a rolling window, use analyst feedback.

**How would you deploy it?** Persist the pipeline, wrap in a low-latency scoring API, log scores and decisions, shadow-mode first, then gradual rollout.

**What would you improve?** Real data benchmark, hyper-parameter search, cross-validation by time blocks, weighted ensemble, SHAP explanations, cost-sensitive thresholds per merchant segment.
