"""End-to-end training + evaluation pipeline.

    python -m src.train

Steps: load -> time split -> fit preprocessing on normal train -> train detectors on
normal train -> pick thresholds on validation -> report on untouched test -> save artifacts.
"""
import json
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from . import config
from .data_gen import load_dataset
from .evaluate import best_f1_threshold, evaluate
from .models import ECDFCalibrator, RankEnsemble, all_detectors
from .preprocess import Preprocessor, engineer, time_split

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
C_FRAUD, C_OK = "#d1495b", "#3b6ea5"


def save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name)
    plt.close(fig)


def eda_plots(df):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    counts = df["Class"].value_counts().sort_index()
    ax[0].bar(["Legit", "Fraud"], counts.values, color=[C_OK, C_FRAUD])
    ax[0].set_yscale("log"); ax[0].set_title(f"Class imbalance (fraud = {100*df.Class.mean():.2f}%)")
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    for cls, col, lab in [(0, C_OK, "Legit"), (1, C_FRAUD, "Fraud")]:
        ax[1].hist(np.log1p(df.loc[df.Class == cls, "Amount"]), bins=50, density=True, alpha=.6, color=col, label=lab)
    ax[1].set_title("Transaction amount (log scale)"); ax[1].set_xlabel("log(1 + Amount)"); ax[1].legend()
    hrs = (df["Time"] % 86400) // 3600
    for cls, col, lab in [(0, C_OK, "Legit"), (1, C_FRAUD, "Fraud")]:
        h = hrs[df.Class == cls].value_counts(normalize=True).sort_index()
        ax[2].plot(h.index, h.values, color=col, label=lab, marker="o", ms=3)
    ax[2].set_title("Share of transactions by hour of day"); ax[2].set_xlabel("hour"); ax[2].legend()
    save(fig, "01_eda_overview.png")

    feats = engineer(df)
    z = (feats - feats[df.Class == 0].mean()) / feats[df.Class == 0].std()
    gap = (z[df.Class == 1].mean()).sort_values(key=abs, ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(gap.index[::-1], gap.values[::-1], color=[C_FRAUD if v > 0 else C_OK for v in gap.values[::-1]])
    ax.set_title("Where fraud differs most from normal\n(mean fraud value in normal-std units)")
    save(fig, "02_feature_shift.png")


def main():
    t0 = time.time()
    df, source = load_dataset()
    print(f"Dataset: {source}  rows={len(df):,}  fraud={df.Class.sum():,} ({100*df.Class.mean():.3f}%)")
    eda_plots(df)

    tr, va, te = time_split(df)
    for n, d in [("train", tr), ("val", va), ("test", te)]:
        print(f"  {n:5s} rows={len(d):>7,}  fraud={int(d.Class.sum()):>4}")
    tr_norm = tr[tr.Class == 0]

    prep = Preprocessor().fit(tr_norm)
    Xn, Xv, Xt = prep.transform(tr_norm), prep.transform(va), prep.transform(te)
    yv, yt = va.Class.values, te.Class.values
    amt_t = te.Amount.values

    detectors, results, val_scores, test_scores, thresholds = [], {}, {}, {}, {}
    for det in all_detectors():
        t = time.time(); det.fit(Xn); detectors.append(det)
        val_scores[det.name], test_scores[det.name] = det.score(Xv), det.score(Xt)
        print(f"  trained {det.name:24s} {time.time()-t:5.1f}s")

    # Ensemble: calibrate each detector on validation-legit scores, average percentiles
    cals = [ECDFCalibrator().fit(val_scores[d.name][yv == 0]) for d in detectors]
    ens = RankEnsemble(detectors, cals)
    val_scores[ens.name], test_scores[ens.name] = ens.score(Xv), ens.score(Xt)

    for name in val_scores:
        thr = best_f1_threshold(yv, val_scores[name])  # chosen on validation only
        thresholds[name] = thr
        results[name] = evaluate(yt, test_scores[name], thr, amt_t)

    # Supervised reference (uses fraud labels; upper-bound context, not an anomaly detector)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample", min_samples_leaf=3,
                                n_jobs=-1, random_state=config.SEED).fit(prep.transform(tr), tr.Class)
    rf_v, rf_t = rf.predict_proba(Xv)[:, 1], rf.predict_proba(Xt)[:, 1]
    results["[Supervised ref] Random Forest"] = evaluate(yt, rf_t, best_f1_threshold(yv, rf_v), amt_t)
    test_scores["[Supervised ref] Random Forest"] = rf_t

    table = pd.DataFrame(results).T
    cols = ["roc_auc", "pr_auc", "precision", "recall", "f1", "precision_at_1pct", "alert_rate",
            "tp", "fp", "fn", "net_saving", "saving_pct"]
    table = table[cols].sort_values("pr_auc", ascending=False)
    config.REPORT_DIR.mkdir(exist_ok=True)
    table.round(4).to_csv(config.REPORT_DIR / "model_comparison.csv")
    pd.set_option("display.width", 200)
    print("\nTEST-SET RESULTS (threshold picked on validation)\n", table.round(3).to_string())

    anomaly_only = [n for n in table.index if not n.startswith("[")]
    best = anomaly_only[0]
    print(f"\nBest anomaly detector by PR-AUC: {best}")

    # ---- plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for n in table.index:
        p, r, _ = precision_recall_curve(yt, test_scores[n]); ax[0].plot(r, p, label=f"{n} ({table.loc[n,'pr_auc']:.2f})",
                                                                         ls="--" if n.startswith("[") else "-")
        f, tp_, _ = roc_curve(yt, test_scores[n]); ax[1].plot(f, tp_, ls="--" if n.startswith("[") else "-")
    ax[0].axhline(yt.mean(), color="grey", lw=.8); ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall (legend: PR-AUC)"); ax[0].legend(fontsize=7)
    ax[1].plot([0, 1], [0, 1], color="grey", lw=.8); ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("Recall")
    ax[1].set_title("ROC")
    save(fig, "03_pr_roc_curves.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    t2 = table[["roc_auc", "pr_auc", "f1"]]
    t2.plot.barh(ax=ax, color=["#8aa6c1", C_FRAUD, "#e0a458"]); ax.invert_yaxis()
    ax.set_title("Model comparison on the test set"); ax.set_xlim(0, 1)
    save(fig, "04_model_comparison.png")

    s = test_scores[best]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(s[yt == 0], bins=60, alpha=.6, color=C_OK, density=True, label="Legit")
    ax[0].hist(s[yt == 1], bins=60, alpha=.6, color=C_FRAUD, density=True, label="Fraud")
    ax[0].axvline(thresholds[best], color="k", ls="--", label="threshold"); ax[0].legend()
    ax[0].set_title(f"{best}: anomaly score distribution")
    cm = confusion_matrix(yt, (s >= thresholds[best]).astype(int))
    ax[1].imshow(cm, cmap="Blues"); ax[1].grid(False)
    for (i, j), v in np.ndenumerate(cm):
        ax[1].text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax[1].set_xticks([0, 1], ["Pred legit", "Pred fraud"]); ax[1].set_yticks([0, 1], ["Actual legit", "Actual fraud"])
    ax[1].set_title("Confusion matrix (test)")
    save(fig, "05_best_model_diagnostics.png")

    # alert-budget curve: recall vs share of transactions reviewed
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for n in [*anomaly_only, "[Supervised ref] Random Forest"]:
        order = np.argsort(-test_scores[n]); cum = np.cumsum(yt[order]) / yt.sum()
        ax.plot(np.arange(1, len(yt) + 1) / len(yt) * 100, cum * 100, ls="--" if n.startswith("[") else "-", label=n)
    ax.set_xlim(0, 10); ax.set_xlabel("% of transactions sent to review"); ax.set_ylabel("% of fraud caught")
    ax.set_title("Alert budget: fraud caught vs review workload"); ax.legend(fontsize=7)
    save(fig, "06_alert_budget.png")

    # ---- artifacts
    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(dict(preprocessor=prep, detectors={d.name: d for d in detectors}, ensemble=ens,
                     thresholds=thresholds, best=best, data_source=source), config.MODEL_DIR / "fraud_detector.joblib")
    (config.REPORT_DIR / "metrics.json").write_text(json.dumps(
        dict(data_source=source, best_anomaly_model=best, thresholds=thresholds, results=results), indent=2))
    print(f"\nDone in {time.time()-t0:.0f}s. Artifacts in models/ and reports/.")


if __name__ == "__main__":
    main()
