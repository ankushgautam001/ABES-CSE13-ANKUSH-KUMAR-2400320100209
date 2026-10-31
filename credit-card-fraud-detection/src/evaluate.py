"""Metrics, threshold selection and business-cost analysis."""
import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score)

from . import config


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[np.argmax(f1)])


def precision_at_k(y, s, frac):
    k = max(1, int(len(y) * frac))
    return float(np.asarray(y)[np.argsort(-s)[:k]].mean())


def cost_analysis(y, s, amount, thr):
    """Cost with the model vs. no model (every fraud goes through)."""
    flag = s >= thr
    y = np.asarray(y).astype(bool)
    missed = amount[y & ~flag].sum()
    review = flag.sum() * config.REVIEW_COST
    baseline = amount[y].sum()
    total = missed + review
    return dict(baseline_loss=float(baseline), loss_with_model=float(total),
                fraud_missed_amount=float(missed), review_cost=float(review),
                net_saving=float(baseline - total),
                saving_pct=float(100 * (baseline - total) / baseline) if baseline else 0.0)


def evaluate(y, s, thr, amount=None):
    y = np.asarray(y)
    pred = (s >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = dict(
        roc_auc=roc_auc_score(y, s), pr_auc=average_precision_score(y, s),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred), f1=f1_score(y, pred),
        precision_at_1pct=precision_at_k(y, s, 0.01),
        alert_rate=float(pred.mean()), tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        threshold=float(thr),
    )
    if amount is not None:
        out.update(cost_analysis(y, s, np.asarray(amount), thr))
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}
