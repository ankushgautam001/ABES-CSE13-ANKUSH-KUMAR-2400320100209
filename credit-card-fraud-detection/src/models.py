"""Unsupervised / semi-supervised anomaly detectors with a common interface.

Every detector is trained on NORMAL transactions only and exposes
    score(X) -> higher means more anomalous.
"""
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import OneClassSVM

from . import config


def _cap(X, n, seed):
    if len(X) <= n:
        return X
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    return X[idx]


class IsolationForestDetector:
    name = "Isolation Forest"

    def fit(self, X):
        self.m = IsolationForest(n_estimators=300, max_samples=512, contamination="auto",
                                 random_state=config.SEED, n_jobs=-1).fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class LOFDetector:
    name = "Local Outlier Factor"

    def fit(self, X):
        self.m = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=-1).fit(
            _cap(X, config.LOF_TRAIN_CAP, config.SEED))
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class OneClassSVMDetector:
    name = "One-Class SVM"

    def fit(self, X):
        self.m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.01).fit(
            _cap(X, config.OCSVM_TRAIN_CAP, config.SEED))
        return self

    def score(self, X):
        return -self.m.decision_function(X)


class MahalanobisDetector:
    """Classical statistical baseline: distance from the normal-data centroid."""
    name = "Mahalanobis (Gaussian)"

    def fit(self, X):
        self.m = LedoitWolf().fit(X)
        return self

    def score(self, X):
        return self.m.mahalanobis(X)


class AutoencoderDetector:
    """Bottleneck autoencoder; anomaly score = reconstruction error."""
    name = "Autoencoder"

    def fit(self, X):
        self.m = MLPRegressor(hidden_layer_sizes=(24, 12, 6, 12, 24), activation="tanh",
                              solver="adam", learning_rate_init=1e-3, batch_size=512,
                              max_iter=60, early_stopping=True, n_iter_no_change=6,
                              random_state=config.SEED).fit(X, X)
        return self

    def score(self, X):
        return ((self.m.predict(X) - X) ** 2).mean(axis=1)


def all_detectors():
    return [MahalanobisDetector(), IsolationForestDetector(), LOFDetector(),
            OneClassSVMDetector(), AutoencoderDetector()]


class ECDFCalibrator:
    """Maps raw scores to 0..1 using the score distribution of known-normal data,
    so heterogeneous detectors can be averaged and reported as a risk score."""

    def fit(self, normal_scores):
        self.ref = np.sort(np.asarray(normal_scores))
        return self

    def transform(self, s):
        return np.searchsorted(self.ref, s, side="right") / len(self.ref)


class RankEnsemble:
    name = "Ensemble (rank avg)"

    def __init__(self, detectors, calibrators):
        self.detectors, self.calibrators = detectors, calibrators

    def score(self, X):
        return np.mean([c.transform(d.score(X)) for d, c in zip(self.detectors, self.calibrators)], axis=0)
