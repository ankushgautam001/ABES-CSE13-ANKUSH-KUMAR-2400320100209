import unittest

import numpy as np

from src import config
from src.data_gen import generate
from src.evaluate import best_f1_threshold, cost_analysis, evaluate, precision_at_k
from src.models import ECDFCalibrator, MahalanobisDetector
from src.preprocess import Preprocessor, engineer, time_split


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = generate(n=6000, fraud_rate=0.02, seed=1)

    def test_schema_and_rate(self):
        self.assertEqual(list(self.df.columns[:1]), ["Time"])
        self.assertIn("Amount", self.df.columns)
        self.assertEqual(int(self.df.Class.sum()), 120)
        self.assertFalse(self.df.isna().any().any())

    def test_split_is_chronological_and_complete(self):
        tr, va, te = time_split(self.df)
        self.assertEqual(len(tr) + len(va) + len(te), len(self.df))
        self.assertLessEqual(tr.Time.max(), va.Time.min())
        self.assertLessEqual(va.Time.max(), te.Time.min())

    def test_features(self):
        f = engineer(self.df)
        self.assertEqual(f.shape[1], 31)
        self.assertTrue(np.isfinite(f.values).all())

    def test_scaler_fit_on_normal_only(self):
        tr, _, _ = time_split(self.df)
        p = Preprocessor().fit(tr[tr.Class == 0])
        self.assertEqual(p.transform(self.df).shape, (len(self.df), 31))


class TestModelsAndMetrics(unittest.TestCase):
    def test_detector_ranks_outliers_higher(self):
        rng = np.random.default_rng(0)
        normal, out = rng.normal(size=(2000, 5)), rng.normal(6, 1, size=(20, 5))
        det = MahalanobisDetector().fit(normal)
        self.assertGreater(det.score(out).min(), np.percentile(det.score(normal), 99))

    def test_ecdf_bounds(self):
        c = ECDFCalibrator().fit(np.arange(100))
        v = c.transform(np.array([-5, 50, 500]))
        self.assertTrue((v >= 0).all() and (v <= 1).all())
        self.assertLess(v[0], v[1]); self.assertLess(v[1], v[2])

    def test_metrics_perfect_scorer(self):
        y = np.array([0] * 95 + [1] * 5); s = y.astype(float)
        thr = best_f1_threshold(y, s)
        m = evaluate(y, s, thr, amount=np.full(100, 100.0))
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (5, 0, 0))
        self.assertAlmostEqual(m["pr_auc"], 1.0)
        self.assertEqual(precision_at_k(y, s, 0.05), 1.0)

    def test_cost_logic(self):
        y = np.array([1, 1, 0, 0]); s = np.array([.9, .1, .8, .1]); amt = np.array([100., 50., 10., 10.])
        c = cost_analysis(y, s, amt, thr=.5)  # flags tx0 and tx2
        self.assertEqual(c["baseline_loss"], 150.0)
        self.assertEqual(c["fraud_missed_amount"], 50.0)
        self.assertEqual(c["review_cost"], 2 * config.REVIEW_COST)


if __name__ == "__main__":
    unittest.main()
