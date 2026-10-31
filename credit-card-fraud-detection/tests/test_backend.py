import unittest

from backend.server import DEMO, score_records


class TestBackend(unittest.TestCase):
    def test_scores_demo_rows(self):
        out = score_records(DEMO.head(30).to_dict(orient="records"))
        self.assertEqual(len(out), 30)
        self.assertTrue(all(0 <= r["risk_score"] <= 100 for r in out))

    def test_rejects_missing_columns(self):
        with self.assertRaises(ValueError):
            score_records([{"Amount": 5}])

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            score_records([])


if __name__ == "__main__":
    unittest.main()
