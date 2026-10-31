"""Fraud detection backend: JSON API + static frontend, standard library only.

    python -m backend.server            # http://localhost:8000
"""
import csv
import io
import json
import mimetypes
import random
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.predict import score_dataframe  # noqa: E402

FRONTEND = ROOT / "frontend"
REQUIRED = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
MAX_ROWS = 20000

BUNDLE = joblib.load(config.MODEL_DIR / "fraud_detector.joblib")
METRICS = json.loads((config.REPORT_DIR / "metrics.json").read_text())
DEMO = pd.read_csv(config.DATA_DIR / "demo_transactions.csv")


def score_records(records):
    if not records:
        raise ValueError("No transactions supplied")
    if len(records) > MAX_ROWS:
        raise ValueError(f"Max {MAX_ROWS} transactions per request")
    df = pd.DataFrame(records)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing[:6])}{'...' if len(missing) > 6 else ''}")
    df[REQUIRED] = df[REQUIRED].apply(pd.to_numeric, errors="raise")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Empty or non-numeric values in required columns")
    res = score_dataframe(df, BUNDLE)
    res["anomaly_score"] = res["anomaly_score"].round(4)
    keep = [c for c in ["Time", "Amount", "Class"] if c in res.columns] + \
           ["anomaly_score", "risk_score", "flagged", "top_reasons"]
    return res[keep].to_dict(orient="records")


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, rel):
        base = FRONTEND if not rel.startswith("figures/") else config.FIG_DIR
        rel = rel[len("figures/"):] if rel.startswith("figures/") else rel
        path = (base / (rel or "index.html")).resolve()
        if base.resolve() not in path.parents or not path.is_file():
            return self._json({"error": "not found"}, 404)
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/health":
            return self._json({"status": "ok", "best_model": BUNDLE["best"], "data_source": BUNDLE["data_source"]})
        if p == "/api/metrics":
            return self._json(METRICS)
        if p == "/api/sample":
            row = DEMO.sample(1, random_state=random.randint(0, 10**6)).iloc[0].to_dict()
            return self._json(row)
        if p == "/api/sample_fraud":
            row = DEMO[DEMO.Class == 1].sample(1, random_state=random.randint(0, 10**6)).iloc[0].to_dict()
            return self._json(row)
        if p == "/api/sample_batch":
            return self._json(DEMO.to_dict(orient="records"))
        self._static(p.lstrip("/"))

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 50_000_000:
                raise ValueError("Payload too large")
            raw = self.rfile.read(n)
            if self.path == "/api/score":
                body = json.loads(raw or b"{}")
                records = body if isinstance(body, list) else [body]
            elif self.path == "/api/score_csv":
                records = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
            else:
                return self._json({"error": "not found"}, 404)
            results = score_records(records)
            return self._json({"count": len(results), "flagged": sum(r["flagged"] for r in results),
                               "model": BUNDLE["best"], "results": results})
        except (ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError) as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:  # pragma: no cover
            return self._json({"error": f"server error: {type(e).__name__}"}, 500)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main(host="127.0.0.1", port=8000):
    print(f"Fraud detection API + dashboard on http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    main(port=port)
