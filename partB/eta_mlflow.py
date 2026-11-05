"""
eta_mlflow.py -- MLflow registry, promotion gate and shadow comparison (DA3408 Module 4, A5 Part B)

Run from the repo root (tracking store = SQLite, from MLFLOW_TRACKING_URI, default sqlite:///mlflow.db):
    python partB/eta_mlflow.py register                    # log v1 + v2, register eta-model, set aliases
    python partB/eta_mlflow.py gate [--min-rel-gain 0.6]   # promotion_gate() on fresh data
    python partB/eta_mlflow.py shadow                      # needs the two `mlflow models serve` below
        mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local
        mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local

Based on Lecture 10 "MLflow_Serving_Exercise" (mlflow_register.py, Task A gate, Task B shadow), changed
for a regression model: MAE instead of accuracy, rush-hour / rain slices, and a latency criterion.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # for eta_data.py in the repo root

import eta_data  # noqa: E402

NAME = "eta-model"
EXPERIMENT = "eta-serving"
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    return mlflow


try:
    from mlflow.pyfunc import PythonModel
except ImportError:                                 # decide() and its tests don't need mlflow
    PythonModel = object


class EtaPyfunc(PythonModel):
    """Wraps a joblib bundle; returns eta_minutes (1 decimal) and the bundle's model_version."""

    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.m, self.cols, self.ver = b["model"], b["feature_columns"], b["version"]

    def predict(self, context, model_input, params=None):
        X = pd.DataFrame(model_input)[self.cols]
        return pd.DataFrame({"eta_minutes": np.round(self.m.predict(X), 1), "model_version": self.ver})


# ---------------------------------------------------------------- B1: register
def register(data_dir=ROOT / "data"):
    import sklearn
    from mlflow import MlflowClient
    from mlflow.models import infer_signature
    from sklearn.metrics import mean_absolute_error

    mlflow = _mlflow()
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    fresh = eta_data.make_data(2000, seed=99)
    X = fresh[eta_data.FEATURES].astype("float64")   # float schema: JSON ints and floats both accepted
    pins = [f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}",
            f"numpy=={np.__version__}", f"joblib=={joblib.__version__}", f"mlflow=={mlflow.__version__}"]
    registered = {}
    for ver in ("v1", "v2"):
        path = str(data_dir / f"eta_{ver}.joblib")
        b = joblib.load(path)
        mae = mean_absolute_error(fresh.eta_minutes, b["model"].predict(fresh[b["feature_columns"]]))
        out_example = pd.DataFrame({"eta_minutes": [42.0], "model_version": [ver]})
        sig = infer_signature(X.head(5), out_example)
        est = b["model"][-1] if hasattr(b["model"], "steps") else b["model"]   # v1 is scaler + Ridge
        with mlflow.start_run(run_name=f"eta-{ver}"):
            mlflow.log_param("algorithm", type(est).__name__)
            mlflow.log_param("bundle_version", ver)
            mlflow.log_metric("fresh_mae", mae)
            info = mlflow.pyfunc.log_model(name="model", python_model=EtaPyfunc(), artifacts={"bundle": path},
                                           signature=sig, input_example=X.head(2), pip_requirements=pins,
                                           registered_model_name=NAME)
        registered[ver] = info.registered_model_version
        print(f"{ver}: fresh MAE {mae:.2f} min -> {NAME} version {info.registered_model_version}")
    client.set_registered_model_alias(NAME, "prod", registered["v1"])
    client.set_registered_model_alias(NAME, "candidate", registered["v2"])
    print(registry_table().to_string(index=False))
    return registered


def registry_table():
    from mlflow import MlflowClient
    _mlflow()
    client = MlflowClient()
    aliases = client.get_registered_model(NAME).aliases           # {alias: version}
    rows = []
    for v in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        bundle = client.get_run(v.run_id).data.params.get("bundle_version", "?")
        rows.append({"version": int(v.version), "bundle": bundle, "run_id": v.run_id[:8],
                     "alias": ", ".join(a for a, n in aliases.items() if str(n) == str(v.version)) or "-"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- B2: gate
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    """Pure promotion rule. prod/cand: dicts with mae, mae_rush, mae_rain, p95_ms.
    Slice regression is relative: cand slice MAE may be at most (1 + max_slice_regression) x prod's."""
    eps = 1e-9                                       # so a value exactly on a threshold isn't lost to float error
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"]
    criteria = {
        "mae_gain": rel_gain >= min_rel_gain - eps,
        "latency_p95": cand["p95_ms"] <= max_p95_ms + eps,
        "rush_slice": cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression) + eps,
        "rain_slice": cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression) + eps,
    }
    return {"criteria": criteria, "promote": all(criteria.values()),
            "rel_gain": round(rel_gain, 4),
            "thresholds": {"min_rel_gain": min_rel_gain, "max_p95_ms": max_p95_ms,
                           "max_slice_regression": max_slice_regression}}


def _evaluate(model, fresh, n_latency=200):
    pred = model.predict(fresh[eta_data.FEATURES].astype("float64"))["eta_minutes"].to_numpy()
    err = np.abs(pred - fresh.eta_minutes.to_numpy())
    rush = fresh.hour.isin(RUSH_HOURS).to_numpy()
    rain = (fresh.rain == 1).to_numpy()
    X = fresh[eta_data.FEATURES].astype("float64")
    for i in range(10):                              # warm-up, not timed
        model.predict(X.iloc[[i]])
    lat = []
    for i in range(n_latency):                       # single-row predictions, like an online request
        t = time.perf_counter()
        model.predict(X.iloc[[i]])
        lat.append((time.perf_counter() - t) * 1000)
    return {"mae": round(float(err.mean()), 3), "mae_rush": round(float(err[rush].mean()), 3),
            "mae_rain": round(float(err[rain].mean()), 3), "p95_ms": round(float(np.percentile(lat, 95)), 3)}


def promotion_gate(**thresholds):
    mlflow = _mlflow()
    fresh = eta_data.make_data(2000, seed=99)        # fresh data, never seen in training (seed 1)
    metrics = {}
    for alias in ("prod", "candidate"):
        m = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")
        metrics[alias] = _evaluate(m, fresh)
    result = decide(metrics["prod"], metrics["candidate"], **thresholds)
    return {"prod": metrics["prod"], "candidate": metrics["candidate"], **result}


# ---------------------------------------------------------------- B3: shadow
def shadow(prod_url="http://127.0.0.1:5101/invocations", cand_url="http://127.0.0.1:5102/invocations", n=200):
    import requests
    rows = pd.read_csv(ROOT / "data" / "eta_fresh.csv")[eta_data.FEATURES].head(n).to_dict(orient="records")
    s = requests.Session()
    out = {"prod": {"pred": [], "lat": [], "ver": set()}, "candidate": {"pred": [], "lat": [], "ver": set()}}
    for rec in rows:                                 # the same request goes to both, one row each
        for arm, url in (("prod", prod_url), ("candidate", cand_url)):
            t = time.perf_counter()
            r = s.post(url, json={"dataframe_records": [rec]}, timeout=10)
            out[arm]["lat"].append((time.perf_counter() - t) * 1000)
            r.raise_for_status()
            p = r.json()["predictions"][0]
            out[arm]["pred"].append(p["eta_minutes"])
            out[arm]["ver"].add(p["model_version"])
    a, b = np.array(out["prod"]["pred"]), np.array(out["candidate"]["pred"])
    diff = np.abs(a - b)
    return {"requests": n,
            "served_versions": {k: sorted(v["ver"]) for k, v in out.items()},
            "mean_abs_diff_min": round(float(diff.mean()), 2),
            "pct_diff_gt_10min": round(float((diff > 10).mean() * 100), 1),
            "max_abs_diff_min": round(float(diff.max()), 1),
            "p95_ms": {k: round(float(np.percentile(v["lat"], 95)), 2) for k, v in out.items()},
            "p50_ms": {k: round(float(np.percentile(v["lat"], 50)), 2) for k, v in out.items()}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register", "table", "gate", "shadow"])
    ap.add_argument("--min-rel-gain", type=float, default=0.03)
    ap.add_argument("--max-p95-ms", type=float, default=10.0)
    ap.add_argument("--max-slice-regression", type=float, default=0.05)
    a = ap.parse_args()
    if a.cmd == "register":
        register()
    elif a.cmd == "table":
        print(registry_table().to_string(index=False))
    elif a.cmd == "gate":
        print(json.dumps(promotion_gate(min_rel_gain=a.min_rel_gain, max_p95_ms=a.max_p95_ms,
                                        max_slice_regression=a.max_slice_regression), indent=2))
    else:
        print(json.dumps(shadow(), indent=2))
