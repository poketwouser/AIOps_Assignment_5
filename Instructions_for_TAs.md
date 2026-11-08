# Instructions for TAs: DA3408 Assignment 5

Kiran Kumar P

How to check each rubric item quickly. All tables and written answers are in [`REPORT.pdf`](report/REPORT.pdf)
(3 pages). If you don't want to run anything, every check below has a raw output file or a screenshot in
[`screenshots/`](screenshots/) (one folder per rubric item; numbers like "03" below are the file prefixes).

## 1. Setup (about 3 minutes)

Run everything from the repository root. CPU only, Python 3.10+.

```bash
conda create -n aiops-m4 python=3.10 -y && conda activate aiops-m4
pip install -r requirements.txt
python eta_data.py        # writes data/ (train/fresh CSVs and the v1/v2 model bundles)
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
```

`eta_data.py` should print v1 fresh-data MAE 11.95 and v2 6.10 ([`0_setup/02`](screenshots/0_setup)).

**Quickest check (no servers needed):** `python -m pytest partA partB -v` gives **28 passed** (15 + 13).

| Port | What |
|---|---|
| 8000 | Part A FastAPI service |
| 5101 / 5102 | `mlflow models serve` prod / candidate |
| 8100 | Ray Serve app |

## 2. Checklist

### A1. Service (12): [`partA/eta_app.py`](partA/eta_app.py)

Start it: `ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --app-dir partA --port 8000`

| Requirement | How to check | Expected | Screenshot |
|---|---|---|---|
| Predict, input order, 0.1 min | `curl -s -X POST localhost:8000/v1/models/eta:predict -H 'Content-Type: application/json' -d '{"instances":[{"distance_km":12,"n_stops":3,"hour":9,"weekday":2,"traffic_index":0.4,"rain":0,"weight_kg":5}]}'` | `predictions`, `model_version: "v1"` | [01](screenshots/A1_service) |
| Request id | add `-i` and `-H 'X-Request-ID: abc'` | `X-Request-ID: abc` echoed; a UUID if not sent, also on errors | 01, 02 |
| Field validation | one row with `"hour":25`, `n_stops` missing and `"distance_km":"x"` | 422 `validation_failed`, one `details` entry per bad field | 03 |
| Cross-field rule | `distance_km > 150` with `n_stops > 20` | 422 "infeasible delivery" | 04 |
| Batch limits | 0 rows and 501 rows | 422 (1 and 500 rows give 200) | 05 |
| `/healthz`, `/metrics` | `curl localhost:8000/healthz`, `curl localhost:8000/metrics` | `{"status":"ok","model_version":"v1"}`; `eta_requests_total` + `eta_request_latency_seconds` | 06 |
| Generic 500 | covered by the monkeypatch test in A2 | body only `{"error":"internal_error","request_id":...}` | |
| Model loaded once | module level `joblib.load` at the top of `eta_app.py` | not inside the handler | |

### A2. Tests (5): [`partA/test_eta_app.py`](partA/test_eta_app.py)

`python -m pytest partA -v` gives **15 passed**. The test names map one-to-one to the brief's list (valid, order,
3 invalid kinds, cross-field, both limits, request id incl. on an error, metrics, 500 via monkeypatch).
Mutation check: breaking the cross-field rule and the header echo makes 4 tests fail
([`A2_tests/02`](screenshots/A2_tests)).

### A3. Load test (8): report §2

Raw results: [`partA/loadtest_results.json`](partA/loadtest_results.json) (c = 1–16),
[`loadtest_v1_c24_c32.json`](partA/loadtest_v1_c24_c32.json), [`loadtest_batch1.json`](partA/loadtest_batch1.json) /
[`loadtest_batch100.json`](partA/loadtest_batch100.json), [`loadtest_v1_2workers.json`](partA/loadtest_v1_2workers.json).
Commands to re-run them are in the [README](README.md#part-a-rest-service).

| Item | Where in the report |
|---|---|
| Table c = 1, 2, 4, 8, 16 | §2 table (plus c = 24, 32 to find the SLO limit) |
| Saturation via Little's Law | X·p50 column ≈ concurrency; saturates at ~520 req/s after c = 2 |
| Capacity under p95 ≤ 80 ms | c = 24 → 78.1 ms (pass), c = 32 → 96.5 ms (fail) |
| Batch 1 vs 100, ms/row | 2.24 ms/row vs 0.032 ms/row (~70× cheaper) |
| Change that doubles capacity | `--workers 2`: p95 at c = 32 drops 96.5 → 40.9 ms ([`A3_load_test/05–07`](screenshots/A3_load_test)) |
| Why p95/p99 | two sentences in §2, after the batch item |

Hardware: AMD Ryzen 7 8840HS, 8 cores, 14 GiB RAM (report §1). The load generator ran on the same laptop.

### B1. Registration (8): `register()` in [`partB/eta_mlflow.py`](partB/eta_mlflow.py)

`python partB/eta_mlflow.py register` logs v1 and v2, registers `eta-model`, sets `@prod` → 1, `@candidate` → 2.
To view: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000` → Models → eta-model.
Signature, pinned requirements and input example: [`B1_registry/03–07`](screenshots/B1_registry).

### B2. Gate (9): `decide()` and `promotion_gate()` in `eta_mlflow.py`

| Check | Command | Expected |
|---|---|---|
| Unit tests | `python -m pytest partB -v` | **13 passed** (each threshold: on, just inside, just outside) |
| Default thresholds | `python partB/eta_mlflow.py gate` | rel. gain 0.489, all 4 criteria true, `promote: true` |
| Stricter gain | `python partB/eta_mlflow.py gate --min-rel-gain 0.60` | `mae_gain: false`, `promote: false` |

Recorded output: [`partB/gate_output.txt`](partB/gate_output.txt).

### B3. Shadow (8): `shadow()` in `eta_mlflow.py`, [`partB/alias_demo.py`](partB/alias_demo.py)

```bash
mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local   # terminal 1
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local   # terminal 2
python partB/eta_mlflow.py shadow       # 200 requests to both
python partB/alias_demo.py              # moves @prod to v2 and back; :5101 still answers v1
```

Expected ([`partB/shadow_output.txt`](partB/shadow_output.txt)): prod serves v1, candidate v2, mean |diff| 9.48 min,
35% of rows differ by more than 10 min, p95 6.34 / 6.91 ms. Decision and the "alias moved, server didn't" answer are in report §3.

### C1. Sticky canary app (12): [`partC/eta_ray.py`](partC/eta_ray.py)

Router + EtaModel v1 (2 replicas) + v2 (autoscaled 1–3) on a 2-CPU Ray. Split by `sha256(customer_id)`, not
`hash()`; the `X-Beta: 1` header forces v2. Optional live run: `python partC/eta_ray.py --weight 0.1` (port 8100).
Screenshots: [`C1_ray_app`](screenshots/C1_ray_app) (sticky arms, `serve status`, `ray status` showing fractional CPUs).

### C2. Verification (8): [`partC/verify_ray.py`](partC/verify_ray.py)

`python partC/verify_ray.py` starts its own Ray, deploys, and runs all checks (~90 s). Expected
([`partC/verify_output.txt`](partC/verify_output.txt)):

| Check | Result |
|---|---|
| (a) share at weight 0.2 | 208 / 1000 = 0.208 (allowed 0.16–0.24) |
| (b) second call | 200 / 200 same arm |
| (c) after delete + redeploy | 200 / 200 same arm |
| (d) weight 0.2 → 0.5 | 0 customers moved back to prod |
| (e) `X-Beta: 1` | 50 / 50 on v2 |

Ends with `ALL 5 CHECKS PASSED`. The `raylet ... graceful_shutdown` line after it is Ray's normal shutdown noise.

### C3. Short answers (5): report §4.1

### D1. Release gate (6): [`partD/smoke_eta.py`](partD/smoke_eta.py)

8 checks (health, version, batch, 3 model-sanity checks, invalid → 422, latency p95). Exit code 0 = pass, 1 = fail.

```bash
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1; echo $?                   # 0
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v2; echo $?                   # 1 (wrong version)
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 0.5; echo $?  # 1 (too slow)
```

Screenshots: [`D1_smoke`](screenshots/D1_smoke). The abort criteria table for the 10% canary is in report §5.

## 3. Not tested

Listed in report §6 (e.g. `/metrics` is per process with 2 workers; the smoke test was not run against the MLflow servers).
