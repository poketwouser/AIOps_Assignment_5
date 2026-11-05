# DA3408 Assignment 5 (Module 4): SwiftParcel ETA service, served four ways

Kiran Kumar P

A delivery-time (ETA) regression model (v1 Ridge, v2 gradient boosting) served as a FastAPI service, registered and
gated in MLflow, released through a sticky Ray Serve canary, and checked by a smoke test before release.
Everything runs on the CPU.

Short version of the results (Ryzen 7 8840HS, 16 threads, 14 GiB):

- FastAPI, 1 worker: saturates at ~520 req/s (peak 620 at concurrency 2). p95 stays under 80 ms up to ~24 requests in flight.
  2 workers: ~1,050 req/s (2.0× at the same concurrency, 1.73× peak to peak).
- Batch 100 costs 0.032 ms/row against 2.24 ms/row for batch 1.
- Gate: v2 cuts MAE from 11.96 to 6.11 min (−49%) with no slice regression → promote. With `min_rel_gain=0.60` → no.
- Shadow: v1 and v2 differ by 9.5 min on average, and 35% of requests differ by more than 10 min. p95 is 6.3 / 6.9 ms.
- Ray canary: 20.8% of 1000 customers on v2 at weight 0.2, 100% sticky after a second call and a redeploy, and none moved back at 0.5.

The report with all tables and answers is [`REPORT.md`](REPORT.md).

---

## Rubric map

| Rubric item (points) | Code | Evidence |
|---|---|---|
| A1 Correct, complete service (12) | [`partA/eta_app.py`](partA/eta_app.py) | [`screenshots/A1_service`](screenshots/A1_service), REPORT §A1 |
| A2 Test suite (5) | [`partA/test_eta_app.py`](partA/test_eta_app.py) (15 tests) | [`screenshots/A2_tests`](screenshots/A2_tests) (pass + the mutation check) |
| A3 Load test and reasoning (8) | [`partA/loadtest_results.json`](partA/loadtest_results.json) + the other `partA/loadtest_*.json` | [`screenshots/A3_load_test`](screenshots/A3_load_test), REPORT §A3 |
| B1 Registration (8) | [`partB/eta_mlflow.py`](partB/eta_mlflow.py) `register()` | [`screenshots/B1_registry`](screenshots/B1_registry), REPORT §B1 |
| B2 Gate (9) | `decide()`, `promotion_gate()`, [`partB/test_gate.py`](partB/test_gate.py) (13 tests) | [`partB/gate_output.txt`](partB/gate_output.txt), [`screenshots/B2_gate`](screenshots/B2_gate) |
| B3 Shadow + reasoning (8) | `shadow()` in `eta_mlflow.py`, [`results/alias_demo.py`](results/alias_demo.py) | [`partB/shadow_output.txt`](partB/shadow_output.txt), [`screenshots/B3_shadow`](screenshots/B3_shadow), REPORT §B3 |
| C1 Sticky canary app (12) | [`partC/eta_ray.py`](partC/eta_ray.py) | [`screenshots/C1_ray_app`](screenshots/C1_ray_app) |
| C2 Verification script (8) | [`partC/verify_ray.py`](partC/verify_ray.py) | [`partC/verify_output.txt`](partC/verify_output.txt), [`screenshots/C2_verify`](screenshots/C2_verify) |
| C3 Short answers (5) | | REPORT §C3 |
| D1 Release gate (6) | [`partD/smoke_eta.py`](partD/smoke_eta.py) | [`screenshots/D1_smoke`](screenshots/D1_smoke), abort criteria in REPORT §D |
| Machine description, "not tested" list | | REPORT §0 and "Not tested" |

---

## Setup (from a clean checkout)

Run every command from the repository root.

```bash
conda create -n aiops-m4 python=3.10 -y && conda activate aiops-m4
pip install -r requirements.txt
python eta_data.py        # data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib, data/eta_v2.joblib
python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1
```

`data/`, `mlflow.db` and `mlruns/` are not in the repo (brief §7). The commands above regenerate them.

## Part A: REST service

```bash
ETA_MODEL=data/eta_v1.joblib uvicorn eta_app:app --app-dir partA --port 8000 --no-access-log   # terminal 1
python -m pytest partA -v                                                                      # 15 tests, no server needed

URL=http://127.0.0.1:8000/v1/models/eta:predict
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --out partA/loadtest_results.json
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 1   --out partA/loadtest_batch1.json
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100 --out partA/loadtest_batch100.json
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 24 32 --duration 4 --out partA/loadtest_v1_c24_c32.json
# the change: restart terminal 1 with --workers 2, then
python load_test.py --url $URL --data data/eta_sample.csv --concurrency 1 2 4 8 16 24 32 --duration 4 --out partA/loadtest_v1_2workers.json
```

## Part B: MLflow

```bash
python partB/eta_mlflow.py register          # logs v1 + v2, registers eta-model, prod -> v1, candidate -> v2
python -m pytest partB -v                    # 13 tests of decide()
{ python partB/eta_mlflow.py gate; python partB/eta_mlflow.py gate --min-rel-gain 0.60; } | tee partB/gate_output.txt

mlflow models serve -m "models:/eta-model@prod"      -p 5101 --env-manager local   # terminal 1
mlflow models serve -m "models:/eta-model@candidate" -p 5102 --env-manager local   # terminal 2
python partB/eta_mlflow.py shadow | tee partB/shadow_output.txt
python results/alias_demo.py                 # moves prod to v2 and back; the running :5101 keeps answering v1
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000                      # optional, for the registry view
```

## Part C: Ray Serve

```bash
python partC/verify_ray.py | tee partC/verify_output.txt     # starts a 2-CPU Ray, deploys, runs checks (a)-(e), ~90 s
python partC/eta_ray.py --weight 0.1                          # optional: keep the app running on :8100 (Ctrl+C to stop)
```

## Part D: smoke test

```bash
python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 --max-p95-ms 80; echo $?   # Part A
python partD/smoke_eta.py --url http://127.0.0.1:8100 --expect-version v1; echo $?                   # Ray app
```

## Ports

| Port | What |
|---|---|
| 8000 / 8001 | Part A FastAPI (v1 / v2) |
| 5000 | MLflow UI |
| 5101 / 5102 | `mlflow models serve` prod / candidate |
| 8100 | Ray Serve app |

## Screenshots

| Folder | What |
|---|---|
| [`0_setup`](screenshots/0_setup) | machine, package versions, `eta_data.py` |
| [`A1_service`](screenshots/A1_service) | 01 predict + generated id, 02 id echoed, 03 422 body with 6 field errors, 04 infeasible, 05 batch 0/1/500/501, 06 healthz + `/metrics`, 07 server log, 08–09 Swagger |
| [`A2_tests`](screenshots/A2_tests) | 01 15 passed, 02 4 tests fail when the code is broken |
| [`A3_load_test`](screenshots/A3_load_test) | 01 sweep, 02 batch 1 vs 100, 03 `top` (1 worker = 1 core), 04 c=24/32 SLO check, 05–07 2 workers |
| [`B1_registry`](screenshots/B1_registry) | 01 register output + table, 02 registry aliases, 03 signature, 04–07 run, MLmodel, pinned requirements, input example, 08 runs |
| [`B2_gate`](screenshots/B2_gate) | 01 13 tests passed, 02 default thresholds → promote, 03 0.60 → no |
| [`B3_shadow`](screenshots/B3_shadow) | 01–02 the two servers, 03 same request to both, 04 shadow metrics, 05 alias moved but the server still answers v1 |
| [`C1_ray_app`](screenshots/C1_ray_app) | 01 app on 2 CPUs, 02 sticky arms + `X-Beta`, 03 `serve status`, 04 `ray status` 0.85/2 CPU |
| [`C2_verify`](screenshots/C2_verify) | `verify_ray.py`, all 5 checks |
| [`D1_smoke`](screenshots/D1_smoke) | 01 Ray app pass, 02 Part A v1 + v2 pass, 03 wrong version and p95 limit → exit 1 |

## Submission zip

```bash
zip -r Module4_<rollno>.zip REPORT.md partA partB partC partD -x "*/__pycache__/*"
```

## Files

```
eta_data.py, load_test.py   starter files (unchanged)
partA/                      eta_app.py, test_eta_app.py, loadtest_*.json
partB/                      eta_mlflow.py, test_gate.py, gate_output.txt, shadow_output.txt
partC/                      eta_ray.py, verify_ray.py, verify_output.txt
partD/                      smoke_eta.py
results/                    alias_demo.py, logs/ (raw output of every run)
screenshots/                one folder per rubric item
REPORT.md                   report
AI_DISCLOSURE.md            AI use (course code of conduct, section 4)
```
