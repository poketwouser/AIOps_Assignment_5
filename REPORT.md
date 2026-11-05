# DA3408 Module 4, Assignment 5: SwiftParcel ETA service

Kiran Kumar P

All numbers below come from runs on my laptop on 5 Nov 2026. The raw outputs are in `partA/*.json`,
`partB/gate_output.txt`, `partB/shadow_output.txt`, `partC/verify_output.txt` and `results/logs/`, and there
are screenshots of every run in `screenshots/` (the map is in the README).

## 0. Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 8840HS, 8 cores / 16 threads |
| RAM | 14 GiB (about 7.7 GiB free during the runs) |
| OS / Python | Ubuntu (Linux 7.0), Python 3.10.22 in a conda env |
| Versions | fastapi 0.162.0, uvicorn 0.54.0, pydantic 2.14.3, scikit-learn 1.7.2, mlflow 3.18.0, ray 2.60.0, prometheus_client 0.26.0 |

The load generator ran on the same laptop as the server, so the two compete for CPU.
`python eta_data.py`: v1 (Ridge) fresh MAE 11.95 min, v2 (GBR) fresh MAE 6.10 min.

---

## Part A: REST service

### A1. `partA/eta_app.py`

| Req. | How |
|---|---|
| 1 | `POST /v1/models/eta:predict`, pydantic `PredictRequest`, predictions `round(x, 1)`, same order as the DataFrame rows |
| 2 | `Field(ge/le/gt)` on every feature, `instances` `min_length=1, max_length=500` |
| 3 | `model_validator` on `Delivery`: `distance_km > 150 and n_stops > 20` → error whose message contains "infeasible" → 422 |
| 4 | HTTP middleware: takes `X-Request-ID` or makes a `uuid4`, keeps it in `request.state`, sets the header on every response (also 422 and 500) |
| 5 | `RequestValidationError` handler → `{"error": "validation_failed", "details": [{field, msg}], "request_id"}`; `field` is the location without `body`, e.g. `instances.0.hour` |
| 6 | Own `CollectorRegistry`: counter `eta_requests_total{path, status}` and histogram `eta_request_latency_seconds` (11 buckets + `+Inf`, `_sum`, `_count`). The middleware skips `/metrics`. `path` is the route template, so random URLs can't create new label values |
| 7 | `GET /healthz`. Inference is in `try/except`, which returns 500 `{"error": "internal_error", "request_id"}` and logs the traceback on the server only |

The bundle is loaded once when the module is imported (`model` is a module variable, so a test can monkeypatch it).
The endpoint is a plain `def`, so FastAPI runs sklearn in its thread pool, as in the lecture's `rest_app.py`.

### A2. Tests (`partA/test_eta_app.py`): 15 passed

Valid request; batch order (3 rows vs the same rows one by one); 5 kinds of invalid input (out of range,
wrong type, missing field, `rain=2`, `weight_kg=0`); cross-field rule including both edges (150/21 and 151/20 are
allowed); batch sizes 0, 500 and 501; request-id echo, generation and on a 422; `/healthz`; the metrics text
(the counter goes up by exactly 1 for 200 and 422, `+Inf`, `_sum`, `_count`, no `/metrics` series); and a generic 500 with a
monkeypatched model whose exception text must not leak.
To check that the tests fail when the code is wrong, I changed `150 → 1500` and removed the header echo in a
copy of the app: **4 tests failed** (`results/logs/pytest_partA_mutation.log`, screenshot A2/02).

### A3. Load test (v1, `uvicorn --workers 1`, batch 1, 4 s per level)

| concurrency | requests | rps | p50 ms | p95 ms | p99 ms | errors | X · p50 |
|---|---|---|---|---|---|---|---|
| 1 | 1700 | 425.0 | 2.0 | 3.8 | 5.2 | 0 | 0.8 |
| 2 | 2478 | **619.5** | 2.9 | 5.0 | 6.3 | 0 | 1.8 |
| 4 | 2062 | 515.5 | 7.5 | 10.9 | 13.9 | 0 | 3.9 |
| 8 | 2086 | 521.5 | 15.0 | 20.8 | 25.7 | 0 | 7.8 |
| 16 | 2091 | 522.8 | 29.4 | 39.6 | 85.3 | 0 | 15.4 |
| 24 (extra) | 1944 | 486.0 | 46.4 | **78.1** | 153.3 | 0 | 22.6 |
| 32 (extra) | 2036 | 509.0 | 60.8 | **96.5** | 129.3 | 0 | 30.9 |

**Saturation.** Throughput stops growing after concurrency 2 and stays at ~520 req/s. The last column
is Little's Law, L = X · W (with p50 as an estimate of W): it comes back to the concurrency I set (7.8 ≈ 8, 15.4 ≈ 16). So once
the server is saturated, more clients don't add throughput, they only wait in the queue: W ≈ N / X_max, and p50 doubles each time N
doubles. `top` during the c=8 run shows the single uvicorn process at ~92% CPU, i.e. one core (the GIL), while
the machine is 88% idle (screenshot A3/03). The service time is 1 / X_max ≈ 1.6–1.9 ms per request. c=2 is a little
faster than c≥4; my reading is that with more threads, the event loop and the worker threads fight over the one GIL.
**The saturation point is concurrency ≈ 2, at about 520–620 req/s.**

**Capacity under p95 ≤ 80 ms.** All the levels in the brief pass, so I worked out the limit and then
checked it. At X_max ≈ 520 req/s, W = N / 520. In my runs p95 ≈ 1.35–1.7 × p50, so p95 reaches 80 ms when p50 ≈ 47–59 ms,
which is N ≈ 520 × (0.047…0.059) ≈ 24–31. Measured: c=24 → p95 78.1 ms (passes), c=32 → 96.5 ms (fails).
**Capacity: about 500 req/s with up to ~24 requests in flight.** I would plan on ~400 req/s (80%) for headroom,
because p99 is already above 80 ms at c=16.

**Batch 1 vs 100 (c=1).** Batch 1: 447 req/s → 2.24 ms per request → **2.24 ms/row**.
Batch 100: 313 req/s → 3.19 ms per request → **0.032 ms/row** (~70× cheaper per row). Most of the cost of a
request is fixed (HTTP + JSON parsing, the pydantic object, building a DataFrame, the sklearn call, the response),
and one extra row only adds (3.19 − 2.24) / 99 ≈ 0.01 ms, because the Ridge pipeline is a vectorised matrix
product. The price is latency: in a batch, the first row waits for the last.

**Change: 2 uvicorn worker processes** (`--workers 2`). The CPU-bound part runs under one GIL per process, and the
laptop has 16 threads, so a second process should double the capacity.

| concurrency | 1 | 2 | 4 | 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|
| rps, 1 worker | 425.0 | 619.5 | 515.5 | 521.5 | 522.8 | 486.0 | 509.0 |
| rps, 2 workers | 429.5 | 597.8 | 810.8 | 1020.0 | 1056.0 | 1072.8 | 1048.0 |
| p95 ms, 2 workers | 3.7 | 5.0 | 8.3 | 12.7 | 21.5 | 31.2 | 40.9 |

Result: at the same concurrency it is **1.96× (c=8) and 2.02× (c=16)**. Peak against peak it is 1072.8 / 619.5 = **1.73×**,
so not quite double. At c=1–2 nothing changes, because one request at a time can't use two processes. p95 at c=32
is 40.9 ms instead of 96.5 ms, so the capacity under the SLO is now ≈ 1,050 req/s. `top` (A3/07) shows both workers
and the load generator at ~100% CPU each, so the Python load generator is now also a limit, and I can't
measure further scaling from this laptop. Side effect: `prometheus_client` counters are per process, so with 2
workers `/metrics` only shows one worker's counts. Fixing that needs `PROMETHEUS_MULTIPROC_DIR` (not done).

**Why p95/p99 and not the mean.** The mean hides the slow tail that some users actually get: at c=16 the p50 is
29 ms but p99 is 85 ms, already over the SLO. Latency is skewed and an SLO is a promise per request
(e.g. "95% under 80 ms"), and the tail is where queueing first shows.

---

## Part B: MLflow registry, gate and shadow

### B1. Registration (`partB/eta_mlflow.py register`)
`EtaPyfunc` wraps the bundle and returns a DataFrame `eta_minutes` (1 decimal) + `model_version`. Each version
is logged with `infer_signature` (7 float64 inputs, 2 outputs), `input_example` (2 rows), pinned
`pip_requirements` (`scikit-learn==1.7.2, pandas==2.3.3, numpy==2.2.6, joblib==1.6.0, mlflow==3.18.0`),
params `algorithm`, `bundle_version` and metric `fresh_mae`. Tracking store: `sqlite:///mlflow.db`.

| version | bundle | algorithm | alias |
|---|---|---|---|
| 1 | v1 | Ridge (scaler + Ridge pipeline) | **prod** |
| 2 | v2 | GradientBoostingRegressor | **candidate** |

### B2. Promotion gate
`decide()` is pure (no I/O, inputs not modified) and returns 4 booleans plus the numbers:
`mae_gain`: (prod − cand) / prod ≥ min_rel_gain; `latency_p95`: cand p95 ≤ max_p95_ms; `rush_slice` and `rain_slice`:
cand slice MAE ≤ prod slice MAE × (1 + max_slice_regression). I read "not more than 0.05" as relative (5%), like
the gain. A 1e-9 tolerance keeps values exactly on a threshold from flipping because of float error.
`promotion_gate()` loads `@prod` and `@candidate` from the registry, evaluates both on `make_data(2000, seed=99)`
(never used in training, which used seed 1) and times 200 single-row `predict` calls after 10 warm-up calls.
`test_gate.py`: **13 passed**: on, just inside and just outside every threshold, a candidate that wins overall but regresses on
rain, custom thresholds, and purity.

| alias | MAE | MAE rush | MAE rain | p95 ms (single row, in-process) |
|---|---|---|---|---|
| prod (v1) | 11.955 | 19.680 | 11.623 | 0.90 |
| candidate (v2) | 6.105 | 8.580 | 6.285 | 1.05 |

Default thresholds: relative gain 0.489 ≥ 0.03, all 4 criteria true → **promote = true**.
With `min_rel_gain=0.60`: `mae_gain` false (0.489 < 0.60), the rest true → **promote = false**. Full JSON: `partB/gate_output.txt`.

### B3. Shadow comparison (200 rows of `eta_fresh.csv`, one row per request, same request to both)

| metric | value |
|---|---|
| mean \|prod − candidate\| | **9.48 min** (max 51.7) |
| requests with \|diff\| > 10 min | **35.0 %** |
| p95 latency, prod :5101 / candidate :5102 | **6.34 ms / 6.91 ms** (p50 4.49 / 4.68) |

The two models disagree a lot: one ETA in three changes by more than 10 minutes. That is expected,
because v2 halves the error on fresh data and the biggest gain is in rush hour (19.7 → 8.6 min), which is where Ridge cannot
fit the rush × traffic interaction. Latency is almost the same and far below the 80 ms SLO, and the gate passes
on every criterion, so **I would promote v2**. But not in one step: with this many visible changes, I would release
it through the sticky canary (10% → 50% → 100% of customers) with the Part D abort criteria, and check its MAE against
real delivery times as they come in.

**Why moving an alias doesn't change a running server:** `mlflow models serve -m models:/eta-model@prod` resolves
the alias to a version once at start-up, downloads that version's artifacts and keeps the model in memory. After that it never asks the
registry again. An alias is only a row in the registry database. I checked this: with `prod` moved to version 2, the
running :5101 server still answered `model_version: v1` (`results/logs/alias_demo.log`, screenshot B3/05).

**Two ways to pick up the new version without dropping requests:** (1) a **rolling / blue-green restart**: start
new server processes (pods) that resolve the alias again, wait for readiness plus the smoke test, move the load
balancer or Kubernetes Service to them, then drain and stop the old ones (`RollingUpdate maxUnavailable: 0`, Lecture 10
Minikube). (2) a **hot swap inside the service**: a background thread polls the registry alias (or gets a webhook),
loads and warms up the new version next to the old one, then swaps the model reference in one step. In-flight requests
finish on the old model. Ray Serve does something similar when the app is redeployed with new arguments: it starts the new replicas before it stops the
old ones.

---

## Part C: Ray Serve sticky canary

### C1. `partC/eta_ray.py`
`EtaModel` loads the bundle once per replica. `EtaModelV1` (prod) has `num_replicas=2`. `EtaModelV2` (candidate) has
`autoscaling_config` min 1 / max 3, `target_ongoing_requests=2`, upscale 3 s, downscale 30 s, metrics interval 2 s,
look-back 10 s. The FastAPI ingress `Router` takes `{"customer_id", "instances"}` and returns the model output plus `"arm"`.
**Sticky split:** `bucket(id) = first 8 bytes of sha256(id) / 2^64`, candidate iff `bucket < weight`. This routes
by customer, not by request, and because the condition is monotonic, raising the weight only moves customers from prod to the candidate. `X-Beta: 1` forces v2.
CPUs: router 0.1 + prod 2 × 0.25 + candidate ≤ 3 × 0.25 = **1.35 of 2 CPUs** (`ray status`: 0.85/2.0 used at
minimum scale). `build(weight=0.1, extra_work_ms=0.0)` returns the bound application.

### C2. `partC/verify_ray.py` (Ray with `num_cpus=2`, 1000 ids `cust-00000…`, weight 0.2): all 5 checks pass

| check | evidence |
|---|---|
| (a) share | 208 / 1000 = 0.208 on v2 (allowed 0.16–0.24) |
| (b) second call | 200 / 200 same arm |
| (c) after `serve.delete` + redeploy | 200 / 200 same arm |
| (d) 0.2 → 0.5 | candidate → prod: **0**, prod → candidate: 312, new share 0.520 |
| (e) `X-Beta: 1` | 50 / 50 on v2 (the same customers without the header: v1) |

### C3. Short answers
1. **`hash()`**: for `str`, Python salts the hash with a random seed per process (`PYTHONHASHSEED`), so each router
   replica and every restart puts customers in different buckets. Stickiness would break across replicas and redeploys,
   which is exactly checks (b)–(d). It isn't guaranteed to be the same across Python versions or platforms either. sha256 is deterministic everywhere and
   spreads ids evenly.
2. **Why stickiness matters.** *Product:* a customer who refreshes the app must not see the ETA jump between two
   models (41 → 55 → 41 min). Consistent promises are the product, and flip-flopping ETAs lose trust and cause support calls.
   *Analysis:* the customer is the unit of randomisation. If one customer saw both arms, their outcomes (late
   deliveries, complaints, cancellations) couldn't be attributed to either model, and observations would no longer be independent.
   With stickiness, the delayed ground truth (the actual delivery time) can be joined back to one arm.
3. **Autoscaling parameters.** One request takes ~2–5 ms, so `target_ongoing_requests=2` scales up as soon as a
   queue starts to form, without adding a replica for every request. Scale up fast (3 s) because canary traffic
   is small and bursty and latency suffers right away. Scale down slowly (30 s) to avoid flapping, as in the lecture. A 10 s look-back with a 2 s
   metrics interval smooths out single spikes. `max_replicas=3` is set by the CPU budget (1.35 ≤ 2 CPUs, which leaves room for
   the proxy and controller), and `min_replicas=1` keeps one warm candidate replica so the first canary customer doesn't hit a cold start.

---

## Part D: Release gate

`partD/smoke_eta.py --url … --expect-version … [--max-p95-ms 80]` runs 8 checks and exits 0 or 1: healthz; the expected `model_version`;
a 20-row batch (200, finite, 0–2000 min); **3 model-sanity checks**: ETA does not go down as distance goes 5 → 40 → 120 km,
traffic 0.1 → 0.5 → 0.9, or stops 1 → 8 → 20; **invalid input (`hour=25`) → 422**; and p95 of 30 single-row requests ≤ limit.
It works against Part A and Part C (same paths, and the body also carries a `customer_id`).
Runs: Part A v1 PASS (p95 2.5 ms), Part A v2 PASS (4.8 ms), Ray app v1 PASS (20.1 ms). It failed with **exit 1** for
a wrong expected version and for an impossible `--max-p95-ms 0.5` (`results/logs/smoke_*.log`).

**Abort criteria for a 10% canary of v2** (comparing the v2 arm with v1 over the same window; any one of them → set the weight to 0):

| # | metric | abort if | window |
|---|---|---|---|
| 1 | 5xx ratio of the v2 arm (`eta_requests_total{status=~"5.."}` / all) | > 1% **or** > 2 × v1's ratio | 5 min rolling, at least 200 v2 requests |
| 2 | v2 p95 latency (`histogram_quantile(0.95, eta_request_latency_seconds)`) | > 80 ms **or** > 1.5 × v1 p95 | 3 consecutive 1-min windows |
| 3 | MAE against real delivery times, v2 vs v1 (overall, rush-hour and rain slices) | v2 overall MAE > v1's, or any slice > v1 + 5% | rolling 1 h, at least 300 completed v2 deliveries |
| 4 | late-delivery rate: share of v2 orders delivered > 15 min after the ETA shown | > v1's rate + 2 percentage points | rolling 1 h, at least 300 completed v2 deliveries |

---

## Not tested

- Autoscaling of the candidate under real load: only the config and the idle state (1 replica) were checked, and I did not run a load test that scales it to 3.
- The abort criteria are a design only. No Prometheus server scraped `/metrics` (the text format was checked by tests and curl), and no alerting was set up.
- Load tests had the client and server on the same laptop. With 2 workers the load generator was CPU-bound, so the 2-worker numbers are a lower bound.
- `/metrics` with `--workers 2` is per process (multiprocess mode not set up).
- `smoke_eta.py` was not run against the MLflow servers (different API: `/invocations`, and MLflow returns 400 with no range checks).
- The shadow test uses 200 offline rows, not mirrored live traffic. The gate's p95 is in-process `pyfunc.predict`, not over HTTP.
- Only Python 3.10 on Linux. No Docker / Kubernetes deployment (not asked).

## Sources and reuse

The starter `eta_data.py` and `load_test.py` are used unchanged. Ideas and small fragments reused, with changes:
Lecture 10 *REST_API_Serving_Exercise* (`rest_app.py`: bundle loaded once, pydantic schema, timing middleware,
thread-pool `def` endpoint, the `--workers` experiment); *MLflow_Serving_Exercise* (`mlflow_register.py`: pyfunc
wrapper, float64 signature, `log_model` + alias; the gate on fresh data and the shadow loop from its Tasks A/B); *Ray_Serve_Exercise*
(`ray_serve_app.py`: model deployment + FastAPI Router + `build_app`, fractional CPUs, `extra_work_ms`, autoscaling
keys); Lecture 10 *Minikube_Serving* slides (the smoke-test checks and `smoke || rollback`, rolling update for B3); Lecture 9
and my Assignment 4 (Prometheus counter and histogram).
AI use is described in `AI_DISCLOSURE.md`.
