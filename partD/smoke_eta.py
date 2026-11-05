"""
smoke_eta.py -- release gate: run against a freshly deployed ETA server before it takes real traffic.

    python partD/smoke_eta.py --url http://127.0.0.1:8000 --expect-version v1 [--max-p95-ms 80]

Works with the Part A FastAPI service and the Part C Ray Serve app (same paths; the body also carries a
customer_id, which Part A ignores). Exit code 0 = all checks passed, 1 = at least one failed.
Idea from the Lecture 10 Minikube slides (smoke_test.py: healthz, expected version, predictions,
p95 bound, bad input -> 422, used as `smoke || rollback`).
"""
import argparse
import math
import sys
import time

import requests

BASE_ROW = {"distance_km": 25.0, "n_stops": 6, "hour": 11, "weekday": 2,
            "traffic_index": 0.4, "rain": 0, "weight_kg": 8.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--n", type=int, default=30, help="single-row requests for the latency check")
    a = ap.parse_args()
    base = a.url.rstrip("/")
    s = requests.Session()
    results = []
    measured = {"p95": float("nan")}

    def predict(rows):
        return s.post(f"{base}/v1/models/eta:predict", json={"customer_id": "smoke-test", "instances": rows},
                      timeout=10)

    def check(name, fn):
        try:
            ok, info = fn()
        except Exception as e:                       # a crash in a check is a failed check
            ok, info = False, f"{type(e).__name__}: {e}"
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:38s} {info}")

    def health():
        r = s.get(f"{base}/healthz", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok", f"HTTP {r.status_code} {r.json()}"

    def version():
        got = predict([BASE_ROW]).json().get("model_version")
        return got == a.expect_version, f"served={got} expected={a.expect_version}"

    def batch():
        rows = [{**BASE_ROW, "distance_km": 2.0 + 7 * i, "n_stops": 1 + i % 12} for i in range(20)]
        r = predict(rows)
        p = r.json().get("predictions", [])
        ok = (r.status_code == 200 and len(p) == 20
              and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 < x < 2000 for x in p))
        return ok, f"HTTP {r.status_code}, {len(p)} predictions, range {min(p, default=0):.1f}-{max(p, default=0):.1f} min"

    def monotone(field, values):
        def fn():
            p = predict([{**BASE_ROW, field: v} for v in values]).json()["predictions"]
            return all(x <= y for x, y in zip(p, p[1:])), f"{field} {values} -> {p}"
        return fn

    def invalid():
        r = predict([{**BASE_ROW, "hour": 25}])
        return r.status_code == 422, f"hour=25 -> HTTP {r.status_code}"

    def latency():
        lat = []
        for _ in range(3):                           # warm-up
            predict([BASE_ROW])
        for _ in range(a.n):
            t = time.perf_counter()
            r = predict([BASE_ROW])
            lat.append((time.perf_counter() - t) * 1000)
            if r.status_code != 200:
                return False, f"HTTP {r.status_code} during latency run"
        lat.sort()
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        measured["p95"] = p95
        return p95 <= a.max_p95_ms, f"p95={p95:.1f} ms over {a.n} requests (limit {a.max_p95_ms} ms)"

    print(f"smoke test: {base}  expect={a.expect_version}  max_p95={a.max_p95_ms} ms")
    check("healthz", health)
    check("expected model version", version)
    check("20-row batch: 200, finite, plausible", batch)
    check("sanity: longer distance not shorter", monotone("distance_km", [5.0, 40.0, 120.0]))
    check("sanity: more traffic not shorter", monotone("traffic_index", [0.1, 0.5, 0.9]))
    check("sanity: more stops not shorter", monotone("n_stops", [1, 8, 20]))
    check("invalid input -> 422", invalid)
    check("latency p95", latency)

    passed = sum(results)
    if passed == len(results):
        print(f"SMOKE TEST PASSED ({passed}/{len(results)}, version={a.expect_version}, p95={measured['p95']:.1f} ms)")
        sys.exit(0)
    print(f"SMOKE TEST FAILED ({passed}/{len(results)} passed) -> do not release / roll back")
    sys.exit(1)


if __name__ == "__main__":
    main()
