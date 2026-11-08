# AI Disclosure: DA3408 Assignment 5

Kiran Kumar P

Required by section 4 of the course *Code of Conduct: Fair and Responsible Use of AI*.

## 1. Tools used

| Tool | Used for |
|---|---|
| Claude (Anthropic), through Claude Code in VS Code | Boilerplate code, library API details (FastAPI, pydantic, MLflow, Ray Serve), debugging, test cases, wording of the report. |

No other AI assistants were used. The v1 (Ridge) and v2 (gradient boosting) ETA models come from the starter
`eta_data.py` .

## 2. How they were used

| Part | What the AI did | What I did |
|---|---|---|
| A: FastAPI service + 15 tests (`partA/`) | Helped with the pydantic schema and validator syntax, the request-id middleware and the `/metrics` counter and histogram code. Suggested edge cases for the tests | Started from the Lecture 10 `rest_app.py` pattern. Chose the endpoints, the field ranges, the infeasible-input rule and the batch limit (500). Ran the tests and broke the code on purpose to check that 4 of them fail |
| A3: load tests and the 2-worker change | Explained Little's law and why one uvicorn worker uses one core (the GIL) | Ran every sweep (c = 1 to 32, batch 1 vs 100). Found the SLO limit from my own numbers (c = 24 passes at 78.1 ms, c = 32 fails at 96.5 ms). Chose the 2-worker change, re-ran it and checked both cores in `top` |
| B: MLflow register / gate / shadow (`partB/`) | Helped with the MLflow API (pyfunc wrapper, signature, aliases) and the boundary cases in `test_gate.py` | Followed the course `mlflow_register.py`. Chose the gate criteria (overall MAE gain, rush and rain slices, p95). Ran both `mlflow models serve` servers, the shadow comparison, and the alias demo showing a running server keeps v1 |
| C: Ray Serve sticky canary + verify script (`partC/`) | Pointed out that Python's `hash()` is salted per process, so `sha256` is used for the customer split. Helped write the checks in `verify_ray.py` | Built on the course `ray_serve_app.py`. Chose the replica counts, the CPU fractions and the autoscaling settings (scale up in 3 s, down in 30 s, max 3 replicas). Ran the verification and checked that no customer moved back to prod when the weight went from 0.2 to 0.5 |
| D: smoke test (`partD/`) | Helped write the request and exit-code code | Chose the 8 checks and the abort criteria for the 10% canary. Ran it against Part A (v1, v2) and the Ray app, and checked that a wrong version and a 0.5 ms p95 limit both exit 1 |
| Report (`report/REPORT.tex`)  | Converted the report to LaTeX and helped with structure and wording | Wrote the reasoning and answers, chose what goes in each table. |

## 3. Impact

Claude made me faster at writing the boilerplate, looking up the FastAPI, MLflow and Ray Serve APIs, and
debugging.
