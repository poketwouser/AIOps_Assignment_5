"""
eta_ray.py -- Ray Serve sticky canary for the ETA model (DA3408 Module 4, A5 Part C)

    client --HTTP--> Router (FastAPI ingress, sticky split by customer_id) --> EtaModelV1 x 2 (prod)
                                                                           +-> EtaModelV2 x 1..3 (candidate, autoscaled)

CPU budget (fits in a 2-CPU Ray): router 0.1 + prod 2 x 0.25 + candidate up to 3 x 0.25 = 1.35 CPU.

Run (from the repo root):  python partC/verify_ray.py      (deploys on a 2-CPU Ray and checks everything)

Based on Lecture 10 "Ray_Serve_Exercise" (ray_serve_app.py: ChurnModel + Router + build_app). Changes:
the split is per customer (stable hash) instead of per request (random.random()), the X-Beta override,
an autoscaled candidate, and the ETA schema.
"""
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve
from ray.serve.handle import DeploymentHandle

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]

CANDIDATE_AUTOSCALING = dict(min_replicas=1, initial_replicas=1, max_replicas=3,
                             target_ongoing_requests=2,      # ~2 in-flight requests per replica
                             upscale_delay_s=3, downscale_delay_s=30,
                             metrics_interval_s=2, look_back_period_s=10)


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 0.25},
                  max_ongoing_requests=16, health_check_period_s=5,
                  logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        # extra_work_ms = fake heavier model (blocking sleep), same idea as the lecture; 0 for real use
        self.extra = extra_work_ms / 1000.0
        b = joblib.load(path)                        # loaded once per replica
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: List[Dict]):
        X = pd.DataFrame(rows)[self.cols]
        preds = [round(float(p), 1) for p in self.model.predict(X)]
        if self.extra:
            time.sleep(self.extra)
        return {"predictions": preds, "model_version": self.version, "served_by": self.replica}

    async def check_health(self):
        if self.model is None:
            raise RuntimeError("model missing")


class Delivery(BaseModel):                           # same ranges as Part A
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)


class Req(BaseModel):
    customer_id: str = Field(min_length=1, max_length=128)
    instances: List[Delivery] = Field(min_length=1, max_length=500)


def bucket(customer_id: str) -> float:
    """Stable position of a customer in [0, 1). sha256 gives the same answer in every process and after
    every restart (Python's hash() is salted per process). Candidate iff bucket < weight, so raising
    the weight only moves customers from prod to the candidate, never back."""
    h = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2**64


api = FastAPI()


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1}, logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class Router:
    def __init__(self, prod: DeploymentHandle, cand: DeploymentHandle, weight: float):
        self.h = {"v1": prod, "v2": cand}
        self.w = weight

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "weight": self.w}

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: Req, x_beta: Optional[str] = Header(default=None)):
        if x_beta == "1":                            # beta testers always get the candidate
            arm = "v2"
        else:
            arm = "v2" if bucket(req.customer_id) < self.w else "v1"
        out = await self.h[arm].remote([d.model_dump() for d in req.instances])
        return {**out, "arm": arm}


def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    """Application graph: Router -> EtaModel v1 (2 replicas) + EtaModel v2 (autoscaled 1..3)."""
    prod = EtaModel.options(name="EtaModelV1", num_replicas=2).bind(str(ROOT / "data/eta_v1.joblib"), extra_work_ms)
    cand = EtaModel.options(name="EtaModelV2", num_replicas=None,
                            autoscaling_config=CANDIDATE_AUTOSCALING).bind(str(ROOT / "data/eta_v2.joblib"), extra_work_ms)
    return Router.bind(prod, cand, weight)


if __name__ == "__main__":
    # standalone server for manual tests / smoke_eta.py (same as the lecture's ray_serve_app.py main)
    import argparse
    import ray
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, default=0.1)
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    ray.init(num_cpus=2, include_dashboard=True, logging_level="ERROR",
             runtime_env={"env_vars": {"PYTHONPATH": str(Path(__file__).resolve().parent)}})
    serve.start(http_options={"host": "127.0.0.1", "port": a.port}, logging_config={"log_level": "WARNING"})
    serve.run(build(weight=a.weight), name="eta", route_prefix="/")
    print(f"Serve is up on http://127.0.0.1:{a.port}  (weight={a.weight}, Ctrl+C to stop)")
    while True:
        time.sleep(3600)
