"""
eta_app.py -- SwiftParcel ETA service (DA3408 Module 4, Assignment 5, Part A)

Run from the repo root:
    ETA_MODEL=data/eta_v1.joblib python -m uvicorn eta_app:app --app-dir partA --port 8000

Endpoints
    POST /v1/models/eta:predict   {"instances": [{7 features}, ...]}  ->  minutes, rounded to 0.1
    GET  /healthz                 liveness + which model version is loaded
    GET  /metrics                 Prometheus text (requests counter + latency histogram)

The structure (bundle loaded once, pydantic schema as the contract, timing middleware) follows
Lecture 10 "REST_API_Serving_Exercise" (rest_app.py). The Prometheus part follows Lecture 9 / my A4.
"""
import os
import time
import uuid
import logging
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram,
                               disable_created_metrics, generate_latest)
from pydantic import BaseModel, Field, model_validator

log = logging.getLogger("eta_app")

MODEL_PATH = os.getenv("ETA_MODEL", "data/eta_v1.joblib")
bundle = joblib.load(MODEL_PATH)                    # loaded once at import, never per request
model = bundle["model"]                             # module level, so tests can monkeypatch it
FEATURES = bundle["feature_columns"]
MODEL_VERSION = bundle["version"]

# own registry, so only my two metrics (no python_gc_* etc.) and no *_created series
disable_created_metrics()
REGISTRY = CollectorRegistry()
REQUESTS = Counter("eta_requests", "HTTP requests by path and status code", ["path", "status"],
                   registry=REGISTRY)
LATENCY = Histogram("eta_request_latency_seconds", "Request latency in seconds (whole request)",
                    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.08, 0.1, 0.25, 0.5, 1.0),
                    registry=REGISTRY)


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        # business rule across two fields: a very long route with many stops can't be done in one trip
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 together with n_stops > 20")
        return self


class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)


app = FastAPI(title="SwiftParcel ETA service")


def _path_label(request: Request) -> str:
    # use the route template, so a scan of random URLs can't create unlimited label values
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:                               # anything the endpoint did not catch
        log.exception("unhandled error, request_id=%s", rid)
        resp = JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    resp.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":              # don't count the scrapes themselves
        LATENCY.observe(time.perf_counter() - t0)
        REQUESTS.labels(path=_path_label(request), status=str(resp.status_code)).inc()
    return resp


@app.exception_handler(RequestValidationError)
async def validation_failed(request: Request, exc: RequestValidationError):
    details = []
    for e in exc.errors():
        loc = [str(x) for x in e["loc"] if x != "body"]
        details.append({"field": ".".join(loc) or "body", "msg": e["msg"]})
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details,
                                                  "request_id": request.state.request_id})


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    # plain def -> FastAPI runs it in its thread pool, so sklearn doesn't block the event loop
    rid = request.state.request_id
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
        preds = [round(float(p), 1) for p in model.predict(X)]
    except Exception:
        log.exception("inference failed, request_id=%s", rid)
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    return {"predictions": preds, "model_version": MODEL_VERSION, "request_id": rid}
