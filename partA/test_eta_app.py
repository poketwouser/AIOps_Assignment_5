"""Tests for eta_app.py. Run from the repo root:  pytest partA -v"""
import os
import re
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ETA_MODEL", str(ROOT / "data" / "eta_v1.joblib"))

from fastapi.testclient import TestClient  # noqa: E402

import eta_app  # noqa: E402

URL = "/v1/models/eta:predict"
GOOD = {"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
        "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}


@pytest.fixture
def client():
    return TestClient(eta_app.app, raise_server_exceptions=False)


def post(client, rows, **kw):
    return client.post(URL, json={"instances": rows}, **kw)


def test_valid_request(client):
    r = post(client, [GOOD])
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "v1"
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]
    assert isinstance(p, float) and p == round(p, 1) and p > 0


def test_batch_order_preserved(client):
    # short, medium and long trips: the answer for each must come back in the same slot
    rows = [{**GOOD, "distance_km": d} for d in (2, 120, 30)]
    r = post(client, rows)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    single = [post(client, [row]).json()["predictions"][0] for row in rows]
    assert preds == single
    assert preds[1] > preds[2] > preds[0]


@pytest.mark.parametrize("bad_row, field", [
    ({**GOOD, "hour": 24}, "instances.0.hour"),                       # out of range
    ({**GOOD, "traffic_index": "heavy"}, "instances.0.traffic_index"),  # wrong type
    ({k: v for k, v in GOOD.items() if k != "weight_kg"}, "instances.0.weight_kg"),  # missing
    ({**GOOD, "rain": 2}, "instances.0.rain"),                        # rain must be 0/1
    ({**GOOD, "weight_kg": 0}, "instances.0.weight_kg"),              # weight must be > 0
])
def test_invalid_input_422(client, bad_row, field):
    r = post(client, [bad_row])
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation_failed"
    assert field in [d["field"] for d in body["details"]]
    assert body["request_id"] == r.headers["X-Request-ID"]


def test_cross_field_infeasible(client):
    r = post(client, [{**GOOD, "distance_km": 151, "n_stops": 21}])
    assert r.status_code == 422
    assert "infeasible" in r.json()["details"][0]["msg"]
    # just on the edge is still allowed (rule is strictly greater than)
    assert post(client, [{**GOOD, "distance_km": 150, "n_stops": 21}]).status_code == 200
    assert post(client, [{**GOOD, "distance_km": 151, "n_stops": 20}]).status_code == 200


def test_batch_size_limits(client):
    assert post(client, []).status_code == 422
    assert post(client, [GOOD] * 501).status_code == 422
    r = post(client, [GOOD] * 500)
    assert r.status_code == 200 and len(r.json()["predictions"]) == 500


def test_request_id_echoed(client):
    r = post(client, [GOOD], headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"


def test_request_id_generated(client):
    r = post(client, [GOOD])
    rid = r.headers["X-Request-ID"]
    uuid.UUID(rid)                                     # raises if it isn't a UUID
    assert r.json()["request_id"] == rid


def test_request_id_on_error(client):
    r = post(client, [{**GOOD, "hour": -1}], headers={"X-Request-ID": "err-9"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-9" and r.json()["request_id"] == "err-9"
    r2 = post(client, [{**GOOD, "hour": -1}])
    assert r2.json()["request_id"] == r2.headers["X-Request-ID"]


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "v1"}
    assert "X-Request-ID" in r.headers


def _counter(text, path, status):
    m = re.search(rf'eta_requests_total{{path="{re.escape(path)}",status="{status}"}} (\S+)', text)
    return float(m.group(1)) if m else 0.0


def test_metrics_output(client):
    before = client.get("/metrics").text
    post(client, [GOOD])
    post(client, [{**GOOD, "hour": 99}])
    text = client.get("/metrics").text
    assert _counter(text, URL, 200) == _counter(before, URL, 200) + 1
    assert _counter(text, URL, 422) == _counter(before, URL, 422) + 1
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert len(re.findall(r"eta_request_latency_seconds_bucket\{le=", text)) >= 5
    assert "eta_request_latency_seconds_sum" in text and "eta_request_latency_seconds_count" in text
    assert 'path="/metrics"' not in text               # scrapes are not counted


def test_generic_500(client, monkeypatch):
    class Broken:
        def predict(self, X):
            raise RuntimeError("secret internal detail /opt/model.joblib")
    monkeypatch.setattr(eta_app, "model", Broken())
    r = post(client, [GOOD], headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert r.json() == {"error": "internal_error", "request_id": "boom-1"}
    assert "secret" not in r.text and "RuntimeError" not in r.text
    assert r.headers["X-Request-ID"] == "boom-1"
