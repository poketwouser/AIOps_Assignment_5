"""
verify_ray.py -- deploys eta_ray.build() on a 2-CPU Ray and checks the sticky canary (A5 Part C2).

    python partC/verify_ray.py | tee partC/verify_output.txt

Checks, each with an assert and printed evidence (1000 distinct customer ids, weight 0.2):
  (a) share routed to the candidate within +-4 points of 0.2
  (b) 200 customers get the same arm on a second call
  (c) the same holds after deleting and redeploying the identical app
  (d) weight 0.2 -> 0.5: nobody on the candidate moves back to prod
  (e) X-Beta: 1 forces the candidate for 50 requests
"""
import sys
import time
from pathlib import Path

import pandas as pd
import ray
import requests
from ray import serve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eta_ray  # noqa: E402

BASE = "http://127.0.0.1:8100"
URL = f"{BASE}/v1/models/eta:predict"
N_CUSTOMERS, WEIGHT, NEW_WEIGHT = 1000, 0.2, 0.5
ROW = pd.read_csv(eta_ray.ROOT / "data/eta_sample.csv").iloc[0].to_dict()
ROW = {k: (int(v) if k in ("n_stops", "hour", "weekday", "rain") else float(v)) for k, v in ROW.items()}
IDS = [f"cust-{i:05d}" for i in range(N_CUSTOMERS)]
sess = requests.Session()


def call(cid, beta=False):
    r = sess.post(URL, json={"customer_id": cid, "instances": [ROW]},
                  headers={"X-Beta": "1"} if beta else {}, timeout=10)
    r.raise_for_status()
    return r.json()


def arms(ids):
    return {c: call(c)["arm"] for c in ids}


def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name="eta", route_prefix="/")
    for _ in range(60):                              # wait until the router answers
        try:
            if sess.get(f"{BASE}/healthz", timeout=2).json().get("weight") == weight:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("app not ready")


def show_deployments():
    st = serve.status().applications["eta"]
    for name, d in st.deployments.items():
        print(f"    {name:12s} {str(d.status).split('.')[-1]:8s} replicas={dict(d.replica_states)}")


def main():
    ray.init(num_cpus=2, include_dashboard=False, logging_level="ERROR", log_to_driver=False,
             runtime_env={"env_vars": {"PYTHONPATH": str(Path(__file__).resolve().parent)}})
    serve.start(http_options={"host": "127.0.0.1", "port": 8100}, logging_config={"log_level": "WARNING"})
    print(f"Ray cluster resources: {ray.cluster_resources().get('CPU')} CPU")

    deploy(WEIGHT)
    print(f"\nDeployed build(weight={WEIGHT}). Deployments:")
    show_deployments()
    cfg = serve.status().applications["eta"].deployments
    used = 2 * 0.25 + 0.1 + 0.25 * eta_ray.CANDIDATE_AUTOSCALING["max_replicas"]
    print(f"    CPU reserved at max scale: {used} of 2.0 | candidate autoscaling: {eta_ray.CANDIDATE_AUTOSCALING}")
    print(f"    Ray available CPU now: {ray.available_resources().get('CPU', 0):.2f}")
    assert used <= 2.0 and set(cfg) == {"Router", "EtaModelV1", "EtaModelV2"}

    print(f"\n(a) share routed to the candidate, {N_CUSTOMERS} distinct customers, weight {WEIGHT}")
    first = arms(IDS)
    share = sum(a == "v2" for a in first.values()) / N_CUSTOMERS
    print(f"    v2 customers: {sum(a == 'v2' for a in first.values())} / {N_CUSTOMERS} = {share:.3f}"
          f"  (allowed {WEIGHT - 0.04:.2f} .. {WEIGHT + 0.04:.2f})")
    assert abs(share - WEIGHT) <= 0.04
    print("    PASS")

    sample = IDS[:200]
    print("\n(b) second call for 200 customers")
    second = arms(sample)
    same = sum(second[c] == first[c] for c in sample)
    print(f"    same arm: {same} / 200   e.g. " +
          ", ".join(f"{c}:{first[c]}->{second[c]}" for c in sample[:4]))
    assert same == 200
    print("    PASS")

    print("\n(c) serve.delete + redeploy the identical app, then call the same 200 customers")
    serve.delete("eta")
    time.sleep(2)
    deploy(WEIGHT)
    show_deployments()
    third = arms(sample)
    same = sum(third[c] == first[c] for c in sample)
    print(f"    same arm after redeploy: {same} / 200")
    assert same == 200
    print("    PASS")

    print(f"\n(d) raise weight {WEIGHT} -> {NEW_WEIGHT}, call all {N_CUSTOMERS} customers again")
    deploy(NEW_WEIGHT)
    after = arms(IDS)
    back = [c for c in IDS if first[c] == "v2" and after[c] == "v1"]
    moved = [c for c in IDS if first[c] == "v1" and after[c] == "v2"]
    new_share = sum(a == "v2" for a in after.values()) / N_CUSTOMERS
    print(f"    candidate -> prod: {len(back)}   prod -> candidate: {len(moved)}   new v2 share: {new_share:.3f}")
    assert not back
    print("    PASS")

    print("\n(e) X-Beta: 1 header, 50 requests from customers that the hash puts on prod")
    deploy(WEIGHT)
    prod_ids = [c for c in IDS if first[c] == "v1"][:50]
    beta = [call(c, beta=True)["arm"] for c in prod_ids]
    plain = [call(c)["arm"] for c in prod_ids[:5]]
    print(f"    with header: {beta.count('v2')} / 50 on v2 | same customers without header: {plain}")
    assert beta.count("v2") == 50 and set(plain) == {"v1"}
    print("    PASS")

    r = call(IDS[0])
    print(f"\nexample response: {r}")
    print("\nALL 5 CHECKS PASSED")
    serve.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()
