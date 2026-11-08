# Moves the prod alias to version 2, asks the running :5101 server which version answers, then moves it back.
import requests
from mlflow import MlflowClient

c = MlflowClient()
body = {"dataframe_records": [{"distance_km": 12.5, "n_stops": 4, "hour": 10, "weekday": 2,
                               "traffic_index": 0.3, "rain": 0, "weight_kg": 5.0}]}
ask = lambda: requests.post("http://127.0.0.1:5101/invocations", json=body).json()["predictions"][0]
print("prod alias ->", c.get_model_version_by_alias("eta-model", "prod").version, "| :5101 answers", ask())
c.set_registered_model_alias("eta-model", "prod", "2")
print("prod alias ->", c.get_model_version_by_alias("eta-model", "prod").version, "| :5101 answers", ask(), "<- still v1")
c.set_registered_model_alias("eta-model", "prod", "1")
print("prod alias moved back ->", c.get_model_version_by_alias("eta-model", "prod").version)
