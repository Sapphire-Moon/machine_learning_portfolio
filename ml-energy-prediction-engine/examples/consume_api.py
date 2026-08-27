"""
Stage 9: Example of an EXTERNAL application consuming the prediction API.

This script knows nothing about pandas, scikit-learn, or the model itself -
it only needs to know the endpoint URL and the JSON shape it expects. This
is exactly how any other program (a dashboard, another team's service, a
mobile app backend) could use this prediction engine.
"""
import requests

API_URL = "http://127.0.0.1:8000/predict"

payload = {
    "source": "Wind",
    "day_name": "Monday",
    "start_hour": 14,
    "day_of_year": 200,
    "year": 2025,
    "production_lag_1": 7200,
    "production_lag_24": 6800,
    "production_lag_168": 6500,
}

response = requests.post(API_URL, json=payload)

print("Status code:", response.status_code)

if response.status_code == 200:
    print("Prediction:", response.json())
else:
    print("Error:", response.json())
