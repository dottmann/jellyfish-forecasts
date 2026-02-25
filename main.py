from fastapi import FastAPI, Query
import json
import os

app = FastAPI()

# ---- Load forecast once at startup ----
file_path = "latest_forecast.json"

if os.path.exists(file_path):
    with open(file_path) as f:
        forecast_data = json.load(f)
else:
    forecast_data = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/jellyfish_forecast")
def get_forecast(date: str = Query(None)):

    if forecast_data is None:
        return {"error": "Forecast file not found"}

    if date is None:
        return forecast_data

    filtered_features = [
        feature for feature in forecast_data["features"]
        if feature["properties"]["forecast_date"] == date
    ]

    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }