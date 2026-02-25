from fastapi import FastAPI, Query
import json
import os

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/jellyfish_forecast")
def get_forecast(date: str = Query(None)):
    file_path = "latest_forecast.json"

    if not os.path.exists(file_path):
        return {"error": "Forecast file not found"}

    with open(file_path) as f:
        data = json.load(f)

    # If no date specified, return full dataset
    if date is None:
        return data

    # Filter features by forecast_date
    filtered_features = [
        feature for feature in data["features"]
        if feature["properties"]["forecast_date"] == date
    ]

    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }