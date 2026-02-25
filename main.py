from fastapi import FastAPI, Query
import json
import os
import math

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

@app.get("/available_dates")
def available_dates():

    if forecast_data is None:
        return {"error": "Forecast file not found"}

    dates = sorted({
        feature["properties"]["forecast_date"]
        for feature in forecast_data["features"]
    })

    return {"available_dates": dates}

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

@app.get("/point_forecast")
def point_forecast(lat: float, lon: float, date: str):

    if forecast_data is None:
        return {"error": "Forecast file not found"}

    # Filter by date first
    daily_features = [
        feature for feature in forecast_data["features"]
        if feature["properties"]["forecast_date"] == date
    ]

    if not daily_features:
        return {"error": "No data for this date"}

    # Find nearest point
    nearest_feature = min(
        daily_features,
        key=lambda feature: (
            (feature["geometry"]["coordinates"][1] - lat) ** 2 +
            (feature["geometry"]["coordinates"][0] - lon) ** 2
        )
    )

    props = nearest_feature["properties"]
    grid_lon, grid_lat = nearest_feature["geometry"]["coordinates"]

    # Simple risk classification based on prob_2
    if props["prob_2"] > 0.6:
        risk_level = "HIGH"
    elif props["prob_2"] > 0.3:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "lat": lat,
        "lon": lon,
        "nearest_point": {
            "grid_lat": grid_lat,
            "grid_lon": grid_lon
        },
        "forecast_date": date,
        "prob_0": props["prob_0"],
        "prob_1": props["prob_1"],
        "prob_2": props["prob_2"],
        "heat": props["heat"],
        "risk_level": risk_level
    }