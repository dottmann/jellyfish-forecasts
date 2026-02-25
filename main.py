from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import os
import logging

app = FastAPI(title="Jellyfish Forecast API")

logging.basicConfig(level=logging.INFO)

file_path = "latest_forecast.json"

forecast_data: Dict[str, Any] | None = None
data_by_date: Dict[str, List[Dict[str, Any]]] = {}
available_dates_cache: List[str] = []


# -----------------------------
# Startup: load + index data
# -----------------------------
@app.on_event("startup")
def load_forecast():

    global forecast_data, data_by_date, available_dates_cache

    if not os.path.exists(file_path):
        logging.error("Forecast file not found.")
        return

    with open(file_path) as f:
        forecast_data = json.load(f)

    # Index by date (performance boost)
    for feature in forecast_data["features"]:
        date = feature["properties"]["forecast_date"]
        data_by_date.setdefault(date, []).append(feature)

    available_dates_cache = sorted(data_by_date.keys())

    logging.info(f"Loaded forecast with {len(forecast_data['features'])} grid points")
    logging.info(f"Available dates: {available_dates_cache}")


# -----------------------------
# Response Models
# -----------------------------
class PointForecastResponse(BaseModel):
    lat: float
    lon: float
    nearest_point: Dict[str, float]
    forecast_date: str
    prob_0: float
    prob_1: float
    prob_2: float
    heat: float
    risk_level: str


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/available_dates")
def available_dates():

    if forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    return {"available_dates": available_dates_cache}


@app.get("/jellyfish_forecast")
def get_forecast(date: str = Query(None)):

    if forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    if date is None:
        return forecast_data

    if date not in data_by_date:
        raise HTTPException(status_code=404, detail="Date not found")

    return {
        "type": "FeatureCollection",
        "features": data_by_date[date]
    }


@app.get("/point_forecast", response_model=PointForecastResponse)
def point_forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date: str = Query(...)
):

    if forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    if date not in data_by_date:
        raise HTTPException(status_code=404, detail="Date not found")

    daily_features = data_by_date[date]

    # Find nearest grid point (fast, no full dataset scan)
    nearest_feature = min(
        daily_features,
        key=lambda feature: (
            (feature["geometry"]["coordinates"][1] - lat) ** 2 +
            (feature["geometry"]["coordinates"][0] - lon) ** 2
        )
    )

    props = nearest_feature["properties"]
    grid_lon, grid_lat = nearest_feature["geometry"]["coordinates"]

    # Risk classification
    heat = props["heat"]

    if heat > 0.8:
        risk_level = "HIGH"
    elif heat > 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PointForecastResponse(
        lat=lat,
        lon=lon,
        nearest_point={
            "grid_lat": grid_lat,
            "grid_lon": grid_lon
        },
        forecast_date=date,
        prob_0=props["prob_0"],
        prob_1=props["prob_1"],
        prob_2=props["prob_2"],
        heat=heat,
        risk_level=risk_level
    )