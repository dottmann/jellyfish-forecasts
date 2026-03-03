from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
from contextlib import asynccontextmanager
import json
import os
import logging
import numpy as np
import mercantile
from PIL import Image, ImageDraw
from io import BytesIO
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm

logging.basicConfig(level=logging.INFO)

FILE_PATH = "latest_forecast.json"


# -------------------------------------------------
# Lifespan handler for startup/shutdown
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):

    if not os.path.exists(FILE_PATH):
        logging.error("Forecast file not found.")
        app.state.forecast_data = None
        app.state.data_by_date = {}
        app.state.available_dates = []
        yield
        return

    with open(FILE_PATH) as f:
        forecast_data = json.load(f)

    data_by_date: Dict[str, List[Dict[str, Any]]] = {}

    for feature in forecast_data["features"]:
        date = feature["properties"]["forecast_date"]
        data_by_date.setdefault(date, []).append(feature)

    app.state.forecast_data = forecast_data
    app.state.data_by_date = data_by_date
    app.state.available_dates = sorted(data_by_date.keys())

    logging.info(f"Loaded {len(forecast_data['features'])} grid points")
    logging.info(f"Available dates: {app.state.available_dates}")

    yield

    # Optional shutdown cleanup
    logging.info("Shutting down API")


# -------------------------------------------------
# Middleware
# -------------------------------------------------

app = FastAPI(
    title="Jellyfish Forecast API",
    lifespan=lifespan
)

# CORS (restrict allow_origins in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your web and app domain before public release
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression (big win for GeoJSON)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# -------------------------------------------------
# Response Model
# -------------------------------------------------
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


# -------------------------------------------------
# Endpoints
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/available_dates")
def available_dates():

    if app.state.forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    return {"available_dates": app.state.available_dates}


@app.get("/jellyfish_forecast")
def jellyfish_forecast(date: str = Query(None)):

    if app.state.forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    if date is None:
        return app.state.forecast_data

    if date not in app.state.data_by_date:
        raise HTTPException(status_code=404, detail="Date not found")

    return {
        "type": "FeatureCollection",
        "features": app.state.data_by_date[date]
    }


@app.get("/point_forecast", response_model=PointForecastResponse)
def point_forecast(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    date: str = Query(...)
):

    if app.state.forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    if date not in app.state.data_by_date:
        raise HTTPException(status_code=404, detail="Date not found")

    daily_features = app.state.data_by_date[date]

    nearest_feature = min(
        daily_features,
        key=lambda feature: (
            (feature["geometry"]["coordinates"][1] - lat) ** 2 +
            (feature["geometry"]["coordinates"][0] - lon) ** 2
        )
    )

    props = nearest_feature["properties"]
    grid_lon, grid_lat = nearest_feature["geometry"]["coordinates"]

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


def _nearest_neighbour_order(coords: np.ndarray) -> List[int]:
    n = len(coords)
    if n == 0:
        return []

    visited = [False] * n
    order = []
    current = 0

    for _ in range(n):
        visited[current] = True
        order.append(current)

        best_dist = float("inf")
        best_next = -1

        for j in range(n):
            if not visited[j]:
                dx = coords[current][0] - coords[j][0]
                dy = coords[current][1] - coords[j][1]
                dist = dx * dx + dy * dy
                if dist < best_dist:
                    best_dist = dist
                    best_next = j

        if best_next == -1:
            break
        current = best_next

    return order

@app.get("/coastal_tile/{z}/{x}/{y}.png")
def coastal_tile(
    z: int,
    x: int,
    y: int,
    date: str = Query(...)
):
    if app.state.forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    if date not in app.state.data_by_date:
        raise HTTPException(status_code=404, detail="Date not found")

    features = app.state.data_by_date[date]

    # Tile bounds
    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    west, south, east, north = bounds

    # Add margin so lines near tile edges are not clipped
    margin = (east - west) * 0.1
    west_m, east_m = west - margin, east + margin
    south_m, north_m = south - margin, north + margin

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Group by island and order points
    islands: Dict[str, List[Dict]] = {}
    for feature in features:
        island = feature["properties"].get("island", "Unknown")
        islands.setdefault(island, []).append(feature)

    # Find global heat range for normalization
    all_heats = [f["properties"]["heat"] for f in features]
    min_heat = min(all_heats)
    max_heat = max(all_heats)
    heat_range = max(max_heat - min_heat, 0.001)

    def heat_to_rgba(heat: float):
        t = float(np.clip((heat - min_heat) / heat_range, 0.0, 1.0))
        if t < 0.5:
            s = t / 0.5
            return (int(255 * s), int(255 * s), 255, 220)
        else:
            s = (t - 0.5) / 0.5
            return (255, int(255 * (1 - s)), 0, 220)

    def geo_to_px(lon, lat):
        px = (lon - west) / (east - west) * size
        py = (north - lat) / (north - south) * size
        return (px, py)

    for island_name, island_features in islands.items():
        coords = np.array([f["geometry"]["coordinates"] for f in island_features])
        ordered_indices = _nearest_neighbour_order(coords)

        for i in range(len(ordered_indices) - 1):
            idx_a = ordered_indices[i]
            idx_b = ordered_indices[i + 1]

            lon_a, lat_a = island_features[idx_a]["geometry"]["coordinates"]
            lon_b, lat_b = island_features[idx_b]["geometry"]["coordinates"]

            # Skip segments completely outside tile (with margin)
            if (max(lon_a, lon_b) < west_m or min(lon_a, lon_b) > east_m or
                    max(lat_a, lat_b) < south_m or min(lat_a, lat_b) > north_m):
                continue

            heat_a = island_features[idx_a]["properties"]["heat"]
            heat_b = island_features[idx_b]["properties"]["heat"]
            avg_heat = (heat_a + heat_b) / 2
            color = heat_to_rgba(avg_heat)

            px_a = geo_to_px(lon_a, lat_a)
            px_b = geo_to_px(lon_b, lat_b)

            draw.line([px_a, px_b], fill=color, width=3)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")