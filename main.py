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

# Tile ranges covering the Balearic Islands per zoom level
TILE_RANGES = {
    6: {"x": range(32, 34), "y": range(24, 26)},
    7: {"x": range(65, 69), "y": range(48, 51)},
    8: {"x": range(130, 137), "y": range(97, 102)},
    9: {"x": range(261, 273), "y": range(194, 204)},
}


# -------------------------------------------------
# Helper
# -------------------------------------------------
def _render_tile(
    z: int,
    x: int,
    y: int,
    date: str,
    ordered_by_date: Dict,
    data_by_date: Dict
) -> bytes:
    features = data_by_date[date]

    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    west, south, east, north = bounds

    margin = (east - west) * 0.1
    west_m, east_m = west - margin, east + margin
    south_m, north_m = south - margin, north + margin

    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    island_lines = ordered_by_date.get(date, [])

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

    for island_data in island_lines:
        island_points = island_data["points"]
        for i in range(len(island_points) - 1):
            lon_a, lat_a = island_points[i]["geometry"]["coordinates"]
            lon_b, lat_b = island_points[i + 1]["geometry"]["coordinates"]

            if (max(lon_a, lon_b) < west_m or min(lon_a, lon_b) > east_m or
                    max(lat_a, lat_b) < south_m or min(lat_a, lat_b) > north_m):
                continue

            heat_a = island_points[i]["properties"]["heat"]
            heat_b = island_points[i + 1]["properties"]["heat"]
            avg_heat = (heat_a + heat_b) / 2
            color = heat_to_rgba(avg_heat)

            px_a = geo_to_px(lon_a, lat_a)
            px_b = geo_to_px(lon_b, lat_b)
            draw.line([px_a, px_b], fill=color, width=6)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


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
        app.state.ordered_by_date = {}
        app.state.tile_cache = {}
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

    # Pre-compute ordered coastal lines per date using point_id
    ordered_by_date: Dict[str, List[Dict]] = {}
    for date, day_features in data_by_date.items():
        islands: Dict[str, List[Dict]] = {}
        for feature in day_features:
            island = feature["properties"].get("island", "Unknown")
            islands.setdefault(island, []).append(feature)

        island_lines = []
        for island_name, island_features in islands.items():
            ordered_points = sorted(
                island_features,
                key=lambda f: f["properties"]["point_id"]
            )
            island_lines.append({"island": island_name, "points": ordered_points})

        ordered_by_date[date] = island_lines

    app.state.ordered_by_date = ordered_by_date
    logging.info("Pre-computed coastal line ordering for all dates")

    # Pre-render all tiles for the Balearic Islands area
    tile_cache: Dict[str, bytes] = {}
    for date in data_by_date.keys():
        for z, ranges in TILE_RANGES.items():
            for x in ranges["x"]:
                for y in ranges["y"]:
                    cache_key = f"{date}/{z}/{x}/{y}"
                    tile_cache[cache_key] = _render_tile(
                        z, x, y, date, ordered_by_date, data_by_date
                    )
        logging.info(f"Pre-rendered tiles for date {date}")

    app.state.tile_cache = tile_cache
    total = len(tile_cache)
    logging.info(f"Tile cache ready: {total} tiles pre-rendered")

    logging.info(f"Loaded {len(forecast_data['features'])} grid points")
    logging.info(f"Available dates: {app.state.available_dates}")

    yield

    logging.info("Shutting down API")


# -------------------------------------------------
# Middleware
# -------------------------------------------------
app = FastAPI(
    title="Jellyfish Forecast API",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# -------------------------------------------------
# Response Models
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
        nearest_point={"grid_lat": grid_lat, "grid_lon": grid_lon},
        forecast_date=date,
        prob_0=props["prob_0"],
        prob_1=props["prob_1"],
        prob_2=props["prob_2"],
        heat=heat,
        risk_level=risk_level
    )


@app.get("/coastal_lines")
def coastal_lines(date: str = Query(...)):
    if app.state.forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    if date not in app.state.ordered_by_date:
        raise HTTPException(status_code=404, detail="Date not found")

    return {"date": date, "islands": app.state.ordered_by_date[date]}


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

    cache_key = f"{date}/{z}/{x}/{y}"
    tile_bytes = app.state.tile_cache.get(cache_key)

    if tile_bytes is not None:
        return Response(content=tile_bytes, media_type="image/png")

    # Fallback: render on the fly for tiles outside pre-rendered range
    tile_bytes = _render_tile(
        z, x, y, date, app.state.ordered_by_date, app.state.data_by_date
    )
    return Response(content=tile_bytes, media_type="image/png")


@app.get("/heatmap_tile/{z}/{x}/{y}.png")
def heatmap_tile(
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

    tile = mercantile.Tile(x=x, y=y, z=z)
    bounds = mercantile.bounds(tile)
    west, south, east, north = bounds

    size = 256
    heat_grid = np.zeros((size, size), dtype=np.float32)

    for feature in features:
        lon, lat = feature["geometry"]["coordinates"]
        heat = feature["properties"]["heat"]

        if west <= lon <= east and south <= lat <= north:
            px = int((lon - west) / (east - west) * size)
            py = int((north - lat) / (north - south) * size)
            if 0 <= px < size and 0 <= py < size:
                heat_grid[py, px] += heat

    if heat_grid.max() > 0:
        heat_grid /= heat_grid.max()

    heat_grid = gaussian_filter(heat_grid, sigma=5)

    colormap = cm.get_cmap("jet")
    rgba_img = colormap(heat_grid)

    img = (rgba_img * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)

    buffer = BytesIO()
    pil_img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.get("/legend.png")
def legend():
    width = 256
    height = 24
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    for px in range(width):
        t = px / (width - 1)
        if t < 0.5:
            s = t / 0.5
            color = (int(255 * s), int(255 * s), 255, 220)
        else:
            s = (t - 0.5) / 0.5
            color = (255, int(255 * (1 - s)), 0, 220)
        draw.line([(px, 0), (px, height)], fill=color)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")