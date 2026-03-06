from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any
from contextlib import asynccontextmanager
import json
import os
import logging
import math
import numpy as np
import mercantile
from PIL import Image, ImageDraw
from io import BytesIO
from scipy.ndimage import gaussian_filter
import matplotlib.cm as cm
import unicodedata

logging.basicConfig(level=logging.INFO)

FILE_PATH = "latest_forecast.json"

# Tile ranges covering the Balearic Islands per zoom level
TILE_RANGES = {
    6: {"x": range(32, 34), "y": range(24, 26)},
    7: {"x": range(65, 69), "y": range(48, 51)},
    8: {"x": range(130, 137), "y": range(97, 102)},
    9: {"x": range(261, 273), "y": range(194, 204)},
}

# Coastal towns and locations in the Balearic Islands
BALEARIC_LOCATIONS = [
    # Mallorca
    {"name": "Palma", "island": "Mallorca", "lat": 39.5696, "lon": 2.6502},
    {"name": "Alcudia", "island": "Mallorca", "lat": 39.8527, "lon": 3.1211},
    {"name": "Port d'Alcudia", "island": "Mallorca", "lat": 39.8419, "lon": 3.1286},
    {"name": "Can Picafort", "island": "Mallorca", "lat": 39.7627, "lon": 3.1575},
    {"name": "Port de Pollença", "island": "Mallorca", "lat": 39.9058, "lon": 3.0814},
    {"name": "Pollença", "island": "Mallorca", "lat": 39.8771, "lon": 3.0142},
    {"name": "Cala Sant Vicenç", "island": "Mallorca", "lat": 39.9217, "lon": 3.0603},
    {"name": "Formentor", "island": "Mallorca", "lat": 39.9594, "lon": 3.1289},
    {"name": "Cala Millor", "island": "Mallorca", "lat": 39.5908, "lon": 3.3897},
    {"name": "Cala Bona", "island": "Mallorca", "lat": 39.6019, "lon": 3.3961},
    {"name": "Porto Cristo", "island": "Mallorca", "lat": 39.5378, "lon": 3.3344},
    {"name": "Cales de Mallorca", "island": "Mallorca", "lat": 39.4767, "lon": 3.3733},
    {"name": "Portocolom", "island": "Mallorca", "lat": 39.5178, "lon": 3.3264},
    {"name": "Cala d'Or", "island": "Mallorca", "lat": 39.3783, "lon": 3.2292},
    {"name": "Portopetro", "island": "Mallorca", "lat": 39.3608, "lon": 3.2244},
    {"name": "Cala Figuera", "island": "Mallorca", "lat": 39.3281, "lon": 3.1644},
    {"name": "Colònia de Sant Jordi", "island": "Mallorca", "lat": 39.3122, "lon": 2.9844},
    {"name": "Sa Ràpita", "island": "Mallorca", "lat": 39.3597, "lon": 2.9578},
    {"name": "s'Arenal", "island": "Mallorca", "lat": 39.4958, "lon": 2.7481},
    {"name": "Magaluf", "island": "Mallorca", "lat": 39.5069, "lon": 2.5486},
    {"name": "Santa Ponça", "island": "Mallorca", "lat": 39.5108, "lon": 2.4803},
    {"name": "Peguera", "island": "Mallorca", "lat": 39.5358, "lon": 2.4469},
    {"name": "Camp de Mar", "island": "Mallorca", "lat": 39.5367, "lon": 2.3969},
    {"name": "Port d'Andratx", "island": "Mallorca", "lat": 39.5392, "lon": 2.3847},
    {"name": "Sant Elm", "island": "Mallorca", "lat": 39.5803, "lon": 2.3506},
    {"name": "Sóller", "island": "Mallorca", "lat": 39.7656, "lon": 2.7153},
    {"name": "Port de Sóller", "island": "Mallorca", "lat": 39.7956, "lon": 2.6958},
    {"name": "Deià", "island": "Mallorca", "lat": 39.7478, "lon": 2.6489},
    {"name": "Valldemossa", "island": "Mallorca", "lat": 39.7128, "lon": 2.6247},
    {"name": "Banyalbufar", "island": "Mallorca", "lat": 39.6894, "lon": 2.5158},
    {"name": "Cala Rajada", "island": "Mallorca", "lat": 39.7081, "lon": 3.4586},
    {"name": "Capdepera", "island": "Mallorca", "lat": 39.7033, "lon": 3.4281},
    {"name": "Artà", "island": "Mallorca", "lat": 39.6983, "lon": 3.3500},
    {"name": "Colònia de Sant Pere", "island": "Mallorca", "lat": 39.7272, "lon": 3.2275},
    {"name": "S'Illot", "island": "Mallorca", "lat": 39.5697, "lon": 3.3633},
    {"name": "Cala Ratjada", "island": "Mallorca", "lat": 39.7081, "lon": 3.4586},
    # Menorca
    {"name": "Maó", "island": "Menorca", "lat": 39.8885, "lon": 4.2656},
    {"name": "Ciutadella", "island": "Menorca", "lat": 39.9994, "lon": 3.8369},
    {"name": "Fornells", "island": "Menorca", "lat": 40.0603, "lon": 4.1319},
    {"name": "Es Mercadal", "island": "Menorca", "lat": 39.9942, "lon": 4.0711},
    {"name": "Arenal d'en Castell", "island": "Menorca", "lat": 40.0514, "lon": 4.1697},
    {"name": "Son Parc", "island": "Menorca", "lat": 40.0458, "lon": 4.1369},
    {"name": "Cala en Porter", "island": "Menorca", "lat": 39.8500, "lon": 4.1294},
    {"name": "Cala Galdana", "island": "Menorca", "lat": 39.9317, "lon": 3.9583},
    {"name": "Son Bou", "island": "Menorca", "lat": 39.8825, "lon": 4.0533},
    {"name": "Binibèquer", "island": "Menorca", "lat": 39.8300, "lon": 4.2536},
    {"name": "Punta Prima", "island": "Menorca", "lat": 39.8194, "lon": 4.2647},
    {"name": "Cala en Bosc", "island": "Menorca", "lat": 39.9514, "lon": 3.8286},
    {"name": "Cala Blanca", "island": "Menorca", "lat": 39.9636, "lon": 3.8397},
    {"name": "Es Castell", "island": "Menorca", "lat": 39.8711, "lon": 4.2789},
    {"name": "Sant Lluís", "island": "Menorca", "lat": 39.8472, "lon": 4.2561},
    # Ibiza
    {"name": "Eivissa", "island": "Ibiza", "lat": 38.9081, "lon": 1.4320},
    {"name": "Sant Antoni de Portmany", "island": "Ibiza", "lat": 38.9800, "lon": 1.3011},
    {"name": "Santa Eulària des Riu", "island": "Ibiza", "lat": 38.9842, "lon": 1.5358},
    {"name": "Portinatx", "island": "Ibiza", "lat": 39.0744, "lon": 1.5297},
    {"name": "Port de Sant Miquel", "island": "Ibiza", "lat": 39.0733, "lon": 1.4444},
    {"name": "Cala de Sant Vicent", "island": "Ibiza", "lat": 39.0578, "lon": 1.5369},
    {"name": "Es Canar", "island": "Ibiza", "lat": 39.0153, "lon": 1.5756},
    {"name": "Cala Llonga", "island": "Ibiza", "lat": 38.9603, "lon": 1.5514},
    {"name": "Talamanca", "island": "Ibiza", "lat": 38.9192, "lon": 1.4533},
    {"name": "Playa d'en Bossa", "island": "Ibiza", "lat": 38.8858, "lon": 1.4072},
    {"name": "Ses Salines", "island": "Ibiza", "lat": 38.8717, "lon": 1.4136},
    {"name": "Cala Tarida", "island": "Ibiza", "lat": 38.9594, "lon": 1.2742},
    {"name": "Cala Vedella", "island": "Ibiza", "lat": 38.9119, "lon": 1.2267},
    {"name": "Cala d'Hort", "island": "Ibiza", "lat": 38.8819, "lon": 1.2297},
    {"name": "Sant Josep de sa Talaia", "island": "Ibiza", "lat": 38.9194, "lon": 1.3056},
    # Formentera
    {"name": "Sant Francesc Xavier", "island": "Formentera", "lat": 38.7019, "lon": 1.4325},
    {"name": "Es Pujols", "island": "Formentera", "lat": 38.7253, "lon": 1.4747},
    {"name": "La Savina", "island": "Formentera", "lat": 38.7342, "lon": 1.4156},
    {"name": "Cala Saona", "island": "Formentera", "lat": 38.6933, "lon": 1.3897},
    {"name": "Es Caló", "island": "Formentera", "lat": 38.7211, "lon": 1.5322},
    {"name": "La Mola", "island": "Formentera", "lat": 38.6708, "lon": 1.5833},
]


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
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
            r = int(32 + (255 - 32) * s)
            g = int(178 + (193 - 178) * s)
            b = int(171 + (7 - 171) * s)
        else:
            s = (t - 0.5) / 0.5
            r = int(255 + (229 - 255) * s)
            g = int(193 + (57 - 193) * s)
            b = int(7 + (53 - 7) * s)
        return (r, g, b, 220)

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
        app.state.searchable_locations = []
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

    # Pre-filter searchable locations to only those within 1km of a prediction point
    first_date = app.state.available_dates[0]
    first_features = data_by_date[first_date]
    pred_coords = [
        (f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0])
        for f in first_features
    ]
    searchable = []
    for loc in BALEARIC_LOCATIONS:
        min_dist = min(
            _haversine_km(loc["lat"], loc["lon"], plat, plon)
            for plat, plon in pred_coords
        )
        if min_dist <= 1.0:
            searchable.append({**loc, "min_dist_km": round(min_dist, 3)})
    app.state.searchable_locations = searchable
    logging.info(f"{len(searchable)} locations within 1km of prediction points")

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


@app.get("/search_locations")
def search_locations(query: str = Query(..., min_length=1)):
    if app.state.forecast_data is None:
        raise HTTPException(status_code=503, detail="Forecast not loaded")

    def _normalize(text: str) -> str:
        return unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode("ascii")

    query_norm = _normalize(query.strip())
    results = [
        {
            "name": loc["name"],
            "island": loc["island"],
            "lat": loc["lat"],
            "lon": loc["lon"],
        }
        for loc in app.state.searchable_locations
        if query_norm in _normalize(loc["name"])
    ]
    return {"query": query, "results": results[:10]}


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
            r = int(32 + (255 - 32) * s)
            g = int(178 + (193 - 178) * s)
            b = int(171 + (7 - 171) * s)
        else:
            s = (t - 0.5) / 0.5
            r = int(255 + (229 - 255) * s)
            g = int(193 + (57 - 193) * s)
            b = int(7 + (53 - 7) * s)
        draw.line([(px, 0), (px, height)], fill=(r, g, b, 220))

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")