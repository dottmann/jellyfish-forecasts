from fastapi import FastAPI
import json
import os

app = FastAPI()

@app.get("/jellyfish_forecast")
def get_forecast():
    file_path = "latest_forecast.json"

    if not os.path.exists(file_path):
        return {"error": "Forecast file not found"}

    with open(file_path) as f:
        data = json.load(f)

    return data