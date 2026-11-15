from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from services.dataset_service import get_dataset_stats
from services.evaluation_service import get_evaluation
from services.model_service import ModelService
from services.prediction_service import parse_csv_signal

logging.basicConfig(level=logging.INFO)
BASE_DIR = Path(__file__).resolve().parent
model_service = ModelService(BASE_DIR / "models" / "binary_stratified")


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    model_service.load()
    yield


app = FastAPI(title="Heartbeat Classifier API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://heartbeat-classifier-2.onrender.com",
],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "models_available": model_service.available}


@app.get("/api/dataset/stats")
def dataset_stats() -> dict:
    return get_dataset_stats()


@app.get("/api/model/info")
def model_info() -> dict:
    return model_service.info()


@app.get("/api/evaluation/ensemble")
def evaluation() -> dict:
    return get_evaluation()


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in {"text/csv", "application/csv", "text/plain", None}:
        raise HTTPException(status_code=415, detail="Upload a CSV file.")
    try:
        signal = parse_csv_signal(await file.read())
        result = model_service.predict(signal)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"filename": file.filename, "signal": signal.tolist(), **result}