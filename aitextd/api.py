"""FastAPI application for the Persian AI-Text Detection API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from aitextd.store import ModelStore

store = ModelStore()


class DetectionRequest(BaseModel):
    text: str
    modelId: str


class DetailsRequest(BaseModel):
    text: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init()
    yield


app = FastAPI(
    title="Persian AI Text Detection API",
    description="Detect whether Persian text is AI-generated or human-written.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/detect")
async def detect(request: DetectionRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="متن نمی‌تواند خالی باشد.")
    try:
        return store.detection(request.text, request.modelId)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/details")
async def details(request: DetailsRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="متن نمی‌تواند خالی باشد.")
    return {"details": store.detail_breakdown(request.text)}


_INTERNAL_FIELDS = {"filename", "runtime", "vectorizer"}


@app.get("/models")
async def get_models():
    # Strip internal/implementation fields before exposing metadata to clients.
    public_models = [
        {k: v for k, v in md.items() if k not in _INTERNAL_FIELDS}
        for md in store.models_dict.values()
    ]
    # The ensemble is a virtual "model" (runs every real model under the hood),
    # listed first so it's the default choice.
    return {"models": [store.get_ensemble_metadata(), *public_models]}
