from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import utils


class DetectionRequest(BaseModel):
    text: str
    modelId: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    utils.init()
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
        return utils.detection(request.text, request.modelId)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/models")
async def get_models():
    # Strip internal fields before exposing metadata to clients.
    public_models = [
        {k: v for k, v in md.items() if k != "filename"}
        for md in utils.models_dict.values()
    ]
    return {"models": public_models}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
