"""FastAPI inference: load the checkpoint once, on the first uploaded image."""

import io
import json
import logging
import os
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from mole_classificator.mole_classificator import Classificator
from mole_classificator.preprocessing import ImagePreprocessor

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def preprocess(raw_bytes, transform=None):
    try:
        with Image.open(io.BytesIO(raw_bytes)) as image:
            return (transform or ImagePreprocessor())(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("The file could not be decoded as an image.") from exc


def get_predictor(app):
    with app.state.model_lock:
        if app.state.model is None:
            checkpoint = app.state.checkpoint
            if not checkpoint.is_file():
                raise HTTPException(503, f"Checkpoint not found: {checkpoint.name}. Train the model first.")
            log.info("Loading checkpoint %s", checkpoint)
            model = Classificator.load_from_checkpoint(str(checkpoint), map_location="cpu", pretrained=False)
            model.eval()
            class_names = model.hparams.get("class_names")
            if class_names is None:
                try:
                    class_names = json.loads(app.state.class_names_path.read_text(encoding="utf-8"))
                except FileNotFoundError as exc:
                    raise HTTPException(503, "class_names.json is missing. Download it together with the checkpoint.") from exc
            if len(class_names) != model.num_classes:
                raise HTTPException(503, "The number of classes does not match the checkpoint.")
            app.state.preprocessor = ImagePreprocessor(
                max_size=model.hparams.max_image_size, normalize=model.hparams.normalize,
            )
            app.state.class_names = class_names
            app.state.model = model
        return app.state.model, app.state.class_names, app.state.preprocessor


class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


def create_app(checkpoint_path=None, class_names_path=None):
    app = FastAPI()
    app.state.checkpoint = Path(checkpoint_path or os.environ.get("MODEL_CHECKPOINT_PATH", ROOT / "model.ckpt"))
    app.state.class_names_path = Path(class_names_path or os.environ.get("CLASS_NAMES_PATH", ROOT / "class_names.json"))
    app.state.model = None
    app.state.model_lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: Request, file: UploadFile = File(...)):
        raw_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Image is too large (maximum 10 MiB).")
        # Reject invalid uploads before loading a potentially expensive checkpoint.
        try:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise HTTPException(400, "The file could not be decoded as an image.") from exc
        model, class_names, transform = get_predictor(request.app)
        try:
            input_tensor = preprocess(raw_bytes, transform)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        with torch.inference_mode():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)
        predictions = [
            Prediction(label=label, probability=float(probabilities[index]))
            for index, label in enumerate(class_names)
        ]
        predictions.sort(key=lambda prediction: prediction.probability, reverse=True)
        log.info("Classified one uploaded image: input=%s", tuple(input_tensor.shape))
        return PredictResponse(predictions=predictions)

    return app


app = create_app()
