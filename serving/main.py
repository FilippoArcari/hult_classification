"""
Servizio di inferenza FastAPI per il classificatore MILK10k.

Nessuna persistenza: l'immagine caricata viene decodificata, trasformata e
scartata all'interno della singola richiesta. Nessun CORS middleware qui:
il frontend Next.js proxa /api/* verso questo servizio via next.config.js.

Avvio locale: uvicorn main:app --reload --port 8000
"""

import io
import json
import os
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

# TODO: sostituisci con l'import reale della tua LightningModule
# (es. from src.models.lightning_module import Milk10kLightningModule)
from src.models.lightning_module import Milk10kLightningModule

CHECKPOINT_PATH = os.environ.get("MODEL_CHECKPOINT_PATH", "checkpoints/best.ckpt")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "checkpoints/class_names.json")

# Dimensione nativa MILK10k: vincolata dall'architettura (num_tokens=252
# richiede esattamente questa griglia spaziale dopo il downsampling /32
# di ConvNeXt-Tiny). Non è un parametro libero: cambiarlo rompe il modello.
TARGET_WIDTH = 600
TARGET_HEIGHT = 450

# TODO: verifica che questi valori coincidano ESATTAMENTE con la
# normalizzazione usata nella pipeline di training (config Hydra / transform
# del DataModule). Un mismatch qui non genera errori, solo probabilità
# sbagliate in modo silenzioso — è il tipo di bug più insidioso da avere.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

_to_tensor_and_normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
)


def resize_and_center_crop(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize preservando l'aspect ratio (il lato che eccede viene coperto
    per intero) e poi center-crop esatto a (target_w, target_h).

    Nota di design: il training non applica resize (MILK10k è nativamente
    600x450), quindi questo step introduce uno shift train/inference per
    qualunque immagine caricata dall'utente che non sia già in quel formato
    esatto. È un compromesso necessario, non un'approssimazione innocua.
    """
    src_w, src_h = image.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def preprocess(raw_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("File non decodificabile come immagine") from exc

    image = resize_and_center_crop(image, TARGET_WIDTH, TARGET_HEIGHT)
    tensor = _to_tensor_and_normalize(image)
    return tensor.unsqueeze(0)  # batch dim: (1, 3, 450, 600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = Milk10kLightningModule.load_from_checkpoint(
        CHECKPOINT_PATH, map_location="cpu"
    )
    model.eval()

    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)

    app.state.model = model
    app.state.class_names = class_names
    yield
    # nessuno shutdown esplicito necessario: i tensori CPU vengono
    # rilasciati dal garbage collector col processo


app = FastAPI(lifespan=lifespan)


class Prediction(BaseModel):
    label: str
    probability: float


class PredictResponse(BaseModel):
    predictions: list[Prediction]


# Endpoint sincrono (non async def): FastAPI/Starlette esegue automaticamente
# le route `def` in un threadpool separato. Il forward pass di un
# ConvNeXt-Tiny + Transformer head su CPU è un carico sincrono e bloccante
# (nessun await al suo interno); se questa fosse `async def`, bloccherebbe
# l'intero event loop e con essa ogni altra richiesta concorrente — per una
# demo single-user locale non cambia nulla in pratica, ma è la ragione
# tecnica per cui la firma è `def`, non una svista.
@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)) -> PredictResponse:
    raw_bytes = file.file.read()

    try:
        input_tensor = preprocess(raw_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = app.state.model
    class_names = app.state.class_names

    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=-1).squeeze(0)

    predictions = [
        Prediction(label=class_names[i], probability=float(probabilities[i]))
        for i in range(len(class_names))
    ]
    predictions.sort(key=lambda p: p.probability, reverse=True)

    return PredictResponse(predictions=predictions)
