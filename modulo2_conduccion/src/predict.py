"""Inferencia para la app web: probabilidades, decisión de alerta y Grad-CAM.

La app solo pasa la imagen tal como la sube el usuario (PIL.Image); aquí se corrige
la orientación EXIF, se convierte a RGB, se redimensiona, se recorta y se normaliza
igual que en la evaluación. Lee models/metadata.json y configs/alertas.yaml, que
escribe evaluate_cnn.py. Solo depende de torch, torchvision, numpy, Pillow, PyYAML
y matplotlib (para colorear el mapa).

Ejemplo:
    from PIL import Image
    from modulo2_conduccion.src import predict
    imagen = Image.open("modulo2_conduccion/data/processed/ejemplos/texting_phone_1.jpg")
    predict.predecir(imagen)            # {'safe_driving': 0.03, ..., 'texting_phone': 0.91, ...}
    predict.decision_alerta(imagen)     # {'alerta': True, 'puntaje': 0.95, 'umbral': 0.42, ...}
    mapa = predict.gradcam(imagen)      # matriz H × W en [0, 1], del tamaño de la imagen
    predict.superponer(imagen, mapa)    # PIL.Image con el mapa encima
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F
import yaml

from .modelos import GradCAM, build_model, test_transform

ROOT = Path(__file__).resolve().parents[1]
_BASE = [ROOT]


def configurar(base):
    """Cambia la carpeta de donde se leen models/ y configs/ (p. ej. la salida de la demo)."""
    _BASE[0] = Path(base)
    _cargar.cache_clear()


@lru_cache(maxsize=1)
def _cargar():
    base = _BASE[0]
    metadata = json.loads((base / "models/metadata.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(base / "models" / metadata["archivo"], map_location="cpu", weights_only=True)
    if checkpoint["classes"] != metadata["clases"]:
        raise ValueError("El orden de clases del modelo no coincide con metadata.json")
    model = build_model(checkpoint["params"], pretrained=False)   # sin descargar pesos de ImageNet
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    alertas = yaml.safe_load((base / "configs/alertas.yaml").read_text(encoding="utf-8"))
    return model, metadata, alertas, test_transform(metadata["tamano_entrada"][0])


def _preparar(imagen):
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    model, metadata, _, transform = _cargar()
    return imagen, transform(imagen)[None], model, metadata


def predecir(imagen: Image.Image) -> dict[str, float]:
    """Devuelve la probabilidad de cada clase (suman 1), en el orden de metadata["clases"]."""
    _, x, model, metadata = _preparar(imagen)
    with torch.inference_mode():
        probs = torch.softmax(model(x), 1)[0].tolist()
    return dict(zip(metadata["clases"], probs))


def decision_alerta(imagen_o_probs) -> dict:
    """Aplica la política de configs/alertas.yaml. Acepta la imagen o el resultado
    de predecir(). Alerta si P(clases con alerta) ≥ umbral recomendado."""
    probs = imagen_o_probs if isinstance(imagen_o_probs, dict) else predecir(imagen_o_probs)
    alertas = _cargar()[2]
    puntaje = float(sum(probs[c] for c in alertas["clases_con_alerta"]))
    return {"alerta": puntaje >= alertas["umbral_recomendado"], "puntaje": puntaje,
            "umbral": alertas["umbral_recomendado"], "clase": max(probs, key=probs.get),
            "clases_con_alerta": alertas["clases_con_alerta"]}


def gradcam(imagen: Image.Image, clase: int | None = None) -> np.ndarray:
    """Mapa Grad-CAM (H × W de la imagen original, valores en [0, 1]) para `clase`
    (índice en metadata["clases"]) o, si es None, para la clase predicha.
    El modelo solo ve el recorte cuadrado central; fuera de él el mapa vale 0."""
    imagen, x, model, metadata = _preparar(imagen)
    cam = GradCAM(model)
    try:
        target = None if clase is None else torch.tensor([clase])
        mapa = cam(x, target)[0]
    finally:
        cam.remove()
    # Llevar el mapa del recorte (size × size) a las coordenadas de la imagen original.
    w, h = imagen.size
    size, short = metadata["tamano_entrada"][0], metadata["redimension_lado_corto"]
    escala = min(w, h) / short
    lado = max(1, round(size * escala))
    ancho_red, alto_red = (short, int(short * h / w)) if w <= h else (int(short * w / h), short)
    x0 = round(int(round((ancho_red - size) / 2.0)) * escala)
    y0 = round(int(round((alto_red - size) / 2.0)) * escala)
    grande = F.interpolate(torch.from_numpy(mapa)[None, None], size=(lado, lado), mode="bilinear",
                           align_corners=False)[0, 0].numpy()
    salida = np.zeros((h, w), dtype=np.float32)
    salida[y0:y0 + lado, x0:x0 + lado] = grande[:max(0, min(lado, h - y0)), :max(0, min(lado, w - x0))]
    return salida


def superponer(imagen: Image.Image, mapa: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Imagen con el mapa de calor (colormap «jet») superpuesto."""
    from matplotlib import colormaps
    imagen = ImageOps.exif_transpose(imagen).convert("RGB")
    color = (colormaps["jet"](mapa)[..., :3] * 255).astype(np.uint8)
    peso = (alpha * (mapa > 0))[..., None]               # fuera del recorte se ve la foto original
    mezcla = np.asarray(imagen, dtype=np.float32) * (1 - peso) + color * peso
    return Image.fromarray(mezcla.astype(np.uint8))


def analizar(imagen: Image.Image) -> dict:
    """Todo lo que muestra la app en una sola llamada."""
    probs = predecir(imagen)
    mapa = gradcam(imagen)
    return {"probabilidades": probs, **decision_alerta(probs), "gradcam": mapa,
            "superpuesta": superponer(imagen, mapa)}