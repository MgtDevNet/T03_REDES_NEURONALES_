"""Funciones de inferencia del módulo 3 para la aplicación web.

Usan el modelo GUARDADO (red de dos torres, `models/dos_torres.pt`); no reentrenan nada.
Siguen la decisión 1: el modelo puntúa los 1000 IDs y al cliente se le muestran LUGARES (el mejor
puntaje de cada lugar). Siguen el caso de uso: K = 3 lugares por defecto y los lugares ya
visitados no se excluyen, se marcan como «Ya lo visitaste, ¿vuelves?».

    recomendar_usuario(user_id, k=3)          → lugares recomendados a un usuario existente
    recomendar_perfil_nuevo(preferencias, k=3) → lugares para un perfil nuevo (sin ID, sin historial)
    historial(user_id)                        → interacciones del usuario (reseñas y visitas)

Ejemplo:
    from modulo3_recomendacion.src.predict import recomendar_perfil_nuevo
    recomendar_perfil_nuevo({"preferencias": ["Beach", "Historical"]})

Solo lee archivos versionados (`data/processed/` y `models/`): funciona en un clon limpio.
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:  # para que funcione importado desde la raíz del repo (la app)
    sys.path.insert(0, str(SRC))

from dos_torres import TIPOS, DosTorres, RedDosTorres, variables_usuario  # noqa: E402

MODULO = SRC.parent
PROCESADOS = MODULO / "data" / "processed"
MODELOS = MODULO / "models"
N_LUGARES = 5
# Las etiquetas del formulario pueden venir como en los datos («Beaches») o como tipo («Beach»).
SINONIMOS = {"Beaches": "Beach", "Beach": "Beach", "Historical": "Historical", "City": "City",
             "Nature": "Nature", "Adventure": "Adventure"}


@lru_cache(maxsize=1)
def _recursos():
    """Carga una sola vez el modelo, el catálogo y las interacciones."""
    meta = json.loads((MODELOS / "dos_torres.json").read_text(encoding="utf-8"))
    red = DosTorres(d=meta["d"], perdida=meta["perdida"], negativos=meta["negativos"],
                    usuario=meta["usuario"], oculta=meta["oculta"])
    red.red = RedDosTorres(red.X_usuario.shape[1], meta["d"], meta["oculta"])
    red.red.load_state_dict(torch.load(MODELOS / "dos_torres.pt"))
    red.red.eval()
    destinos = pd.read_csv(PROCESADOS / "destinos.csv").set_index("DestinationID").sort_index()
    usuarios = pd.read_csv(PROCESADOS / "usuarios.csv").set_index("UserID")
    inter = pd.read_csv(PROCESADOS / "interacciones.csv")
    return red, meta, destinos, usuarios, inter


def _validar_k(k):
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= N_LUGARES:
        raise ValueError(f"k debe ser un entero entre 1 y {N_LUGARES} (hay {N_LUGARES} lugares); se recibió {k!r}")


def _lugares(puntajes, ids_vistos, k):
    """De los puntajes de los 1000 IDs a una tabla de k lugares (mejor ID de cada lugar)."""
    _, _, destinos, _, _ = _recursos()
    p = pd.Series(puntajes, index=destinos.index)
    if ids_vistos:
        p = p.drop(index=list(ids_vistos), errors="ignore")  # no repetir el MISMO ID
    mejor_id = p.groupby(destinos.Name.reindex(p.index)).idxmax()
    lugares_vistos = set(destinos.loc[list(ids_vistos), "Name"]) if ids_vistos else set()
    filas = []
    for lugar, i in mejor_id.items():
        d = destinos.loc[i]
        filas.append({"lugar": lugar, "tipo": d.Type, "estado": d.State, "mejor_epoca": d.BestTimeToVisit,
                      "destination_id": int(i), "puntaje": float(p[i]),
                      "ya_visitado": lugar in lugares_vistos,
                      "mensaje": "Ya lo visitaste, ¿vuelves?" if lugar in lugares_vistos else ""})
    t = pd.DataFrame(filas).sort_values("puntaje", ascending=False).head(k).reset_index(drop=True)
    t.insert(0, "posicion", range(1, len(t) + 1))
    return t


def historial(user_id) -> pd.DataFrame:
    """Interacciones del usuario (reseñas y visitas), con el lugar y la calificación."""
    _, _, destinos, usuarios, inter = _recursos()
    if user_id not in usuarios.index:
        raise ValueError(f"No existe el usuario {user_id!r} (los IDs van de 1 a {usuarios.index.max()})")
    h = inter[inter.UserID == user_id].merge(destinos[["Type", "State"]], left_on="DestinationID", right_index=True)
    return h.rename(columns={"Type": "tipo", "State": "estado"})[
        ["DestinationID", "lugar", "tipo", "estado", "fuente", "calificacion"]].reset_index(drop=True)


def recomendar_usuario(user_id, k: int = 3) -> pd.DataFrame:
    """Lugares recomendados a un usuario existente. No repite los IDs exactos de su historial,
    pero sí puede recomendar un lugar ya visitado (marcado «¿vuelves?»)."""
    _validar_k(k)
    red, _, _, _, _ = _recursos()
    vistos = set(historial(user_id).DestinationID)
    return _lugares(red.puntuar([user_id])[0], vistos, k)


def recomendar_perfil_nuevo(preferencias: dict, k: int = 3) -> pd.DataFrame:
    """Lugares para un cliente nuevo, sin ID ni historial (el perfil no se guarda).

    preferencias: {"preferencias": ["Beach", "Historical"],      # obligatorio, ≥ 1 tipo
                   "genero": "Female", "adultos": 2, "ninos": 0}  # opcionales
    El modelo guardado solo usa las preferencias (fue lo único que ayudó, sección 7.4 del
    informe); los demás campos se aceptan y se validan, pero no cambian el resultado.
    """
    _validar_k(k)
    red, meta, _, _, _ = _recursos()
    elegidas = preferencias.get("preferencias", [])
    if isinstance(elegidas, str):
        elegidas = [x.strip() for x in elegidas.split(",")]
    desconocidas = [x for x in elegidas if x not in SINONIMOS]
    if not elegidas or desconocidas:
        raise ValueError(f"Hay que elegir al menos una preferencia entre {TIPOS}; desconocidas: {desconocidas}")
    tipos = {SINONIMOS[x] for x in elegidas}
    perfil = {f"pref_{t}": int(t in tipos) for t in TIPOS}
    perfil.update({"Gender": preferencias.get("genero", "Female"), "NumberOfAdults": preferencias.get("adultos", 1),
                   "NumberOfChildren": preferencias.get("ninos", 0)})
    x = variables_usuario(pd.DataFrame([perfil]), meta["usuario"])
    return _lugares(red.puntuar_variables(x)[0], set(), k)
