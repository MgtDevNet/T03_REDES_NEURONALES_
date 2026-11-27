"""Utilidades compartidas de la app: rutas del repositorio y carga de los módulos.

La app solo importa las funciones de inferencia de cada módulo (`moduloX/src/predict.py`), como
dice app/README.md. Si un módulo todavía no tiene su predict.py, su página lo indica en lugar de
fallar, y se activa sola cuando el archivo aparezca.
"""

import importlib
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# Enlaces que pide el enunciado. Completar cuando existan.
ENLACES = {
    "Aplicación web": "https://travesia-t03.streamlit.app",
    "Repositorio": "https://github.com/MgtDevNet/T03_REDES_NEURONALES_",
    "Reporte técnico": None,
    "Video promocional": None,
    "Videos de contribución individual": None,
}

MODULOS = {
    1: {"carpeta": "modulo1_demanda", "funciones": ["predecir"], "titulo": "Demanda y operación"},
    2: {"carpeta": "modulo2_conduccion", "funciones": ["predecir", "gradcam"], "titulo": "Conducción distractiva"},
    3: {"carpeta": "modulo3_recomendacion", "funciones": ["recomendar_usuario", "recomendar_perfil_nuevo", "historial"],
        "titulo": "Recomendación de destinos"},
}


def estado_modulo(n):
    """(listo, mensaje): si el predict.py del módulo existe y tiene las funciones del contrato."""
    m = MODULOS[n]
    ruta = RAIZ / m["carpeta"] / "src" / "predict.py"
    if not ruta.exists():
        return False, f"Falta `{m['carpeta']}/src/predict.py`."
    try:
        mod = importlib.import_module(f"{m['carpeta']}.src.predict")
    except Exception as e:  # el archivo existe pero no se puede importar
        return False, f"`{m['carpeta']}/src/predict.py` existe pero no se pudo importar: {e}"
    faltan = [f for f in m["funciones"] if not hasattr(mod, f)]
    if faltan:
        return False, f"A `{m['carpeta']}/src/predict.py` le faltan las funciones: {', '.join(faltan)}."
    return True, "Listo"


def cargar_modulo(n):
    return importlib.import_module(f"{MODULOS[n]['carpeta']}.src.predict")
