"""Dimensionamiento operativo de despachos, flota y conductores (funcion pura).

No depende de ningun modelo de pronostico: recibe una serie de demanda diaria
y un diccionario de parametros operativos (cargado desde configs/operacion.yaml).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "operacion.yaml"


def cargar_params_default() -> dict[str, Any]:
    """Devuelve parametros operativos por defecto (coinciden con configs/operacion.yaml)."""
    return {
        "capacidad_sillas": 40,
        "ocupacion_objetivo": 0.80,
        "dist_linea_recta_km": 120.0,
        "factor_sinuosidad_vial": 1.35,
        "velocidad_comercial_kmh": 55.0,
        "tiempo_terminal_horas": 1.0,
        "tiempo_ciclo_horas": None,  # Si es None, se estima desde dist_linea_recta_km
        "horas_operacion_dia": 16.0,
        "jornada_max_horas_dia": 8.0,
        "dias_trabajo_semana": 6.0,
        "factor_ausentismo": 0.08,
    }


def estimar_tiempo_ciclo_horas(
    dist_linea_recta_km: float,
    factor_sinuosidad_vial: float = 1.35,
    velocidad_comercial_kmh: float = 55.0,
    tiempo_terminal_horas: float = 1.0,
) -> float:
    """Estima el tiempo de ciclo ida y vuelta (horas) para un bus en la ruta."""
    dist_vial_km = max(float(dist_linea_recta_km), 10.0) * float(factor_sinuosidad_vial)
    tiempo_sentido = dist_vial_km / max(float(velocidad_comercial_kmh), 15.0)
    return round(2.0 * (tiempo_sentido + float(tiempo_terminal_horas)), 2)


def dimensionar(demanda: pd.Series, params: dict[str, Any]) -> pd.DataFrame:
    """Calcula despachos, vehiculos y conductores diarios a partir de la demanda estimada.

    Formulas:
    - capacidad_efectiva = capacidad_sillas * ocupacion_objetivo
    - despachos_dia      = ceil(demanda / capacidad_efectiva)
    - vueltas_dia        = max(0.5, horas_operacion_dia / tiempo_ciclo_horas)
                           (en rutas largas >16h un vehiculo realiza <1 vuelta/dia)
    - vehiculos_dia      = ceil(despachos_dia / vueltas_dia)
    - turnos_por_bus     = horas_operacion_dia / jornada_max_horas_dia
    - factor_reserva     = (7 / dias_trabajo_semana) * (1 + factor_ausentismo)
    - conductores_dia    = ceil(vehiculos_dia * turnos_por_bus * factor_reserva)
    """
    p = {**cargar_params_default(), **params}
    cap_ef = max(float(p["capacidad_sillas"]) * float(p["ocupacion_objetivo"]), 1.0)

    if p.get("tiempo_ciclo_horas") is not None and float(p["tiempo_ciclo_horas"]) > 0:
        t_ciclo = float(p["tiempo_ciclo_horas"])
    else:
        t_ciclo = estimar_tiempo_ciclo_horas(
            p["dist_linea_recta_km"],
            p["factor_sinuosidad_vial"],
            p["velocidad_comercial_kmh"],
            p["tiempo_terminal_horas"],
        )

    horas_op = max(float(p["horas_operacion_dia"]), 1.0)
    vueltas_dia = max(0.5, horas_op / t_ciclo)

    turnos_bus = horas_op / max(float(p["jornada_max_horas_dia"]), 1.0)
    factor_descanso = 7.0 / max(float(p["dias_trabajo_semana"]), 1.0)
    factor_reserva = factor_descanso * (1.0 + max(float(p["factor_ausentismo"]), 0.0))

    dem_vals = np.maximum(np.asarray(demanda, dtype=float), 0.0)
    despachos = np.ceil(dem_vals / cap_ef).astype(int)
    vehiculos = np.ceil(despachos / vueltas_dia).astype(int)
    conductores = np.ceil(vehiculos * turnos_bus * factor_reserva).astype(int)

    return pd.DataFrame(
        {
            "demanda": np.round(dem_vals, 1),
            "despachos": despachos,
            "vehiculos": vehiculos,
            "conductores": conductores,
            "tiempo_ciclo_h": t_ciclo,
            "vueltas_dia": round(vueltas_dia, 2),
        },
        index=demanda.index,
    )
