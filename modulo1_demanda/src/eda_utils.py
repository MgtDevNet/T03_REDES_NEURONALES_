"""Utilidades de carga, limpieza, DIVIPOLA y analisis exploratorio para el Modulo 1.

Fuente principal de datos:
- Superintendencia de Transporte (datos.gov.co, dataset eh75-8ah6):
  "Operacion de pasajeros y despacho de vehiculos en la modalidad de
  transporte de pasajeros por carretera".
- DANE - DIVIPOLA (datos.gov.co, dataset gdxc-w37w):
  "DIVIPOLA - Codigos municipios" (con coordenadas geograficas oficiales).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import requests

MODULE_DIR = Path(__file__).resolve().parents[1]
RAW_PARQUET = MODULE_DIR / "data" / "raw" / "operacion_pasajeros.parquet"
PROCESSED_DIR = MODULE_DIR / "data" / "processed"
FIGURES_DIR = MODULE_DIR / "figures"
DIVIPOLA_CSV = PROCESSED_DIR / "divipola.csv"

DIVIPOLA_DATASET_ID = "gdxc-w37w"
DIVIPOLA_URL = f"https://www.datos.gov.co/resource/{DIVIPOLA_DATASET_ID}.json"

# Rango valido del calendario operativo (excluye fechas futuras con errores de digitacion como 6202)
FECHA_MIN_VALIDA = date(2019, 1, 1)
FECHA_MAX_VALIDA = date(2026, 12, 31)

# Festivos oficiales de Colombia (Ley 51 de 1983 - Ley Emiliani + fijos + Semana Santa)
# Fuente primaria: Ley 51 de 1983, Congreso de la Republica de Colombia.
FESTIVOS_COLOMBIA: dict[int, list[tuple[str, str]]] = {
    2021: [
        ("2021-01-01", "Ano Nuevo"),
        ("2021-01-11", "Reyes Magos"),
        ("2021-03-22", "San Jose"),
        ("2021-04-01", "Jueves Santo"),
        ("2021-04-02", "Viernes Santo"),
        ("2021-05-01", "Dia del Trabajo"),
        ("2021-05-17", "Ascension del Senor"),
        ("2021-06-07", "Corpus Christi"),
        ("2021-06-14", "Sagrado Corazon"),
        ("2021-07-05", "San Pedro y San Pablo"),
        ("2021-07-20", "Independencia"),
        ("2021-08-07", "Batalla de Boyaca"),
        ("2021-08-16", "Asuncion de la Virgen"),
        ("2021-10-18", "Dia de la Raza"),
        ("2021-11-01", "Todos los Santos"),
        ("2021-11-15", "Independencia de Cartagena"),
        ("2021-12-08", "Inmaculada Concepcion"),
        ("2021-12-25", "Navidad"),
    ],
    2022: [
        ("2022-01-01", "Ano Nuevo"),
        ("2022-01-10", "Reyes Magos"),
        ("2022-03-21", "San Jose"),
        ("2022-04-14", "Jueves Santo"),
        ("2022-04-15", "Viernes Santo"),
        ("2022-05-01", "Dia del Trabajo"),
        ("2022-05-30", "Ascension del Senor"),
        ("2022-06-20", "Corpus Christi"),
        ("2022-06-27", "Sagrado Corazon"),
        ("2022-07-04", "San Pedro y San Pablo"),
        ("2022-07-20", "Independencia"),
        ("2022-08-07", "Batalla de Boyaca"),
        ("2022-08-15", "Asuncion de la Virgen"),
        ("2022-10-17", "Dia de la Raza"),
        ("2022-11-07", "Todos los Santos"),
        ("2022-11-14", "Independencia de Cartagena"),
        ("2022-12-08", "Inmaculada Concepcion"),
        ("2022-12-25", "Navidad"),
    ],
    2023: [
        ("2023-01-01", "Ano Nuevo"),
        ("2023-01-09", "Reyes Magos"),
        ("2023-03-20", "San Jose"),
        ("2023-04-06", "Jueves Santo"),
        ("2023-04-07", "Viernes Santo"),
        ("2023-05-01", "Dia del Trabajo"),
        ("2023-05-22", "Ascension del Senor"),
        ("2023-06-12", "Corpus Christi"),
        ("2023-06-19", "Sagrado Corazon"),
        ("2023-07-03", "San Pedro y San Pablo"),
        ("2023-07-20", "Independencia"),
        ("2023-08-07", "Batalla de Boyaca"),
        ("2023-08-21", "Asuncion de la Virgen"),
        ("2023-10-16", "Dia de la Raza"),
        ("2023-11-06", "Todos los Santos"),
        ("2023-11-13", "Independencia de Cartagena"),
        ("2023-12-08", "Inmaculada Concepcion"),
        ("2023-12-25", "Navidad"),
    ],
    2024: [
        ("2024-01-01", "Ano Nuevo"),
        ("2024-01-08", "Reyes Magos"),
        ("2024-03-25", "San Jose"),
        ("2024-03-28", "Jueves Santo"),
        ("2024-03-29", "Viernes Santo"),
        ("2024-05-01", "Dia del Trabajo"),
        ("2024-05-13", "Ascension del Senor"),
        ("2024-06-03", "Corpus Christi"),
        ("2024-06-10", "Sagrado Corazon"),
        ("2024-07-01", "San Pedro y San Pablo"),
        ("2024-07-20", "Independencia"),
        ("2024-08-07", "Batalla de Boyaca"),
        ("2024-08-19", "Asuncion de la Virgen"),
        ("2024-10-14", "Dia de la Raza"),
        ("2024-11-04", "Todos los Santos"),
        ("2024-11-11", "Independencia de Cartagena"),
        ("2024-12-08", "Inmaculada Concepcion"),
        ("2024-12-25", "Navidad"),
    ],
    2025: [
        ("2025-01-01", "Ano Nuevo"),
        ("2025-01-06", "Reyes Magos"),
        ("2025-03-24", "San Jose"),
        ("2025-04-17", "Jueves Santo"),
        ("2025-04-18", "Viernes Santo"),
        ("2025-05-01", "Dia del Trabajo"),
        ("2025-06-02", "Ascension del Senor"),
        ("2025-06-23", "Corpus Christi"),
        ("2025-06-30", "Sagrado Corazon"),
        ("2025-07-20", "Independencia"),
        ("2025-08-07", "Batalla de Boyaca"),
        ("2025-08-18", "Asuncion de la Virgen"),
        ("2025-10-13", "Dia de la Raza"),
        ("2025-11-03", "Todos los Santos"),
        ("2025-11-17", "Independencia de Cartagena"),
        ("2025-12-08", "Inmaculada Concepcion"),
        ("2025-12-25", "Navidad"),
    ],
    2026: [
        ("2026-01-01", "Ano Nuevo"),
        ("2026-01-12", "Reyes Magos"),
        ("2026-03-23", "San Jose"),
        ("2026-04-02", "Jueves Santo"),
        ("2026-04-03", "Viernes Santo"),
        ("2026-05-01", "Dia del Trabajo"),
        ("2026-05-18", "Ascension del Senor"),
        ("2026-06-08", "Corpus Christi"),
        ("2026-06-15", "Sagrado Corazon"),
        ("2026-06-29", "San Pedro y San Pablo"),
        ("2026-07-20", "Independencia"),
        ("2026-08-07", "Batalla de Boyaca"),
        ("2026-08-17", "Asuncion de la Virgen"),
        ("2026-10-12", "Dia de la Raza"),
        ("2026-11-02", "Todos los Santos"),
        ("2026-11-16", "Independencia de Cartagena"),
        ("2026-12-08", "Inmaculada Concepcion"),
        ("2026-12-25", "Navidad"),
    ],
}


def scan_operacion(parquet_path: Path | str = RAW_PARQUET) -> pl.LazyFrame:
    """Devuelve un LazyFrame de Polars apuntando al archivo Parquet crudo."""
    return pl.scan_parquet(Path(parquet_path))


def load_divipola(
    csv_path: Path | str = DIVIPOLA_CSV,
    force_download: bool = False,
) -> pl.DataFrame:
    """Carga el catalogo oficial DIVIPOLA (DANE) desde disco o lo descarga de datos.gov.co.

    Incluye codigos de departamento y municipio (5 digitos), nombres oficiales,
    tipo de municipio y coordenadas geograficas (latitud, longitud en WGS84).
    """
    path = Path(csv_path)
    if path.exists() and not force_download:
        return pl.read_csv(
            path,
            schema_overrides={
                "cod_dpto": pl.String,
                "cod_mpio": pl.String,
                "latitud": pl.Float64,
                "longitud": pl.Float64,
            },
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(DIVIPOLA_URL, params={"$limit": 5000}, timeout=60)
    resp.raise_for_status()
    rows = resp.json()

    clean_rows = []
    for r in rows:
        lat_raw = str(r.get("latitud", "")).replace(",", ".").strip()
        lon_raw = str(r.get("longitud", "")).replace(",", ".").strip()
        try:
            lat = float(lat_raw)
            lon = float(lon_raw)
        except ValueError:
            lat, lon = None, None
        clean_rows.append(
            {
                "cod_dpto": str(r.get("cod_dpto", "")).zfill(2),
                "dpto": str(r.get("dpto", "")).strip(),
                "cod_mpio": str(r.get("cod_mpio", "")).zfill(5),
                "nom_mpio": str(r.get("nom_mpio", "")).strip(),
                "tipo_municipio": str(r.get("tipo_municipio", "")).strip(),
                "latitud": lat,
                "longitud": lon,
            }
        )

    df = pl.DataFrame(clean_rows).unique(subset=["cod_mpio"]).sort("cod_mpio")
    df.write_csv(path)
    return df


def build_festivos_df(anios: Iterable[int] = range(2021, 2027)) -> pl.DataFrame:
    """Construye tabla de festivos oficiales de Colombia para los anios indicados."""
    rows = []
    for y in anios:
        for fecha_str, nombre in FESTIVOS_COLOMBIA.get(y, []):
            rows.append({"fecha": date.fromisoformat(fecha_str), "festivo_nombre": nombre})
    return pl.DataFrame(rows).sort("fecha")


def audit_calidad_datos(lf: pl.LazyFrame) -> dict[str, pl.DataFrame]:
    """Ejecuta auditoria integral de calidad y cobertura temporal sobre el dataset crudo."""
    fechas_anomalas = (
        lf.filter(
            (pl.col("fecha_despacho") < FECHA_MIN_VALIDA)
            | (pl.col("fecha_despacho") > FECHA_MAX_VALIDA)
        )
        .collect()
    )

    cobertura_mensual = (
        lf.filter(pl.col("fecha_despacho").is_between(date(2021, 1, 1), FECHA_MAX_VALIDA))
        .with_columns([
            pl.col("fecha_despacho").dt.year().alias("anio"),
            pl.col("fecha_despacho").dt.month().alias("mes"),
        ])
        .group_by(["anio", "mes"])
        .agg([
            pl.len().alias("registros"),
            pl.col("despachos").sum().alias("despachos"),
            pl.col("pasajeros").sum().alias("pasajeros"),
            pl.col("terminal").n_unique().alias("terminales_activas"),
            pl.col("fecha_despacho").n_unique().alias("dias_reportados"),
        ])
        .sort(["anio", "mes"])
        .collect()
    )

    resumen_tipo_despacho = (
        lf.filter(pl.col("fecha_despacho").is_between(FECHA_MIN_VALIDA, FECHA_MAX_VALIDA))
        .group_by("tipo_despacho")
        .agg([
            pl.len().alias("registros"),
            pl.col("despachos").sum().alias("total_despachos"),
            pl.col("pasajeros").sum().alias("total_pasajeros"),
            (pl.col("pasajeros") == 0).sum().alias("reg_cero_pasajeros"),
            ((pl.col("pasajeros") == 0).mean() * 100).round(2).alias("pct_cero_pasajeros"),
            (pl.col("despachos") > 1).sum().alias("reg_despachos_gt1"),
            ((pl.col("despachos") > 1).mean() * 100).round(2).alias("pct_despachos_gt1"),
            (pl.col("pasajeros") / pl.col("despachos")).mean().round(2).alias("pax_por_despacho_medio"),
            (pl.col("pasajeros") / pl.col("despachos")).quantile(0.99).round(2).alias("pax_por_despacho_p99"),
            (pl.col("pasajeros") / pl.col("despachos")).max().round(2).alias("pax_por_despacho_max"),
        ])
        .collect()
    )

    resumen_clase_vehiculo = (
        lf.filter(
            pl.col("fecha_despacho").is_between(date(2021, 1, 1), date(2023, 12, 31))
            & (pl.col("tipo_despacho") == "ORIGEN")
        )
        .with_columns((pl.col("pasajeros") / pl.col("despachos")).alias("pax_por_desp"))
        .group_by("clase_vehiculo")
        .agg([
            pl.len().alias("registros"),
            pl.col("despachos").sum().alias("total_despachos"),
            pl.col("pasajeros").sum().alias("total_pasajeros"),
            ((pl.col("pasajeros") == 0).mean() * 100).round(2).alias("pct_cero_pax"),
            pl.col("pax_por_desp").mean().round(2).alias("pax_desp_media"),
            pl.col("pax_por_desp").median().round(2).alias("pax_desp_mediana"),
            pl.col("pax_por_desp").quantile(0.95).round(2).alias("pax_desp_p95"),
            pl.col("pax_por_desp").quantile(0.99).round(2).alias("pax_desp_p99"),
            pl.col("pax_por_desp").max().round(2).alias("pax_desp_max"),
        ])
        .sort("total_pasajeros", descending=True)
        .collect()
    )

    return {
        "fechas_anomalas": fechas_anomalas,
        "cobertura_mensual": cobertura_mensual,
        "resumen_tipo_despacho": resumen_tipo_despacho,
        "resumen_clase_vehiculo": resumen_clase_vehiculo,
    }


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Distancia geodésica en kilometros (formula de Haversine) entre pares de coordenadas."""
    r = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def top_rutas_con_divipola(
    lf: pl.LazyFrame,
    divipola: pl.DataFrame,
    fecha_inicio: date = date(2021, 1, 1),
    fecha_fin: date = date(2023, 12, 31),
    solo_origen: bool = True,
    top_n: int = 30,
) -> pl.DataFrame:
    """Obtiene las rutas de mayor volumen enriquecidas con nombres, departamentos y distancia."""
    q = lf.filter(pl.col("fecha_despacho").is_between(fecha_inicio, fecha_fin))
    if solo_origen:
        q = q.filter(pl.col("tipo_despacho") == "ORIGEN")

    agg = (
        q.group_by(["municipio_origen_ruta", "municipio_destino_ruta"])
        .agg([
            pl.len().alias("registros"),
            pl.col("despachos").sum().alias("despachos"),
            pl.col("pasajeros").sum().alias("pasajeros"),
            pl.col("fecha_despacho").n_unique().alias("dias_con_datos"),
            ((pl.col("pasajeros") == 0).mean() * 100).round(2).alias("pct_cero_pax"),
            (pl.col("pasajeros").sum() / pl.col("despachos").sum()).round(2).alias("pax_por_despacho"),
        ])
        .sort("pasajeros", descending=True)
        .head(top_n)
        .collect()
    )

    div_o = divipola.select([
        pl.col("cod_mpio").alias("municipio_origen_ruta"),
        pl.col("nom_mpio").alias("origen_nombre"),
        pl.col("dpto").alias("origen_dpto"),
        pl.col("latitud").alias("lat_origen"),
        pl.col("longitud").alias("lon_origen"),
    ])
    div_d = divipola.select([
        pl.col("cod_mpio").alias("municipio_destino_ruta"),
        pl.col("nom_mpio").alias("destino_nombre"),
        pl.col("dpto").alias("destino_dpto"),
        pl.col("latitud").alias("lat_destino"),
        pl.col("longitud").alias("lon_destino"),
    ])

    enriched = agg.join(div_o, on="municipio_origen_ruta", how="left").join(
        div_d, on="municipio_destino_ruta", how="left"
    )
    dist = haversine_km(
        enriched["lat_origen"].to_numpy(),
        enriched["lon_origen"].to_numpy(),
        enriched["lat_destino"].to_numpy(),
        enriched["lon_destino"].to_numpy(),
    )
    total_dias = (fecha_fin - fecha_inicio).days + 1
    return enriched.with_columns([
        pl.Series("dist_linea_recta_km", np.round(dist, 1)),
        (pl.col("dias_con_datos") / total_dias * 100).round(1).alias("cobertura_dias_pct"),
        (pl.col("origen_nombre") + " -> " + pl.col("destino_nombre")).alias("ruta_nombre"),
        (pl.col("municipio_origen_ruta") + "_" + pl.col("municipio_destino_ruta")).alias("ruta_id"),
    ])


def construir_serie_diaria(
    lf: pl.LazyFrame,
    origen: str,
    destino: str,
    fecha_inicio: date = date(2021, 1, 1),
    fecha_fin: date = date(2023, 12, 31),
    solo_origen: bool = True,
) -> pd.DataFrame:
    """Construye serie diaria continua (calendario completo) para una ruta origen -> destino."""
    q = lf.filter(
        (pl.col("municipio_origen_ruta") == origen)
        & (pl.col("municipio_destino_ruta") == destino)
        & pl.col("fecha_despacho").is_between(fecha_inicio, fecha_fin)
    )
    if solo_origen:
        q = q.filter(pl.col("tipo_despacho") == "ORIGEN")

    diaria = (
        q.group_by("fecha_despacho")
        .agg([
            pl.col("pasajeros").sum().alias("pasajeros"),
            pl.col("despachos").sum().alias("despachos"),
            pl.len().alias("registros"),
            (pl.col("pasajeros") == 0).sum().alias("reg_cero_pax"),
        ])
        .sort("fecha_despacho")
        .collect()
        .to_pandas()
    )
    diaria["fecha_despacho"] = pd.to_datetime(diaria["fecha_despacho"])

    calendario = pd.DataFrame({"fecha": pd.date_range(fecha_inicio, fecha_fin, freq="D")})
    df = calendario.merge(diaria, left_on="fecha", right_on="fecha_despacho", how="left").drop(
        columns=["fecha_despacho"]
    )
    df["dia_sin_registro"] = df["registros"].isna()
    for col in ["pasajeros", "despachos", "registros", "reg_cero_pax"]:
        df[col] = df[col].fillna(0).astype(int)

    df["anio"] = df["fecha"].dt.year
    df["mes"] = df["fecha"].dt.month
    df["dia_semana"] = df["fecha"].dt.dayofweek  # 0=Lunes ... 6=Domingo
    df["es_fin_semana"] = df["dia_semana"].isin([5, 6])

    festivos = set(pd.to_datetime(build_festivos_df(range(fecha_inicio.year, fecha_fin.year + 1))["fecha"].to_list()))
    df["es_festivo"] = df["fecha"].isin(festivos)
    df["es_puente"] = (
        df["es_festivo"]
        | ((df["dia_semana"] == 6) & (df["fecha"] + pd.Timedelta(days=1)).isin(festivos))
        | ((df["dia_semana"] == 5) & (df["fecha"] + pd.Timedelta(days=2)).isin(festivos))
        | ((df["dia_semana"] == 4) & (df["fecha"] + pd.Timedelta(days=3)).isin(festivos))
    )
    return df
