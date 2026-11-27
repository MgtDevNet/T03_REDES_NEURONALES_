"""Construye las series diarias limpias y el catalogo de rutas para el Modulo 1.

Criterios de seleccion de rutas (configurables):
- Al menos 500 dias con registros ORIGEN en el bloque 2021-2023.
- Promedio >= 25 pasajeros/dia en el bloque 2021-2023.
- Al menos 60 dias con registros ORIGEN en el bloque reciente 2026 (ene-mar).
- Exclusion de anomalias DIVIPOLA documentadas (ej. 76001_47570) o sin coordenadas.

Genera en data/processed/:
- rutas.csv: metadatos legibles, codigos DIVIPOLA, coordenadas, distancia y sentido.
- series_diarias.parquet: serie diaria por ruta en bloques continuos validos
  (2021-01-01..2023-12-31, 2025-01-01..2025-11-29, 2026-01-01..2026-03-30),
  con columnas separadas para ORIGEN y TRANSITO, cruzada con calendario_diario.csv.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from calendario import CALENDARIO_DIARIO_CSV, exportar_calendario_diario_csv, exportar_festivos_csv
from eda_utils import PROCESSED_DIR, haversine_km, load_divipola, scan_operacion

SERIES_PARQUET = PROCESSED_DIR / "series_diarias.parquet"
RUTAS_CSV = PROCESSED_DIR / "rutas.csv"

MIN_DIAS_2021_2023 = 500
MIN_MEDIA_PAX_DIA = 25.0
MIN_DIAS_2026 = 60
RUTAS_EXCLUIDAS = {"76001_47570"}  # Anomalia DIVIPOLA Cali -> Puebloviejo (878 km en buseta)

# Bloques continuos validos (excluye hueco estructural 2024, corte dic-2025 y parcial abr-2026)
BLOQUES_VALIDOS: list[tuple[str, date, date]] = [
    ("B1_2021_2023", date(2021, 1, 1), date(2023, 12, 31)),
    ("B2_2025", date(2025, 1, 1), date(2025, 11, 29)),
    ("B3_2026", date(2026, 1, 1), date(2026, 3, 30)),
]


def seleccionar_rutas(
    lf: pl.LazyFrame,
    divipola: pl.DataFrame,
    min_dias_b1: int = MIN_DIAS_2021_2023,
    min_media_pax: float = MIN_MEDIA_PAX_DIA,
    min_dias_2026: int = MIN_DIAS_2026,
) -> pd.DataFrame:
    """Selecciona rutas activas segun umbrales y construye data/processed/rutas.csv."""
    dias_b1 = (BLOQUES_VALIDOS[0][2] - BLOQUES_VALIDOS[0][1]).days + 1  # 1095

    stats_b1 = (
        lf.filter(pl.col("fecha_despacho").is_between(BLOQUES_VALIDOS[0][1], BLOQUES_VALIDOS[0][2]))
        .with_columns((pl.col("municipio_origen_ruta") + "_" + pl.col("municipio_destino_ruta")).alias("ruta_id"))
        .group_by(["ruta_id", "municipio_origen_ruta", "municipio_destino_ruta"])
        .agg([
            pl.col("pasajeros").filter(pl.col("tipo_despacho") == "ORIGEN").sum().alias("pax_origen_b1"),
            pl.col("pasajeros").filter(pl.col("tipo_despacho") == "TRANSITO").sum().alias("pax_transito_b1"),
            pl.col("fecha_despacho").filter(pl.col("tipo_despacho") == "ORIGEN").n_unique().alias("dias_origen_b1"),
        ])
        .filter(
            (pl.col("dias_origen_b1") >= min_dias_b1)
            & (pl.col("pax_origen_b1") >= min_media_pax * dias_b1)
            & (~pl.col("ruta_id").is_in(list(RUTAS_EXCLUIDAS)))
        )
        .collect()
    )

    stats_b3 = (
        lf.filter(
            pl.col("fecha_despacho").is_between(BLOQUES_VALIDOS[2][1], BLOQUES_VALIDOS[2][2])
            & (pl.col("tipo_despacho") == "ORIGEN")
        )
        .with_columns((pl.col("municipio_origen_ruta") + "_" + pl.col("municipio_destino_ruta")).alias("ruta_id"))
        .group_by("ruta_id")
        .agg(pl.col("fecha_despacho").n_unique().alias("dias_origen_2026"))
        .collect()
    )

    sel = (
        stats_b1.join(stats_b3, on="ruta_id", how="inner")
        .filter(pl.col("dias_origen_2026") >= min_dias_2026)
        .rename({"municipio_origen_ruta": "divipola_origen", "municipio_destino_ruta": "divipola_destino"})
    )

    div_o = divipola.select([
        pl.col("cod_mpio").alias("divipola_origen"),
        pl.col("nom_mpio").alias("origen"),
        pl.col("dpto").alias("dpto_origen"),
        pl.col("latitud").alias("lat_o"),
        pl.col("longitud").alias("lon_o"),
    ])
    div_d = divipola.select([
        pl.col("cod_mpio").alias("divipola_destino"),
        pl.col("nom_mpio").alias("destino"),
        pl.col("dpto").alias("dpto_destino"),
        pl.col("latitud").alias("lat_d"),
        pl.col("longitud").alias("lon_d"),
    ])

    enriched = (
        sel.join(div_o, on="divipola_origen", how="inner")
        .join(div_d, on="divipola_destino", how="inner")
        .filter(pl.col("lat_o").is_not_null() & pl.col("lat_d").is_not_null())
        .to_pandas()
    )

    enriched["dist_linea_recta_km"] = np.round(
        haversine_km(
            enriched["lat_o"].to_numpy(),
            enriched["lon_o"].to_numpy(),
            enriched["lat_d"].to_numpy(),
            enriched["lon_d"].to_numpy(),
        ),
        1,
    )
    enriched["media_pax_origen_dia"] = np.round(enriched["pax_origen_b1"] / dias_b1, 1)
    enriched["media_pax_transito_dia"] = np.round(enriched["pax_transito_b1"] / dias_b1, 1)
    enriched["ruta_nombre"] = enriched["origen"] + " -> " + enriched["destino"]

    rutas_set = set(enriched["ruta_id"])
    enriched["ruta_inversa"] = enriched["divipola_destino"] + "_" + enriched["divipola_origen"]
    enriched["tiene_regreso"] = enriched["ruta_inversa"].isin(rutas_set)
    enriched["sentido"] = np.where(
        enriched["divipola_origen"] < enriched["divipola_destino"], "Ida", "Regreso"
    )
    enriched["par_id"] = np.where(
        enriched["divipola_origen"] < enriched["divipola_destino"],
        enriched["ruta_id"],
        enriched["ruta_inversa"],
    )

    cols = [
        "ruta_id", "origen", "destino", "divipola_origen", "divipola_destino",
        "dpto_origen", "dpto_destino", "sentido", "par_id", "tiene_regreso",
        "dist_linea_recta_km", "ruta_nombre", "dias_origen_b1", "dias_origen_2026",
        "media_pax_origen_dia", "media_pax_transito_dia",
    ]
    df_out = enriched[cols].sort_values("media_pax_origen_dia", ascending=False).reset_index(drop=True)
    RUTAS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(RUTAS_CSV, index=False, encoding="utf-8")
    return df_out


def construir_series_diarias(
    lf: pl.LazyFrame,
    rutas_df: pd.DataFrame,
    calendario_csv: Path | str = CALENDARIO_DIARIO_CSV,
) -> pl.DataFrame:
    """Construye series diarias continuas por bloque con ORIGEN y TRANSITO separados."""
    if not Path(calendario_csv).exists():
        exportar_festivos_csv()
        exportar_calendario_diario_csv(out_path=calendario_csv)

    cal_pl = pl.read_csv(calendario_csv).with_columns(pl.col("fecha").str.to_date("%Y-%m-%d"))
    rutas_ids = rutas_df["ruta_id"].tolist()

    # Agregacion unica en Polars sobre todas las rutas seleccionadas (ORIGEN y TRANSITO separados)
    agg = (
        lf.filter(
            pl.col("fecha_despacho").is_between(BLOQUES_VALIDOS[0][1], BLOQUES_VALIDOS[-1][2])
        )
        .with_columns((pl.col("municipio_origen_ruta") + "_" + pl.col("municipio_destino_ruta")).alias("ruta_id"))
        .filter(pl.col("ruta_id").is_in(rutas_ids))
        .group_by(["ruta_id", "fecha_despacho"])
        .agg([
            pl.col("pasajeros").filter(pl.col("tipo_despacho") == "ORIGEN").sum().fill_null(0).alias("pasajeros_origen"),
            pl.col("despachos").filter(pl.col("tipo_despacho") == "ORIGEN").sum().fill_null(0).alias("despachos_origen"),
            (pl.col("tipo_despacho") == "ORIGEN").sum().alias("registros_origen"),
            ((pl.col("tipo_despacho") == "ORIGEN") & (pl.col("pasajeros") == 0)).sum().alias("reg_cero_origen"),
            pl.col("pasajeros").filter(pl.col("tipo_despacho") == "TRANSITO").sum().fill_null(0).alias("pasajeros_transito"),
            pl.col("despachos").filter(pl.col("tipo_despacho") == "TRANSITO").sum().fill_null(0).alias("despachos_transito"),
            (pl.col("tipo_despacho") == "TRANSITO").sum().alias("registros_transito"),
        ])
        .rename({"fecha_despacho": "fecha"})
        .collect()
    )

    # Grilla calendario continua por bloque x ruta
    bloques_frames = []
    for bloque_id, f_ini, f_fin in BLOQUES_VALIDOS:
        fechas_b = pl.date_range(f_ini, f_fin, "1d", eager=True).alias("fecha")
        df_b = pl.DataFrame({"fecha": fechas_b}).with_columns(pl.lit(bloque_id).alias("bloque_id"))
        bloques_frames.append(df_b)
    grid_fechas = pl.concat(bloques_frames)
    grid_rutas = pl.DataFrame({"ruta_id": rutas_ids})
    grid = grid_rutas.join(grid_fechas, how="cross")

    series = (
        grid.join(agg, on=["ruta_id", "fecha"], how="left")
        .with_columns([
            pl.col("registros_origen").is_null().cast(pl.Int8).alias("dia_sin_registro_origen"),
            pl.col("pasajeros_origen").fill_null(0).cast(pl.Int32),
            pl.col("despachos_origen").fill_null(0).cast(pl.Int32),
            pl.col("registros_origen").fill_null(0).cast(pl.Int32),
            pl.col("reg_cero_origen").fill_null(0).cast(pl.Int32),
            pl.col("pasajeros_transito").fill_null(0).cast(pl.Int32),
            pl.col("despachos_transito").fill_null(0).cast(pl.Int32),
            pl.col("registros_transito").fill_null(0).cast(pl.Int32),
        ])
        .with_columns(pl.col("pasajeros_origen").alias("pasajeros"))
        .join(cal_pl, on="fecha", how="left")
        .sort(["ruta_id", "fecha"])
    )

    SERIES_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    series.write_parquet(SERIES_PARQUET, compression="zstd")
    return series.to_pandas()


if __name__ == "__main__":
    lf = scan_operacion()
    div = load_divipola()
    rutas_df = seleccionar_rutas(lf, div)
    series_df = construir_series_diarias(lf, rutas_df)
    print(f"Guardado {RUTAS_CSV} ({len(rutas_df)} rutas | con par regreso: {rutas_df['tiene_regreso'].sum()})")
    print(f"Guardado {SERIES_PARQUET} ({len(series_df):,} filas | {SERIES_PARQUET.stat().st_size / (1024*1024):.1f} MB)")
    print(series_df.groupby("bloque_id")[["pasajeros_origen", "pasajeros_transito"]].agg(["count", "mean", "sum"]).round(1))
