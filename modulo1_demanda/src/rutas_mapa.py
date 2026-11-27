"""Analisis geoespacial de rutas en Colombia y factibilidad de rastreo de buses.

Provee funciones reutilizables para:
1. Mapear las 52 terminales terrestres a sus coordenadas oficiales DIVIPOLA (DANE).
2. Reconstruir corredores reales con paradas intermedias (ORIGEN -> TRANSITO -> DESTINO).
3. Evaluar cuantitativamente la factibilidad de seguir un mismo bus sin ID de vehiculo
   (colisiones horarias, agregacion previa, correlacion cruzada de tiempos de viaje y
   emparejamiento aproximado).
4. Generar mapas interactivos de Colombia (Plotly) sobre cartografia real.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import polars as pl

from eda_utils import haversine_km


def mapear_terminales_divipola(
    lf: pl.LazyFrame,
    divipola: pl.DataFrame,
    fecha_inicio: date = date(2021, 1, 1),
    fecha_fin: date = date(2023, 12, 31),
) -> pl.DataFrame:
    """Asocia cada terminal terrestre a su municipio DIVIPOLA principal y volumen operativo."""
    volumen = (
        lf.filter(pl.col("fecha_despacho").is_between(fecha_inicio, fecha_fin))
        .group_by("terminal")
        .agg([
            pl.len().alias("registros"),
            pl.col("despachos").sum().alias("despachos"),
            pl.col("pasajeros").sum().alias("pasajeros"),
            (pl.col("tipo_despacho") == "ORIGEN").sum().alias("reg_origen"),
            (pl.col("tipo_despacho") == "TRANSITO").sum().alias("reg_transito"),
        ])
        .collect()
    )

    mpio_principal = (
        lf.filter(pl.col("tipo_despacho") == "ORIGEN")
        .group_by(["terminal", "municipio_origen_ruta"])
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .group_by("terminal")
        .first()
        .select(["terminal", pl.col("municipio_origen_ruta").alias("cod_mpio")])
        .collect()
    )

    return (
        volumen.join(mpio_principal, on="terminal", how="left")
        .join(
            divipola.select(["cod_mpio", "nom_mpio", "dpto", "latitud", "longitud"]),
            on="cod_mpio",
            how="left",
        )
        .sort("pasajeros", descending=True)
    )


def obtener_paradas_corredor(
    lf: pl.LazyFrame,
    divipola: pl.DataFrame,
    terminales_df: pl.DataFrame,
    origen: str,
    destino: str,
    fecha_inicio: date = date(2021, 1, 1),
    fecha_fin: date = date(2023, 12, 31),
    min_despachos: int = 100,
) -> pd.DataFrame:
    """Reconstruye las paradas observadas (ORIGEN y TRANSITO) de una ruta en orden geografico."""
    agg = (
        lf.filter(
            (pl.col("municipio_origen_ruta") == origen)
            & (pl.col("municipio_destino_ruta") == destino)
            & pl.col("fecha_despacho").is_between(fecha_inicio, fecha_fin)
        )
        .group_by(["terminal", "tipo_despacho"])
        .agg([
            pl.len().alias("registros"),
            pl.col("despachos").sum().alias("despachos"),
            pl.col("pasajeros").sum().alias("pasajeros"),
            ((pl.col("pasajeros") == 0).mean() * 100).round(1).alias("pct_cero_pax"),
        ])
        .filter(pl.col("despachos") >= min_despachos)
        .collect()
    )

    paradas = agg.join(
        terminales_df.select(["terminal", "cod_mpio", "nom_mpio", "dpto", "latitud", "longitud"]),
        on="terminal",
        how="left",
    ).to_pandas()

    orig_row = divipola.filter(pl.col("cod_mpio") == origen).to_pandas().iloc[0]
    dest_row = divipola.filter(pl.col("cod_mpio") == destino).to_pandas().iloc[0]

    paradas["dist_desde_origen_km"] = np.round(
        haversine_km(
            np.full(len(paradas), orig_row["latitud"]),
            np.full(len(paradas), orig_row["longitud"]),
            paradas["latitud"].to_numpy(),
            paradas["longitud"].to_numpy(),
        ),
        1,
    )
    paradas["dist_hacia_destino_km"] = np.round(
        haversine_km(
            paradas["latitud"].to_numpy(),
            paradas["longitud"].to_numpy(),
            np.full(len(paradas), dest_row["latitud"]),
            np.full(len(paradas), dest_row["longitud"]),
        ),
        1,
    )
    # Ordenar primero ORIGEN y luego las terminales de TRANSITO por distancia creciente desde el origen
    paradas["orden_tipo"] = np.where(paradas["tipo_despacho"] == "ORIGEN", 0, 1)
    paradas = paradas.sort_values(["orden_tipo", "dist_desde_origen_km"]).drop(columns=["orden_tipo"]).reset_index(drop=True)
    return paradas


def estimar_rezago_viaje_corredor(
    lf: pl.LazyFrame,
    origen: str,
    destino: str,
    terminal_origen: str,
    terminal_transito: str,
    fecha_inicio: date = date(2023, 1, 1),
    fecha_fin: date = date(2023, 12, 31),
    max_lag_horas: int = 18,
) -> pd.DataFrame:
    """Estima el tiempo empirico de viaje entre terminal ORIGEN y TRANSITO por correlacion cruzada horaria."""
    sub = (
        lf.filter(
            (pl.col("municipio_origen_ruta") == origen)
            & (pl.col("municipio_destino_ruta") == destino)
            & pl.col("fecha_despacho").is_between(fecha_inicio, fecha_fin)
            & pl.col("terminal").is_in([terminal_origen, terminal_transito])
        )
        .with_columns(pl.col("hora_despacho").cast(pl.Int32).alias("hora"))
        .group_by(["terminal", "tipo_despacho", "fecha_despacho", "hora"])
        .agg(pl.col("despachos").sum().alias("despachos"))
        .collect()
        .to_pandas()
    )

    if sub.empty:
        return pd.DataFrame(columns=["lag_horas", "correlacion"])

    sub["ts"] = pd.to_datetime(sub["fecha_despacho"]) + pd.to_timedelta(sub["hora"], unit="h")
    idx = pd.date_range(
        pd.Timestamp(fecha_inicio),
        pd.Timestamp(fecha_fin) + pd.Timedelta(hours=23),
        freq="h",
    )

    s_orig = (
        sub[(sub["terminal"] == terminal_origen) & (sub["tipo_despacho"] == "ORIGEN")]
        .set_index("ts")["despachos"]
        .reindex(idx, fill_value=0)
    )
    s_tran = (
        sub[(sub["terminal"] == terminal_transito) & (sub["tipo_despacho"] == "TRANSITO")]
        .set_index("ts")["despachos"]
        .reindex(idx, fill_value=0)
    )

    rows = []
    for lag in range(0, max_lag_horas + 1):
        # Correlacion entre salida en ORIGEN en t y paso por TRANSITO en t + lag
        corr = s_orig.corr(s_tran.shift(-lag))
        rows.append({"lag_horas": lag, "correlacion": round(float(corr), 4) if pd.notna(corr) else np.nan})
    return pd.DataFrame(rows)


def evaluar_factibilidad_rastreo(
    lf: pl.LazyFrame,
    fecha_inicio: date = date(2023, 1, 1),
    fecha_fin: date = date(2023, 12, 31),
) -> dict[str, Any]:
    """Cuantifica las barreras y alcances reales para rastrear un mismo bus sin ID de vehiculo."""
    base = lf.filter(pl.col("fecha_despacho").is_between(fecha_inicio, fecha_fin))

    # 1. Colisiones horarias (multiples buses identicos saliendo en la misma hora)
    colisiones = (
        base.filter(pl.col("tipo_despacho") == "ORIGEN")
        .group_by([
            "terminal",
            "municipio_origen_ruta",
            "municipio_destino_ruta",
            "clase_vehiculo",
            "nivel_servicio",
            "fecha_despacho",
            "hora_despacho",
        ])
        .agg([
            pl.len().alias("filas_bloque"),
            pl.col("despachos").sum().alias("buses_bloque"),
        ])
        .select([
            pl.len().alias("bloques_hora_totales"),
            (pl.col("buses_bloque") == 1).mean().alias("prop_bloque_1_bus"),
            (pl.col("buses_bloque") > 1).mean().alias("prop_bloque_multiples_buses"),
            (pl.col("buses_bloque") >= 5).mean().alias("prop_bloque_5omas_buses"),
            pl.col("buses_bloque").mean().alias("promedio_buses_por_bloque"),
            pl.col("buses_bloque").max().alias("max_buses_en_1_hora"),
        ])
        .collect()
    )

    # 2. Cobertura ORIGEN vs TRANSITO por ruta
    rutas_ot = (
        base.group_by(["municipio_origen_ruta", "municipio_destino_ruta"])
        .agg([
            (pl.col("tipo_despacho") == "ORIGEN").sum().alias("reg_origen"),
            (pl.col("tipo_despacho") == "TRANSITO").sum().alias("reg_transito"),
            pl.col("terminal").n_unique().alias("n_terminales"),
            pl.col("pasajeros").sum().alias("pasajeros"),
        ])
        .collect()
    )

    resumen_ot = pd.DataFrame([
        {
            "categoria": "Solo ORIGEN (sin paradas intermedias registradas)",
            "n_rutas": int(rutas_ot.filter((pl.col("reg_origen") > 0) & (pl.col("reg_transito") == 0)).height),
        },
        {
            "categoria": "Ambas (ORIGEN y al menos 1 terminal TRANSITO)",
            "n_rutas": int(rutas_ot.filter((pl.col("reg_origen") > 0) & (pl.col("reg_transito") > 0)).height),
        },
        {
            "categoria": "Solo TRANSITO (origen sin terminal o sin reporte)",
            "n_rutas": int(rutas_ot.filter((pl.col("reg_origen") == 0) & (pl.col("reg_transito") > 0)).height),
        },
    ])
    resumen_ot["pct_rutas"] = (resumen_ot["n_rutas"] / resumen_ot["n_rutas"].sum() * 100).round(2)

    # 3. Top corredores con multiples terminales (ORIGEN + TRANSITO)
    top_corredores_ot = (
        rutas_ot.filter((pl.col("reg_origen") > 0) & (pl.col("reg_transito") > 0) & (pl.col("n_terminales") >= 3))
        .sort("pasajeros", descending=True)
        .head(15)
    )

    return {
        "colisiones": colisiones,
        "resumen_origen_transito": resumen_ot,
        "top_corredores_multiterminal": top_corredores_ot,
    }


def emparejar_buses_dia_ejemplo(
    lf: pl.LazyFrame,
    origen: str,
    destino: str,
    terminal_origen: str,
    terminal_transito: str,
    fecha: date,
    lag_esperado_horas: int,
    tolerancia_horas: int = 1,
) -> dict[str, Any]:
    """Prueba heuristica de emparejamiento 1-a-1 ORIGEN -> TRANSITO en un dia especifico.

    Intenta asociar cada despacho individual de ORIGEN (expandiendo despachos > 1) con un
    despacho de TRANSITO de la misma (clase_vehiculo, nivel_servicio) dentro de
    [t_origen + lag_esperado - tolerancia, t_origen + lag_esperado + tolerancia].
    """
    sub = (
        lf.filter(
            (pl.col("municipio_origen_ruta") == origen)
            & (pl.col("municipio_destino_ruta") == destino)
            & pl.col("fecha_despacho").is_between(fecha, fecha + timedelta(days=1))
            & pl.col("terminal").is_in([terminal_origen, terminal_transito])
        )
        .with_columns(pl.col("hora_despacho").cast(pl.Int32).alias("hora"))
        .collect()
        .to_pandas()
    )
    sub["fecha_dt"] = pd.to_datetime(sub["fecha_despacho"]).dt.date
    sub["ts"] = pd.to_datetime(sub["fecha_despacho"]) + pd.to_timedelta(sub["hora"], unit="h")

    # Expandir filas con despachos > 1 en despachos individuales
    orig_df = sub[(sub["terminal"] == terminal_origen) & (sub["tipo_despacho"] == "ORIGEN") & (sub["fecha_dt"] == fecha)]
    tran_df = sub[(sub["terminal"] == terminal_transito) & (sub["tipo_despacho"] == "TRANSITO")]

    orig_buses = orig_df.loc[orig_df.index.repeat(orig_df["despachos"])].reset_index(drop=True)
    tran_buses = tran_df.loc[tran_df.index.repeat(tran_df["despachos"])].reset_index(drop=True)

    usados: set[int] = set()
    matches = []
    for i, row in orig_buses.iterrows():
        t_min = row["ts"] + pd.Timedelta(hours=lag_esperado_horas - tolerancia_horas)
        t_max = row["ts"] + pd.Timedelta(hours=lag_esperado_horas + tolerancia_horas)
        cand = tran_buses[
            (~tran_buses.index.isin(usados))
            & (tran_buses["clase_vehiculo"] == row["clase_vehiculo"])
            & (tran_buses["nivel_servicio"] == row["nivel_servicio"])
            & (tran_buses["ts"] >= t_min)
            & (tran_buses["ts"] <= t_max)
        ].copy()
        n_candidatos = len(cand)
        if n_candidatos > 0:
            cand["diff_h"] = (cand["ts"] - (row["ts"] + pd.Timedelta(hours=lag_esperado_horas))).abs()
            best_idx = cand["diff_h"].idxmin()
            usados.add(int(best_idx))
            best = tran_buses.loc[best_idx]
            matches.append({
                "hora_origen": row["ts"],
                "clase": row["clase_vehiculo"],
                "nivel": row["nivel_servicio"],
                "pax_origen": row["pasajeros"],
                "hora_transito": best["ts"],
                "pax_transito": best["pasajeros"],
                "candidatos_en_ventana": n_candidatos,
                "estado": "Unico" if n_candidatos == 1 else "Ambiguo",
            })
        else:
            matches.append({
                "hora_origen": row["ts"],
                "clase": row["clase_vehiculo"],
                "nivel": row["nivel_servicio"],
                "pax_origen": row["pasajeros"],
                "hora_transito": pd.NaT,
                "pax_transito": np.nan,
                "candidatos_en_ventana": 0,
                "estado": "Sin emparejar",
            })

    res_df = pd.DataFrame(matches)
    conteo = res_df["estado"].value_counts().to_dict() if not res_df.empty else {}
    return {
        "total_buses_origen": len(orig_buses),
        "total_buses_transito_48h": len(tran_buses),
        "resumen_estados": conteo,
        "detalle": res_df,
    }


def mapa_red_nacional_colombia(
    top_rutas: pl.DataFrame,
    terminales_df: pl.DataFrame,
    titulo: str = "Red nacional de transporte intermunicipal — Top rutas y 52 terminales (2021-2023)",
) -> go.Figure:
    """Mapa interactivo de Colombia (SVG Scattergeo, sin WebGL) con terminales y rutas principales."""
    rutas_pd = top_rutas.to_pandas()
    term_pd = terminales_df.to_pandas()

    fig = go.Figure()

    max_pax = rutas_pd["pasajeros"].max() if len(rutas_pd) else 1
    shown_normal = False
    for _, r in rutas_pd.iterrows():
        if pd.isna(r["lat_origen"]) or pd.isna(r["lat_destino"]):
            continue
        es_anomalia = r["ruta_id"] == "76001_47570"
        width = 1.2 + 4.2 * (r["pasajeros"] / max_pax)
        color = "#d62728" if es_anomalia else "rgba(31, 119, 180, 0.55)"
        dash = "dash" if es_anomalia else "solid"
        etiqueta = (
            f"<b>{r['ruta_nombre']}</b>{' [ANOMALÍA DIVIPOLA]' if es_anomalia else ''}<br>"
            f"Pasajeros (2021-2023): {r['pasajeros']:,.0f}<br>"
            f"Despachos: {r['despachos']:,.0f} | Pax/desp: {r['pax_por_despacho']:.1f}<br>"
            f"Distancia línea recta: {r['dist_linea_recta_km']:.0f} km"
        )
        show_leg = bool(es_anomalia) or (not shown_normal)
        if not es_anomalia:
            shown_normal = True
        fig.add_trace(
            go.Scattergeo(
                mode="lines",
                lon=[r["lon_origen"], r["lon_destino"]],
                lat=[r["lat_origen"], r["lat_destino"]],
                line=dict(width=width, color=color, dash=dash),
                hoverinfo="text",
                text=etiqueta,
                name="Anomalía DIVIPOLA (76001->47570)" if es_anomalia else "Top rutas (ORIGEN)",
                showlegend=show_leg,
            )
        )

    # Terminales terrestres
    sizes = 6 + 18 * np.sqrt(term_pd["pasajeros"] / term_pd["pasajeros"].max())
    fig.add_trace(
        go.Scattergeo(
            mode="markers",
            lon=term_pd["longitud"],
            lat=term_pd["latitud"],
            marker=dict(
                size=sizes,
                color=term_pd["pasajeros"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Pasajeros<br>(2021-2023)"),
                line=dict(width=0.7, color="white"),
                opacity=0.92,
            ),
            text=[
                f"<b>{r['terminal']}</b><br>{r['nom_mpio']} ({r['dpto']})<br>"
                f"Pasajeros: {r['pasajeros']:,.0f}<br>Despachos: {r['despachos']:,.0f}"
                for _, r in term_pd.iterrows()
            ],
            hoverinfo="text",
            name="Terminales terrestres (52)",
        )
    )

    fig.update_geos(
        scope="south america",
        resolution=50,
        showcountries=True,
        countrycolor="#666666",
        showcoastlines=True,
        coastlinecolor="#444444",
        showland=True,
        landcolor="#f4f4f0",
        showocean=True,
        oceancolor="#e6f2fa",
        showrivers=True,
        rivercolor="#b3d9f2",
        lataxis_range=[-4.5, 13.0],
        lonaxis_range=[-79.5, -66.5],
    )
    fig.update_layout(
        title=titulo,
        margin=dict(l=10, r=10, t=50, b=10),
        height=680,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.88)"),
    )
    return fig


def mapa_corredor_real_paradas(
    paradas_df: pd.DataFrame,
    divipola: pl.DataFrame,
    origen: str,
    destino: str,
    titulo: str,
) -> go.Figure:
    """Mapa interactivo (SVG Scattergeo, sin WebGL) de un corredor real: ORIGEN -> TRANSITO -> DESTINO."""
    dest_row = divipola.filter(pl.col("cod_mpio") == destino).to_pandas().iloc[0]

    lats = paradas_df["latitud"].tolist() + [dest_row["latitud"]]
    lons = paradas_df["longitud"].tolist() + [dest_row["longitud"]]

    fig = go.Figure()

    # Linea del corredor siguiendo las terminales observadas
    fig.add_trace(
        go.Scattergeo(
            mode="lines",
            lat=lats,
            lon=lons,
            line=dict(width=3.5, color="#1f77b4"),
            name="Trayectoria entre terminales observadas",
            hoverinfo="skip",
        )
    )

    # Marcadores de ORIGEN y TRANSITO
    colores = {"ORIGEN": "#2ca02c", "TRANSITO": "#ff7f0e"}
    for tipo in ["ORIGEN", "TRANSITO"]:
        sub = paradas_df[paradas_df["tipo_despacho"] == tipo]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scattergeo(
                mode="markers+text",
                lat=sub["latitud"],
                lon=sub["longitud"],
                marker=dict(
                    size=14 if tipo == "ORIGEN" else 11,
                    color=colores[tipo],
                    line=dict(width=1, color="white"),
                ),
                text=sub["nom_mpio"],
                textposition="top right",
                textfont=dict(size=10, color="#222222"),
                hovertext=[
                    f"<b>{r['terminal']} ({r['tipo_despacho']})</b><br>"
                    f"Municipio: {r['nom_mpio']}, {r['dpto']}<br>"
                    f"Despachos: {r['despachos']:,.0f}<br>"
                    f"Pasajeros abordados: {r['pasajeros']:,.0f}<br>"
                    f"% cero pasajeros: {r['pct_cero_pax']}%<br>"
                    f"Dist. desde origen: {r['dist_desde_origen_km']} km"
                    for _, r in sub.iterrows()
                ],
                hoverinfo="text",
                name=f"Parada {tipo}",
            )
        )

    # Marcador de DESTINO final
    fig.add_trace(
        go.Scattergeo(
            mode="markers+text",
            lat=[dest_row["latitud"]],
            lon=[dest_row["longitud"]],
            marker=dict(size=15, color="#d62728", line=dict(width=1, color="white")),
            text=[f"DESTINO: {dest_row['nom_mpio']}"],
            textposition="bottom right",
            textfont=dict(size=10, color="#a00000"),
            hovertext=[f"<b>DESTINO FINAL: {dest_row['nom_mpio']} ({dest_row['dpto']})</b><br>DIVIPOLA: {destino}"],
            hoverinfo="text",
            name="Destino final",
        )
    )

    lat_pad = max(1.2, (max(lats) - min(lats)) * 0.35)
    lon_pad = max(1.5, (max(lons) - min(lons)) * 0.45)

    fig.update_geos(
        scope="south america",
        resolution=50,
        showcountries=True,
        countrycolor="#666666",
        showcoastlines=True,
        coastlinecolor="#444444",
        showland=True,
        landcolor="#f4f4f0",
        showocean=True,
        oceancolor="#e6f2fa",
        showrivers=True,
        rivercolor="#b3d9f2",
        lataxis_range=[min(lats) - lat_pad, max(lats) + lat_pad],
        lonaxis_range=[min(lons) - lon_pad, max(lons) + lon_pad],
    )
    fig.update_layout(
        title=titulo,
        margin=dict(l=10, r=10, t=50, b=10),
        height=600,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.88)"),
    )
    return fig
