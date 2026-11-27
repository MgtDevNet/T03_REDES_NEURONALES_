"""Modelos de referencia para pronostico diario de demanda a 30 dias (Modulo 1).

Compara de forma sencilla, reproducible y sin fuga de informacion:
1. Ingenuo Estacional (mismo dia de la semana mas reciente conocido en el origen t).
2. GLM Poisson General (global): un unico modelo para las 848 rutas (intercepto por ruta,
   coeficientes de calendario y rezagos compartidos).
3. GLM Poisson Especifico (por ruta): un modelo independiente para cada una de las 848 rutas
   (cada ruta aprende su propio patron de dia de semana, puentes, estacionalidad y rezagos).

Ambos modelos GLM se calibran con cuantiles conformalizados sobre residuos de validacion
para P05, P10, P50, P75, P90 y P95 (IC 80% = [P10, P90] e IC 90% = [P05, P95]).

Artefactos generados:
- models/modelo_referencia.joblib (contiene el modelo ganador + resumen de comparacion)
- results/metricas_referencia.csv
- results/metricas_por_ruta.csv
- results/pronosticos_30d.csv (horizonte 30 dias de marzo 2026 para las 848 rutas)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODULE_DIR = Path(__file__).resolve().parents[1]
SERIES_PARQUET = MODULE_DIR / "data" / "processed" / "series_diarias.parquet"
RUTAS_CSV = MODULE_DIR / "data" / "processed" / "rutas.csv"
MODELS_DIR = MODULE_DIR / "models"
RESULTS_DIR = MODULE_DIR / "results"

MODEL_PATH = MODELS_DIR / "modelo_referencia.joblib"
PRONOSTICOS_30D_CSV = RESULTS_DIR / "pronosticos_30d.csv"
METRICAS_REF_CSV = RESULTS_DIR / "metricas_referencia.csv"
METRICAS_RUTA_CSV = RESULTS_DIR / "metricas_por_ruta.csv"

VENTANAS_TEST: list[tuple[str, str, str]] = [
    ("W1_Oct2025_Receso", "2025-10-01", "2025-10-30"),
    ("W2_Nov2025_Puentes", "2025-10-31", "2025-11-29"),
    ("W3_Mar2026_Reciente", "2026-03-01", "2026-03-30"),
]

NUM_FEATURES = [
    "log_lag_35",
    "log_lag_42",
    "log_lag_49",
    "log_lag_56",
    "log_roll7_lag30",
    "log_roll28_lag30",
    "log_std28_lag30",
    "log_transito_roll28_lag30",
    "sin_anio",
    "cos_anio",
    "sin_sem",
    "cos_sem",
    "es_fin_semana",
    "es_festivo",
    "es_vispera_festivo",
    "es_puente",
    "es_semana_santa",
    "es_receso_octubre",
    "es_temporada_fin_anio",
    "es_temporada_mitad_anio",
    "log_dist_km",
]
NUM_FEATURES_RUTA = [c for c in NUM_FEATURES if c != "log_dist_km"]
CAT_FEATURES_GLOBAL = ["ruta_id", "dia_semana_num", "mes"]
CUANTILES = [0.05, 0.10, 0.50, 0.75, 0.90, 0.95]


def preparar_features_sin_fuga(
    series_path: Path | str = SERIES_PARQUET,
    rutas_path: Path | str = RUTAS_CSV,
) -> pd.DataFrame:
    """Construye tabla de features estrictamente causales (rezagos >= 30 dias dentro de cada bloque)."""
    rutas = pl.read_csv(rutas_path).select(["ruta_id", "dist_linea_recta_km", "sentido"])
    df = (
        pl.read_parquet(series_path)
        .sort(["ruta_id", "bloque_id", "fecha"])
        .join(rutas, on="ruta_id", how="left")
    )

    g = ["ruta_id", "bloque_id"]
    df = df.with_columns([
        pl.col("pasajeros_origen").shift(7).over(g).alias("lag_7"),
        pl.col("pasajeros_origen").shift(14).over(g).alias("lag_14"),
        pl.col("pasajeros_origen").shift(21).over(g).alias("lag_21"),
        pl.col("pasajeros_origen").shift(28).over(g).alias("lag_28"),
        pl.col("pasajeros_origen").shift(35).over(g).alias("lag_35"),
        pl.col("pasajeros_origen").shift(42).over(g).alias("lag_42"),
        pl.col("pasajeros_origen").shift(49).over(g).alias("lag_49"),
        pl.col("pasajeros_origen").shift(56).over(g).alias("lag_56"),
        pl.col("pasajeros_origen").rolling_mean(7).shift(30).over(g).alias("roll7_lag30"),
        pl.col("pasajeros_origen").rolling_mean(28).shift(30).over(g).alias("roll28_lag30"),
        pl.col("pasajeros_origen").rolling_std(28).shift(30).over(g).alias("std28_lag30"),
        pl.col("pasajeros_transito").rolling_mean(28).shift(30).over(g).alias("transito_roll28_lag30"),
    ])

    df = df.with_columns([
        pl.col("lag_35").log1p().alias("log_lag_35"),
        pl.col("lag_42").log1p().alias("log_lag_42"),
        pl.col("lag_49").log1p().alias("log_lag_49"),
        pl.col("lag_56").log1p().alias("log_lag_56"),
        pl.col("roll7_lag30").log1p().alias("log_roll7_lag30"),
        pl.col("roll28_lag30").log1p().alias("log_roll28_lag30"),
        pl.col("std28_lag30").fill_null(0.0).log1p().alias("log_std28_lag30"),
        pl.col("transito_roll28_lag30").fill_null(0.0).log1p().alias("log_transito_roll28_lag30"),
        (2.0 * np.pi * pl.col("dia_anio") / 365.25).sin().alias("sin_anio"),
        (2.0 * np.pi * pl.col("dia_anio") / 365.25).cos().alias("cos_anio"),
        (2.0 * np.pi * pl.col("dia_semana_num") / 7.0).sin().alias("sin_sem"),
        (2.0 * np.pi * pl.col("dia_semana_num") / 7.0).cos().alias("cos_sem"),
        pl.col("dist_linea_recta_km").fill_null(50.0).log1p().alias("log_dist_km"),
    ])

    df = df.filter(pl.col("log_lag_56").is_not_null() & pl.col("log_roll28_lag30").is_not_null())
    pdf = df.to_pandas()
    pdf["fecha"] = pd.to_datetime(pdf["fecha"])
    return pdf


def asignar_particion_y_horizonte(df: pd.DataFrame) -> pd.DataFrame:
    """Asigna particion temporal (train / val / test) y horizonte h (1..30) en test."""
    out = df.copy()
    out["split"] = "unused"
    out["ventana_test"] = ""
    out["horizonte"] = np.nan

    mask_train = (out["fecha"] <= "2023-12-31") | (
        (out["fecha"] >= "2025-01-01") & (out["fecha"] <= "2025-07-31")
    )
    out.loc[mask_train, "split"] = "train"

    mask_val = (out["fecha"] >= "2025-08-01") & (out["fecha"] <= "2025-09-30")
    out.loc[mask_val, "split"] = "val"

    for nombre, f_ini, f_fin in VENTANAS_TEST:
        m = (out["fecha"] >= f_ini) & (out["fecha"] <= f_fin)
        out.loc[m, "split"] = "test"
        out.loc[m, "ventana_test"] = nombre
        out.loc[m, "horizonte"] = (out.loc[m, "fecha"] - pd.Timestamp(f_ini)).dt.days + 1

    return out


def predecir_ingenuo_estacional(df_test: pd.DataFrame) -> np.ndarray:
    """Ingenuo estacional sin fuga: mismo dia de la semana mas reciente conocido en el origen t."""
    h = df_test["horizonte"].to_numpy()
    pred = np.where(
        h <= 7,
        df_test["lag_7"].to_numpy(),
        np.where(
            h <= 14,
            df_test["lag_14"].to_numpy(),
            np.where(
                h <= 21,
                df_test["lag_21"].to_numpy(),
                np.where(h <= 28, df_test["lag_28"].to_numpy(), df_test["lag_35"].to_numpy()),
            ),
        ),
    )
    return np.maximum(np.nan_to_num(pred, nan=0.0), 0.0)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def calcular_metricas(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mase_denom: float,
    cuantiles_dict: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mase = mae / max(mase_denom, 1e-6)
    res = {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MASE": round(mase, 4)}

    if cuantiles_dict is not None:
        for q, key in [(0.50, "p50"), (0.75, "p75"), (0.90, "p90"), (0.95, "p95")]:
            res[f"Pinball_{key}"] = round(pinball_loss(y_true, cuantiles_dict[key], q), 2)
        cob80 = np.mean((y_true >= cuantiles_dict["p10"]) & (y_true <= cuantiles_dict["p90"])) * 100
        cob90 = np.mean((y_true >= cuantiles_dict["p05"]) & (y_true <= cuantiles_dict["p95"])) * 100
        res["Cobertura_IC80_pct"] = round(float(cob80), 2)
        res["Cobertura_IC90_pct"] = round(float(cob90), 2)
    return res


def _calibrar_cuantiles_por_tier(
    y_val_log: np.ndarray,
    val_pred_log: np.ndarray,
    val_tiers: np.ndarray,
    test_pred_log: np.ndarray,
    test_tiers: np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Calibra cuantiles conformalizados en escala log1p por tercil de volumen."""
    val_resid = y_val_log - val_pred_log
    offsets: dict[str, dict[str, float]] = {}
    for tier in ["baja", "media", "alta"]:
        r_t = val_resid[val_tiers == tier]
        offsets[tier] = {f"p{int(q * 100):02d}": float(np.quantile(r_t, q)) for q in CUANTILES}

    preds_q: dict[str, np.ndarray] = {}
    for q in CUANTILES:
        key = f"p{int(q * 100):02d}"
        offs = np.array([offsets[t][key] for t in test_tiers])
        preds_q[key] = np.maximum(np.expm1(np.clip(test_pred_log + offs, 0.0, 12.0)), 0.0)
    return offsets, preds_q


def _entrenar_poisson_por_ruta(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    alpha: float = 0.2,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Ajusta un GLM Poisson independiente (especifico) para cada una de las rutas."""
    prep_ruta = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES_RUTA),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["dia_semana_num", "mes"]),
        ]
    )
    X_tr_all = prep_ruta.fit_transform(train_df[NUM_FEATURES_RUTA + ["dia_semana_num", "mes"]])
    X_va_all = prep_ruta.transform(val_df[NUM_FEATURES_RUTA + ["dia_semana_num", "mes"]])
    X_te_all = prep_ruta.transform(test_df[NUM_FEATURES_RUTA + ["dia_semana_num", "mes"]])

    y_tr_all = train_df["pasajeros_origen"].to_numpy(dtype=float)

    tr_groups = train_df.groupby("ruta_id").indices
    va_groups = val_df.groupby("ruta_id").indices
    te_groups = test_df.groupby("ruta_id").indices

    pred_val = np.zeros(len(val_df), dtype=float)
    pred_test = np.zeros(len(test_df), dtype=float)
    modelos_ruta: dict[str, PoissonRegressor] = {}

    for rid, tr_idx in tr_groups.items():
        m = PoissonRegressor(alpha=alpha, max_iter=200)
        m.fit(X_tr_all[tr_idx], y_tr_all[tr_idx])
        modelos_ruta[str(rid)] = m
        if rid in va_groups:
            vi = va_groups[rid]
            pred_val[vi] = np.maximum(m.predict(X_va_all[vi]), 0.0)
        if rid in te_groups:
            ti = te_groups[rid]
            pred_test[ti] = np.maximum(m.predict(X_te_all[ti]), 0.0)

    artefacto_ruta = {"preprocessor": prep_ruta, "modelos_por_ruta": modelos_ruta, "alpha": alpha}
    return artefacto_ruta, pred_val, pred_test


def entrenar_y_evaluar() -> dict[str, Any]:
    """Entrena y compara Ingenuo, GLM Poisson General (global) y GLM Poisson Especifico (por ruta)."""
    df = preparar_features_sin_fuga()
    df = asignar_particion_y_horizonte(df)

    train_df = df[(df["split"] == "train") & (df["dia_sin_registro_origen"] == 0)].reset_index(drop=True)
    val_df = df[(df["split"] == "val") & (df["dia_sin_registro_origen"] == 0)].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    mase_denom_global = float(np.mean(np.abs(train_df["pasajeros_origen"] - train_df["lag_7"])))
    y_train = train_df["pasajeros_origen"].to_numpy(dtype=float)
    y_val = val_df["pasajeros_origen"].to_numpy(dtype=float)
    y_val_log = np.log1p(y_val)
    y_test = test_df["pasajeros_origen"].to_numpy(dtype=float)

    ruta_media_train = train_df.groupby("ruta_id")["pasajeros_origen"].mean()
    q1_vol, q2_vol = float(ruta_media_train.quantile(1 / 3)), float(ruta_media_train.quantile(2 / 3))

    def asignar_tier(rutas_series: pd.Series) -> np.ndarray:
        medias = rutas_series.map(ruta_media_train).fillna(q1_vol).to_numpy()
        return np.where(medias <= q1_vol, "baja", np.where(medias <= q2_vol, "media", "alta"))

    val_tiers = asignar_tier(val_df["ruta_id"])
    test_tiers = asignar_tier(test_df["ruta_id"])

    # 1. Ingenuo Estacional
    pred_naive_test = predecir_ingenuo_estacional(test_df)

    # 2. GLM Poisson General (global)
    prep_global = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES_GLOBAL),
        ]
    )
    glm_general = Pipeline([
        ("prep", prep_global),
        ("model", PoissonRegressor(alpha=1e-3, max_iter=300)),
    ])
    glm_general.fit(train_df[NUM_FEATURES + CAT_FEATURES_GLOBAL], y_train)
    pred_val_gen = np.maximum(glm_general.predict(val_df[NUM_FEATURES + CAT_FEATURES_GLOBAL]), 0.0)
    pred_test_gen = np.maximum(glm_general.predict(test_df[NUM_FEATURES + CAT_FEATURES_GLOBAL]), 0.0)
    mae_val_gen = float(np.mean(np.abs(y_val - pred_val_gen)))

    offsets_gen, q_test_gen = _calibrar_cuantiles_por_tier(
        y_val_log, np.log1p(pred_val_gen), val_tiers, np.log1p(pred_test_gen), test_tiers
    )

    # 3. GLM Poisson Especifico (un modelo por ruta)
    glm_esp_art, pred_val_esp, pred_test_esp = _entrenar_poisson_por_ruta(train_df, val_df, test_df, alpha=0.2)
    mae_val_esp = float(np.mean(np.abs(y_val - pred_val_esp)))

    offsets_esp, q_test_esp = _calibrar_cuantiles_por_tier(
        y_val_log, np.log1p(pred_val_esp), val_tiers, np.log1p(pred_test_esp), test_tiers
    )

    print(
        f"Validacion MAE | GLM Poisson General (global): {mae_val_gen:.2f} | "
        f"GLM Poisson Especifico (por ruta): {mae_val_esp:.2f}"
    )

    # Seleccionar ganador por MAE en validacion
    if mae_val_esp <= mae_val_gen:
        nombre_ganador = "GLM_Poisson_Especifico_PorRuta"
        q_ganador = q_test_esp
        modelo_guardado = glm_esp_art
        offsets_ganador = offsets_esp
    else:
        nombre_ganador = "GLM_Poisson_General_Global"
        q_ganador = q_test_gen
        modelo_guardado = glm_general
        offsets_ganador = offsets_gen

    # Metricas agregadas y por bloque de horizonte para los 3 enfoques
    filas_metricas = []
    for grupo, mask in [
        ("Global (dias 1-30)", np.ones(len(test_df), dtype=bool)),
        ("Horizonte dias 1-7", test_df["horizonte"].to_numpy() <= 7),
        ("Horizonte dias 8-14", (test_df["horizonte"].to_numpy() >= 8) & (test_df["horizonte"].to_numpy() <= 14)),
        ("Horizonte dias 15-30", test_df["horizonte"].to_numpy() >= 15),
    ]:
        filas_metricas.append({
            "modelo": "Ingenuo_Estacional",
            "tipo": "Base",
            "segmento": grupo,
            **calcular_metricas(y_test[mask], pred_naive_test[mask], mase_denom_global),
        })
        q_g_sub = {k: v[mask] for k, v in q_test_gen.items()}
        filas_metricas.append({
            "modelo": "GLM_Poisson_General_Global",
            "tipo": "General (global)",
            "segmento": grupo,
            **calcular_metricas(y_test[mask], q_g_sub["p50"], mase_denom_global, q_g_sub),
        })
        q_e_sub = {k: v[mask] for k, v in q_test_esp.items()}
        filas_metricas.append({
            "modelo": "GLM_Poisson_Especifico_PorRuta",
            "tipo": "Específico (por ruta)",
            "segmento": grupo,
            **calcular_metricas(y_test[mask], q_e_sub["p50"], mase_denom_global, q_e_sub),
        })

    metricas_df = pd.DataFrame(filas_metricas)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metricas_df.to_csv(METRICAS_REF_CSV, index=False, encoding="utf-8")

    # Metricas por ruta (comparando General vs Especifico)
    test_eval = test_df[["ruta_id", "fecha", "ventana_test", "horizonte", "pasajeros_origen"]].copy()
    test_eval["pred_naive"] = pred_naive_test
    test_eval["p50_general"] = np.round(q_test_gen["p50"], 1)
    test_eval["p50_especifico"] = np.round(q_test_esp["p50"], 1)
    for k in ["p05", "p10", "p50", "p75", "p90", "p95"]:
        test_eval[k] = np.round(q_ganador[k], 1)

    mase_ruta_map = (
        train_df.assign(err_naive=(train_df["pasajeros_origen"] - train_df["lag_7"]).abs())
        .groupby("ruta_id")["err_naive"]
        .mean()
        .to_dict()
    )
    filas_ruta = []
    for rid, g_r in test_eval.groupby("ruta_id"):
        yt = g_r["pasajeros_origen"].to_numpy(dtype=float)
        denom = mase_ruta_map.get(rid, mase_denom_global)
        m_n = calcular_metricas(yt, g_r["pred_naive"].to_numpy(), denom)
        m_g = calcular_metricas(yt, g_r["p50_general"].to_numpy(), denom)
        m_e = calcular_metricas(yt, g_r["p50_especifico"].to_numpy(), denom)
        m_w = calcular_metricas(
            yt,
            g_r["p50"].to_numpy(),
            denom,
            {k: g_r[k].to_numpy() for k in ["p05", "p10", "p50", "p75", "p90", "p95"]},
        )
        filas_ruta.append({
            "ruta_id": rid,
            "MAE_naive": m_n["MAE"],
            "MAE_general": m_g["MAE"],
            "MAE_especifico": m_e["MAE"],
            "MAE_ref": m_w["MAE"],
            "RMSE_ref": m_w["RMSE"],
            "MASE_ref": m_w["MASE"],
            "Cobertura_IC80_pct": m_w["Cobertura_IC80_pct"],
            "Cobertura_IC90_pct": m_w["Cobertura_IC90_pct"],
        })
    metricas_ruta_df = pd.DataFrame(filas_ruta)
    metricas_ruta_df.to_csv(METRICAS_RUTA_CSV, index=False, encoding="utf-8")

    # Guardar pronosticos 30d (Marzo 2026) con el modelo ganador + columna p50 del otro enfoque para transparencia
    mar26 = test_eval[test_eval["ventana_test"] == "W3_Mar2026_Reciente"].copy()
    mar26["modelo"] = nombre_ganador
    mar26["fecha"] = mar26["fecha"].dt.strftime("%Y-%m-%d")
    mar26 = mar26.rename(columns={"pasajeros_origen": "real"})[
        ["ruta_id", "fecha", "modelo", "p05", "p10", "p50", "p75", "p90", "p95", "p50_general", "p50_especifico", "real"]
    ].sort_values(["ruta_id", "fecha"])
    mar26.to_csv(PRONOSTICOS_30D_CSV, index=False, encoding="utf-8")

    # Guardar artefacto en models/
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artefacto = {
        "nombre_modelo": nombre_ganador,
        "modelo": modelo_guardado,
        "offsets_cuantiles": offsets_ganador,
        "ruta_media_train": ruta_media_train.to_dict(),
        "umbrales_tier": (q1_vol, q2_vol),
        "mae_val_general": mae_val_gen,
        "mae_val_especifico": mae_val_esp,
    }
    joblib.dump(artefacto, MODEL_PATH, compress=3)

    return {
        "nombre_ganador": nombre_ganador,
        "metricas": metricas_df,
        "metricas_por_ruta": metricas_ruta_df,
        "mar26": mar26,
    }


if __name__ == "__main__":
    res = entrenar_y_evaluar()
    print(f"\nGanador guardado en {MODEL_PATH}: {res['nombre_ganador']} ({MODEL_PATH.stat().st_size / 1024:.1f} KB)")
    print("\n=== METRICAS EN PRUEBA: INGENUO vs GENERAL (GLOBAL) vs ESPECIFICO (POR RUTA) ===")
    print(res["metricas"].to_string(index=False))
