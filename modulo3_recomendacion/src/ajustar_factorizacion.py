"""Ajuste y evaluación final de la factorización de matrices con BPR (tarea 4).

Fase 1 — AJUSTE (aprende con entrenamiento, se mide en validación):
    9 configuraciones (d ∈ {8, 16, 32} × λ ∈ {0,01; 0,1; 1}) × 3 semillas. Cada corrida para
    cuando NDCG@10 en validación no mejora en 10 épocas (máx. 200) y guarda su mejor época.
    Se elige la configuración con mayor NDCG@10 medio. Si ni la mejor supera al azar en
    validación, se elige la más simple (d = 8, λ = 1): no tiene sentido escoger entre ruidos.
    Número de épocas para la fase 2 = mediana de las mejores épocas de esa configuración.

Fase 2 — FINAL (aprende con entrenamiento + validación, se mide en prueba una sola vez):
    La configuración elegida, con el número de épocas fijo, 30 semillas.

Comparaciones (regla del paso 7: los usuarios son una muestra):
    - contra el azar: prueba binomial unilateral con K = 10;
    - contra la popularidad: prueba PAREADA por usuario. Para cada usuario se calcula la
      diferencia (acierto medio de la factorización − acierto medio de la popularidad) sobre las
      30 semillas, y se hace una prueba de permutación de signos (10 000 permutaciones): bajo H0
      no hay diferencia, así que el signo de cada diferencia es como lanzar una moneda.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/ajustar_factorizacion.py

Salidas: results/tarea4/fm_*.csv, results/tarea4/RESUMEN_factorizacion.md,
         models/factorizacion_bpr.npz (+ .json) y models/popularidad.npz.
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from evaluacion import N_DESTINOS, cargar_escenario, evaluar
from factorizacion import FactorizacionBPR
from inventario import _a_markdown
from referencias import Azar, Popularidad, prueba_lugar

MODULO = Path(__file__).resolve().parents[1]
SALIDA = MODULO / "results" / "tarea4"
MODELOS = MODULO / "models"
DS, LAMS = (8, 16, 32), (0.01, 0.1, 1.0)
SEMILLAS_AJUSTE, N_SEMILLAS = 3, 30
EPOCAS_MAX, PACIENCIA = 200, 10

torch.set_num_threads(1)  # corridas pequeñas: un hilo es más rápido y más reproducible


def p_azar(aprende, evalua, k=10):
    n_u = evalua.UserID.map(aprende.groupby("UserID").size()).fillna(0).to_numpy()
    return float(np.mean(k / (N_DESTINOS - n_u)))


def fase_ajuste():
    aprende, val = cargar_escenario("ajuste")
    filas = []
    for d, lam in itertools.product(DS, LAMS):
        for s in range(SEMILLAS_AJUSTE):
            t0 = time.perf_counter()
            m = FactorizacionBPR(d=d, lam=lam, epocas=EPOCAS_MAX, semilla=s).ajustar(aprende, validacion=val, paciencia=PACIENCIA)
            r = evaluar(m.puntuar, aprende, val, semilla=s, ks=(10,))[0].iloc[0]
            filas.append({"d": d, "lambda": lam, "semilla": s, "mejor_epoca": m.mejor_epoca,
                          "epocas_corridas": len(m.historia), "NDCG@10_val": r.NDCG,
                          "HitRate@10_val": r.HitRate, "segundos": time.perf_counter() - t0,
                          "parametros": m.parametros()})
            print(f"d={d:>2} λ={lam:<4} semilla={s}: mejor época {m.mejor_epoca:>3}, "
                  f"HitRate@10 val {100 * r.HitRate:.2f} %, {filas[-1]['segundos']:.1f} s")
    corridas = pd.DataFrame(filas)
    config = (corridas.groupby(["d", "lambda"])
              .agg(NDCG_val_media=("NDCG@10_val", "mean"), HitRate_val_media=("HitRate@10_val", "mean"),
                   HitRate_val_de=("HitRate@10_val", "std"), mediana_mejor_epoca=("mejor_epoca", "median"),
                   segundos_total=("segundos", "sum"), parametros=("parametros", "first"))
              .reset_index().sort_values("NDCG_val_media", ascending=False))

    # ¿La mejor configuración supera al azar en validación? (binomial unilateral, K = 10)
    mejor = config.iloc[0]
    n = len(val)
    aciertos = int(round(mejor.HitRate_val_media * n))
    p0 = p_azar(aprende, val)
    p_valor = stats.binomtest(aciertos, n, p0, alternative="greater").pvalue
    if p_valor < 0.05:
        elegida, motivo = mejor, f"mayor NDCG@10 medio en validación y supera al azar (p-valor = {p_valor:.3f})"
    else:
        elegida = config[(config.d == min(DS)) & (config["lambda"] == max(LAMS))].iloc[0]
        motivo = (f"ninguna configuración supera al azar en validación (la mejor, d = {int(mejor.d)}, "
                  f"λ = {mejor['lambda']:g}, acierta {aciertos} de {n} frente a {n * p0:.1f} esperados; "
                  f"p-valor = {p_valor:.3f}), así que se elige la más simple")
    decision = {"d": int(elegida.d), "lambda": float(elegida["lambda"]),
                "epocas": int(round(elegida.mediana_mejor_epoca)), "motivo": motivo,
                "configuraciones": len(config), "corridas": len(corridas),
                "segundos_ajuste": round(float(corridas.segundos.sum()), 1)}
    return corridas, config, decision


def fase_final(decision):
    aprende, pru = cargar_escenario("final")
    filas, aciertos_fm, aciertos_pop, tiempos = [], [], [], []
    lugar_fm, lugar_pop = [], []  # posición del lugar escondido, por semilla
    for s in range(N_SEMILLAS):
        t0 = time.perf_counter()
        m = FactorizacionBPR(d=decision["d"], lam=decision["lambda"], epocas=decision["epocas"], semilla=s).ajustar(aprende)
        tiempos.append(time.perf_counter() - t0)
        res, por_u = evaluar(m.puntuar, aprende, pru, semilla=s)
        res.insert(0, "semilla", s)
        filas.append(res)
        aciertos_fm.append((por_u.rango < 10).to_numpy())
        lugar_fm.append(por_u.rango_lugar.to_numpy())
        if s == 0:
            modelo_0 = m
        pop = Popularidad().ajustar(aprende)
        por_u_pop = evaluar(pop.puntuar, aprende, pru, semilla=s)[1]
        aciertos_pop.append((por_u_pop.rango < 10).to_numpy())
        lugar_pop.append(por_u_pop.rango_lugar.to_numpy())
    por_semilla = pd.concat(filas, ignore_index=True)

    # Comparaciones con K = 10
    n = len(pru)
    p0 = p_azar(aprende, pru)
    hit_fm = np.mean(aciertos_fm, axis=0)    # acierto medio por usuario (30 semillas)
    hit_pop = np.mean(aciertos_pop, axis=0)
    aciertos = int(round(hit_fm.sum()))
    p_vs_azar = stats.binomtest(aciertos, n, p0, alternative="greater").pvalue
    dif = hit_fm - hit_pop
    rng = np.random.default_rng(0)
    signos = rng.choice([-1, 1], size=(10_000, n))
    nulas = (signos * dif).mean(axis=1)
    p_vs_pop = float(np.mean(np.abs(nulas) >= abs(dif.mean())))
    comparacion = {
        "usuarios": n, "aciertos_fm_media": round(float(hit_fm.sum()), 1),
        "aciertos_popularidad_media": round(float(hit_pop.sum()), 1),
        "esperados_por_azar": round(n * p0, 1), "p_valor_vs_azar": round(p_vs_azar, 3),
        "usuarios_donde_fm_acierta_mas": int((dif > 0).sum()),
        "usuarios_donde_popularidad_acierta_mas": int((dif < 0).sum()),
        "p_valor_vs_popularidad_pareada": round(p_vs_pop, 3),
        "segundos_por_entrenamiento": round(float(np.mean(tiempos)), 2),
    }
    l3_fm, l1_fm = np.mean(np.array(lugar_fm) < 3, axis=0), np.mean(np.array(lugar_fm) < 1, axis=0)
    comparacion.update(prueba_lugar(l3_fm, l1_fm))
    dif3 = l3_fm - np.mean(np.array(lugar_pop) < 3, axis=0)
    nulas3 = (rng.choice([-1, 1], size=(10_000, n)) * dif3).mean(axis=1)
    comparacion["lugar@3: p-valor vs popularidad (pareada)"] = round(float(np.mean(np.abs(nulas3) >= abs(dif3.mean()))), 3)
    return por_semilla, comparacion, modelo_0, aprende


def guardar_modelos(modelo, decision, aprende):
    MODELOS.mkdir(parents=True, exist_ok=True)
    P, Q = modelo.factores()
    np.savez(MODELOS / "factorizacion_bpr.npz", P=P, Q=Q)
    meta = {**{k: decision[k] for k in ("d", "lambda", "epocas")}, "semilla": 0,
            "aprende_con": "entrenamiento + validación", "parametros": modelo.parametros(),
            "uso": "puntaje(u, i) = P[u-1] · Q[i-1]; UserID y DestinationID empiezan en 1"}
    (MODELOS / "factorizacion_bpr.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    pop = Popularidad().ajustar(aprende)
    np.savez(MODELOS / "popularidad.npz", conteo=pop.conteo)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    corridas, config, decision = fase_ajuste()
    print(f"\nElegida: d = {decision['d']}, λ = {decision['lambda']:g}, {decision['epocas']} épocas — {decision['motivo']}\n")
    por_semilla, comparacion, modelo_0, aprende = fase_final(decision)
    guardar_modelos(modelo_0, decision, aprende)

    corridas.to_csv(SALIDA / "fm_ajuste_corridas.csv", index=False)
    config.to_csv(SALIDA / "fm_ajuste_configuraciones.csv", index=False)
    por_semilla.to_csv(SALIDA / "fm_final_por_semilla.csv", index=False)
    pd.DataFrame([comparacion]).to_csv(SALIDA / "fm_comparacion.csv", index=False)
    (SALIDA / "fm_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    t = stats.t.ppf(0.975, N_SEMILLAS - 1)
    MET = ["HitRate", "Precision", "NDCG", "Cobertura", "HitRate_lugar@3", "HitRate_lugar@1"]
    resumen = por_semilla.groupby("K")[MET].agg(["mean", "std"])
    filas = []
    for k in resumen.index:
        fila = {"K": k}
        for m in MET:
            media, de = resumen.loc[k, (m, "mean")], resumen.loc[k, (m, "std")]
            fila[f"{m} (%)"] = round(100 * media, 2)
            if m == "HitRate":
                fila["IC 95 % entre semillas"] = f"{100 * (media - t * de / np.sqrt(N_SEMILLAS)):.2f}–{100 * (media + t * de / np.sqrt(N_SEMILLAS)):.2f}"
        filas.append(fila)
    final = pd.DataFrame(filas)
    final.to_csv(SALIDA / "fm_final_resumen.csv", index=False)

    cfg = config.copy()
    for c in ("NDCG_val_media", "HitRate_val_media", "HitRate_val_de"):
        cfg[c] = (100 * cfg[c]).round(2)
    cfg["segundos_total"] = cfg.segundos_total.round(1)
    texto = [
        "# Factorización de matrices con BPR (tarea 4)\n",
        "_Generado por `src/ajustar_factorizacion.py`. No editar a mano._\n",
        f"## 1. Ajuste: {decision['configuraciones']} configuraciones × {SEMILLAS_AJUSTE} semillas "
        f"({decision['corridas']} corridas, {decision['segundos_ajuste']} s)\n",
        "Valores en %. Ordenadas por NDCG@10 medio en validación (326 usuarios).\n",
        _a_markdown(cfg) + "\n",
        f"**Elegida:** d = {decision['d']}, λ = {decision['lambda']:g}, {decision['epocas']} épocas. "
        f"Motivo: {decision['motivo']}.\n",
        f"## 2. Evaluación final: {N_SEMILLAS} semillas, entrenamiento + validación → prueba\n",
        _a_markdown(final) + "\n",
        "## 3. Comparaciones con K = 10 (los usuarios como muestra)\n",
        _a_markdown(pd.DataFrame([comparacion])) + "\n",
        f"_Tiempo total: {time.perf_counter() - t0:.0f} s. Modelo guardado (semilla 0): "
        "`models/factorizacion_bpr.npz`._\n",
    ]
    (SALIDA / "RESUMEN_factorizacion.md").write_text("\n".join(texto), encoding="utf-8")
    print(cfg.to_string(index=False))
    print("\n" + final.to_string(index=False))
    print("\n" + pd.Series(comparacion).to_string())
    print(f"\nTiempo total: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
