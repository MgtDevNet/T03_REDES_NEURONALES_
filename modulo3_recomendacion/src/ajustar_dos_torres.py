"""Ajuste y evaluación final de la red de dos torres (tarea 5). Reglas fijadas ANTES de correr
(DECISIONES.md, «Detalle del ajuste de la red»):

Fase 1 — AJUSTE (entrenamiento → validación):
    d ∈ {2, 4, 8} × pérdida ∈ {BCE, BPR} × negativos ∈ {1, 4} × usuario ∈ {preferencias, completo}
    = 24 configuraciones × 3 semillas. Parada temprana por HitRate@3 por lugar en validación.
    Se elige la de mayor HitRate@3 por lugar medio; si no supera al azar en validación (binomial
    contra 60 %), la más simple: d = 2, preferencias, 1 negativo, BPR.

Fase 2 — FINAL (entrenamiento + validación → prueba), 30 semillas, épocas fijas (mediana).

Comparaciones:
    - PRINCIPAL (fijada de antemano): HitRate@3 por lugar contra el azar, binomial; junto con
      HitRate@1 por lugar, con Bonferroni m = 2.
    - Secundarias: HitRate@10 por ID contra el azar; pareadas por usuario (permutación de signos)
      contra la popularidad y la factorización en HitRate@3 por lugar.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/ajustar_dos_torres.py
"""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from ajustar_factorizacion import p_azar
from dos_torres import DosTorres
from evaluacion import cargar_escenario, evaluar
from factorizacion import FactorizacionBPR
from inventario import _a_markdown
from referencias import Popularidad, prueba_lugar

MODULO = Path(__file__).resolve().parents[1]
SALIDA = MODULO / "results" / "tarea5"
MODELOS = MODULO / "models"
DS, PERDIDAS, NEGATIVOS, USUARIO = (2, 4, 8), ("bce", "bpr"), (1, 4), ("preferencias", "completo")
MAS_SIMPLE = {"d": 2, "perdida": "bpr", "negativos": 1, "usuario": "preferencias"}
SEMILLAS_AJUSTE, N_SEMILLAS = 3, 30
METRICA = "HitRate_lugar@3"

torch.set_num_threads(1)


def fase_ajuste():
    aprende, val = cargar_escenario("ajuste")
    filas = []
    for d, perdida, neg, usuario in itertools.product(DS, PERDIDAS, NEGATIVOS, USUARIO):
        for s in range(SEMILLAS_AJUSTE):
            t0 = time.perf_counter()
            m = DosTorres(d=d, perdida=perdida, negativos=neg, usuario=usuario, semilla=s).ajustar(aprende, validacion=val)
            r = evaluar(m.puntuar, aprende, val, semilla=s, ks=(10,))[0].iloc[0]
            filas.append({"d": d, "perdida": perdida, "negativos": neg, "usuario": usuario, "semilla": s,
                          "mejor_epoca": m.mejor_epoca, "Lugar@3_val": r[METRICA], "Lugar@1_val": r["HitRate_lugar@1"],
                          "HitRate@10_val": r.HitRate, "segundos": time.perf_counter() - t0, "parametros": m.parametros()})
        g = pd.DataFrame(filas[-SEMILLAS_AJUSTE:])
        print(f"d={d} {perdida.upper():3} neg={neg} {usuario:<12}: Lugar@3 val {100 * g['Lugar@3_val'].mean():.1f} %")
    corridas = pd.DataFrame(filas)
    claves = ["d", "perdida", "negativos", "usuario"]
    config = (corridas.groupby(claves)
              .agg(Lugar3_val=("Lugar@3_val", "mean"), Lugar1_val=("Lugar@1_val", "mean"),
                   HitRate10_val=("HitRate@10_val", "mean"), mediana_mejor_epoca=("mejor_epoca", "median"),
                   segundos_total=("segundos", "sum"), parametros=("parametros", "first"))
              .reset_index().sort_values("Lugar3_val", ascending=False))
    mejor = config.iloc[0]
    n = len(val)
    k = int(round(mejor.Lugar3_val * n))
    p_val = stats.binomtest(k, n, 0.6, alternative="greater").pvalue
    if p_val < 0.05:
        elegida, motivo = mejor, f"mayor HitRate@3 por lugar en validación y supera al azar ({k} de {n}; p-valor = {p_val:.3f})"
    else:
        elegida = config[(config.d == MAS_SIMPLE["d"]) & (config.perdida == MAS_SIMPLE["perdida"])
                         & (config.negativos == MAS_SIMPLE["negativos"]) & (config.usuario == MAS_SIMPLE["usuario"])].iloc[0]
        motivo = (f"ninguna supera al azar en validación (la mejor, d = {int(mejor.d)}, {mejor.perdida.upper()}, "
                  f"{int(mejor.negativos)} neg., {mejor.usuario}, acierta {k} de {n} frente a {0.6 * n:.1f} "
                  f"esperados; p-valor = {p_val:.3f}), así que se elige la más simple")
    decision = {"d": int(elegida.d), "perdida": elegida.perdida, "negativos": int(elegida.negativos),
                "usuario": elegida.usuario, "epocas": int(round(elegida.mediana_mejor_epoca)), "motivo": motivo,
                "configuraciones": len(config), "corridas": len(corridas),
                "segundos_ajuste": round(float(corridas.segundos.sum()), 1)}
    return corridas, config, decision


def permutacion_pareada(dif, rng, n_perm=10_000):
    nulas = (rng.choice([-1, 1], size=(n_perm, len(dif))) * dif).mean(axis=1)
    return float(np.mean(np.abs(nulas) >= abs(dif.mean())))


def fase_final(decision):
    aprende, pru = cargar_escenario("final")
    fm_dec = json.loads((MODULO / "results" / "tarea4" / "fm_decision.json").read_text(encoding="utf-8"))
    filas, tiempos = [], []
    pos = {"red": [], "pop": [], "fm": []}  # (rango por ID, rango por lugar) por semilla
    for s in range(N_SEMILLAS):
        t0 = time.perf_counter()
        red = DosTorres(d=decision["d"], perdida=decision["perdida"], negativos=decision["negativos"],
                        usuario=decision["usuario"], semilla=s, epocas=decision["epocas"]).ajustar(aprende)
        tiempos.append(time.perf_counter() - t0)
        res, por_u = evaluar(red.puntuar, aprende, pru, semilla=s)
        res.insert(0, "semilla", s)
        filas.append(res)
        pos["red"].append(por_u)
        if s == 0:
            red_0 = red
        pos["pop"].append(evaluar(Popularidad().ajustar(aprende).puntuar, aprende, pru, semilla=s)[1])
        fm = FactorizacionBPR(d=fm_dec["d"], lam=fm_dec["lambda"], epocas=fm_dec["epocas"], semilla=s).ajustar(aprende)
        pos["fm"].append(evaluar(fm.puntuar, aprende, pru, semilla=s)[1])
    por_semilla = pd.concat(filas, ignore_index=True)

    def media(modelo, col, corte):
        return np.mean([(p[col] < corte).to_numpy() for p in pos[modelo]], axis=0)

    n = len(pru)
    rng = np.random.default_rng(0)
    l3, l1, h10 = media("red", "rango_lugar", 3), media("red", "rango_lugar", 1), media("red", "rango", 10)
    comp = {"usuarios": n, **prueba_lugar(l3, l1)}
    k10 = int(round(h10.sum()))
    comp["ID@10: aciertos / esperados"] = f"{k10} / {n * p_azar(aprende, pru):.1f}"
    comp["ID@10: p-valor vs azar"] = round(stats.binomtest(k10, n, p_azar(aprende, pru), alternative="greater").pvalue, 3)
    for otro, nombre in (("pop", "popularidad"), ("fm", "factorización")):
        comp[f"lugar@3: red − {nombre} (puntos)"] = round(100 * (l3.mean() - media(otro, "rango_lugar", 3).mean()), 2)
        comp[f"lugar@3: p-valor vs {nombre} (pareada)"] = round(permutacion_pareada(l3 - media(otro, "rango_lugar", 3), rng), 3)
    comp["segundos_por_entrenamiento"] = round(float(np.mean(tiempos)), 2)
    comp["parametros"] = red_0.parametros()
    return por_semilla, comp, red_0


def guardar(red, decision):
    MODELOS.mkdir(parents=True, exist_ok=True)
    torch.save(red.red.state_dict(), MODELOS / "dos_torres.pt")
    meta = {**{k: decision[k] for k in ("d", "perdida", "negativos", "usuario", "epocas")}, "semilla": 0,
            "oculta": red.oculta, "aprende_con": "entrenamiento + validación", "parametros": red.parametros(),
            "uso": "dos_torres.DosTorres(**config) y luego red.red.load_state_dict(torch.load(...)); "
                   "puntuar_variables(variables_usuario(perfil, usuario)) sirve para perfiles nuevos"}
    (MODELOS / "dos_torres.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    corridas, config, decision = fase_ajuste()
    print(f"\nElegida: {decision}\n")
    por_semilla, comp, red_0 = fase_final(decision)
    guardar(red_0, decision)
    pd.DataFrame(red_0.historia).to_csv(SALIDA / "red_curva_final_semilla0.csv", index=False)

    corridas.to_csv(SALIDA / "red_ajuste_corridas.csv", index=False)
    config.to_csv(SALIDA / "red_ajuste_configuraciones.csv", index=False)
    por_semilla.to_csv(SALIDA / "red_final_por_semilla.csv", index=False)
    pd.DataFrame([comp]).to_csv(SALIDA / "red_comparacion.csv", index=False)
    (SALIDA / "red_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    MET = ["HitRate", "Precision", "NDCG", "Cobertura", "HitRate_lugar@3", "HitRate_lugar@1"]
    final = (100 * por_semilla.groupby("K")[MET].mean()).round(2).reset_index()
    final.to_csv(SALIDA / "red_final_resumen.csv", index=False)
    cfg = config.copy()
    for c in ("Lugar3_val", "Lugar1_val", "HitRate10_val"):
        cfg[c] = (100 * cfg[c]).round(1)
    cfg["segundos_total"] = cfg.segundos_total.round(1)
    texto = [
        "# Red de dos torres (tarea 5)\n", "_Generado por `src/ajustar_dos_torres.py`. No editar a mano._\n",
        f"## 1. Ajuste: {decision['configuraciones']} configuraciones × {SEMILLAS_AJUSTE} semillas "
        f"({decision['corridas']} corridas, {decision['segundos_ajuste']} s)\n",
        "Valores en %, validación (326 usuarios). Ordenadas por HitRate@3 por lugar.\n", _a_markdown(cfg) + "\n",
        f"**Elegida:** {decision}\n",
        f"## 2. Evaluación final: {N_SEMILLAS} semillas, entrenamiento + validación → prueba (valores en %)\n",
        _a_markdown(final) + "\n", "## 3. Comparaciones\n", _a_markdown(pd.DataFrame([comp])) + "\n",
        f"_Tiempo total: {time.perf_counter() - t0:.0f} s._\n",
    ]
    (SALIDA / "RESUMEN.md").write_text("\n".join(texto), encoding="utf-8")
    print(cfg.head(10).to_string(index=False))
    print("\n" + final.to_string(index=False))
    print("\n" + pd.Series(comp).to_string())
    print(f"\nTiempo total: {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
