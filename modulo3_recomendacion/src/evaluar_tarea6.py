"""Tarea 6 (parte 1): desempeño por grupo de usuarios y distribución sobre las 30 semillas.

Vuelve a entrenar y evaluar los 4 modelos en el momento final (entrenamiento + validación →
prueba), con las configuraciones ya elegidas, y guarda el resultado de CADA usuario. Así se puede:
- separar a los usuarios en dos grupos según cuántas interacciones tenían para aprender:
      «pocas»  = 1 interacción   (280 usuarios)
      «muchas» = 2 o más         (326 usuarios)
- dibujar la distribución de las métricas sobre las 30 semillas (diagramas de caja).

Comprobación: las medias generales deben coincidir con las Tablas 16 y 19 del informe.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/evaluar_tarea6.py

Salidas: results/tarea6/{por_usuario.csv.gz, por_semilla.csv, grupos.csv, RESUMEN.md}
         figures/tarea6_semillas.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from descriptivo import GRIS_REF  # noqa: F401  (aplica el estilo común de las figuras)
from dos_torres import DosTorres
from evaluacion import N_DESTINOS, cargar_escenario, evaluar
from factorizacion import FactorizacionBPR
from inventario import _a_markdown
from referencias import Azar, Popularidad

MODULO = Path(__file__).resolve().parents[1]
SALIDA = MODULO / "results" / "tarea6"
FIGURAS = MODULO / "figures"
N_SEMILLAS = 30
COLORES = {"Azar": "#2a78d6", "Popularidad": "#eb6834", "Factorización (BPR)": "#1baf7a", "Red de dos torres": "#eda100"}

torch.set_num_threads(1)


def modelos(semilla):
    fm = json.loads((MODULO / "results" / "tarea4" / "fm_decision.json").read_text(encoding="utf-8"))
    red = json.loads((MODULO / "results" / "tarea5" / "red_decision.json").read_text(encoding="utf-8"))
    return {
        "Azar": Azar(semilla=1000 + semilla),
        "Popularidad": Popularidad(),
        "Factorización (BPR)": FactorizacionBPR(d=fm["d"], lam=fm["lambda"], epocas=fm["epocas"], semilla=semilla),
        "Red de dos torres": DosTorres(d=red["d"], perdida=red["perdida"], negativos=red["negativos"],
                                       usuario=red["usuario"], epocas=red["epocas"], semilla=semilla),
    }


def correr():
    aprende, pru = cargar_escenario("final")
    por_usuario, por_semilla = [], []
    for s in range(N_SEMILLAS):
        for nombre, m in modelos(s).items():
            res, pu = evaluar(m.ajustar(aprende).puntuar, aprende, pru, semilla=s)
            pu.insert(0, "semilla", s)
            pu.insert(0, "modelo", nombre)
            por_usuario.append(pu)
            r = res.set_index("K").loc[10]
            por_semilla.append({"modelo": nombre, "semilla": s, "HitRate@10": r.HitRate,
                                "Lugar@3": r["HitRate_lugar@3"], "Lugar@1": r["HitRate_lugar@1"]})
    return pd.concat(por_usuario, ignore_index=True), pd.DataFrame(por_semilla), aprende, pru


def grupos(pu, aprende, pru):
    n_u = pru.UserID.map(aprende.groupby("UserID").size())
    p_id = dict(zip(pru.UserID, 10 / (N_DESTINOS - n_u)))  # probabilidad del azar por ID
    pu = pu.assign(grupo=np.where(pu.n_aprendidas == 1, "pocas (1)", "muchas (≥ 2)"),
                   l3=pu.rango_lugar < 3, l1=pu.rango_lugar < 1, h10=pu.rango < 10)
    filas = []
    for (modelo, grupo), g in pu.groupby(["modelo", "grupo"], sort=False):
        por_u = g.groupby("UserID")[["l3", "l1", "h10"]].mean()  # acierto medio por usuario (30 semillas)
        n = len(por_u)
        k3 = int(round(por_u.l3.sum()))
        filas.append({
            "modelo": modelo, "grupo": grupo, "usuarios": n,
            "Lugar@3 (%)": round(100 * por_u.l3.mean(), 1), "Lugar@1 (%)": round(100 * por_u.l1.mean(), 1),
            "HitRate@10 (%)": round(100 * por_u.h10.mean(), 2),
            "Lugar@3: aciertos / esperados": f"{k3} / {0.6 * n:.1f}",
            "Lugar@3: p-valor vs azar": round(stats.binomtest(k3, n, 0.6, alternative="greater").pvalue, 3),
            "HitRate@10 esperado del azar (%)": round(100 * np.mean([p_id[u] for u in por_u.index]), 2),
        })
    return pd.DataFrame(filas)


def figura(ps):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    orden = list(COLORES)
    for ax, (met, azar, titulo) in zip(axes, (("Lugar@3", 60, "HitRate@3 por lugar (métrica principal)"),
                                             ("HitRate@10", 1.0, "HitRate@10 por ID"))):
        datos = [100 * ps[ps.modelo == m][met].to_numpy() for m in orden]
        cajas = ax.boxplot(datos, vert=True, patch_artist=True, widths=0.55,
                           medianprops={"color": "#0b0b0b", "linewidth": 1.2})
        for caja, m in zip(cajas["boxes"], orden):
            caja.set_facecolor(COLORES[m])
            caja.set_alpha(0.85)
        ax.axhline(azar, ls=":", color=GRIS_REF, lw=1.2)
        ax.text(0.52, azar, "azar", va="bottom", ha="left", fontsize=9, color=GRIS_REF)
        ax.set_xticks(range(1, 5), ["Azar", "Popularidad", "Factorización", "Red"])
        ax.set_ylabel("%")
        ax.set_title(titulo, loc="left")
        ax.grid(axis="x", visible=False)
    fig.suptitle("Variación de los resultados en las 30 semillas (prueba, 606 usuarios)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Cada caja muestra solo cuánto cambia el resultado al cambiar la semilla (siempre los mismos 606 "
             "usuarios); no incluye la incertidumbre por muestreo de usuarios.", fontsize=8.5, color=GRIS_REF)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    ruta = FIGURAS / "tarea6_semillas.png"
    fig.savefig(ruta, dpi=200)
    return ruta


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    pu, ps, aprende, pru = correr()
    pu.to_csv(SALIDA / "por_usuario.csv.gz", index=False)
    ps.to_csv(SALIDA / "por_semilla.csv", index=False)
    gr = grupos(pu, aprende, pru)
    gr.to_csv(SALIDA / "grupos.csv", index=False)

    # Comprobación contra las Tablas 16 y 19
    general = (100 * ps.groupby("modelo", sort=False)[["Lugar@3", "Lugar@1", "HitRate@10"]].mean()).round(2)
    desv = (100 * ps.groupby("modelo", sort=False)[["Lugar@3", "HitRate@10"]].std(ddof=1)).round(2)
    general = general.join(desv.add_suffix(" DE")).reset_index()
    general.to_csv(SALIDA / "general.csv", index=False)
    ruta = figura(ps)

    texto = ["# Tarea 6 — grupos de usuarios y 30 semillas\n", "_Generado por `src/evaluar_tarea6.py`. No editar a mano._\n",
             "## 1. Medias generales (deben coincidir con las Tablas 16 y 19) y desviación entre semillas (%)\n",
             _a_markdown(general) + "\n",
             "## 2. Por grupo: usuarios con pocas (1) o muchas (≥ 2) interacciones para aprender\n",
             "Acierto medio por usuario sobre las 30 semillas. p-valor: binomial unilateral contra el 60 % "
             "del azar (sin corregir; es un análisis descriptivo por grupo).\n",
             _a_markdown(gr) + "\n", f"Figura: `{ruta.relative_to(MODULO)}`\n"]
    (SALIDA / "RESUMEN.md").write_text("\n".join(texto), encoding="utf-8")
    print(general.to_string(index=False))
    print()
    print(gr.to_string(index=False))


if __name__ == "__main__":
    main()
