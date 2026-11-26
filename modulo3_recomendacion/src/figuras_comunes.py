"""Figuras comunes a los tres módulos, aplicadas al módulo 3.

1. figures/comun_arquitectura.png   — arquitectura de la red de dos torres elegida, con las
                                       dimensiones y los parámetros entrenables de cada capa.
2. figures/comun_busqueda.png       — búsqueda de hiperparámetros de la factorización (9
                                       configuraciones) y de la red (24), en validación.
3. figures/comun_curvas.png         — curvas de aprendizaje de las configuraciones elegidas:
                                       pérdida de entrenamiento y métrica de validación por época
                                       (sin parada temprana, 60 épocas, media de 3 semillas).
(La cuarta figura común, los diagramas de caja de las 30 semillas, es figures/tarea6_semillas.png.)

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/figuras_comunes.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from descriptivo import GRIS_REF, TEXTO_2  # noqa: F401  (estilo común)
from dos_torres import DosTorres
from evaluacion import cargar_escenario
from factorizacion import FactorizacionBPR

MODULO = Path(__file__).resolve().parents[1]
RES = MODULO / "results"
FIG = MODULO / "figures"
AZUL, NARANJA, AGUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
EPOCAS, SEMILLAS = 60, 3

torch.set_num_threads(1)


# ------------------------------------------------------------------ 1. Arquitectura
def caja(ax, x, y, w, h, texto, color, sub=""):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none", alpha=0.18))
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="none", ec=color, lw=1.4))
    ax.text(x, y + (0.09 if sub else 0), texto, ha="center", va="center", fontsize=9.5, fontweight="bold")
    if sub:
        ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=8.2, color=TEXTO_2)


def sup(n):
    """Número como superíndice (ℝ² en lugar de ℝ^2)."""
    return str(n).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))


def flecha(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, color=GRIS_REF, lw=1.1))


def figura_arquitectura():
    meta = json.loads((MODULO / "models" / "dos_torres.json").read_text(encoding="utf-8"))
    d, h = meta["d"], meta["oculta"]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    xu, xd = 2.6, 7.4
    # Torre del usuario
    caja(ax, xu, 6.2, 3.2, 0.7, "Preferencias del usuario", AZUL, "5 casillas 0/1 (Beach … Adventure)")
    caja(ax, xu, 4.8, 3.2, 0.7, f"Capa oculta: {h} neuronas, ReLU", AZUL, f"5 × {h} pesos + {h} sesgos = {5 * h + h}")
    caja(ax, xu, 3.4, 3.2, 0.7, f"Vector del usuario u ∈ ℝ{sup(d)}", AZUL, f"{h} × {d} + {d} = {h * d + d}")
    # Torre del destino
    caja(ax, xd - 0.95, 6.2, 1.5, 0.7, "Lugar (1 de 5)", NARANJA, f"embedding 5 × {d} = {5 * d}")
    caja(ax, xd + 0.95, 6.2, 1.5, 0.7, "Popularity", NARANJA, "estandarizada")
    caja(ax, xd, 4.8, 3.2, 0.7, f"Capa oculta: {h} neuronas, ReLU", NARANJA, f"{d + 1} × {h} + {h} = {(d + 1) * h + h}")
    caja(ax, xd, 3.4, 3.2, 0.7, f"Vector del destino v ∈ ℝ{sup(d)}", NARANJA, f"{h} × {d} + {d} = {h * d + d}")
    # Unión
    caja(ax, 5, 1.9, 3.0, 0.7, "Puntaje s = u · v", AGUA, "producto punto, sin parámetros")
    caja(ax, 5, 0.65, 4.6, 0.75, "Pérdida BCE con 1 negativo por positivo", AGUA,
         "σ(s) = probabilidad de «visitó»; destino al azar no visitado = 0")
    for x in (xu,):
        flecha(ax, x, 5.83, x, 5.17)
        flecha(ax, x, 4.43, x, 3.77)
    flecha(ax, xd - 0.95, 5.83, xd - 0.5, 5.17)
    flecha(ax, xd + 0.95, 5.83, xd + 0.5, 5.17)
    flecha(ax, xd, 4.43, xd, 3.77)
    flecha(ax, xu, 3.03, 4.2, 2.27)
    flecha(ax, xd, 3.03, 5.8, 2.27)
    flecha(ax, 5, 1.53, 5, 1.05)
    ax.text(xu, 6.85, "Torre del usuario", ha="center", fontsize=11, fontweight="bold", color=AZUL)
    ax.text(xd, 6.85, "Torre del destino", ha="center", fontsize=11, fontweight="bold", color=NARANJA)
    total = (5 * h + h) + (h * d + d) + 5 * d + ((d + 1) * h + h) + (h * d + d)
    ax.text(9.95, 0.1, f"Parámetros entrenables: {total}", ha="right", fontsize=9.5, color=TEXTO_2)
    fig.suptitle(f"Arquitectura de la red de dos torres elegida (d = {d})", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ruta = FIG / "comun_arquitectura.png"
    fig.savefig(ruta, dpi=200)
    assert total == meta["parametros"], (total, meta["parametros"])
    return ruta


# ------------------------------------------------------------------ 2. Búsqueda de hiperparámetros
def figura_busqueda():
    fm = pd.read_csv(RES / "tarea4" / "fm_ajuste_corridas.csv")
    red = pd.read_csv(RES / "tarea5" / "red_ajuste_corridas.csv")
    fm_dec = json.loads((RES / "tarea4" / "fm_decision.json").read_text(encoding="utf-8"))
    red_dec = json.loads((RES / "tarea5" / "red_decision.json").read_text(encoding="utf-8"))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})

    # Factorización: mapa de calor d × λ del HitRate@10 medio en validación
    m = (100 * fm.groupby(["d", "lambda"])["HitRate@10_val"].mean()).unstack()
    im = ax1.imshow(m.values, cmap="Blues", vmin=0, vmax=max(2.5, m.values.max()), aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            elegida = m.index[i] == fm_dec["d"] and m.columns[j] == fm_dec["lambda"]
            ax1.text(j, i, f"{m.values[i, j]:.2f}" + ("\n(elegida)" if elegida else ""), ha="center", va="center",
                     fontsize=9, fontweight="bold" if elegida else "normal")
    ax1.set_xticks(range(m.shape[1]), [f"{v:g}" for v in m.columns])
    ax1.set_yticks(range(m.shape[0]), m.index)
    ax1.set_xlabel("regularización λ")
    ax1.set_ylabel("dimensión d")
    ax1.grid(False)
    ax1.set_title("Factorización: HitRate@10 (%) en validación\n(azar ≈ 1 %; media de 3 semillas)", loc="left")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    # Red: cada configuración (media y rango de 3 semillas) del HitRate@3 por lugar
    g = red.groupby(["d", "perdida", "negativos", "usuario"])["Lugar@3_val"].agg(["mean", "min", "max"]).reset_index()
    g = g.sort_values("mean").reset_index(drop=True)
    y = np.arange(len(g))
    colores = np.where(g.usuario == "preferencias", AZUL, NARANJA)
    ax2.hlines(y, 100 * g["min"], 100 * g["max"], color=colores, alpha=0.45, lw=2)
    ax2.scatter(100 * g["mean"], y, color=colores, s=22, zorder=3)
    eleg = g[(g.d == red_dec["d"]) & (g.perdida == red_dec["perdida"]) & (g.negativos == red_dec["negativos"]) & (g.usuario == red_dec["usuario"])].index[0]
    ax2.scatter(100 * g.loc[eleg, "mean"], eleg, s=90, facecolor="none", edgecolor="#0b0b0b", lw=1.4, zorder=4)
    ax2.annotate("elegida", (100 * g.loc[eleg, "mean"], eleg), xytext=(-62, 2), textcoords="offset points", fontsize=8.5, va="center")
    ax2.axvline(60, ls=":", color=GRIS_REF)
    ax2.text(60.15, -1.3, "azar 60 %", fontsize=8.5, color=GRIS_REF)
    ax2.set_yticks(y, [f"d={r.d} · {r.perdida.upper()} · {r.negativos} neg." for r in g.itertuples()], fontsize=7.5)
    ax2.set_ylim(-2, len(g))
    ax2.set_xlabel("HitRate@3 por lugar (%): punto = media; línea = mín.–máx. (3 semillas)")
    ax2.set_title("Red: 24 configuraciones", loc="left")
    ax2.scatter([], [], color=AZUL, label="solo preferencias")
    ax2.scatter([], [], color=NARANJA, label="perfil completo")
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Búsqueda de hiperparámetros (validación, 326 usuarios)", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ruta = FIG / "comun_busqueda.png"
    fig.savefig(ruta, dpi=200)
    return ruta


# ------------------------------------------------------------------ 3. Curvas de aprendizaje
def curvas():
    aprende, val = cargar_escenario("ajuste")
    fm_dec = json.loads((RES / "tarea4" / "fm_decision.json").read_text(encoding="utf-8"))
    red_dec = json.loads((RES / "tarea5" / "red_decision.json").read_text(encoding="utf-8"))
    filas = []
    for s in range(SEMILLAS):
        fm = FactorizacionBPR(d=fm_dec["d"], lam=fm_dec["lambda"], epocas=EPOCAS, semilla=s).ajustar(aprende, validacion=val, paciencia=10 ** 6)
        filas += [{"modelo": "Factorización", "semilla": s, "epoca": h["epoca"], "perdida": h["perdida"], "perdida_val": h["perdida_val"], "metrica": h["NDCG@10_val"]} for h in fm.historia]
        red = DosTorres(d=red_dec["d"], perdida=red_dec["perdida"], negativos=red_dec["negativos"], usuario=red_dec["usuario"],
                        epocas=EPOCAS, semilla=s).ajustar(aprende, validacion=val, paciencia=10 ** 6)
        filas += [{"modelo": "Red", "semilla": s, "epoca": h["epoca"], "perdida": h["perdida"], "perdida_val": h["perdida_val"], "metrica": h["HitRate_lugar@3_val"]} for h in red.historia]
    t = pd.DataFrame(filas)
    t.to_csv(RES / "tarea6" / "curvas_aprendizaje.csv", index=False)
    return t, fm_dec["epocas"], red_dec["epocas"]


def figura_curvas(t, ep_fm, ep_red):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), sharex=True)
    ndcg_azar = 100 / 1000 * sum(1 / np.log2(r + 1) for r in range(1, 11))  # NDCG@10 esperado del azar
    for fila, (modelo, met, ep, azar, color) in enumerate((("Factorización", "NDCG@10 en validación (%)", ep_fm, ndcg_azar, AGUA),
                                                           ("Red", "HitRate@3 por lugar en validación (%)", ep_red, 60, AMARILLO))):
        g = t[t.modelo == modelo].groupby("epoca")[["perdida", "perdida_val", "metrica"]].agg(["mean", "min", "max"])
        for col, (campo, titulo, escala) in enumerate((("perdida", "Pérdida de entrenamiento y validación", 1), ("metrica", met, 100))):
            ax = axes[fila, col]
            ax.fill_between(g.index, escala * g[(campo, "min")], escala * g[(campo, "max")], color=color, alpha=0.2, lw=0)
            ax.plot(g.index, escala * g[(campo, "mean")], color=color, lw=2, label="entrenamiento" if campo == "perdida" else None)
            if campo == "perdida":
                ax.fill_between(g.index, g[("perdida_val", "min")], g[("perdida_val", "max")], color=GRIS_REF, alpha=0.15, lw=0)
                ax.plot(g.index, g[("perdida_val", "mean")], color=GRIS_REF, lw=1.8, ls="--", label="validación")
                ax.legend(fontsize=8.5, loc="upper right")
            ax.axvline(ep, ls="--", color="#0b0b0b", lw=1)
            ax.text(ep + 0.8, ax.get_ylim()[1], f"parada temprana: época {ep}", va="top", fontsize=8.5)
            if campo == "perdida":
                ax.axhline(np.log(2), ls=":", color=GRIS_REF)
                ax.text(EPOCAS, np.log(2), "ln 2 (moneda al aire)", ha="right", va="bottom", fontsize=8.5, color=GRIS_REF)
            if campo == "metrica" and azar:
                ax.axhline(azar, ls=":", color=GRIS_REF)
                ax.text(EPOCAS, azar, "azar", ha="right", va="bottom", fontsize=8.5, color=GRIS_REF)
            ax.set_title(f"{modelo}: {titulo}", loc="left")
            if fila == 1:
                ax.set_xlabel("época")
    fig.suptitle("Curvas de aprendizaje de las configuraciones elegidas (media y rango de 3 semillas)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ruta = FIG / "comun_curvas.png"
    fig.savefig(ruta, dpi=200)
    return ruta


def main():
    print(figura_arquitectura().relative_to(MODULO))
    print(figura_busqueda().relative_to(MODULO))
    t, ep_fm, ep_red = curvas()
    print(figura_curvas(t, ep_fm, ep_red).relative_to(MODULO))
    r = t.groupby(["modelo", "epoca"])[["perdida", "metrica"]].mean().reset_index()
    print(r[r.epoca.isin([1, 5, 8, 20, 60])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
