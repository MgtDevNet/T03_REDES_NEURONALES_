"""Figura de la tarea 4: los tres modelos de referencia frente al azar.

Lee los resultados ya calculados (no reentrena nada):
    results/tarea4/referencias_resumen.csv   (azar y popularidad)
    results/tarea4/fm_final_por_semilla.csv  (factorización)

Izquierda: HitRate@K de cada modelo (media de 30 semillas, prueba) y la banda del 95 % que
           produciría el azar con 606 usuarios (binomial). Si un modelo cae dentro de la banda,
           no se distingue del azar.
Derecha:   cobertura@10, la única medida en la que los modelos se diferencian.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/figuras_tarea4.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from descriptivo import GRIS_REF, REJILLA, TEXTO_2  # noqa: F401  (aplica también el estilo común)
from evaluacion import KS, N_DESTINOS, cargar_escenario

MODULO = Path(__file__).resolve().parents[1]
RES = MODULO / "results" / "tarea4"
FIGURAS = MODULO / "figures"
# Paleta categórica en orden fijo (guía de visualización): azul, naranja, aguamarina.
COLORES = {"Azar": "#2a78d6", "Popularidad": "#eb6834", "Factorización (BPR)": "#1baf7a"}
MARCAS = {"Azar": "o", "Popularidad": "s", "Factorización (BPR)": "D"}


def datos():
    ref = pd.read_csv(RES / "referencias_resumen.csv")
    ref = ref[ref.escenario == "final"][["modelo", "K", "HitRate_%", "Cobertura_%"]]
    fm = pd.read_csv(RES / "fm_final_por_semilla.csv").groupby("K")[["HitRate", "Cobertura"]].mean().reset_index()
    fm = pd.DataFrame({"modelo": "Factorización (BPR)", "K": fm.K,
                       "HitRate_%": 100 * fm.HitRate, "Cobertura_%": 100 * fm.Cobertura})
    return pd.concat([ref, fm], ignore_index=True)


def banda_azar():
    """Media y banda del 95 % del HitRate@K del azar para los 606 usuarios de prueba."""
    aprende, pru = cargar_escenario("final")
    n_u = pru.UserID.map(aprende.groupby("UserID").size()).fillna(0).to_numpy()
    n = len(pru)
    filas = []
    for k in KS:
        p0 = float(np.mean(k / (N_DESTINOS - n_u)))
        lo, hi = stats.binom.ppf([0.025, 0.975], n, p0)
        filas.append({"K": k, "esperado": 100 * p0, "bajo": 100 * lo / n, "alto": 100 * hi / n})
    return pd.DataFrame(filas)


def main():
    d, banda = datos(), banda_azar()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.25, 1]})

    x = np.arange(len(KS))
    ax1.fill_between(x, banda.bajo, banda.alto, color=REJILLA, alpha=0.9, lw=0,
                     label="Rango que produce el azar (95 %)")
    ax1.plot(x, banda.esperado, ls=":", color=GRIS_REF, lw=1.2, label="Esperado por azar")
    for modelo, color in COLORES.items():
        g = d[d.modelo == modelo].sort_values("K")
        ax1.plot(x, g["HitRate_%"], color=color, lw=2, marker=MARCAS[modelo], ms=6,
                 markeredgecolor="white", markeredgewidth=1, label=modelo)
    ax1.set_xticks(x, [str(k) for k in KS])
    ax1.set_xlabel("K (tamaño de la lista)")
    ax1.set_ylabel("HitRate@K (%)")
    ax1.set_title("Aciertos según el tamaño de la lista", loc="left")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_ylim(bottom=0)

    c = d[d.K == 10].set_index("modelo").loc[list(COLORES)]
    y = np.arange(len(c))[::-1]
    ax2.barh(y, c["Cobertura_%"], color=[COLORES[m] for m in c.index], height=0.6)
    for yi, v in zip(y, c["Cobertura_%"]):
        ax2.text(127, yi, f"{v:.1f} %", va="center", ha="right", fontsize=10)
    ax2.set_yticks(y, c.index)
    ax2.set_xlim(0, 128)
    ax2.set_xticks([0, 25, 50, 75, 100])
    ax2.set_xlabel("% de los 1000 destinos en algún top-10")
    ax2.set_title("Cobertura@10", loc="left")
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Ningún modelo de referencia acierta más que el azar; solo difieren en cobertura",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ruta = FIGURAS / "tarea4_referencias.png"
    fig.savefig(ruta, dpi=200)
    print(ruta.relative_to(MODULO.parent))
    print(banda.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
