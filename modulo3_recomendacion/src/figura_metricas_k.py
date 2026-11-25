"""Figura mínima del módulo: Precision, Recall y NDCG en función de K, para cada modelo.

Lee los resultados ya calculados (prueba, 606 usuarios, media de 30 semillas); no reentrena.
Incluye la referencia del azar teórico: Recall@K ≈ K/1000, Precision@K ≈ 1/1000 y
NDCG@K = (1/1000) Σ_{r=1..K} 1/log2(r+1) (cada posición tiene la misma probabilidad).

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/figura_metricas_k.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from descriptivo import GRIS_REF  # noqa: F401  (estilo común)
from evaluacion import KS

MODULO = Path(__file__).resolve().parents[1]
RES = MODULO / "results"
COLORES = {"Azar": "#2a78d6", "Popularidad": "#eb6834", "Factorización (BPR)": "#1baf7a", "Red de dos torres": "#eda100"}
MARCAS = {"Azar": "o", "Popularidad": "s", "Factorización (BPR)": "D", "Red de dos torres": "^"}


def datos():
    ref = pd.read_csv(RES / "tarea4" / "referencias_por_semilla.csv")
    ref = ref[ref.escenario == "final"].drop(columns="escenario")
    fm = pd.read_csv(RES / "tarea4" / "fm_final_por_semilla.csv").assign(modelo="Factorización (BPR)")
    red = pd.read_csv(RES / "tarea5" / "red_final_por_semilla.csv").assign(modelo="Red de dos torres")
    todo = pd.concat([ref, fm, red], ignore_index=True)
    return (100 * todo.groupby(["modelo", "K"])[["Precision", "Recall", "NDCG"]].mean()).reset_index()


def main():
    d = datos()
    d.round(3).to_csv(RES / "tarea6" / "metricas_por_k.csv", index=False)
    k = np.array(KS)
    azar = {"Precision": np.full(len(k), 100 / 1000), "Recall": 100 * k / 1000,
            "NDCG": np.array([100 / 1000 * sum(1 / np.log2(r + 1) for r in range(1, kk + 1)) for kk in k])}
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    x = np.arange(len(k))
    for ax, met in zip(axes, ("Precision", "Recall", "NDCG")):
        ax.plot(x, azar[met], ls=":", color=GRIS_REF, lw=1.3, label="Azar teórico")
        for modelo, color in COLORES.items():
            g = d[d.modelo == modelo].sort_values("K")
            ax.plot(x, g[met], color=color, marker=MARCAS[modelo], ms=5, lw=1.8,
                    markeredgecolor="white", markeredgewidth=0.8, label=modelo)
        ax.set_xticks(x, [str(v) for v in k])
        ax.set_xlabel("K")
        ax.set_title(f"{met}@K (%)", loc="left")
        ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=8.5, loc="upper left")
    fig.suptitle("Precision, Recall y NDCG por ID: los cuatro modelos siguen la curva del azar",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Prueba, 606 usuarios, media de 30 semillas. Con un solo destino escondido por usuario, "
             "Recall@K = HitRate@K y Precision@K = Recall@K / K.", fontsize=8.5, color=GRIS_REF)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    ruta = MODULO / "figures" / "tarea6_metricas_k.png"
    fig.savefig(ruta, dpi=200)
    print(ruta.relative_to(MODULO.parent))
    print(d[d.K == 10].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
