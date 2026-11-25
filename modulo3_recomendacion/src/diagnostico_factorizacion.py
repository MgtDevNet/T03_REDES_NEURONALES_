"""Diagnóstico: ¿por qué la factorización tiene una cobertura «alta»?

Con regularización fuerte (λ = 1), cada vez que un destino aparece en un trío de entrenamiento su
vector se encoge. Hipótesis: los destinos que más aparecen terminan con vectores más pequeños
(puntajes más bajos) y el top-10 se llena de destinos que el modelo nunca vio.

Usa el modelo guardado (models/factorizacion_bpr.npz, semilla 0) y los 606 usuarios de prueba.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/diagnostico_factorizacion.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from evaluacion import N_DESTINOS, cargar_escenario, enmascarar
from inventario import _a_markdown

MODULO = Path(__file__).resolve().parents[1]
SALIDA = MODULO / "results" / "tarea4"


def main():
    aprende, pru = cargar_escenario("final")
    f = np.load(MODULO / "models" / "factorizacion_bpr.npz")
    P, Q = f["P"], f["Q"]
    conteo = aprende.DestinationID.value_counts().reindex(range(1, N_DESTINOS + 1), fill_value=0).to_numpy()
    norma = np.linalg.norm(Q, axis=1)
    t1 = pd.DataFrame([{"interacciones del destino al aprender": k, "destinos": int((conteo == k).sum()),
                        "norma media de su vector": round(float(norma[conteo == k].mean()), 3)}
                       for k in sorted(set(conteo))])
    items_de = aprende.groupby("UserID").DestinationID.apply(list).to_dict()
    usuarios = pru.UserID.to_numpy()
    S = enmascarar(P[usuarios - 1] @ Q.T, [items_de.get(u, []) for u in usuarios])
    top10 = np.argsort(-S, axis=1)[:, :10]
    frios = np.flatnonzero(conteo == 0)
    t2 = pd.DataFrame([{"% del top-10 que son destinos nunca vistos": round(100 * np.isin(top10, frios).mean(), 1),
                        "% de destinos nunca vistos en el catálogo": round(100 * len(frios) / N_DESTINOS, 1)}])
    (SALIDA / "RESUMEN_diagnostico_factorizacion.md").write_text("\n".join([
        "# Diagnóstico de la cobertura de la factorización\n", "_Generado por `src/diagnostico_factorizacion.py`._\n",
        "## 1. Tamaño del vector de cada destino según cuántas veces apareció al aprender\n", _a_markdown(t1) + "\n",
        "## 2. Qué recomienda (top-10 de los 606 usuarios de prueba)\n", _a_markdown(t2) + "\n"]), encoding="utf-8")
    t1.to_csv(SALIDA / "fm_diagnostico_normas.csv", index=False)
    t2.to_csv(SALIDA / "fm_diagnostico_top10.csv", index=False)
    print(t1.to_string(index=False)); print(); print(t2.to_string(index=False))


if __name__ == "__main__":
    main()
