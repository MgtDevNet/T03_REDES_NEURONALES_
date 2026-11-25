"""Modelos de referencia sin entrenamiento: azar y popularidad (tarea 4, decisiones 7.1 y 7.2).

Los dos se miden con el evaluador común (`evaluacion.py`) en los dos momentos (ajuste y final) y
con 30 semillas.

- **Azar:** un puntaje aleatorio por destino. Es el piso: cualquier modelo útil debe superarlo.
- **Popularidad:** el puntaje de un destino es cuántas interacciones tiene en lo aprendido. A todos
  los usuarios les recomienda lo mismo (salvo lo que ya tienen). Los empates (muchos destinos con
  el mismo conteo) los rompe el evaluador al azar, con la semilla.

Ninguno tiene parámetros entrenables: el azar no guarda nada y la popularidad guarda 1000 conteos,
que no se aprenden por optimización.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/referencias.py

Salidas en modulo3_recomendacion/results/tarea4/:
    referencias_por_semilla.csv   una fila por escenario × modelo × semilla × K
    referencias_resumen.csv       media, desviación e intervalo del 95 % sobre las 30 semillas
    RESUMEN.md
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluacion import KS, N_DESTINOS, cargar_escenario, evaluar
from inventario import _a_markdown

MODULO = Path(__file__).resolve().parents[1]
SALIDA = MODULO / "results" / "tarea4"
N_SEMILLAS = 30
METRICAS = ["HitRate", "Recall", "Precision", "NDCG", "Cobertura", "HitRate_lugar@1", "HitRate_lugar@3"]


class Azar:
    """Puntaje aleatorio uniforme para cada (usuario, destino)."""

    nombre = "Azar"

    def __init__(self, semilla):
        self.rng = np.random.default_rng(semilla)

    def ajustar(self, aprende):
        return self  # no aprende nada

    def puntuar(self, usuarios):
        return self.rng.random((len(usuarios), N_DESTINOS))


class Popularidad:
    """Puntaje = número de interacciones del destino en lo aprendido (igual para todos)."""

    nombre = "Popularidad"

    def __init__(self, semilla=None):
        self.conteo = None

    def ajustar(self, aprende):
        c = aprende.DestinationID.value_counts().reindex(range(1, N_DESTINOS + 1), fill_value=0)
        self.conteo = c.to_numpy(dtype=float)
        return self

    def puntuar(self, usuarios):
        return np.tile(self.conteo, (len(usuarios), 1))


def correr(modelos=(Azar, Popularidad)):
    filas = []
    for escenario in ("ajuste", "final"):
        aprende, evalua = cargar_escenario(escenario)
        for Modelo in modelos:
            inicio = time.perf_counter()
            for s in range(N_SEMILLAS):
                modelo = Modelo(semilla=1000 + s).ajustar(aprende)
                resumen, _ = evaluar(modelo.puntuar, aprende, evalua, semilla=s)
                resumen.insert(0, "semilla", s)
                resumen.insert(0, "modelo", Modelo.nombre)
                resumen.insert(0, "escenario", escenario)
                filas.append(resumen)
            print(f"{escenario:>6} · {Modelo.nombre:<11} 30 semillas en {time.perf_counter() - inicio:.1f} s")
    return pd.concat(filas, ignore_index=True)


def resumir(por_semilla):
    """Media, desviación estándar e intervalo del 95 % (t de Student, 29 gl) sobre las semillas."""
    t = stats.t.ppf(0.975, N_SEMILLAS - 1)
    filas = []
    for (esc, mod, k), g in por_semilla.groupby(["escenario", "modelo", "K"], sort=False):
        fila = {"escenario": esc, "modelo": mod, "K": k}
        for m in METRICAS:
            media, de = g[m].mean(), g[m].std(ddof=1)
            mitad = t * de / np.sqrt(len(g))
            fila[f"{m}_%"] = round(100 * media, 2)
            fila[f"{m}_de_%"] = round(100 * de, 2)
            fila[f"{m}_ic95_%"] = f"{100 * (media - mitad):.2f}–{100 * (media + mitad):.2f}"
        filas.append(fila)
    return pd.DataFrame(filas)


def tabla_k10(resumen):
    """Vista compacta para el informe: K = 10, las métricas principales."""
    r = resumen[resumen.K == 10]
    return pd.DataFrame({
        "escenario": r.escenario, "modelo": r.modelo,
        "HitRate@10 (%)": r["HitRate_%"], "IC 95 %": r["HitRate_ic95_%"],
        "NDCG@10 (%)": r["NDCG_%"], "Precision@10 (%)": r["Precision_%"],
        "Cobertura@10 (%)": r["Cobertura_%"],
        "HitRate_lugar@3 (%)": r["HitRate_lugar@3_%"], "HitRate_lugar@1 (%)": r["HitRate_lugar@1_%"],
    })


def prueba_lugar(aciertos_por_usuario_l3, aciertos_por_usuario_l1):
    """Contra el azar por lugar (decisión 7.5): binomial unilateral contra 60 % (@3) y 20 % (@1),
    con Bonferroni m = 2. Aciertos = media de las semillas, redondeada."""
    n = len(aciertos_por_usuario_l3)
    fila = {}
    for nombre, hits, p0 in (("@3", aciertos_por_usuario_l3, 0.6), ("@1", aciertos_por_usuario_l1, 0.2)):
        k = int(round(float(np.sum(hits))))
        p = stats.binomtest(k, n, p0, alternative="greater").pvalue
        fila[f"lugar{nombre}: aciertos / esperados"] = f"{k} / {n * p0:.1f}"
        fila[f"lugar{nombre}: p-valor corregido (m = 2)"] = round(min(1.0, 2 * p), 3)
    return fila


def contra_azar(por_semilla):
    """¿El modelo acierta más que el azar? Prueba binomial unilateral con K = 10.

    El IC 95 % entre semillas solo mide cuánto cambia el resultado al cambiar la semilla; NO tiene
    en cuenta que los usuarios evaluados son una muestra (606 o 326 casos). Esta prueba sí:
    bajo H0 cada usuario acierta con la probabilidad del azar, p0 = media de 10 / (1000 − n_u),
    donde n_u son los destinos que el usuario ya tiene (se quitan del orden).
    Aciertos observados = media de aciertos en las 30 semillas, redondeada.
    """
    filas = []
    for escenario in ("ajuste", "final"):
        aprende, evalua = cargar_escenario(escenario)
        n_u = evalua.UserID.map(aprende.groupby("UserID").size()).fillna(0).to_numpy()
        p0 = float(np.mean(10 / (N_DESTINOS - n_u)))
        n = len(evalua)
        for modelo, g in por_semilla[(por_semilla.escenario == escenario) & (por_semilla.K == 10)].groupby("modelo", sort=False):
            aciertos = int(round(g.HitRate.mean() * n))
            prueba = stats.binomtest(aciertos, n, p0, alternative="greater")
            lugar = prueba_lugar(np.full(n, g["HitRate_lugar@3"].mean()), np.full(n, g["HitRate_lugar@1"].mean()))
            filas.append({**lugar, 
                "escenario": escenario, "modelo": modelo, "usuarios": n,
                "aciertos (media de 30 semillas)": aciertos,
                "esperados por azar": round(n * p0, 1),
                "error estándar por muestreo (pp)": round(100 * np.sqrt(p0 * (1 - p0) / n), 2),
                "p-valor (H1: acierta más que el azar)": round(prueba.pvalue, 3),
            })
    t = pd.DataFrame(filas)
    lugar = [c for c in t.columns if c.startswith("lugar")]
    return t[[c for c in t.columns if c not in lugar] + lugar]


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    por_semilla = correr()
    resumen = resumir(por_semilla)
    por_semilla.to_csv(SALIDA / "referencias_por_semilla.csv", index=False)
    resumen.to_csv(SALIDA / "referencias_resumen.csv", index=False)

    k10 = tabla_k10(resumen)
    azar = contra_azar(por_semilla)
    azar.to_csv(SALIDA / "referencias_contra_azar.csv", index=False)
    curvas = resumen.pivot_table(index=["escenario", "modelo"], columns="K", values="HitRate_%", sort=False)
    curvas.columns = [f"HitRate@{k} (%)" for k in curvas.columns]
    curvas = curvas.reset_index()

    texto = [
        "# Modelos de referencia: azar y popularidad (tarea 4)\n",
        "_Generado por `src/referencias.py`. No editar a mano._\n",
        f"Media de {N_SEMILLAS} semillas. IC 95 % = media ± t(0,975; 29) · DE / √30. "
        "Recall@K = HitRate@K porque cada usuario tiene un solo destino escondido.\n",
        "## 1. Resumen con K = 10\n", _a_markdown(k10) + "\n",
        "## 2. HitRate según K\n", _a_markdown(curvas) + "\n",
        "## 3. ¿Acierta más que el azar? (K = 10, prueba binomial unilateral)\n",
        "El IC 95 % de la sección 1 solo mide la variación entre semillas, no que los usuarios "
        "evaluados sean una muestra. Esta prueba sí lo tiene en cuenta.\n",
        _a_markdown(azar) + "\n",
        "_CSV: `results/tarea4/referencias_resumen.csv` (todas las métricas y K) y "
        "`referencias_por_semilla.csv`._\n",
    ]
    (SALIDA / "RESUMEN.md").write_text("\n".join(texto), encoding="utf-8")
    print("\n" + k10.to_string(index=False))
    print("\n" + curvas.to_string(index=False))
    print("\n" + azar.to_string(index=False))


if __name__ == "__main__":
    main()
