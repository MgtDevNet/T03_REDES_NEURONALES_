"""Evidencia para las decisiones de la tarea 4 (modelos de referencia).

Antes de programar los modelos, mide lo que cada uno PUEDE lograr con estos datos:
- cuánto acertaría el azar (el piso);
- cuántos empates tiene la popularidad (si casi todos los destinos tienen el mismo conteo,
  el orden «por popularidad» es casi arbitrario);
- cuántas veces KNN y la factorización de matrices pueden siquiera puntuar el destino escondido.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/evidencia_tarea4.py

Salidas en modulo3_recomendacion/results/tarea4_evidencia/: tabla_01..03.csv + RESUMEN.md
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from inventario import _a_markdown

MODULO = Path(__file__).resolve().parents[1]
PROCESADOS = MODULO / "data" / "processed"
SALIDA = MODULO / "results" / "tarea4_evidencia"
K = 10
N_DESTINOS = 1000


def escenarios(inter):
    """Los dos momentos de la evaluación: ajustar (ent → val) y evaluar una vez (ent+val → prueba)."""
    ent = inter[inter.conjunto == "entrenamiento"]
    val = inter[inter.conjunto == "validación"]
    pru = inter[inter.conjunto == "prueba"]
    return {
        "ajuste: entrenamiento → validación": (ent, val),
        "final: entrenamiento+validación → prueba": (pd.concat([ent, val]), pru),
    }


def analizar(aprende, evalua):
    """Qué fracción de los casos evaluados puede puntuar cada familia de modelos."""
    items_de = aprende.groupby("UserID").DestinationID.apply(set).to_dict()
    usuarios_de = aprende.groupby("DestinationID").UserID.apply(set).to_dict()
    vistos = set(aprende.DestinationID)

    # Ítem-ítem: el destino escondido i recibe puntaje > 0 si comparte al menos un usuario
    # (distinto de u) con algún destino de entrenamiento de u.
    vecinos_item = defaultdict(set)
    for items in items_de.values():
        for i in items:
            vecinos_item[i] |= items - {i}

    azar, frio, knn_ii, knn_uu = [], [], [], []
    for u, i in zip(evalua.UserID, evalua.DestinationID):
        propios = items_de.get(u, set())
        azar.append(K / (N_DESTINOS - len(propios)))  # P(acierto) ordenando al azar los no vistos
        frio.append(i not in vistos)
        knn_ii.append(any(i in vecinos_item[j] for j in propios))
        # Usuario-usuario: algún vecino (comparte ≥ 1 destino con u) tiene el destino escondido.
        vecinos = set().union(*(usuarios_de[j] for j in propios)) - {u} if propios else set()
        knn_uu.append(any(i in items_de[v] for v in vecinos))
    n = len(evalua)
    return {
        "casos evaluados": n,
        "HitRate@10 esperado del azar (%)": round(100 * np.mean(azar), 2),
        "destino escondido nunca visto al aprender (%)": round(100 * np.mean(frio), 1),
        "techo de un modelo solo de IDs (%)": round(100 * (1 - np.mean(frio)), 1),
        "KNN ítem–ítem puede puntuarlo (%)": round(100 * np.mean(knn_ii), 1),
        "KNN usuario–usuario puede puntuarlo (%)": round(100 * np.mean(knn_uu), 1),
    }


def empates_popularidad(aprende):
    """Cuántos destinos hay con cada conteo de interacciones (los empates hacen arbitrario el orden)."""
    c = aprende.DestinationID.value_counts().reindex(range(1, N_DESTINOS + 1), fill_value=0)
    t = c.value_counts().sort_index(ascending=False).rename_axis("interacciones").reset_index(name="destinos")
    t["destinos acumulados"] = t.destinos.cumsum()
    return t


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    inter = pd.read_csv(PROCESADOS / "interacciones.csv")
    esc = escenarios(inter)

    t1 = pd.DataFrame({nombre: analizar(a, e) for nombre, (a, e) in esc.items()}).rename_axis("medida").reset_index()
    aprende_final = esc["final: entrenamiento+validación → prueba"][0]
    t2 = empates_popularidad(aprende_final)
    # Densidad de la matriz con la que aprende cada modelo
    t3 = pd.DataFrame([{
        "escenario": nombre, "interacciones": len(a), "usuarios con ≥ 1": a.UserID.nunique(),
        "destinos con ≥ 1": a.DestinationID.nunique(),
        "densidad (%)": round(100 * len(a) / (999 * N_DESTINOS), 3),
    } for nombre, (a, _) in esc.items()])

    secciones = [
        ("1. Qué puede lograr cada familia de modelos", t1,
         "Para cada caso evaluado (un usuario y su destino escondido) se mide si el modelo puede, en "
         "principio, darle un puntaje mayor que 0 al destino escondido. Si no puede, ese caso es un "
         "fallo seguro, entrene como entrene."),
        ("2. Empates en la popularidad (entrenamiento+validación)", t2,
         "Destinos agrupados por su número de interacciones. Para llenar un top-10 por popularidad hay "
         "que desempatar dentro del último grupo que entra."),
        ("3. Tamaño y densidad de la matriz con la que se aprende", t3, ""),
    ]
    partes = ["# Evidencia para la tarea 4 — modelos de referencia\n",
              "_Generado por `src/evidencia_tarea4.py`. No editar a mano._\n"]
    for i, (titulo, tabla, texto) in enumerate(secciones, start=1):
        tabla.to_csv(SALIDA / f"tabla_{i:02d}.csv", index=False)
        partes += [f"## {titulo}\n", texto + "\n" if texto else "", _a_markdown(tabla) + "\n"]
        print(f"\n=== {titulo} ===\n{tabla.to_string(index=False)}")
    (SALIDA / "RESUMEN.md").write_text("\n".join(partes), encoding="utf-8")


if __name__ == "__main__":
    main()
