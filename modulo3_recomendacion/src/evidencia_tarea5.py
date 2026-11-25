"""Evidencia para las decisiones de la tarea 5 (red neuronal con variables de contenido).

Antes de diseñar la red, mide qué puede aportar cada variable de contenido con estos datos:
1. Qué variables de los destinos son redundantes (dicen lo mismo que el lugar).
2. Cuántos perfiles de usuario distintos hay (cuánto pueden personalizar las variables del usuario).
3. El máximo que puede lograr un modelo que acierte el LUGAR perfectamente: como cada lugar tiene
   200 IDs, saber el lugar no basta para acertar el ID exacto en un top-10.
4. Si el perfil o el historial del usuario predicen el lugar que visitará (contra el 20 % del azar).

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/evidencia_tarea5.py

Salidas en modulo3_recomendacion/results/tarea5_evidencia/: tabla_01..04.csv + RESUMEN.md
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from evaluacion import N_DESTINOS, cargar_escenario, evaluar
from inventario import _a_markdown

MODULO = Path(__file__).resolve().parents[1]
PROCESADOS = MODULO / "data" / "processed"
SALIDA = MODULO / "results" / "tarea5_evidencia"
PERFIL = ["Preferences", "Gender", "NumberOfAdults", "NumberOfChildren"]


def t1_redundancia(destinos):
    """Para cada variable del destino: cuántos valores distintos toma dentro de cada lugar."""
    filas = []
    for v in ["Type", "State", "BestTimeToVisit", "Popularity"]:
        por_lugar = destinos.groupby("Name")[v].nunique()
        filas.append({"variable": v, "valores distintos por lugar (mín–máx)": f"{por_lugar.min()}–{por_lugar.max()}",
                      "¿aporta algo además del lugar?": "no: la fija el lugar" if por_lugar.max() == 1
                      else "sí: cambia entre IDs del mismo lugar"})
    return pd.DataFrame(filas)


def t2_perfiles(usuarios):
    filas = [{"variables": v, "perfiles distintos": usuarios.drop_duplicates(v).shape[0]}
             for v in (["Preferences"], ["Preferences", "Gender"], PERFIL)]
    t = pd.DataFrame(filas)
    t["variables"] = t.variables.apply(lambda xs: " + ".join(xs))
    return t


def t3_maximo_por_lugar(destinos):
    """Oráculo: un modelo que SIEMPRE sabe el lugar del destino escondido y ordena sus 200 IDs
    al azar (o por Popularity). Es el máximo que puede lograr el contenido del destino."""
    lugar = destinos.set_index("DestinationID").Name.reindex(range(1, N_DESTINOS + 1)).to_numpy()
    popul = destinos.set_index("DestinationID").Popularity.reindex(range(1, N_DESTINOS + 1)).to_numpy()
    filas = []
    for escenario in ("ajuste", "final"):
        aprende, evalua = cargar_escenario(escenario)
        objetivo = dict(zip(evalua.UserID, evalua.DestinationID))
        for nombre, extra in (("sabe el lugar; IDs del lugar al azar", 0.0),
                              ("sabe el lugar; IDs del lugar por Popularity", 1e-3)):
            def oraculo(usuarios, extra=extra):
                correcto = np.array([lugar[objetivo[u] - 1] for u in usuarios])
                return (lugar[None, :] == correcto[:, None]).astype(float) + extra * popul[None, :]
            hr = np.mean([evaluar(oraculo, aprende, evalua, semilla=s, ks=(10,))[0].HitRate.iloc[0] for s in range(30)])
            filas.append({"escenario": escenario, "modelo de referencia (oráculo)": nombre,
                          "HitRate@10 (%)": round(100 * hr, 2)})
    return pd.DataFrame(filas)


def t4_predecir_lugar(usuarios, destinos):
    """¿El perfil o el historial predicen el LUGAR del destino escondido? (final: ent+val → prueba)."""
    aprende, evalua = cargar_escenario("final")
    nombre = destinos.set_index("DestinationID").Name
    a = aprende.assign(lugar=aprende.DestinationID.map(nombre)).merge(usuarios[["UserID"] + PERFIL], on="UserID")
    e = evalua.assign(lugar=evalua.DestinationID.map(nombre)).merge(usuarios[["UserID"] + PERFIL], on="UserID")
    n = len(e)
    filas = []

    # (a) Perfil: para cada combinación de variables, el lugar más frecuente en lo aprendido.
    for v in (["Preferences"], PERFIL):
        mayoritario = a.groupby(v).lugar.agg(lambda s: s.value_counts().index[0])
        pred = e.set_index(v).index.map(mayoritario)
        k = int((pred == e.lugar.to_numpy()).sum())
        filas.append({"predictor del lugar": "perfil: " + " + ".join(v) + " → lugar más frecuente del perfil",
                      "aciertos": k, "usuarios": n, "acierto (%)": round(100 * k / n, 1), "azar (%)": 20.0,
                      "p-valor (H1: acierta más que el azar)": round(stats.binomtest(k, n, 0.2, alternative="greater").pvalue, 3)})

    # (b) Historial: ¿el lugar escondido es uno de los lugares que el usuario ya tiene en lo aprendido?
    lugares_de = a.groupby("UserID").lugar.apply(set)
    m = e.UserID.map(lugares_de.apply(len)).to_numpy()
    k = int(sum(l in lugares_de[u] for u, l in zip(e.UserID, e.lugar)))
    esperado = float(np.sum(m / 5))  # si el lugar escondido fuera al azar: P = (lugares distintos ya vistos) / 5
    filas.append({"predictor del lugar": "historial: el lugar escondido es uno que ya visitó",
                  "aciertos": k, "usuarios": n, "acierto (%)": round(100 * k / n, 1),
                  "azar (%)": round(100 * esperado / n, 1),
                  "p-valor (H1: acierta más que el azar)": round(float(stats.poisson_binom.sf(k - 1, m / 5))
                                                                 if hasattr(stats, "poisson_binom") else
                                                                 stats.binomtest(k, n, esperado / n, alternative="greater").pvalue, 3)})
    return pd.DataFrame(filas)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    destinos = pd.read_csv(PROCESADOS / "destinos.csv")
    usuarios = pd.read_csv(PROCESADOS / "usuarios.csv")
    secciones = [
        ("1. ¿Qué variables del destino aportan algo además del lugar?", t1_redundancia(destinos)),
        ("2. ¿Cuántos perfiles de usuario distintos hay?", t2_perfiles(usuarios)),
        ("3. Máximo que puede lograr el contenido del destino (oráculo que siempre sabe el lugar, 30 semillas)",
         t3_maximo_por_lugar(destinos)),
        ("4. ¿El perfil o el historial predicen el lugar? (prueba, 606 usuarios)", t4_predecir_lugar(usuarios, destinos)),
    ]
    partes = ["# Evidencia para la tarea 5 — red neuronal con contenido\n",
              "_Generado por `src/evidencia_tarea5.py`. No editar a mano._\n"]
    for i, (titulo, tabla) in enumerate(secciones, start=1):
        tabla.to_csv(SALIDA / f"tabla_{i:02d}.csv", index=False)
        partes += [f"## {titulo}\n", _a_markdown(tabla) + "\n"]
        print(f"\n=== {titulo} ===\n{tabla.to_string(index=False)}")
    (SALIDA / "RESUMEN.md").write_text("\n".join(partes), encoding="utf-8")


if __name__ == "__main__":
    main()
