"""Evaluador común del módulo 3 (DECISIONES.md, decisión 7.4).

TODOS los modelos (azar, popularidad, factorización de matrices y la red) se miden con este mismo
código. Eso es lo que hace justa la comparación.

Cómo funciona, para cada usuario evaluado (uno con un destino escondido):
1. El modelo da un puntaje a los 1000 destinos.
2. Se quitan los destinos que el usuario ya tiene en lo aprendido (puntaje = −∞). El destino
   escondido nunca está ahí (lo garantiza preprocesar.py), así que no se pierde ningún acierto.
3. Se ordenan los destinos de mayor a menor puntaje. Los EMPATES se rompen al azar con una
   semilla: si no, un modelo con muchos empates (la popularidad) quedaría mejor o peor según el
   orden de los IDs, que no significa nada.
4. Se mira en qué posición quedó el destino escondido (rango 0 = el primero).

Métricas con UN destino relevante por usuario (dejar uno fuera), para un corte K:
    HitRate@K   = 1 si rango < K, si no 0
    Recall@K    = HitRate@K          (hay 1 relevante: o se recupera o no)
    Precision@K = HitRate@K / K      (de K recomendados, a lo sumo 1 es relevante)
    NDCG@K      = 1 / log2(rango + 2) si rango < K, si no 0   (el ideal, IDCG, vale 1)
Cobertura@K = fracción de los 1000 destinos que aparece en al menos un top-K.
Cada métrica se promedia sobre los usuarios evaluados.

Métricas por LUGAR (decisión 7.5; lo que ve el cliente):
    Cada uno de los 5 lugares recibe el mejor puntaje entre sus IDs (sin los ya aprendidos);
    se ordenan los lugares (empates al azar) y se mira la posición del lugar escondido.
    HitRate_lugar@3 (PRINCIPAL) = 1 si está entre los 3 primeros   (azar: 60 %)
    HitRate_lugar@1             = 1 si queda primero               (azar: 20 %)
"""

from pathlib import Path

import numpy as np
import pandas as pd

MODULO = Path(__file__).resolve().parents[1]
PROCESADOS = MODULO / "data" / "processed"
N_DESTINOS = 1000
KS = (1, 3, 5, 10, 20)
KS_LUGAR = (1, 3)


def lugar_de_cada_id():
    """Arreglo de 1000 posiciones: índice (0–4) del lugar de cada DestinationID, en orden alfabético."""
    destinos = pd.read_csv(PROCESADOS / "destinos.csv").set_index("DestinationID").reindex(range(1, N_DESTINOS + 1))
    nombres = sorted(destinos.Name.unique())
    return destinos.Name.map({n: i for i, n in enumerate(nombres)}).to_numpy(), nombres


def cargar_escenario(nombre):
    """Devuelve (aprende, evalua) para uno de los dos momentos de la evaluación.

    - "ajuste": aprende con entrenamiento, se mide en validación (para elegir configuraciones).
    - "final":  aprende con entrenamiento + validación, se mide en prueba (una sola vez).
    """
    inter = pd.read_csv(PROCESADOS / "interacciones.csv")
    ent = inter[inter.conjunto == "entrenamiento"]
    if nombre == "ajuste":
        return ent, inter[inter.conjunto == "validación"]
    if nombre == "final":
        return pd.concat([ent, inter[inter.conjunto == "validación"]]), inter[inter.conjunto == "prueba"]
    raise ValueError(f"escenario desconocido: {nombre!r} (usa 'ajuste' o 'final')")


def enmascarar(puntajes, aprendidos):
    """Copia de los puntajes con −∞ en los destinos que cada usuario ya tiene aprendidos."""
    p = np.array(puntajes, dtype=float, copy=True)
    for fila, ids in enumerate(aprendidos):
        if len(ids):
            p[fila, np.asarray(list(ids)) - 1] = -np.inf
    return p


def rango_por_lugar(p_enmascarado, objetivo, lugar_id, rng):
    """Posición (0 = primero) del lugar del destino escondido entre los 5 lugares."""
    n_lugares = lugar_id.max() + 1
    por_lugar = np.stack([p_enmascarado[:, lugar_id == l].max(axis=1) for l in range(n_lugares)], axis=1)
    orden = np.lexsort((rng.random(por_lugar.shape), -por_lugar), axis=1)
    return np.argmax(orden == lugar_id[objetivo][:, None], axis=1)


def ordenar(puntajes, aprendidos, rng):
    """Orden de los destinos para cada usuario (fila), con empates rotos al azar.

    puntajes:   matriz (usuarios × destinos); la columna j es el DestinationID j + 1.
    aprendidos: lista con, para cada fila, los DestinationID que el usuario ya tiene.
    Devuelve una matriz de índices de columna, del mejor al peor.
    """
    p = enmascarar(puntajes, aprendidos)
    desempate = rng.random(p.shape)
    # lexsort ordena por la ÚLTIMA clave primero: puntaje descendente y, si empata, el azar.
    return np.lexsort((desempate, -p), axis=1)


def evaluar(puntuar, aprende, evalua, semilla=0, ks=KS):
    """Mide un modelo. `puntuar(user_ids)` debe devolver la matriz de puntajes (usuarios × 1000).

    Devuelve (resumen, por_usuario):
      resumen      DataFrame con una fila por K: HitRate, Recall, Precision, NDCG y Cobertura.
      por_usuario  DataFrame con el rango del destino escondido de cada usuario y cuántas
                   interacciones tenía para aprender (sirve para los grupos de la tarea 6).
    """
    rng = np.random.default_rng(semilla)
    usuarios = evalua.UserID.to_numpy()
    objetivo = evalua.DestinationID.to_numpy() - 1  # columna del destino escondido
    items_de = aprende.groupby("UserID").DestinationID.apply(list).to_dict()
    aprendidos = [items_de.get(u, []) for u in usuarios]

    puntajes = np.asarray(puntuar(usuarios), dtype=float)
    assert puntajes.shape == (len(usuarios), N_DESTINOS), f"forma inesperada: {puntajes.shape}"
    orden = ordenar(puntajes, aprendidos, rng)
    rango = np.argmax(orden == objetivo[:, None], axis=1)  # posición del escondido (0 = primero)
    # Por lugar (se calcula DESPUÉS, con el mismo rng: no cambia nada de lo calculado por ID).
    lugar_id, _ = lugar_de_cada_id()
    rango_lugar = rango_por_lugar(enmascarar(puntajes, aprendidos), objetivo, lugar_id, rng)

    filas = []
    for k in ks:
        acierto = rango < k
        filas.append({
            "K": k,
            "HitRate": acierto.mean(),
            "Recall": acierto.mean(),
            "Precision": (acierto / k).mean(),
            "NDCG": np.where(acierto, 1 / np.log2(rango + 2), 0.0).mean(),
            "Cobertura": len(np.unique(orden[:, :k])) / N_DESTINOS,
        })
    for fila in filas:  # las métricas por lugar no dependen de K: se repiten en cada fila
        for kl in KS_LUGAR:
            fila[f"HitRate_lugar@{kl}"] = (rango_lugar < kl).mean()
    por_usuario = pd.DataFrame({
        "UserID": usuarios, "DestinationID": objetivo + 1, "rango": rango, "rango_lugar": rango_lugar,
        "n_aprendidas": [len(a) for a in aprendidos],
    })
    return pd.DataFrame(filas), por_usuario
