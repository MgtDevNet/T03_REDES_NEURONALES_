"""Pruebas del evaluador común (`evaluacion.py`).

Cada prueba compara lo que calcula el código con un resultado hecho A MANO, para no confiar a
ciegas en las fórmulas (la «Cacería de la alucinación» pide buscar errores de código y de
matemáticas). Si algo falla, el script se detiene con un mensaje.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/probar_evaluacion.py
"""

import numpy as np
import pandas as pd

import evaluacion as ev
from evaluacion import N_DESTINOS, evaluar, cargar_escenario


def con_puntajes(tabla):
    """Modelo de juguete: devuelve puntajes fijos, uno por usuario."""
    return lambda usuarios: np.array([tabla[u] for u in usuarios], dtype=float)


def prueba_ejemplo_a_mano():
    """3 usuarios; puntajes decrecientes 1000, 999, ... para que el orden sea el de los IDs."""
    base = np.arange(N_DESTINOS, 0, -1, dtype=float)  # destino 1 el mejor, destino 1000 el peor
    aprende = pd.DataFrame({"UserID": [1, 2, 2], "DestinationID": [1, 1, 2]})
    # Usuario 1 tiene el destino 1 aprendido → se quita; escondido el 2 → queda en posición 0.
    # Usuario 2 tiene 1 y 2 aprendidos → se quitan; escondido el 5 → posiciones: 3,4,5 → rango 2.
    # Usuario 3 no tiene nada aprendido; escondido el 12 → rango 11.
    evalua = pd.DataFrame({"UserID": [1, 2, 3], "DestinationID": [2, 5, 12]})
    resumen, por_usuario = evaluar(con_puntajes({1: base, 2: base, 3: base}), aprende, evalua, ks=(1, 3, 10, 20))
    assert por_usuario.rango.tolist() == [0, 2, 11], por_usuario.rango.tolist()

    r = resumen.set_index("K")
    # HitRate: K=1 → solo el usuario 1; K=3 → usuarios 1 y 2; K=10 → 1 y 2; K=20 → los tres.
    assert np.allclose(r.HitRate.tolist(), [1 / 3, 2 / 3, 2 / 3, 1.0])
    assert np.allclose(r.Recall, r.HitRate)
    # Precision@3 = (1/3 + 1/3 + 0) / 3 usuarios = 2/9
    assert np.isclose(r.loc[3, "Precision"], 2 / 9)
    # NDCG@3 = (1/log2(2) + 1/log2(4) + 0) / 3 = (1 + 0,5) / 3 = 0,5
    assert np.isclose(r.loc[3, "NDCG"], 0.5), r.loc[3, "NDCG"]
    # NDCG@20 = (1 + 0,5 + 1/log2(13)) / 3
    assert np.isclose(r.loc[20, "NDCG"], (1 + 0.5 + 1 / np.log2(13)) / 3)
    # Cobertura@1: los tops son 2, 3 y 1 → 3 destinos distintos de 1000.
    assert np.isclose(r.loc[1, "Cobertura"], 3 / 1000)
    print("OK  ejemplo a mano: rangos [0, 2, 11]; HitRate, Precision, NDCG y cobertura coinciden")


def prueba_lugar_a_mano():
    """Lugar de cada ID por orden alfabético: 0 Goa, 1 Jaipur, 2 Kerala, 3 Leh, 4 Taj Mahal.
    El ID i pertenece a uno de ellos según destinos.csv; se usa ese mapa real."""
    lugar_id, nombres = ev.lugar_de_cada_id()
    assert nombres == ["Goa Beaches", "Jaipur City", "Kerala Backwaters", "Leh Ladakh", "Taj Mahal"], nombres
    # Puntaje = 10 para los IDs de Kerala, 5 para Taj Mahal, 1 para Goa y 0 para el resto.
    base = np.select([lugar_id == 2, lugar_id == 4, lugar_id == 0], [10.0, 5.0, 1.0], 0.0)
    id_taj = int(np.flatnonzero(lugar_id == 4)[0]) + 1
    id_jaipur = int(np.flatnonzero(lugar_id == 1)[0]) + 1
    aprende = pd.DataFrame({"UserID": [], "DestinationID": []}, dtype=int)
    # Usuario 1: escondido un Taj Mahal → lugares ordenados Kerala, Taj, Goa, ... → rango_lugar 1.
    # Usuario 2: escondido un Jaipur (puntaje 0, empatado con Leh) → posición 3 o 4.
    evalua = pd.DataFrame({"UserID": [1, 2], "DestinationID": [id_taj, id_jaipur]})
    resumen, por_u = evaluar(lambda us: np.tile(base, (len(us), 1)), aprende, evalua)
    assert por_u.rango_lugar.iloc[0] == 1 and por_u.rango_lugar.iloc[1] in (3, 4), por_u.rango_lugar.tolist()
    assert np.isclose(resumen["HitRate_lugar@1"].iloc[0], 0.0)
    assert np.isclose(resumen["HitRate_lugar@3"].iloc[0], 0.5)
    print("OK  lugar a mano: Taj Mahal queda 2.º (acierta @3, no @1); Jaipur 4.º o 5.º; HitRate_lugar@3 = 0,5")


def prueba_empates():
    """Un modelo que empata todo debe dar posiciones al azar: el escondido cae en el top-10 ≈ 1 %."""
    iguales = np.zeros(N_DESTINOS)
    evalua = pd.DataFrame({"UserID": np.arange(1, 2001), "DestinationID": 500})
    aprende = pd.DataFrame({"UserID": [], "DestinationID": []}, dtype=int)
    resumen, por_usuario = evaluar(lambda us: np.tile(iguales, (len(us), 1)), aprende, evalua, ks=(10,))
    hr = resumen.HitRate.iloc[0]
    # 2000 usuarios con p = 10/1000: error estándar = sqrt(0,01 · 0,99 / 2000) ≈ 0,0022
    assert abs(hr - 0.01) < 4 * 0.0022, hr
    assert 0 <= por_usuario.rango.min() and por_usuario.rango.max() < N_DESTINOS
    print(f"OK  empates: HitRate@10 = {hr:.4f} con todo empatado (esperado 0,01 ± 0,009)")


def prueba_reproducible():
    """Misma semilla → mismo resultado; otra semilla → otros desempates."""
    iguales = lambda us: np.zeros((len(us), N_DESTINOS))
    evalua = pd.DataFrame({"UserID": np.arange(1, 51), "DestinationID": 7})
    aprende = pd.DataFrame({"UserID": [], "DestinationID": []}, dtype=int)
    a = evaluar(iguales, aprende, evalua, semilla=3)[1].rango
    b = evaluar(iguales, aprende, evalua, semilla=3)[1].rango
    c = evaluar(iguales, aprende, evalua, semilla=4)[1].rango
    assert a.equals(b) and not a.equals(c)
    print("OK  reproducible: misma semilla, mismo resultado")


def prueba_datos_reales():
    """Con los datos reales, el azar debe acercarse al 1 % que predice T4-1."""
    for escenario in ("ajuste", "final"):
        aprende, evalua = cargar_escenario(escenario)
        hrs = []
        for s in range(30):
            rng = np.random.default_rng(1000 + s)
            azar = lambda us, rng=rng: rng.random((len(us), N_DESTINOS))
            hrs.append(evaluar(azar, aprende, evalua, semilla=s)[0].set_index("K").loc[10, "HitRate"])
        hl = [evaluar(lambda us, r=np.random.default_rng(2000 + s): r.random((len(us), N_DESTINOS)),
                      aprende, evalua, semilla=s)[0].iloc[0] for s in range(30)]
        l1, l3 = np.mean([h["HitRate_lugar@1"] for h in hl]), np.mean([h["HitRate_lugar@3"] for h in hl])
        print(f"OK  azar por lugar ({escenario}): HitRate_lugar@1 = {100 * l1:.1f} % (esperado 20), "
              f"HitRate_lugar@3 = {100 * l3:.1f} % (esperado 60)")
        assert abs(l1 - 0.2) < 0.03 and abs(l3 - 0.6) < 0.03
        print(f"OK  azar en datos reales ({escenario}, {len(evalua)} usuarios, 30 semillas): "
              f"HitRate@10 = {100 * np.mean(hrs):.2f} % (desv. {100 * np.std(hrs, ddof=1):.2f}); esperado ≈ 1,0 %")
        assert abs(np.mean(hrs) - 0.01) < 0.004


if __name__ == "__main__":
    prueba_ejemplo_a_mano()
    prueba_lugar_a_mano()
    prueba_empates()
    prueba_reproducible()
    prueba_datos_reales()
    print("\nTodas las pruebas del evaluador pasaron.")
