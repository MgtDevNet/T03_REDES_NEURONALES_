"""Prueba de la factorización de matrices con DATOS DE JUGUETE de respuesta conocida.

Por qué: si con los datos reales la factorización queda igual al azar, puede ser porque los datos
no tienen señal (lo esperado, tarea 2) o porque el código tiene un error. Con datos reales no se
distingue. Aquí se fabrican datos CON señal; si el código está bien, debe acertar mucho más que
el azar y que la popularidad.

Datos de juguete (mismo tamaño que los reales: 999 usuarios, 1000 destinos):
- 50 grupos de 20 destinos (el destino i pertenece al grupo (i − 1) mod 50).
- Cada usuario pertenece a un grupo y visita 4 destinos distintos de SU grupo, al azar.
- Partición dejar-uno-fuera: 1 para validación, 1 para prueba, 2 para entrenamiento.
Lo mejor que puede hacer un modelo es adivinar el grupo y recomendar sus otros destinos: con 20
destinos por grupo, 3 ya vistos y K = 10, el techo de HitRate@10 es 10/17 ≈ 59 %. El azar da ≈ 1 %.

Configuración de la prueba: d = 16, λ = 0,1. Un primer intento con λ = 0,0001 acertó solo 12,8 %
porque el modelo memorizaba (pérdida 0,007); al subir λ: 0,001 → 15,2 %, 0,01 → 48,4 %,
0,1 → 58,4 %. Esto mostró que la rejilla planeada (λ ≤ 0,01) era demasiado débil para datos de
este tamaño (DECISIONES.md, paso 7).

Criterio de éxito: al menos 2/3 del techo (≈ 40 %), es decir, unas 40 veces el azar.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/probar_factorizacion.py
"""

import time

import numpy as np
import pandas as pd

from evaluacion import N_DESTINOS, evaluar
from factorizacion import N_USUARIOS, FactorizacionBPR
from referencias import Azar, Popularidad

N_GRUPOS, POR_USUARIO = 50, 4


def datos_de_juguete(semilla=0):
    rng = np.random.default_rng(semilla)
    destinos_del_grupo = {g: np.arange(g, N_DESTINOS, N_GRUPOS) + 1 for g in range(N_GRUPOS)}
    filas = []
    for u in range(1, N_USUARIOS + 1):
        g = rng.integers(N_GRUPOS)
        elegidos = rng.choice(destinos_del_grupo[g], size=POR_USUARIO, replace=False)
        for pos, i in enumerate(elegidos):
            conj = "prueba" if pos == 0 else "validación" if pos == 1 else "entrenamiento"
            filas.append((u, int(i), conj))
    inter = pd.DataFrame(filas, columns=["UserID", "DestinationID", "conjunto"])
    ent = inter[inter.conjunto == "entrenamiento"]
    val = inter[inter.conjunto == "validación"]
    pru = inter[inter.conjunto == "prueba"]
    return ent, val, pru


def main():
    ent, val, pru = datos_de_juguete()
    aprende = pd.concat([ent, val])
    print(f"Datos de juguete: {len(aprende)} interacciones para aprender, {len(pru)} usuarios evaluados\n")

    filas = []
    for modelo in (Azar(semilla=0), Popularidad()):
        r = evaluar(modelo.ajustar(aprende).puntuar, aprende, pru, semilla=0)[0].set_index("K")
        filas.append({"modelo": modelo.nombre, "HitRate@10 (%)": 100 * r.loc[10, "HitRate"],
                      "NDCG@10 (%)": 100 * r.loc[10, "NDCG"], "segundos": 0.0})

    inicio = time.perf_counter()
    fm = FactorizacionBPR(d=16, lam=0.1, epocas=200, semilla=0).ajustar(ent, validacion=val)
    print(f"Ajuste con validación: mejor época = {fm.mejor_epoca}")
    fm_final = FactorizacionBPR(d=16, lam=0.1, epocas=fm.mejor_epoca, semilla=0).ajustar(aprende)
    seg = time.perf_counter() - inicio
    r = evaluar(fm_final.puntuar, aprende, pru, semilla=0)[0].set_index("K")
    filas.append({"modelo": fm_final.nombre, "HitRate@10 (%)": 100 * r.loc[10, "HitRate"],
                  "NDCG@10 (%)": 100 * r.loc[10, "NDCG"], "segundos": seg})

    tabla = pd.DataFrame(filas).round(2)
    print(tabla.to_string(index=False))

    h = pd.DataFrame(fm_final.historia)
    print("\nPérdida de entrenamiento (debe bajar):",
          ", ".join(f"época {e}: {p:.3f}" for e, p in zip(h.epoca, h.perdida) if e in (1, 5, 10, 20, len(h))))
    print(f"Parámetros entrenables: {fm_final.parametros()} = (999 + 1000) × 16 = {(999 + 1000) * 16}")

    hr_fm = tabla.set_index("modelo").loc[fm_final.nombre, "HitRate@10 (%)"]
    assert h.perdida.iloc[-1] < h.perdida.iloc[0], "la pérdida no bajó"
    assert fm_final.parametros() == (999 + 1000) * 16
    assert hr_fm > 2 / 3 * 100 * 10 / 17, f"la factorización debería acertar mucho más que el azar y acertó {hr_fm:.1f} %"
    print(f"\nOK: con datos que SÍ tienen señal, la factorización acierta {hr_fm:.1f} % "
          f"(techo ≈ 59 %, azar ≈ 1 %). El código aprende.")


if __name__ == "__main__":
    main()
