"""Prueba de la red de dos torres con DATOS SIMULADOS de respuesta conocida.

Igual que con la factorización (sección 6.3 del informe): si con los datos reales la red queda
igual al azar, hay que poder descartar que sea un error del código.

Datos simulados: los mismos usuarios (con sus preferencias reales), las mismas cantidades y la
misma partición, pero cada interacción se reemplaza por un ID de un lugar elegido así:
    - con probabilidad 0,9, uno de los 2 tipos que el usuario dice preferir (al azar entre ellos);
    - con probabilidad 0,1, cualquiera de los 5 lugares.
Aquí las preferencias SÍ permiten predecir el lugar. Lo mejor que puede hacer un modelo es poner
primero sus 2 lugares preferidos:
    HitRate@3 por lugar  ≈ 0,9 + 0,1 · 3/5 = 96 %     (azar 60 %)
    HitRate@1 por lugar  ≈ 0,9 · 1/2 + 0,1 · 1/5 = 47 %   (azar 20 %)
Criterio de éxito: Lugar@3 ≥ 85 % y Lugar@1 ≥ 35 %.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/probar_dos_torres.py
"""

import time

import numpy as np
import pandas as pd

from dos_torres import TIPOS, DosTorres
from evaluacion import PROCESADOS, evaluar, lugar_de_cada_id


def datos_simulados(semilla=0, p_preferido=0.9):
    rng = np.random.default_rng(semilla)
    inter = pd.read_csv(PROCESADOS / "interacciones.csv")
    usuarios = pd.read_csv(PROCESADOS / "usuarios.csv").set_index("UserID")
    destinos = pd.read_csv(PROCESADOS / "destinos.csv")
    lugar_id, nombres = lugar_de_cada_id()
    tipo_a_lugar = {t: nombres.index(n) for n, t in destinos.drop_duplicates("Name")[["Name", "Type"]].values}
    ids_de_lugar = {l: np.flatnonzero(lugar_id == l) + 1 for l in range(5)}
    nuevos, vistos = [], set()
    for u in inter.UserID:
        preferidos = [tipo_a_lugar[t] for t in TIPOS if usuarios.loc[u, f"pref_{t}"] == 1]
        lugar = rng.choice(preferidos) if rng.random() < p_preferido else rng.integers(5)
        while True:  # sin repetir (usuario, destino)
            i = int(rng.choice(ids_de_lugar[lugar]))
            if (u, i) not in vistos:
                vistos.add((u, i))
                nuevos.append(i)
                break
    return inter.assign(DestinationID=nuevos)


def main():
    inter = datos_simulados()
    ent = inter[inter.conjunto == "entrenamiento"]
    val = inter[inter.conjunto == "validación"]
    pru = inter[inter.conjunto == "prueba"]
    aprende = pd.concat([ent, val])

    t0 = time.perf_counter()
    filas = []
    for perdida in ("bpr", "bce"):
        m = DosTorres(d=4, perdida=perdida, negativos=1, semilla=0).ajustar(ent, validacion=val)
        final = DosTorres(d=4, perdida=perdida, negativos=1, semilla=0, epocas=m.mejor_epoca).ajustar(aprende)
        r = evaluar(final.puntuar, aprende, pru, semilla=0)[0].set_index("K")
        h = pd.DataFrame(final.historia)
        filas.append({"pérdida": perdida.upper(), "mejor época": m.mejor_epoca,
                      "pérdida inicial": round(h.perdida.iloc[0], 3), "pérdida final": round(h.perdida.iloc[-1], 3),
                      "Lugar@3 (%)": round(100 * r.loc[10, "HitRate_lugar@3"], 1),
                      "Lugar@1 (%)": round(100 * r.loc[10, "HitRate_lugar@1"], 1),
                      "HitRate@10 (%)": round(100 * r.loc[10, "HitRate"], 2),
                      "parámetros": final.parametros()})
    tabla = pd.DataFrame(filas)
    print(f"Datos simulados: {len(aprende)} interacciones para aprender, {len(pru)} usuarios evaluados "
          f"({time.perf_counter() - t0:.1f} s)\n")
    print(tabla.to_string(index=False))
    print("\nMáximo posible: Lugar@3 ≈ 96 %, Lugar@1 ≈ 47 %. Azar: 60 % y 20 %.")
    for _, f in tabla.iterrows():
        assert f["pérdida final"] < f["pérdida inicial"], "la pérdida no bajó"
        assert f["Lugar@3 (%)"] >= 85 and f["Lugar@1 (%)"] >= 35, f"no aprendió con {f['pérdida']}"
    print("\nOK: con datos en los que las preferencias predicen el lugar, la red lo aprende (BPR y BCE).")


if __name__ == "__main__":
    main()
