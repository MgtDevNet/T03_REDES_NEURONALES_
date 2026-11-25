"""Verificación de las cifras del informe (informe/index.html) contra los datos y los resultados.

Cada comprobación recalcula una cifra desde los datos crudos/procesados o la lee de las tablas
generadas por los scripts, y la compara con el valor escrito en el informe. Imprime OK o DIFIERE.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/verificar_informe.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from inventario import cargar, interacciones_unidas

MODULO = Path(__file__).resolve().parents[1]
RES = MODULO / "results"
PROC = MODULO / "data" / "processed"
filas = []


def chequear(seccion, que, informe, calculado, tol=0.0):
    ok = (abs(float(informe) - float(calculado)) <= tol) if isinstance(informe, (int, float)) else str(informe) == str(calculado)
    filas.append({"sección": seccion, "qué": que, "informe": informe, "calculado": calculado, "estado": "OK" if ok else "DIFIERE"})


def main():
    destinos, usuarios, resenas, historial = cargar()
    inter = interacciones_unidas(resenas, historial)
    proc = pd.read_csv(PROC / "interacciones.csv")
    nombre = destinos.set_index("DestinationID").Name

    # ---------------- 1. Datos ----------------
    chequear("1.2", "usuarios", 999, len(usuarios))
    chequear("1.2", "usuarios con ≥ 1 interacción", 858, inter.UserID.nunique())
    chequear("1.2", "destinos con ≥ 1 interacción", 866, inter.DestinationID.nunique())
    chequear("1.2", "reseñas", 999, len(resenas))
    chequear("1.2", "historial", 999, len(historial))
    chequear("1.2", "densidad unión (%)", 0.20, round(100 * len(inter) / (999 * 1000), 2))
    chequear("1.2", "interacciones por usuario activo", 2.33, round(len(inter) / inter.UserID.nunique(), 2))
    por_u = inter.groupby("UserID").size().reindex(usuarios.UserID, fill_value=0)
    por_d = inter.groupby("DestinationID").size().reindex(destinos.DestinationID, fill_value=0)
    chequear("1.2", "usuarios con 1 interacción", 252, int((por_u == 1).sum()))
    chequear("1.2", "destinos sin interacciones", 134, int((por_d == 0).sum()))
    chequear("1.2/3.3", "máximo de interacciones por usuario", 7, int(por_u.max()))
    chequear("3.3", "mediana de interacciones por usuario", 2, float(por_u.median()))
    g = usuarios.Gender.value_counts()
    chequear("1.2 Tabla 3", "mujeres / hombres", "496 / 503", f"{g['Female']} / {g['Male']}")
    chequear("1.2 Tabla 3", "Popularity mín–máx", "7,5–9,5", f"{destinos.Popularity.min():.1f}–{destinos.Popularity.max():.1f}".replace(".", ","))
    chequear("1.2 Tabla 3", "nombres distintos de usuario", 20, usuarios.Name.nunique())
    chequear("1.3 Tabla 4", "visitas por fecha (3 fechas)", "333/333/333",
             "/".join(map(str, historial.VisitDate.value_counts().sort_index().tolist())))
    chequear("1.3 Tabla 4", "reseñas en el historial del mismo usuario", 0,
             len(resenas.merge(historial, on=["UserID", "DestinationID"])))

    # ---------------- 2. Caso de uso ----------------
    t14 = pd.read_csv(RES / "inventario" / "tabla_14.csv")
    print("Tabla I-14 (para la Tabla 6):\n", t14.to_string(index=False), "\n")

    # ---------------- 3. Descriptivo ----------------
    d1 = pd.read_csv(RES / "descriptivo" / "tabla_01.csv")
    print("Tabla D-1 (para la Tabla 7 y los p-valores 0,28 / 0,998):\n", d1.to_string(index=False), "\n")
    lug = inter.DestinationID.map(nombre).value_counts(normalize=True) * 100
    chequear("3.2", "lugar con menos interacciones (%)", 18.8, round(lug.min(), 1))
    chequear("3.2", "lugar con más interacciones (%)", 21.1, round(lug.max(), 1))
    d3 = pd.read_csv(RES / "descriptivo" / "tabla_03.csv")
    for etiqueta, inf in (("0 (sin actividad)", 141), ("exactamente 1", 252), ("2 o menos", 673), ("3 o más", 326), ("5 o más", 52)):
        chequear("3.3 Tabla 9", etiqueta, inf, int(d3.set_index(d3.columns[0]).loc[etiqueta, "usuarios"]))

    # ---------------- 4. Partición ----------------
    ent = proc[proc.conjunto == "entrenamiento"]
    def resumen_usuario(u):
        x = proc[proc.UserID == u]
        return {c: ", ".join(f"{l} ({i})" for l, i in zip(x[x.conjunto == c].lugar, x[x.conjunto == c].DestinationID)) or "—"
                for c in ("entrenamiento", "validación", "prueba")}
    for u, texto in ((6, {"entrenamiento": "Jaipur City (963)"}), (1, {"entrenamiento": "Goa Beaches (467)", "prueba": "Taj Mahal (941)"}),
                     (16, {"validación": "Leh Ladakh (40)", "prueba": "Jaipur City (113)"})):
        r = resumen_usuario(u)
        for c, v in texto.items():
            chequear("4 Tabla 11", f"usuario {u}, {c}", v, r[c])
    r16 = resumen_usuario(16)["entrenamiento"]
    chequear("4 Tabla 11", "usuario 16, entrenamiento (contiene 970 y 176)", True, ("(970)" in r16) and ("(176)" in r16))
    chequear("4", "destino 963: interacciones en entrenamiento", 1, int((ent.DestinationID == 963).sum()))
    chequear("4", "interacciones ent + val", 1392, int((proc.conjunto != "prueba").sum()))
    chequear("4", "tamaños ent / val / prueba", "1066 / 326 / 606",
             " / ".join(str((proc.conjunto == c).sum()) for c in ("entrenamiento", "validación", "prueba")))

    # ---------------- 6 y 7. Modelos ----------------
    fm = json.loads((RES / "tarea4" / "fm_decision.json").read_text(encoding="utf-8"))
    chequear("6.4", "segundos del ajuste de la factorización", 23,
             round(pd.read_csv(RES / "tarea4" / "fm_ajuste_corridas.csv").segundos.sum()), tol=0)
    chequear("6.4", "configuración elegida", "d=8, λ=1, 7 épocas", f"d={fm['d']}, λ={fm['lambda']:g}, {fm['epocas']} épocas")
    red = json.loads((RES / "tarea5" / "red_decision.json").read_text(encoding="utf-8"))
    chequear("7.4", "segundos del ajuste de la red", 64, round(red["segundos_ajuste"]))
    rr = pd.read_csv(RES / "tarea5" / "red_final_resumen.csv").set_index("K").loc[10]
    ref = pd.read_csv(RES / "tarea4" / "referencias_resumen.csv")
    ref = ref[(ref.escenario == "final") & (ref.K == 10)].set_index("modelo")
    fmr = pd.read_csv(RES / "tarea4" / "fm_final_resumen.csv").set_index("K").loc[10]
    lugar3 = {"Azar": ref.loc["Azar", "HitRate_lugar@3_%"], "Popularidad": ref.loc["Popularidad", "HitRate_lugar@3_%"],
              "Factorización": fmr["HitRate_lugar@3 (%)"], "Red": rr["HitRate_lugar@3"]}
    lugar1 = {"Azar": ref.loc["Azar", "HitRate_lugar@1_%"], "Popularidad": ref.loc["Popularidad", "HitRate_lugar@1_%"],
              "Factorización": fmr["HitRate_lugar@1 (%)"], "Red": rr["HitRate_lugar@1"]}
    mas_lejos = max((m for m in lugar3 if m != "Azar"), key=lambda m: (lugar3[m] - 60) + (lugar1[m] - 20))
    chequear("7.5", "modelo que más se aleja del azar en las dos métricas por lugar", "Red", mas_lejos)
    chequear("7.5", "¿la red supera al azar en Lugar@3 y en Lugar@1?", True, lugar3["Red"] > 60 and lugar1["Red"] > 20)

    t = pd.DataFrame(filas)
    pd.set_option("display.width", 200, "display.max_colwidth", 70)
    print(t.to_string(index=False))
    print(f"\n{(t.estado == 'OK').sum()} OK, {(t.estado != 'OK').sum()} DIFIERE")


if __name__ == "__main__":
    main()
