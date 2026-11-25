"""Preguntas de negocio del módulo 3.

P1. Usuarios nuevos: ¿qué recomendar a un cliente sin historial?
    Se simulan clientes nuevos con los 606 usuarios de prueba: se reparten en 5 grupos; para cada
    grupo se BORRA todo su historial del entrenamiento (el modelo nunca los vio) y se les recomienda
    solo con su perfil. Se mide si el lugar escondido queda entre los 3 recomendados. 30 semillas.
    Estrategias: azar, popularidad, regla «lo que más visita tu perfil» y la red con preferencias.

P2. Concentración: ¿las recomendaciones concentran a los clientes en pocos destinos?
    Con los modelos finales (entrenamiento + validación) y los 606 usuarios de prueba: qué % de
    clientes recibe cada lugar como 1.ª recomendación y entre sus 3; factor de capacidad =
    % de clientes que lo reciben primero / 20 % (demanda pareja actual). Por ID: qué parte de las
    recomendaciones top-10 se lleva cada destino, ordenados por popularidad (figura).

P3. Campaña de un solo destino este mes (noviembre).
    Candidatos: lugares en temporada en noviembre según BestTimeToVisit. Grupo: los clientes a
    quienes la red (modelo guardado) les recomienda ese lugar en primer lugar. Conversiones
    esperadas = clientes del grupo × tasa estimada en entrenamiento (proporción de las
    interacciones de ese grupo que van a ese lugar). Se comprueba en prueba.
    Supuestos: conversión = el próximo viaje del cliente es a ese lugar; la campaña no cambia la
    conducta (se estima la propensión, no el efecto del anuncio); un viaje por cliente en el mes;
    el público son los 999 clientes del conjunto; los datos son sintéticos.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/preguntas_negocio.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from descriptivo import GRIS_REF  # noqa: F401  (estilo común)
from dos_torres import DosTorres
from ejemplos_embeddings import cargar_red
from evaluacion import N_DESTINOS, PROCESADOS, cargar_escenario, enmascarar, evaluar, lugar_de_cada_id
from factorizacion import FactorizacionBPR
from inventario import _a_markdown
from referencias import Azar, Popularidad, prueba_lugar

MODULO = Path(__file__).resolve().parents[1]
SALIDA = MODULO / "results" / "negocio"
FIGURAS = MODULO / "figures"
N_SEMILLAS, N_PLIEGUES = 30, 5
COLORES = {"Azar": "#2a78d6", "Popularidad": "#eb6834", "Factorización (BPR)": "#1baf7a", "Red de dos torres": "#eda100"}
MES = 11  # noviembre
MESES = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

torch.set_num_threads(1)
USUARIOS = pd.read_csv(PROCESADOS / "usuarios.csv")
LUGAR_ID, NOMBRES = lugar_de_cada_id()
RED_CFG = json.loads((MODULO / "results" / "tarea5" / "red_decision.json").read_text(encoding="utf-8"))
FM_CFG = json.loads((MODULO / "results" / "tarea4" / "fm_decision.json").read_text(encoding="utf-8"))


class ReglaPerfil:
    """Puntaje de un destino = % de las interacciones de los usuarios con el MISMO perfil de
    preferencias que fueron a su lugar (en lo aprendido). Sin entrenamiento."""

    nombre = "Regla: lo que más visita tu perfil"

    def ajustar(self, aprende):
        x = aprende.merge(USUARIOS[["UserID", "Preferences"]], on="UserID")
        x["lugar"] = LUGAR_ID[x.DestinationID.to_numpy() - 1]
        self.tasa = pd.crosstab(x.Preferences, x.lugar, normalize="index")
        self.perfil = USUARIOS.set_index("UserID").Preferences
        return self

    def puntuar(self, usuarios):
        return np.stack([self.tasa.loc[self.perfil[u]].reindex(range(5), fill_value=0).to_numpy()[LUGAR_ID] for u in usuarios])


def red(semilla, epocas=None):
    return DosTorres(d=RED_CFG["d"], perdida=RED_CFG["perdida"], negativos=RED_CFG["negativos"],
                     usuario=RED_CFG["usuario"], epocas=epocas or RED_CFG["epocas"], semilla=semilla)


# ----------------------------------------------------------------- P1
def p1_usuarios_nuevos():
    aprende, pru = cargar_escenario("final")
    pliegues = np.array_split(np.random.default_rng(0).permutation(pru.UserID.to_numpy()), N_PLIEGUES)
    filas = []
    for s in range(N_SEMILLAS):
        for f, nuevos in enumerate(pliegues):
            ent = aprende[~aprende.UserID.isin(nuevos)]        # borrar TODO el historial de los «nuevos»
            ev = pru[pru.UserID.isin(nuevos)]
            estrategias = {"Azar": Azar(semilla=1000 + s), "Popularidad": Popularidad(),
                           ReglaPerfil.nombre: ReglaPerfil(), "Red de dos torres": red(s)}
            for nombre, m in estrategias.items():
                _, pu = evaluar(m.ajustar(ent).puntuar, ent, ev, semilla=s)
                assert (pu.n_aprendidas == 0).all()             # de verdad son nuevos
                filas.append(pu.assign(estrategia=nombre, semilla=s))
    pu = pd.concat(filas, ignore_index=True)
    n = pru.UserID.nunique()
    tabla, aciertos = [], {}
    for nombre, g in pu.groupby("estrategia", sort=False):
        por_u = g.assign(l3=g.rango_lugar < 3, l1=g.rango_lugar < 1, h10=g.rango < 10).groupby("UserID")[["l3", "l1", "h10"]].mean()
        aciertos[nombre] = por_u.l3
        tabla.append({"estrategia": nombre, "Lugar@3 (%)": round(100 * por_u.l3.mean(), 1),
                      "Lugar@1 (%)": round(100 * por_u.l1.mean(), 1), "HitRate@10 (%)": round(100 * por_u.h10.mean(), 2),
                      **prueba_lugar(por_u.l3.to_numpy(), por_u.l1.to_numpy())})
    t = pd.DataFrame(tabla)
    rng = np.random.default_rng(0)
    dif = (aciertos["Red de dos torres"] - aciertos["Popularidad"]).to_numpy()
    p_pareada = float(np.mean(np.abs((rng.choice([-1, 1], size=(10_000, n)) * dif).mean(1)) >= abs(dif.mean())))
    return t, {"usuarios simulados como nuevos": n, "red − popularidad en Lugar@3 (puntos)": round(100 * dif.mean(), 1),
               "p-valor pareado red vs popularidad": round(p_pareada, 3)}


# ----------------------------------------------------------------- P2
def orden_lugares(puntajes, aprendidos, rng):
    p = enmascarar(puntajes, aprendidos)
    por_lugar = np.stack([p[:, LUGAR_ID == l].max(1) for l in range(5)], axis=1)
    return np.lexsort((rng.random(por_lugar.shape), -por_lugar), axis=1)  # lugares del mejor al peor


def orden_ids(puntajes, aprendidos, rng, k=10):
    p = enmascarar(puntajes, aprendidos)
    return np.lexsort((rng.random(p.shape), -p), axis=1)[:, :k]


def p2_concentracion():
    aprende, pru = cargar_escenario("final")
    usuarios = pru.UserID.to_numpy()
    items_de = aprende.groupby("UserID").DestinationID.apply(list).to_dict()
    aprendidos = [items_de.get(u, []) for u in usuarios]
    primero, top3, ids10 = {}, {}, {}
    for s in range(N_SEMILLAS):
        modelos = {"Azar": Azar(semilla=1000 + s), "Popularidad": Popularidad(),
                   "Factorización (BPR)": FactorizacionBPR(d=FM_CFG["d"], lam=FM_CFG["lambda"], epocas=FM_CFG["epocas"], semilla=s),
                   "Red de dos torres": red(s)}
        for nombre, m in modelos.items():
            P = m.ajustar(aprende).puntuar(usuarios)
            rng = np.random.default_rng(s)
            o = orden_lugares(P, aprendidos, rng)
            primero.setdefault(nombre, []).append(np.bincount(o[:, 0], minlength=5) / len(usuarios))
            top3.setdefault(nombre, []).append(np.bincount(o[:, :3].ravel(), minlength=5) / len(usuarios))
            ids10.setdefault(nombre, []).append(np.bincount(orden_ids(P, aprendidos, rng).ravel(), minlength=N_DESTINOS))
    filas = []
    for nombre in primero:
        p1 = 100 * np.mean(primero[nombre], axis=0)
        p3 = 100 * np.mean(top3[nombre], axis=0)
        for l in range(5):
            filas.append({"modelo": nombre, "lugar": NOMBRES[l], "% de clientes que lo reciben 1.º": round(p1[l], 1),
                          "% que lo reciben entre sus 3": round(p3[l], 1),
                          "factor de capacidad (1.º / 20 %)": round(p1[l] / 20, 2)})
    por_lugar = pd.DataFrame(filas)
    # Concentración por ID, con los destinos ordenados por popularidad (interacciones en lo aprendido)
    conteo = aprende.DestinationID.value_counts().reindex(range(1, N_DESTINOS + 1), fill_value=0).to_numpy()
    orden_pop = np.argsort(-conteo, kind="stable")
    curvas, resumen = {}, []
    for nombre, lista in ids10.items():
        rec = np.mean(lista, axis=0)
        curvas[nombre] = np.insert(np.cumsum(rec[orden_pop]) / rec.sum(), 0, 0)
        x = np.sort(rec)
        n = len(x)
        gini = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * x) / (n * x.sum()))
        resumen.append({"modelo": nombre, "Gini de las recomendaciones top-10 por ID": round(gini, 3),
                        "% de las recomendaciones que va al 10 % de destinos más populares": round(100 * curvas[nombre][100], 1),
                        "cobertura@10 (%)": round(100 * np.mean(rec > 0), 1),
                        "máximo factor de capacidad por lugar": por_lugar[por_lugar.modelo == nombre]["factor de capacidad (1.º / 20 %)"].max()})
    demanda = np.insert(np.cumsum(conteo[orden_pop]) / conteo.sum(), 0, 0)
    return por_lugar, pd.DataFrame(resumen), curvas, demanda


def figura_p2(curvas, demanda, por_lugar):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    x = np.linspace(0, 100, N_DESTINOS + 1)
    ax1.plot(x, 100 * demanda, color=GRIS_REF, ls=":", lw=1.4, label="Demanda real (interacciones)")
    for nombre, c in curvas.items():
        ax1.plot(x, 100 * c, color=COLORES[nombre], lw=2, label=nombre)
    ax1.set_xlim(0, 50)
    ax1.set_xlabel("% de destinos, de más a menos popular")
    ax1.set_ylabel("% acumulado de las recomendaciones top-10")
    ax1.set_title("Concentración según la popularidad (por ID)", loc="left")
    ax1.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(0.04, 0.93))
    orden = list(COLORES)
    maxf = por_lugar.groupby("modelo")["factor de capacidad (1.º / 20 %)"].max().reindex(orden)
    y = np.arange(len(orden))[::-1]
    ax2.barh(y, maxf.values, color=[COLORES[m] for m in orden], height=0.6)
    for yi, (m, v) in zip(y, maxf.items()):
        l = por_lugar[(por_lugar.modelo == m)].sort_values("factor de capacidad (1.º / 20 %)").iloc[-1].lugar
        ax2.text(v + 0.08, yi, f"×{v:.1f}  ({l})", va="center", fontsize=9)
    ax2.axvline(1, color=GRIS_REF, ls=":", lw=1.2)
    ax2.text(1.06, y.min() - 0.55, "×1 = demanda actual", fontsize=8.5, color=GRIS_REF, va="center")
    ax2.set_yticks(y, ["Azar", "Popularidad", "Factorización", "Red"])
    ax2.set_xlim(0, maxf.max() * 1.6)
    ax2.set_xlabel("Capacidad necesaria (× la actual)")
    ax2.set_title("Lugar que más clientes recibiría primero", loc="left")
    ax2.set_ylim(y.min() - 0.9, y.max() + 0.5)
    ax2.grid(axis="y", visible=False)
    fig.suptitle("Pregunta 2: la popularidad y la red concentran a los clientes; el azar no",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ruta = FIGURAS / "negocio_concentracion.png"
    fig.savefig(ruta, dpi=200)
    return ruta


# ----------------------------------------------------------------- P3
def en_temporada(rango, mes=MES):
    a, b = (MESES[x] for x in rango.split("-"))
    return a <= mes <= b if a <= b else (mes >= a or mes <= b)


def p3_campana():
    aprende, pru = cargar_escenario("final")
    destinos = pd.read_csv(PROCESADOS / "destinos.csv").drop_duplicates("Name").set_index("Name")
    red_g, _ = cargar_red()  # el modelo guardado (el de la app)
    todos = USUARIOS.UserID.to_numpy()
    primero = orden_lugares(red_g.puntuar(todos), [[] for _ in todos], np.random.default_rng(0))[:, 0]
    lugar_de = dict(zip(todos, primero))
    x = aprende.assign(lugar=LUGAR_ID[aprende.DestinationID.to_numpy() - 1])
    y = pru.assign(lugar=LUGAR_ID[pru.DestinationID.to_numpy() - 1])
    filas = []
    for l, nombre in enumerate(NOMBRES):
        temporada = destinos.loc[nombre, "BestTimeToVisit"]
        grupo = [u for u in todos if lugar_de[u] == l]
        n = len(grupo)
        base = float((x.lugar == l).mean())                       # tasa en todos los clientes
        if n == 0:
            filas.append({"lugar": nombre, "temporada": temporada, "¿en temporada en noviembre?": "sí" if en_temporada(temporada) else "no",
                          "clientes del grupo (la red se lo recomienda 1.º)": 0})
            continue
        tasa = float((x[x.UserID.isin(grupo)].lugar == l).mean())  # tasa del grupo, en entrenamiento
        yy = y[y.UserID.isin(grupo)]
        k = int((yy.lugar == l).sum())
        ic = stats.binomtest(k, len(yy), 0.5).proportion_ci(method="wilson")
        perfiles = USUARIOS[USUARIOS.UserID.isin(grupo)].Preferences.value_counts().to_dict()
        filas.append({
            "lugar": nombre, "temporada": temporada, "¿en temporada en noviembre?": "sí" if en_temporada(temporada) else "no",
            "clientes del grupo (la red se lo recomienda 1.º)": n,
            "perfiles del grupo": "; ".join(f"{p} ({c})" for p, c in perfiles.items()),
            "tasa en entrenamiento (%)": round(100 * tasa, 1), "tasa en todos los clientes (%)": round(100 * base, 1),
            "conversiones esperadas en el grupo": round(n * tasa, 1),
            "conversiones si se mostrara a un grupo al azar del mismo tamaño": round(n * base, 1),
            "comprobación en prueba: aciertos / usuarios": f"{k} / {len(yy)}",
            "comprobación en prueba: tasa (%) e IC 95 %": f"{100 * k / len(yy):.1f} ({100 * ic.low:.1f}–{100 * ic.high:.1f})",
        })
    return pd.DataFrame(filas)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    t1, c1 = p1_usuarios_nuevos()
    por_lugar, r2, curvas, demanda = p2_concentracion()
    ruta = figura_p2(curvas, demanda, por_lugar)
    t3 = p3_campana()
    for nombre, t in (("p1_usuarios_nuevos", t1), ("p2_por_lugar", por_lugar), ("p2_resumen", r2), ("p3_campana", t3)):
        t.to_csv(SALIDA / f"{nombre}.csv", index=False)
    texto = ["# Preguntas de negocio\n", "_Generado por `src/preguntas_negocio.py`. No editar a mano._\n",
             "## P1. Usuarios nuevos (606 usuarios de prueba simulados como nuevos, 5 pliegues, 30 semillas)\n",
             _a_markdown(t1) + "\n", _a_markdown(pd.DataFrame([c1])) + "\n",
             "## P2. Concentración (modelos finales, 606 usuarios, 30 semillas)\n", _a_markdown(r2) + "\n", _a_markdown(por_lugar) + "\n",
             f"Figura: `{ruta.relative_to(MODULO)}`\n",
             "## P3. Campaña de un solo destino en noviembre (red guardada; 999 clientes)\n", _a_markdown(t3) + "\n"]
    (SALIDA / "RESUMEN.md").write_text("\n".join(texto), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(t1.to_string(index=False)); print(c1, "\n")
    print(r2.to_string(index=False)); print(por_lugar.to_string(index=False), "\n")
    print(t3.T.to_string())


if __name__ == "__main__":
    main()
