"""Tarea 6 (parte 2): usuarios de ejemplo y proyección de los embeddings de destinos.

1. Usuarios de ejemplo: 5 usuarios de prueba con perfiles distintos, elegidos con reglas fijas
   (el de menor UserID que cumple cada una):
     - uno de cada perfil de preferencias, sin niños;
     - uno que viaja con niños;
     - el que más interacciones tiene para aprender.
   Para cada uno: su perfil, su historial (lo aprendido), el lugar escondido y los 3 lugares que
   le recomienda la red guardada (models/dos_torres.pt), marcando los ya visitados («¿vuelves?»).
   Se usa el modelo GUARDADO, cargado igual que lo hará la app.

2. Embeddings de destinos:
     - Red (d = 2): el vector de cada destino ya está en 2D y se grafica directamente, junto con
       el vector de cada uno de los 3 perfiles de preferencias (el puntaje es su producto punto).
     - Factorización (d = 8): proyección t-SNE a 2D de los vectores de los 1000 destinos.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/ejemplos_embeddings.py

Salidas: results/tarea6/ejemplos.csv, results/tarea6/RESUMEN_ejemplos.md, figures/tarea6_embeddings.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE

from descriptivo import GRIS_REF, TEXTO_2  # noqa: F401  (estilo común)
from dos_torres import TIPOS, DosTorres, variables_usuario
from evaluacion import PROCESADOS, cargar_escenario, enmascarar, lugar_de_cada_id
from inventario import _a_markdown

MODULO = Path(__file__).resolve().parents[1]
SALIDA = MODULO / "results" / "tarea6"
FIGURAS = MODULO / "figures"
# 5 categorías: colores en orden fijo + marcadores distintos + etiquetas directas (no solo color).
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARCAS = ["o", "s", "^", "D", "v"]


def cargar_red():
    meta = json.loads((MODULO / "models" / "dos_torres.json").read_text(encoding="utf-8"))
    red = DosTorres(d=meta["d"], perdida=meta["perdida"], negativos=meta["negativos"],
                    usuario=meta["usuario"], oculta=meta["oculta"])
    red.red = __import__("dos_torres").RedDosTorres(red.X_usuario.shape[1], meta["d"], meta["oculta"])
    red.red.load_state_dict(torch.load(MODULO / "models" / "dos_torres.pt"))
    red.red.eval()
    return red, meta


def elegir_ejemplos(usuarios, pru, n_aprendidas):
    candidatos = usuarios[usuarios.UserID.isin(pru.UserID)].assign(n=lambda d: d.UserID.map(n_aprendidas)).sort_values("UserID")
    reglas = [(f"perfil «{p}», sin niños", (candidatos.Preferences == p) & (candidatos.NumberOfChildren == 0))
              for p in sorted(candidatos.Preferences.unique())]
    reglas.append(("viaja con niños", candidatos.NumberOfChildren > 0))
    elegidos = []
    for motivo, filtro in reglas:
        u = candidatos[filtro & ~candidatos.UserID.isin([e[0] for e in elegidos])].UserID.iloc[0]
        elegidos.append((int(u), motivo))
    maximo = candidatos[~candidatos.UserID.isin([e[0] for e in elegidos])].sort_values(["n", "UserID"], ascending=[False, True]).iloc[0]
    elegidos.append((int(maximo.UserID), f"el que más interacciones tiene para aprender ({int(maximo.n)})"))
    return elegidos


def ejemplos(red):
    usuarios = pd.read_csv(PROCESADOS / "usuarios.csv")
    aprende, pru = cargar_escenario("final")
    lugar_id, nombres = lugar_de_cada_id()
    nombre_id = pd.read_csv(PROCESADOS / "destinos.csv").set_index("DestinationID").Name
    n_aprendidas = aprende.groupby("UserID").size()
    filas = []
    for u, motivo in elegir_ejemplos(usuarios, pru, n_aprendidas):
        perfil = usuarios.set_index("UserID").loc[u]
        hist = aprende[aprende.UserID == u]
        visitados = set(hist.DestinationID.map(nombre_id))
        p = enmascarar(red.puntuar([u]), [hist.DestinationID.tolist()])[0]
        por_lugar = {nombres[l]: p[lugar_id == l].max() for l in range(5)}
        top3 = sorted(por_lugar, key=por_lugar.get, reverse=True)[:3]
        escondido = nombre_id[int(pru[pru.UserID == u].DestinationID.iloc[0])]
        filas.append({
            "UserID": u, "por qué se eligió": motivo,
            "perfil": f"{perfil.Preferences}; {perfil.Gender}; {perfil.NumberOfAdults} adulto(s), {perfil.NumberOfChildren} niño(s)",
            "historial (lo aprendido)": "; ".join(f"{nombre_id[i]} ({f})" for i, f in zip(hist.DestinationID, hist.fuente)),
            "recomendación de la red (3 lugares)": ", ".join(f"{l}{' (¿vuelves?)' if l in visitados else ''}" for l in top3),
            "lugar escondido": escondido,
            "¿acierta? (Lugar@3)": "sí" if escondido in top3 else "no",
        })
    return pd.DataFrame(filas)


def figura(red, meta):
    lugar_id, nombres = lugar_de_cada_id()
    with torch.no_grad():
        V = red.red.vec_destino(red.lugar, red.pop).numpy()
        perfiles = pd.DataFrame([{f"pref_{t}": int(t in combo) for t in TIPOS}
                                 for combo in (("Beach", "Historical"), ("City", "Historical"), ("Nature", "Adventure"))])
        U = red.red.vec_usuario(torch.as_tensor(variables_usuario(perfiles, meta["usuario"]))).numpy()
    Q = np.load(MODULO / "models" / "factorizacion_bpr.npz")["Q"]
    Z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(Q)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    for l, nombre in enumerate(nombres):
        m = lugar_id == l
        for ax, X in ((ax1, V), (ax2, Z)):
            ax.scatter(X[m, 0], X[m, 1], s=10, color=COLORES[l], marker=MARCAS[l], alpha=0.75, lw=0, label=nombre)
        extremo = V[m][np.argmax(np.linalg.norm(V[m], axis=1))]  # punta de la curva del lugar
        ax1.annotate(nombre, extremo, xytext=(4, -10), textcoords="offset points", fontsize=8.5, color=TEXTO_2)
    escala = np.abs(V).max() / np.abs(U).max() * 0.9
    for (x, y), etiqueta in zip(U * escala, ("Beaches + Historical", "City + Historical", "Nature + Adventure")):
        ax1.annotate("", xy=(x, y), xytext=(0, 0), arrowprops={"arrowstyle": "->", "color": "#0b0b0b", "lw": 1.2})
        ax1.text(x, y, " " + etiqueta, fontsize=8, ha="left", va="center")
    ax1.axhline(0, color=GRIS_REF, lw=0.5)
    ax1.axvline(0, color=GRIS_REF, lw=0.5)
    ax1.set_title(f"Red de dos torres (d = {meta['d']}, sin proyectar)", loc="left")
    ax1.set_xlabel("dimensión 1")
    ax1.set_ylabel("dimensión 2")
    ax2.set_title("Factorización (d = 8, proyección t-SNE)", loc="left")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    ax2.legend(fontsize=8, loc="lower left", markerscale=1.6, ncols=1)
    fig.suptitle("Embeddings de destinos: la red separa por lugar (por diseño); la factorización, no",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.01, "Izquierda: las flechas son los vectores de los 3 perfiles de preferencias (escalados); el "
             "puntaje de un destino es el producto punto con la flecha.", fontsize=8.5, color=GRIS_REF)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    ruta = FIGURAS / "tarea6_embeddings.png"
    fig.savefig(ruta, dpi=200)
    return ruta, V, U


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    red, meta = cargar_red()
    t = ejemplos(red)
    t.to_csv(SALIDA / "ejemplos.csv", index=False)
    ruta, V, U = figura(red, meta)
    lugar_id, nombres = lugar_de_cada_id()
    puntaje = pd.DataFrame(U @ np.stack([V[lugar_id == l].mean(0) for l in range(5)]).T,
                           index=["Beaches + Historical", "City + Historical", "Nature + Adventure"], columns=nombres).round(3)
    (SALIDA / "RESUMEN_ejemplos.md").write_text("\n".join([
        "# Tarea 6 — usuarios de ejemplo y embeddings\n", "_Generado por `src/ejemplos_embeddings.py`. No editar a mano._\n",
        "## 1. Usuarios de ejemplo (red guardada, semilla 0)\n", _a_markdown(t) + "\n",
        "## 2. Puntaje medio de cada lugar para cada perfil (producto punto u · v)\n", _a_markdown(puntaje.reset_index()) + "\n",
        f"Figura: `{ruta.relative_to(MODULO)}`\n"]), encoding="utf-8")
    pd.set_option("display.width", 250, "display.max_colwidth", 80)
    print(t.to_string(index=False))
    print("\nPuntaje medio por lugar y perfil:\n", puntaje.to_string())


if __name__ == "__main__":
    main()
