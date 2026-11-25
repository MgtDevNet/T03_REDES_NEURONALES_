"""Funciones de apoyo para el notebook `notebooks/experimentos_modulo_3.ipynb`.

El notebook define los hiperparámetros; aquí está la lógica para entrenar, evaluar y comparar los
cuatro modelos con el MISMO evaluador y la MISMA partición que el informe.

Modelos disponibles (clave → clase):
    "Azar"            sin hiperparámetros
    "Popularidad"     sin hiperparámetros
    "Factorización"   d, lam, epocas, lr, lote
    "Red"             d, perdida ("bce"/"bpr"), negativos, usuario ("preferencias"/"completo"),
                      epocas, lr, decaimiento, oculta, lote
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

from dos_torres import DosTorres
from evaluacion import KS, N_DESTINOS, cargar_escenario, enmascarar, evaluar, lugar_de_cada_id
from factorizacion import FactorizacionBPR
from referencias import Azar, Popularidad

torch.set_num_threads(1)
# Estilo común de las figuras del proyecto (el mismo de descriptivo.py). No se importa
# descriptivo.py porque fija el backend «Agg», que impediría ver los gráficos en el notebook.
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#e4e3df", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
    "text.color": "#0b0b0b", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
CLASES = {"Azar": Azar, "Popularidad": Popularidad, "Factorización": FactorizacionBPR, "Red": DosTorres}


def crear(tipo, semilla, **hp):
    """Crea un modelo nuevo del tipo indicado con sus hiperparámetros."""
    if tipo == "Azar":
        return Azar(semilla=1000 + semilla)
    if tipo == "Popularidad":
        return Popularidad()
    return CLASES[tipo](semilla=semilla, **hp)


def entrenar_y_evaluar(modelos, escenario="ajuste", semillas=3):
    """Entrena y evalúa cada modelo con varias semillas.

    modelos: {"nombre que quieras": {"tipo": "Red", "hp": {...}}, ...}
    escenario: "ajuste" (aprende con entrenamiento, mide en validación: PARA EXPERIMENTAR) o
               "final" (aprende con entrenamiento + validación, mide en prueba: solo al final).
    Devuelve (por_semilla, por_usuario): métricas por modelo × semilla × K, y la posición del
    destino escondido de cada usuario (para las pruebas pareadas).
    """
    aprende, evalua = cargar_escenario(escenario)
    filas, usuarios = [], []
    for nombre, cfg in modelos.items():
        for s in range(semillas):
            t0 = time.perf_counter()
            m = crear(cfg["tipo"], s, **cfg.get("hp", {})).ajustar(aprende)
            seg = time.perf_counter() - t0
            res, pu = evaluar(m.puntuar, aprende, evalua, semilla=s)
            params = m.parametros() if hasattr(m, "parametros") else 0
            filas.append(res.assign(modelo=nombre, semilla=s, segundos=seg, parametros=params))
            usuarios.append(pu.assign(modelo=nombre, semilla=s))
        print(f"✓ {nombre}: {semillas} semilla(s)")
    return pd.concat(filas, ignore_index=True), pd.concat(usuarios, ignore_index=True)


def tabla_comparativa(por_semilla, por_usuario, k=10):
    """Una fila por modelo: medias (± desviación entre semillas) y pruebas contra el azar.

    p-valores: binomial unilateral (H0: acierta como el azar; H1: acierta más), sin corregir.
    """
    escenario_n = por_usuario.UserID.nunique()
    d = por_semilla[por_semilla.K == k]
    filas = []
    for nombre, g in d.groupby("modelo", sort=False):
        pu = por_usuario[por_usuario.modelo == nombre]
        acierto_l3 = (pu.rango_lugar < 3).groupby(pu.UserID).mean().sum()
        acierto_id = (pu.rango < k).groupby(pu.UserID).mean().sum()
        n_u = pu.groupby("UserID").n_aprendidas.first()
        p0 = float(np.mean(k / (N_DESTINOS - n_u)))
        filas.append({
            "modelo": nombre,
            f"HitRate@{k} (%)": f"{100 * g.HitRate.mean():.2f} ± {100 * g.HitRate.std(ddof=1):.2f}" if len(g) > 1 else f"{100 * g.HitRate.mean():.2f}",
            f"NDCG@{k} (%)": round(100 * g.NDCG.mean(), 2),
            f"Precision@{k} (%)": round(100 * g.Precision.mean(), 3),
            f"Recall@{k} (%)": round(100 * g.Recall.mean(), 2),
            f"Cobertura@{k} (%)": round(100 * g.Cobertura.mean(), 1),
            "Lugar@3 (%)": round(100 * g["HitRate_lugar@3"].mean(), 1),
            "Lugar@1 (%)": round(100 * g["HitRate_lugar@1"].mean(), 1),
            "p-valor Lugar@3 vs azar": round(stats.binomtest(int(round(acierto_l3)), escenario_n, 0.6, alternative="greater").pvalue, 3),
            f"p-valor HitRate@{k} vs azar": round(stats.binomtest(int(round(acierto_id)), escenario_n, p0, alternative="greater").pvalue, 3),
            "parámetros": int(g.parametros.iloc[0]),
            "segundos por entrenamiento": round(g.segundos.mean(), 2),
        })
    return pd.DataFrame(filas)


def comparar_pareado(por_usuario, modelo_a, modelo_b, metrica="Lugar@3", n_perm=10_000):
    """¿A acierta más que B en los MISMOS usuarios? Prueba de permutación de signos (bilateral)."""
    col, corte = ("rango_lugar", 3) if metrica == "Lugar@3" else ("rango_lugar", 1) if metrica == "Lugar@1" else ("rango", 10)
    a = (por_usuario[por_usuario.modelo == modelo_a][col] < corte).groupby(por_usuario.UserID).mean()
    b = (por_usuario[por_usuario.modelo == modelo_b][col] < corte).groupby(por_usuario.UserID).mean()
    dif = (a - b.reindex(a.index)).to_numpy()
    nulas = (np.random.default_rng(0).choice([-1, 1], size=(n_perm, len(dif))) * dif).mean(1)
    p = float(np.mean(np.abs(nulas) >= abs(dif.mean())))
    return {"comparación": f"{modelo_a} − {modelo_b}", "métrica": metrica,
            "diferencia (puntos)": round(100 * dif.mean(), 2), "p-valor (pareada)": round(p, 3)}


def grafico_metricas_k(por_semilla):
    """Precision, Recall, NDCG por ID en función de K y las dos métricas por lugar, por modelo."""
    medias = por_semilla.groupby(["modelo", "K"], sort=False)[["Precision", "Recall", "NDCG"]].mean().reset_index()
    lugar = por_semilla.groupby("modelo", sort=False)[["HitRate_lugar@3", "HitRate_lugar@1"]].mean()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    x = np.arange(len(KS))
    for i, met in enumerate(("Precision", "Recall", "NDCG")):
        for j, (nombre, g) in enumerate(medias.groupby("modelo", sort=False)):
            axes[i].plot(x, 100 * g[met], marker="o", ms=4, lw=1.8, color=COLORES[j % 8], label=nombre)
        axes[i].set_xticks(x, [str(k) for k in KS])
        axes[i].set_title(f"{met}@K (%)", loc="left")
        axes[i].set_xlabel("K")
    axes[1].plot(x, 100 * np.array(KS) / 1000, ls=":", color="#52514e", label="azar teórico")
    axes[0].legend(fontsize=8)
    ancho = 0.38
    xi = np.arange(len(lugar))
    axes[3].bar(xi - ancho / 2, 100 * lugar["HitRate_lugar@3"], ancho, color="#2a78d6", label="Lugar@3")
    axes[3].bar(xi + ancho / 2, 100 * lugar["HitRate_lugar@1"], ancho, color="#eb6834", label="Lugar@1")
    axes[3].axhline(60, ls=":", color="#52514e")
    axes[3].axhline(20, ls=":", color="#52514e")
    axes[3].set_xticks(xi, lugar.index, rotation=20, ha="right", fontsize=8)
    axes[3].set_title("Por lugar (%) · azar 60 y 20", loc="left")
    axes[3].set_ylim(0, 85)
    axes[3].legend(fontsize=8, loc="upper right", ncols=2)
    fig.tight_layout()
    return fig


def barrido(tipo, hp_base, parametro, valores, escenario="ajuste", semillas=3, metrica="HitRate_lugar@3"):
    """Cambia UN hiperparámetro y mide cómo cambia la métrica (media ± desviación entre semillas)."""
    modelos = {f"{parametro}={v}": {"tipo": tipo, "hp": {**hp_base, parametro: v}} for v in valores}
    ps, _ = entrenar_y_evaluar(modelos, escenario, semillas)
    t = ps[ps.K == 10].groupby("modelo", sort=False)[metrica].agg(["mean", "std"]).reset_index()
    t.insert(1, parametro, valores)
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.errorbar([str(v) for v in valores], 100 * t["mean"], yerr=100 * t["std"].fillna(0), marker="o", capsize=4, color="#2a78d6")
    if metrica == "HitRate_lugar@3":
        ax.axhline(60, ls=":", color="#52514e", label="azar")
    elif metrica == "HitRate_lugar@1":
        ax.axhline(20, ls=":", color="#52514e", label="azar")
    ax.set_xlabel(parametro)
    ax.set_ylabel(f"{metrica} (%)")
    ax.set_title(f"{tipo}: efecto de «{parametro}» ({escenario})", loc="left")
    fig.tight_layout()
    return t.assign(mean=lambda x: (100 * x["mean"]).round(2), std=lambda x: (100 * x["std"]).round(2)).rename(
        columns={"mean": f"{metrica} media (%)", "std": "desviación entre semillas"}), fig


def curva_aprendizaje(tipo, hp, semilla=0, metrica_val=None):
    """Entrena con entrenamiento y mide en validación en cada época: pérdida y métrica."""
    aprende, val = cargar_escenario("ajuste")
    m = crear(tipo, semilla, **hp)
    if tipo == "Red":
        m.ajustar(aprende, validacion=val, paciencia=10 ** 6)
        col = "HitRate_lugar@3_val"
    else:
        m.ajustar(aprende, validacion=val, paciencia=10 ** 6)
        col = "NDCG@10_val"
    h = pd.DataFrame(m.historia)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.4))  # dos medidas distintas: dos paneles
    ax1.plot(h.epoca, h.perdida, color="#2a78d6")
    ax1.set_xlabel("época")
    ax1.set_title("Pérdida en entrenamiento", loc="left")
    ax2.plot(h.epoca, 100 * h[col], color="#eb6834")
    ax2.axvline(m.mejor_epoca, ls=":", color="#52514e")
    ax2.set_xlabel("época")
    ax2.set_title(f"{col} (%) · mejor época {m.mejor_epoca}", loc="left")
    fig.suptitle(f"{tipo}: curva de aprendizaje", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return h, fig


def recomendaciones_de(user_id, modelos, escenario="final"):
    """Los 3 lugares que cada modelo le recomendaría a un usuario (entrenando una vez, semilla 0)."""
    aprende, evalua = cargar_escenario(escenario)
    lugar_id, nombres = lugar_de_cada_id()
    vistos = aprende[aprende.UserID == user_id].DestinationID.tolist()
    escondido = evalua[evalua.UserID == user_id].DestinationID
    filas = []
    for nombre, cfg in modelos.items():
        m = crear(cfg["tipo"], 0, **cfg.get("hp", {})).ajustar(aprende)
        p = enmascarar(m.puntuar([user_id]), [vistos])[0]
        por_lugar = {nombres[l]: p[lugar_id == l].max() for l in range(5)}
        filas.append({"modelo": nombre, "recomienda (3 lugares)": ", ".join(sorted(por_lugar, key=por_lugar.get, reverse=True)[:3])})
    t = pd.DataFrame(filas)
    info = {"historial (lo aprendido)": ", ".join(nombres[lugar_id[i - 1]] for i in vistos) or "—",
            "lugar escondido": nombres[lugar_id[int(escondido.iloc[0]) - 1]] if len(escondido) else "— (no se evalúa)"}
    return t, info
