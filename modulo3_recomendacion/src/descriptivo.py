"""Análisis descriptivo del módulo 3 (tarea 2 del enunciado).

El enunciado pide: distribución de interacciones por usuario y por destino,
concentración en los destinos populares, proporción de usuarios con pocas
interacciones e hipótesis sobre qué explica las preferencias.

Idea central: cada patrón se compara con lo que produciría el PURO AZAR. Si los
datos se parecen al azar, el patrón no dice nada sobre preferencias reales.

Usa TODAS las interacciones (unión de reseñas + historial, 1998), porque describe
el dataset; la partición no interviene aquí.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/descriptivo.py

Salidas:
    modulo3_recomendacion/figures/descriptivo_*.png
    modulo3_recomendacion/results/descriptivo/tabla_*.csv y RESUMEN.md
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from inventario import PREFERENCIA_A_TIPO, _a_markdown, cargar, interacciones_unidas

MODULO = Path(__file__).resolve().parents[1]
FIGURAS = MODULO / "figures"
SALIDA = MODULO / "results" / "descriptivo"
SEMILLA = 0
N_SIMULACIONES = 2000

# Estilo (guía de visualización): un color para lo observado, gris para la
# referencia «azar», superficie clara, rejilla tenue, marcas delgadas.
AZUL = "#2a78d6"
GRIS_REF = "#52514e"
TEXTO = "#0b0b0b"
TEXTO_2 = "#52514e"
SUPERFICIE = "#fcfcfb"
REJILLA = "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO_2, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
    "text.color": TEXTO, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def gini(conteos):
    """Coeficiente de Gini de un vector de conteos (0 = reparto parejo, 1 = todo en uno)."""
    x = np.sort(np.asarray(conteos, dtype=float))
    n = len(x)
    k = np.arange(1, n + 1)
    return float(np.sum((2 * k - n - 1) * x) / (n * x.sum()))


def participacion_top(conteos, fraccion):
    """Qué % de las interacciones se lleva el `fraccion` superior de los ítems."""
    x = np.sort(np.asarray(conteos))[::-1]
    k = max(1, int(round(fraccion * len(x))))
    return float(100 * x[:k].sum() / x.sum())


def poisson_esperado(n_entidades, media, maximo):
    """Conteos esperados de entidades con 0, 1, ..., maximo-1 y ≥ maximo interacciones."""
    p = stats.poisson.pmf(np.arange(maximo), media)
    p = np.append(p, 1 - p.sum())
    return n_entidades * p


def bonferroni(p, m):
    return min(1.0, p * m)


# --------------------------------------------------------------------------
# Secciones
# --------------------------------------------------------------------------
def s1_distribucion(inter, n_usuarios, n_destinos):
    """Interacciones por usuario y por destino frente a una Poisson (azar)."""
    filas, pruebas, datos_fig = [], [], {}
    maximo = 7  # se agrupa «7 o más»
    for nombre, col, total in [("usuario", "UserID", n_usuarios), ("destino", "DestinationID", n_destinos)]:
        por = inter.groupby(col).size()
        todos = np.zeros(total, dtype=int)
        todos[: len(por)] = por.values  # incluye entidades con 0 interacciones
        obs = np.array([np.sum(todos == k) for k in range(maximo)] + [np.sum(todos >= maximo)])
        media = len(inter) / total
        esp = poisson_esperado(total, media, maximo)
        # Bondad de ajuste: se agrupan las colas con esperado < 5 (regla habitual).
        obs_g, esp_g = list(obs[:6]), list(esp[:6])
        obs_g.append(obs[6:].sum()); esp_g.append(esp[6:].sum())
        chi = stats.chisquare(obs_g, esp_g, ddof=1)  # ddof=1: la media se estimó de los datos
        pruebas.append((nombre, media, chi.statistic, chi.pvalue))
        for k in range(maximo + 1):
            filas.append({"por": nombre, "n_interacciones": f"{k}" if k < maximo else f"≥{maximo}",
                          "observado": int(obs[k]), "esperado_azar_poisson": round(float(esp[k]), 1)})
        datos_fig[nombre] = (obs, esp, media)
    explicacion = (
        "Cuántos usuarios y cuántos destinos tienen 0, 1, 2, … interacciones, comparado con lo que "
        "se esperaría si cada interacción se asignara **al azar** (distribución de Poisson con la misma "
        "media). En datos reales la cola es mucho más larga que la de Poisson: unos pocos usuarios muy "
        "activos y unos pocos destinos muy populares concentran gran parte de la actividad.\n\n"
        + "\n".join(f"- Por {n}: media = {m:.2f}; bondad de ajuste a Poisson χ² = {c:.2f}, p = {p:.3f} "
                    f"({'no se distingue del azar' if p > 0.05 else 'se aleja del azar'})"
                    for n, m, c, p in pruebas)
    )
    return ("1. Interacciones por usuario y por destino (cola larga)", explicacion, pd.DataFrame(filas)), datos_fig


def figura_cola_larga(datos_fig):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=False)
    for ax, (nombre, titulo) in zip(ejes, [("usuario", "Interacciones por usuario"), ("destino", "Interacciones por destino (ID)")]):
        obs, esp, media = datos_fig[nombre]
        x = np.arange(len(obs))
        etiquetas = [str(k) for k in range(len(obs) - 1)] + [f"≥{len(obs) - 1}"]
        ax.bar(x, obs, width=0.62, color=AZUL, label="Observado", zorder=2)
        ax.plot(x, esp, color=GRIS_REF, linestyle="--", linewidth=2, marker="o", markersize=5,
                label=f"Esperado si fuera al azar (Poisson, media {media:.1f})", zorder=3)
        ax.set_xticks(x, etiquetas)
        ax.set_xlabel("Número de interacciones")
        ax.set_ylabel(f"Número de {nombre}s")
        ax.set_title(titulo, loc="left")
        ax.grid(axis="x", visible=False)
    manejadores, etiquetas = ejes[0].get_legend_handles_labels()
    fig.legend(manejadores, etiquetas, loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=2, fontsize=9)
    fig.suptitle("Interacciones por usuario y por destino: muy parecidas a lo que produciría el azar", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    ruta = FIGURAS / "descriptivo_cola_larga.png"
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    return ruta


def s2_concentracion(inter, destinos):
    """¿Unos pocos destinos concentran las interacciones? Observado frente a azar."""
    rng = np.random.default_rng(SEMILLA)
    n_ids = len(destinos)
    por_id = inter.groupby("DestinationID").size().reindex(destinos.DestinationID, fill_value=0).values
    lugares = inter.merge(destinos[["DestinationID", "Name"]], on="DestinationID")
    por_lugar = lugares.groupby("Name").size()

    # Azar: las mismas 1998 interacciones repartidas uniformemente entre los ítems.
    sim_id = rng.multinomial(len(inter), np.full(n_ids, 1 / n_ids), size=N_SIMULACIONES)
    sim_lugar = rng.multinomial(len(inter), np.full(5, 1 / 5), size=N_SIMULACIONES)

    def fila(nivel, obs, sims):
        g_sim = np.array([gini(s) for s in sims])
        t10 = np.array([participacion_top(s, 0.10) for s in sims]) if len(obs) > 5 else None
        return {
            "nivel": nivel,
            "gini_observado": round(gini(obs), 3),
            "gini_azar_media": round(g_sim.mean(), 3),
            "gini_azar_int95": f"{np.percentile(g_sim, 2.5):.3f}–{np.percentile(g_sim, 97.5):.3f}",
            "p_(azar≥observado)": round(float(np.mean(g_sim >= gini(obs))), 3),
            "top10%_observado_%": round(participacion_top(obs, 0.10), 1) if t10 is not None else "—",
            "top10%_azar_%": round(float(t10.mean()), 1) if t10 is not None else "—",
        }

    tabla = pd.DataFrame([fila("1000 IDs", por_id, sim_id), fila("5 lugares", por_lugar.values, sim_lugar)])
    lugares_txt = ", ".join(f"{k}: {v} ({100 * v / len(inter):.1f} %)" for k, v in por_lugar.sort_values(ascending=False).items())
    explicacion = (
        "El **coeficiente de Gini** resume la concentración (0 = todos los destinos reciben lo mismo; 1 = un "
        "solo destino se lleva todo). Se compara con 2000 simulaciones en las que las mismas 1998 "
        "interacciones se reparten **al azar** entre los destinos. Si el Gini observado cae dentro del "
        "intervalo del azar, la concentración que vemos es la que el azar produce por sí solo.\n\n"
        f"- Interacciones por lugar: {lugares_txt}. Al azar, cada lugar tendría ≈ 20 %."
    )
    # Evidencia de cómo se construye el intervalo del azar (se cita en teoría 22.9 y en el informe).
    g_id = np.array([gini(s) for s in sim_id])
    g_obs = gini(por_id)
    explicacion += (
        "\n\n**Cómo se construye el intervalo del azar (por ID):**\n"
        f"- Simulación 1, interacciones de los IDs 1 a 10: {sim_id[0][:10].tolist()}.\n"
        f"- Gini de las simulaciones 1 a 5: {', '.join(f'{g:.3f}' for g in g_id[:5])}.\n"
        f"- Mínimo y máximo de los {N_SIMULACIONES} Gini: {g_id.min():.3f} y {g_id.max():.3f}.\n"
        f"- Ordenados, posición 50 y 1950 (percentiles 2,5 y 97,5): "
        f"{np.sort(g_id)[49]:.3f} y {np.sort(g_id)[1949]:.3f}.\n"
        f"- Simulaciones con Gini ≥ observado ({g_obs:.3f}): {int((g_id >= g_obs).sum())} de "
        f"{N_SIMULACIONES} → p-valor = {(g_id >= g_obs).mean():.2f}."
    )
    datos_fig = (por_id, sim_id)
    return ("2. Concentración en los destinos populares", explicacion, tabla), datos_fig, por_lugar


def figura_concentracion(datos_fig, por_lugar, total):
    por_id, sim_id = datos_fig

    def lorenz(c):
        x = np.sort(c)
        return np.insert(np.cumsum(x) / x.sum(), 0, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.15, 1]})
    eje_x = np.linspace(0, 1, len(por_id) + 1)
    curvas = np.array([lorenz(s) for s in sim_id[:500]])
    ax1.fill_between(eje_x, np.percentile(curvas, 2.5, axis=0), np.percentile(curvas, 97.5, axis=0),
                     color=GRIS_REF, alpha=0.18, linewidth=0, label="Rango del azar (95 %)")
    ax1.plot([0, 1], [0, 1], color=GRIS_REF, linestyle=":", linewidth=1.2, label="Reparto perfectamente parejo")
    ax1.plot(eje_x, lorenz(por_id), color=AZUL, linewidth=2, label="Observado (1000 IDs)")
    ax1.set_xlabel("Fracción de destinos (de menos a más interacciones)")
    ax1.set_ylabel("Fracción acumulada de interacciones")
    ax1.set_title("Curva de Lorenz por destino (ID)", loc="left")
    ax1.legend(loc="upper left", fontsize=8.5)

    orden = por_lugar.sort_values()
    pct = 100 * orden / total
    ax2.barh(orden.index, pct, height=0.6, color=AZUL, zorder=2)
    ax2.vlines(20, -0.5, len(orden) - 0.5, color=GRIS_REF, linestyle="--", linewidth=2)  # termina bajo el rótulo
    ax2.text(20, len(orden) - 0.45, "20 % = reparto parejo", ha="center", va="bottom", fontsize=8.5, color=TEXTO_2)
    for y, v in enumerate(pct):  # columna de valores alineada, lejos de la línea del 20 %
        ax2.text(27.5, y, f"{v:.1f} %", va="center", ha="right", fontsize=9, color=TEXTO)
    ax2.set_xlim(0, 28)
    ax2.set_ylim(-0.5, len(orden) - 0.1)
    ax2.set_xticks([0, 5, 10, 15, 20])
    ax2.set_xlabel("% de las interacciones")
    ax2.set_title("Interacciones por lugar real", loc="left")
    ax2.grid(axis="y", visible=False)
    fig.suptitle("No hay destinos «estrella»: la concentración es la del azar", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    ruta = FIGURAS / "descriptivo_concentracion.png"
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    return ruta


def s3_pocas_interacciones(inter, n_usuarios):
    por = inter.groupby("UserID").size().reindex(range(1, n_usuarios + 1), fill_value=0)
    filas = []
    for etiqueta, mascara in [("0 (sin actividad)", por == 0), ("exactamente 1", por == 1), ("1 o menos", por <= 1),
                              ("2 o menos", por <= 2), ("3 o más", por >= 3), ("5 o más", por >= 5)]:
        filas.append({"usuarios_con": etiqueta, "usuarios": int(mascara.sum()), "%_del_total": round(100 * mascara.mean(), 1)})
    return (
        "3. Proporción de usuarios con pocas interacciones",
        f"Sobre los {n_usuarios} usuarios del catálogo (unión de reseñas + historial). La mediana es "
        f"{int(por.median())} interacciones por usuario y el máximo es {int(por.max())}.",
        pd.DataFrame(filas),
    )


def s4_hipotesis(inter, usuarios, destinos, historial):
    """Hipótesis sobre qué explica las preferencias, cada una con H0, H1, su prueba y su decisión."""
    m = inter.merge(usuarios, on="UserID").merge(destinos[["DestinationID", "Type", "Popularity", "BestTimeToVisit", "Name"]], on="DestinationID")
    pruebas = []

    def chi2(fila, h_id, txt, h0, h1, nombre, det=""):
        """Chi-cuadrado de independencia variable × tipo de destino; registra estadístico y validez."""
        tabla_c = pd.crosstab(fila, m.Type)
        t = stats.chi2_contingency(tabla_c)
        est = (f"χ² = {t.statistic:.2f}, gl = {t.dof} (tabla {tabla_c.shape[0]}×{tabla_c.shape[1]}; "
               f"mínimo esperado {t.expected_freq.min():.0f} ≥ 5)")
        pruebas.append((h_id, txt, h0, h1, f"chi-cuadrado de independencia {nombre} × tipo", est, t.pvalue, det))

    # H1: las preferencias declaradas explican el tipo de destino.
    prefs = m.Preferences.str.split(", ").apply(lambda xs: {PREFERENCIA_A_TIPO[x] for x in xs})
    coincide = np.mean([ti in p for ti, p in zip(m.Type, prefs)])
    chi2(m.Preferences, "H1", "Las personas visitan los tipos de destino que dicen preferir",
         "el tipo visitado es independiente del perfil de preferencias", "el tipo visitado depende del perfil",
         "perfil", f"coinciden {100 * coincide:.1f} % (azar ≈ 40 %)")

    # H2: quienes viajan con niños prefieren otros tipos (p. ej. playa/naturaleza).
    con = m[m.NumberOfChildren > 0].Type.isin(["Beach", "Nature"]).mean()
    sin = m[m.NumberOfChildren == 0].Type.isin(["Beach", "Nature"]).mean()
    chi2(m.NumberOfChildren > 0, "H2", "Quienes viajan con niños eligen más playa y naturaleza",
         "el tipo visitado es independiente de viajar con niños", "el tipo visitado depende de viajar con niños",
         "con/sin niños", f"playa+naturaleza: con niños {100 * con:.1f} %, sin niños {100 * sin:.1f} %")

    # H3: viajar en pareja (2 adultos) frente a solo cambia el tipo de destino.
    chi2(m.NumberOfAdults, "H3", "Viajar solo o con otro adulto cambia el tipo de destino",
         "el tipo visitado es independiente del número de adultos", "el tipo visitado depende del número de adultos",
         "adultos (1 o 2)")

    # H4: el género cambia el tipo de destino.
    chi2(m.Gender, "H4", "El género cambia el tipo de destino",
         "el tipo visitado es independiente del género", "el tipo visitado depende del género", "género")

    # H5: los destinos con mayor Popularity reciben más interacciones.
    por_id = inter.groupby("DestinationID").size().reindex(destinos.DestinationID, fill_value=0)
    rho, p = stats.spearmanr(destinos.Popularity, por_id.values)
    pruebas.append(("H5", "Los destinos con mayor `Popularity` reciben más interacciones",
                    "ρ = 0 (no hay relación monótona)", "ρ ≠ 0 (bilateral)",
                    "correlación de Spearman (1000 IDs)", f"ρ = {rho:.3f}", p, ""))

    # H6: las visitas ocurren en la mejor temporada del destino (solo historial: tiene fecha).
    meses = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

    def en_temporada(rango, mes):
        a, b = (meses[x] for x in rango.split("-"))
        return a <= mes <= b if a <= b else (mes >= a or mes <= b)

    h = historial.merge(destinos[["DestinationID", "BestTimeToVisit"]], on="DestinationID")
    mes = pd.to_datetime(h.VisitDate).dt.month.to_numpy()
    temporada = h.BestTimeToVisit.to_numpy()
    obs = np.mean([en_temporada(r, mm) for r, mm in zip(temporada, mes)])
    rng = np.random.default_rng(SEMILLA)
    sims = np.array([np.mean([en_temporada(r, mm) for r, mm in zip(rng.permutation(temporada), mes)]) for _ in range(N_SIMULACIONES)])
    p = float(np.mean(sims >= obs))
    pruebas.append(("H6", "Las visitas ocurren en la mejor temporada del destino",
                    f"el % de visitas en temporada es el del azar ({100 * sims.mean():.1f} %)",
                    "el % en temporada es mayor que el del azar (unilateral)",
                    f"permutación, {N_SIMULACIONES} repeticiones (solo historial)",
                    f"en temporada {100 * obs:.1f} %", p, f"en temporada {100 * obs:.1f} % (azar {100 * sims.mean():.1f} %)"))

    m_pruebas = len(pruebas)
    tabla = pd.DataFrame([{
        "hipótesis": h_id, "enunciado": txt, "H0": h0, "H1": h1, "prueba": prueba, "estadístico": est,
        "p_valor": round(p, 3), "p_valor_corregido": round(bonferroni(p, m_pruebas), 3),
        "decisión (α = 0,05)": ("se rechaza H0" if bonferroni(p, m_pruebas) < 0.05
                                else "no se rechaza H0: no hay evidencia suficiente"),
        "detalle": det,
    } for h_id, txt, h0, h1, prueba, est, p, det in pruebas])
    explicacion = (
        "Hipótesis razonables sobre qué haría que un viajero prefiera un destino. Para cada una: hipótesis nula "
        "(H0, «no hay relación»), alternativa (H1), prueba y estadístico. Como se hacen **varias pruebas a la vez**, "
        "aumenta la probabilidad de encontrar un «resultado significativo» por casualidad; por eso se aplica la "
        f"**corrección de Bonferroni**: p-valor corregido = mín(1, {m_pruebas} × p-valor). Se rechaza H0 solo si el "
        f"p-valor corregido es menor que 0,05 (equivale a p-valor < 0,05/{m_pruebas} = {0.05 / m_pruebas:.4f})."
    )
    return "4. Hipótesis sobre qué explica las preferencias", explicacion, tabla


# --------------------------------------------------------------------------
def main():
    FIGURAS.mkdir(parents=True, exist_ok=True)
    SALIDA.mkdir(parents=True, exist_ok=True)
    destinos, usuarios, resenas, historial = cargar()
    inter = interacciones_unidas(resenas, historial)

    sec1, fig1_datos = s1_distribucion(inter, len(usuarios), len(destinos))
    sec2, fig2_datos, por_lugar = s2_concentracion(inter, destinos)
    sec3 = s3_pocas_interacciones(inter, len(usuarios))
    sec4 = s4_hipotesis(inter, usuarios, destinos, historial)
    rutas = [figura_cola_larga(fig1_datos), figura_concentracion(fig2_datos, por_lugar, len(inter))]

    partes = ["# Análisis descriptivo — Módulo 3 (tarea 2)\n",
              "_Generado por `src/descriptivo.py`. No editar a mano: volver a ejecutar el script._\n",
              "Figuras: " + ", ".join(f"`figures/{r.name}`" for r in rutas) + "\n"]
    for i, (titulo, explicacion, tabla) in enumerate([sec1, sec2, sec3, sec4], start=1):
        tabla.to_csv(SALIDA / f"tabla_{i:02d}.csv", index=False)
        partes += [f"## {titulo}\n", explicacion + "\n", _a_markdown(tabla) + "\n",
                   f"_CSV: `results/descriptivo/tabla_{i:02d}.csv`_\n"]
        print(f"\n=== {titulo} ===\n{explicacion}\n{tabla.to_string(index=False)}")
    (SALIDA / "RESUMEN.md").write_text("\n".join(partes), encoding="utf-8")
    print("\nFiguras:", *[r.relative_to(MODULO.parent) for r in rutas], sep="\n  ")


if __name__ == "__main__":
    main()
