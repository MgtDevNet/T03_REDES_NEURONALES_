"""Inventario y análisis descriptivo de los datos del módulo 3.

Reproduce TODA la evidencia que respalda las decisiones de los pasos 1 y 2
(ver DECISIONES.md). Cada sección termina en una tabla que se guarda en
results/inventario/ como CSV, y además se escribe un resumen legible en
results/inventario/RESUMEN.md para citar en el informe.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/inventario.py

Requisitos: los 4 CSV del dataset de Kaggle descomprimidos en
modulo3_recomendacion/data/raw/ (no se versionan; ver README del módulo).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODULO = Path(__file__).resolve().parents[1]
RAW = MODULO / "data" / "raw"
SALIDA = MODULO / "results" / "inventario"

# Nombres de las preferencias del usuario -> tipo de destino equivalente.
# Las preferencias vienen en plural o con otra forma ("Beaches") y los tipos
# en singular ("Beach"), así que hay que emparejarlas explícitamente.
PREFERENCIA_A_TIPO = {
    "Beaches": "Beach",
    "Historical": "Historical",
    "City": "City",
    "Nature": "Nature",
    "Adventure": "Adventure",
}


# --------------------------------------------------------------------------
# Carga
# --------------------------------------------------------------------------
def cargar():
    """Lee los 4 CSV y devuelve (destinos, usuarios, resenas, historial)."""
    destinos = pd.read_csv(RAW / "Expanded_Destinations.csv")
    usuarios = pd.read_csv(RAW / "Final_Updated_Expanded_Users.csv")
    resenas = pd.read_csv(RAW / "Final_Updated_Expanded_Reviews.csv")
    historial = pd.read_csv(RAW / "Final_Updated_Expanded_UserHistory.csv")
    return destinos, usuarios, resenas, historial


def interacciones_unidas(resenas, historial):
    """Une reseñas e historial en una sola tabla de interacciones.

    Columnas: UserID, DestinationID, calificacion, fuente.
    """
    r = resenas[["UserID", "DestinationID", "Rating"]].rename(columns={"Rating": "calificacion"})
    h = historial[["UserID", "DestinationID", "ExperienceRating"]].rename(
        columns={"ExperienceRating": "calificacion"}
    )
    return pd.concat(
        [r.assign(fuente="reseña"), h.assign(fuente="historial")], ignore_index=True
    )


# --------------------------------------------------------------------------
# Secciones del análisis. Cada una devuelve (titulo, explicacion, tabla).
# --------------------------------------------------------------------------
def s01_tamanos_y_calidad(destinos, usuarios, resenas, historial):
    filas = []
    for nombre, df, clave in [
        ("Destinos", destinos, "DestinationID"),
        ("Usuarios", usuarios, "UserID"),
        ("Reseñas", resenas, "ReviewID"),
        ("Historial", historial, "HistoryID"),
    ]:
        fila = {
            "tabla": nombre,
            "filas": len(df),
            "columnas": df.shape[1],
            "celdas_nulas": int(df.isna().sum().sum()),
            "ids_repetidos": int(df[clave].duplicated().sum()),
        }
        if "UserID" in df and nombre in ("Reseñas", "Historial"):
            fila["pares_usuario_destino_repetidos"] = int(
                df.duplicated(["UserID", "DestinationID"]).sum()
            )
            fila["usuarios_inexistentes"] = int((~df.UserID.isin(usuarios.UserID)).sum())
            fila["destinos_inexistentes"] = int(
                (~df.DestinationID.isin(destinos.DestinationID)).sum()
            )
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    return (
        "1. Tamaño y calidad de las tablas",
        "Cuántas filas tiene cada archivo y si hay datos faltantes, IDs repetidos o "
        "interacciones que apunten a usuarios/destinos que no existen.",
        tabla,
    )


def s02_patrones_ciclicos(destinos, usuarios, resenas, historial):
    """¿Alguna columna queda totalmente determinada por (ID-1) mod k?"""

    def ciclo(df, id_col, col, k):
        grupos = df.groupby((df[id_col] - 1) % k)[col].nunique()
        return bool((grupos == 1).all())

    filas = [
        ("Destinos", "Name", destinos.Name.nunique(), "DestinationID", 5, ciclo(destinos, "DestinationID", "Name", 5)),
        ("Destinos", "Type", destinos.Type.nunique(), "DestinationID", 5, ciclo(destinos, "DestinationID", "Type", 5)),
        ("Destinos", "State", destinos.State.nunique(), "DestinationID", 5, ciclo(destinos, "DestinationID", "State", 5)),
        ("Destinos", "BestTimeToVisit", destinos.BestTimeToVisit.nunique(), "DestinationID", 5, ciclo(destinos, "DestinationID", "BestTimeToVisit", 5)),
        ("Destinos", "Popularity", destinos.Popularity.nunique(), "DestinationID", 5, ciclo(destinos, "DestinationID", "Popularity", 5)),
        ("Usuarios", "Preferences", usuarios.Preferences.nunique(), "UserID", 3, ciclo(usuarios, "UserID", "Preferences", 3)),
        ("Usuarios", "Name", usuarios.Name.nunique(), "UserID", 20, ciclo(usuarios, "UserID", "Name", 20)),
        ("Usuarios", "Email", usuarios.Email.nunique(), "UserID", 20, ciclo(usuarios, "UserID", "Email", 20)),
        ("Reseñas", "ReviewText", resenas.ReviewText.nunique(), "ReviewID", 3, ciclo(resenas, "ReviewID", "ReviewText", 3)),
        ("Historial", "VisitDate", historial.VisitDate.nunique(), "HistoryID", 3, ciclo(historial, "HistoryID", "VisitDate", 3)),
    ]
    tabla = pd.DataFrame(
        filas,
        columns=["tabla", "columna", "valores_distintos", "id", "k", "determinada_por_(id-1)_mod_k"],
    )
    return (
        "2. Valores repetidos en ciclos exactos (evidencia de datos sintéticos)",
        "Para cada columna se cuentan sus valores distintos y se verifica si su valor queda "
        "totalmente determinado por la posición del ID en un ciclo de k (por ejemplo, IDs 1, 6, "
        "11, … siempre son el mismo destino). En datos reales esto no ocurre.",
        tabla,
    )


def s03_lugares_reales(destinos):
    tabla = (
        destinos.groupby(["Name", "State", "Type", "BestTimeToVisit"])
        .agg(
            ids=("DestinationID", "size"),
            popularity_min=("Popularity", "min"),
            popularity_max=("Popularity", "max"),
        )
        .reset_index()
        .round(2)
    )
    return (
        "3. Los 1000 destinos son 5 lugares",
        "Se agrupan los destinos por todos sus atributos descriptivos. Solo aparecen 5 "
        "combinaciones, cada una con 200 IDs; lo único que cambia entre copias es Popularity.",
        tabla,
    )


def s04_distribucion_notas(resenas, historial):
    filas = []
    pruebas = []
    for nombre, serie in [("Reseñas (Rating)", resenas.Rating), ("Historial (ExperienceRating)", historial.ExperienceRating)]:
        conteo = serie.value_counts().sort_index()
        prueba = stats.chisquare(conteo.values)  # H0: las 5 notas son igual de probables
        pruebas.append((nombre, prueba.statistic, prueba.pvalue, serie.mean()))
        for nota, n in conteo.items():
            filas.append({"tabla": nombre, "nota": nota, "conteo": n, "porcentaje": round(100 * n / len(serie), 1)})
    tabla = pd.DataFrame(filas).pivot(index="nota", columns="tabla", values=["conteo", "porcentaje"])
    tabla.columns = [f"{b} — {a}" for a, b in tabla.columns]
    tabla = tabla.reset_index()
    extra = "\n".join(
        f"- {n}: media = {m:.2f}; chi-cuadrado de uniformidad χ² = {c:.2f}, p = {p:.3f}"
        for n, c, p, m in pruebas
    )
    return (
        "4. Distribución de las calificaciones",
        "Si las 5 notas fueran igual de probables, cada una tendría ~20 %. La prueba "
        "chi-cuadrado de bondad de ajuste contrasta H0: «distribución uniforme». Un p > 0,05 "
        "significa que no hay evidencia para rechazar que las notas se sortearon al azar.\n\n" + extra,
        tabla,
    )


def s05_nota_vs_popularidad(destinos, resenas, historial):
    filas = []
    for nombre, df, col in [("Reseñas", resenas, "Rating"), ("Historial", historial, "ExperienceRating")]:
        m = df.merge(destinos, on="DestinationID")
        r, p = stats.pearsonr(m[col], m.Popularity)
        rho, p_s = stats.spearmanr(m[col], m.Popularity)
        filas.append({"tabla": nombre, "n": len(m), "pearson": round(r, 3), "p_pearson": round(p, 3),
                      "spearman": round(rho, 3), "p_spearman": round(p_s, 3)})
    return (
        "5. ¿La nota se relaciona con la popularidad del destino?",
        "Se une cada interacción con su destino y se correlaciona la nota con Popularity. "
        "Pearson mide relación lineal; Spearman, si a más de una tiende a haber más de la otra. "
        "Valores cercanos a 0 con p alto = no hay relación.",
        pd.DataFrame(filas),
    )


def s06_preferencias_vs_tipo(destinos, usuarios, historial, resenas):
    """¿Los usuarios visitan/reseñan destinos del tipo que dicen preferir?"""
    inter = interacciones_unidas(resenas, historial)
    m = inter.merge(usuarios[["UserID", "Preferences"]], on="UserID").merge(
        destinos[["DestinationID", "Type"]], on="DestinationID"
    )
    prefs = m.Preferences.str.split(", ").apply(lambda xs: {PREFERENCIA_A_TIPO[x] for x in xs})
    m["coincide"] = [t in p for t, p in zip(m.Type, prefs)]

    # Probabilidad de coincidir por azar: cada perfil tiene 2 de los 5 tipos y
    # los tipos están igual de representados en el catálogo (200 IDs cada uno).
    esperado = 2 / 5
    filas = []
    for fuente, g in [("reseña", m[m.fuente == "reseña"]), ("historial", m[m.fuente == "historial"]), ("unión", m)]:
        k, n = int(g.coincide.sum()), len(g)
        p = stats.binomtest(k, n, esperado, alternative="greater").pvalue
        filas.append({"fuente": fuente, "interacciones": n, "coinciden": k,
                      "observado_%": round(100 * k / n, 1), "esperado_por_azar_%": 100 * esperado,
                      "p_binomial(observado>azar)": round(p, 3)})
    # Prueba de permutación (no depende de suponer el 40 %): se barajan los tipos
    # de destino entre las interacciones, rompiendo cualquier relación con el
    # perfil, y se mira con qué frecuencia el azar iguala o supera lo observado.
    rng = np.random.default_rng(0)
    observado = m.coincide.mean()
    tipos = m.Type.to_numpy()
    simulados = np.array([
        np.mean([t in p for t, p in zip(rng.permutation(tipos), prefs)]) for _ in range(5000)
    ])
    p_perm = (simulados >= observado).mean()

    por_perfil = m.groupby("Preferences").coincide.agg(["size", "mean"]).reset_index()
    por_perfil.columns = ["perfil", "interacciones", "coinciden_%"]
    por_perfil["coinciden_%"] = (100 * por_perfil["coinciden_%"]).round(1)

    cruce = pd.crosstab(m.Preferences, m.Type)
    chi = stats.chi2_contingency(cruce)
    veredicto = (
        "hay evidencia estadística de relación, pero el efecto es pequeño"
        if chi.pvalue < 0.05 else "no hay evidencia de relación"
    )
    explicacion = (
        "Cada perfil declara 2 de los 5 tipos de destino, así que si los usuarios eligieran al "
        "azar, ~40 % de sus interacciones coincidiría con sus preferencias. Tres pruebas "
        "contrastan si lo observado supera al azar: binomial (contra 40 %), permutación "
        "(barajar los tipos 5000 veces) y chi-cuadrado de independencia perfil × tipo.\n\n"
        f"- Permutación (unión): azar promedio = {100 * simulados.mean():.1f} %, "
        f"observado = {100 * observado:.1f} %, p = {p_perm:.3f}\n"
        f"- Chi-cuadrado de independencia (unión): χ² = {chi.statistic:.2f}, gl = {chi.dof}, "
        f"p = {chi.pvalue:.3f} → {veredicto}.\n\n"
        "Coincidencia por perfil (unión):\n\n" + _a_markdown(por_perfil) +
        "\n\nTabla cruzada perfil × tipo de destino (unión):\n\n" + _a_markdown(cruce.reset_index())
    )
    return "6. ¿Las preferencias declaradas explican lo que visitan?", explicacion, pd.DataFrame(filas)


def s07_texto_vs_tipo(destinos, resenas):
    m = resenas.merge(destinos[["DestinationID", "Type"]], on="DestinationID")
    cruce = pd.crosstab(m.Type, m.ReviewText)
    chi = stats.chi2_contingency(cruce)
    return (
        "7. ¿El texto de la reseña corresponde al destino?",
        "Si los textos fueran reales, «Loved the beaches!» aparecería sobre todo en playas. "
        f"Chi-cuadrado de independencia: χ² = {chi.statistic:.2f}, gl = {chi.dof}, p = {chi.pvalue:.3f}.",
        cruce.reset_index(),
    )


def s08_solapamiento(resenas, historial):
    pares_r = set(zip(resenas.UserID, resenas.DestinationID))
    pares_h = set(zip(historial.UserID, historial.DestinationID))
    tabla = pd.DataFrame([
        {"medida": "pares (usuario, destino) en ambas tablas", "valor": len(pares_r & pares_h)},
        {"medida": "usuarios con interacciones en ambas tablas", "valor": len(set(resenas.UserID) & set(historial.UserID))},
        {"medida": "usuarios solo en reseñas", "valor": len(set(resenas.UserID) - set(historial.UserID))},
        {"medida": "usuarios solo en historial", "valor": len(set(historial.UserID) - set(resenas.UserID))},
    ])
    return (
        "8. Solapamiento entre reseñas e historial",
        "Si los datos fueran reales, quien reseña un destino normalmente lo visitó. Aquí ningún "
        "par usuario–destino se repite entre tablas (por eso unirlas no crea duplicados).",
        tabla,
    )


def s09_granularidad(destinos, resenas, historial):
    """Decisión 1: ¿ítem = 1000 IDs o 5 lugares?"""
    inter = interacciones_unidas(resenas, historial).merge(destinos[["DestinationID", "Name"]], on="DestinationID")
    por_id = inter.groupby("DestinationID").size().reindex(destinos.DestinationID, fill_value=0)
    por_lugar = inter.groupby("Name").size()
    lugares_por_usuario = inter.drop_duplicates(["UserID", "Name"]).groupby("UserID").size()
    repiten = int((inter.groupby("UserID").size() > lugares_por_usuario).sum())

    # ¿El ítem que se esconde para prueba existe en entrenamiento? (1 interacción
    # aleatoria por usuario con >= 2; semilla fija para que sea reproducible)
    cuenta = inter.groupby("UserID").size()
    elegibles = inter[inter.UserID.isin(cuenta[cuenta >= 2].index)]
    prueba = elegibles.groupby("UserID").sample(1, random_state=0)
    entreno = inter.drop(prueba.index)
    visto = prueba.DestinationID.isin(entreno.DestinationID).mean()

    tabla = pd.DataFrame([
        {"aspecto": "ítems en el catálogo", "A_1000_IDs": 1000, "B_5_lugares": 5},
        {"aspecto": "ítems sin ninguna interacción", "A_1000_IDs": int((por_id == 0).sum()), "B_5_lugares": 0},
        {"aspecto": "ítems con 1 sola interacción", "A_1000_IDs": int((por_id == 1).sum()), "B_5_lugares": 0},
        {"aspecto": "interacciones del ítem más popular", "A_1000_IDs": int(por_id.max()), "B_5_lugares": int(por_lugar.max())},
        {"aspecto": "interacciones del ítem menos popular (con ≥1)", "A_1000_IDs": int(por_id[por_id > 0].min()), "B_5_lugares": int(por_lugar.min())},
        {"aspecto": "usuarios que repetirían ítem al agrupar", "A_1000_IDs": 0, "B_5_lugares": repiten},
        {"aspecto": "K máximo con sentido", "A_1000_IDs": "10 o más", "B_5_lugares": "≤ 4"},
        {"aspecto": "HitRate@10 de un recomendador al azar", "A_1000_IDs": "≈ 1 %", "B_5_lugares": "100 % con K = 5 (trivial)"},
        {"aspecto": "% de ítems de prueba vistos en entrenamiento", "A_1000_IDs": f"{100 * visto:.0f} %", "B_5_lugares": "100 %"},
    ])
    explicacion = (
        "Compara las dos formas de definir «destino». Lugares distintos por usuario (opción B): "
        + ", ".join(f"{k} lugar(es): {v} usuarios" for k, v in lugares_por_usuario.value_counts().sort_index().items())
        + "."
    )
    return "9. Decisión 1 — ¿Qué es un destino? (1000 IDs frente a 5 lugares)", explicacion, tabla


def s10_definiciones_interaccion(resenas, historial):
    """Decisiones 2 y 3: ¿qué tablas y qué calificaciones cuentan?"""
    inter = interacciones_unidas(resenas, historial)
    opciones = [
        ("Solo reseñas", inter[inter.fuente == "reseña"]),
        ("Solo historial", inter[inter.fuente == "historial"]),
        ("Unión", inter),
        ("Unión, solo nota ≥ 3", inter[inter.calificacion >= 3]),
        ("Unión, solo nota ≥ 4", inter[inter.calificacion >= 4]),
    ]
    filas = []
    for nombre, df in opciones:
        c = df.groupby("UserID").size()
        filas.append({
            "definición": nombre,
            "interacciones": len(df),
            "usuarios": df.UserID.nunique(),
            "destinos": df.DestinationID.nunique(),
            "usuarios_con_1": int((c == 1).sum()),
            "usuarios_evaluables_(≥2)": int((c >= 2).sum()),
            "media_por_usuario": round(c.mean(), 2),
        })
    return (
        "10. Decisiones 2 y 3 — ¿Qué cuenta como interacción?",
        "Para evaluar a un usuario se necesitan al menos 2 interacciones (una para entrenar y "
        "otra escondida para prueba). La tabla muestra cuántos usuarios cumplen eso con cada "
        "definición posible.",
        pd.DataFrame(filas),
    )


def s11_cola_larga(resenas, historial):
    inter = interacciones_unidas(resenas, historial)
    por_usuario = inter.groupby("UserID").size().value_counts().sort_index()
    por_destino = inter.groupby("DestinationID").size().value_counts().sort_index()
    tabla = pd.DataFrame({
        "n_interacciones": sorted(set(por_usuario.index) | set(por_destino.index)),
    })
    tabla["usuarios_con_n"] = tabla.n_interacciones.map(por_usuario).fillna(0).astype(int)
    tabla["destinos_con_n"] = tabla.n_interacciones.map(por_destino).fillna(0).astype(int)
    total_u, total_d = 999, 1000
    sin_u = total_u - inter.UserID.nunique()
    sin_d = total_d - inter.DestinationID.nunique()
    tabla = pd.concat([pd.DataFrame([{"n_interacciones": 0, "usuarios_con_n": sin_u, "destinos_con_n": sin_d}]), tabla])
    densidad = len(inter) / (total_u * total_d)
    return (
        "11. Interacciones por usuario y por destino (unión)",
        f"Cuántos usuarios y destinos tienen 0, 1, 2, … interacciones. Densidad de la matriz "
        f"usuario–destino: {len(inter)} / ({total_u} × {total_d}) = {100 * densidad:.2f} %.",
        tabla,
    )


def s12_cifras_clave(destinos, usuarios, resenas, historial):
    """Las cifras que pide el enunciado en la sección «Datos»."""
    inter = interacciones_unidas(resenas, historial)
    m, n = len(usuarios), len(destinos)
    filas = [
        ("usuarios en el catálogo", m),
        ("usuarios con ≥ 1 interacción", inter.UserID.nunique()),
        ("destinos (IDs) en el catálogo", n),
        ("lugares reales distintos", destinos.Name.nunique()),
        ("destinos (IDs) con ≥ 1 interacción", inter.DestinationID.nunique()),
        ("interacciones: reseñas", len(resenas)),
        ("interacciones: historial", len(historial)),
        ("interacciones: unión", len(inter)),
        ("escala de las calificaciones", f"{inter.calificacion.min()}–{inter.calificacion.max()} (enteros)"),
        ("densidad (solo reseñas)", f"{100 * len(resenas) / (m * n):.2f} %"),
        ("densidad (solo historial)", f"{100 * len(historial) / (m * n):.2f} %"),
        ("densidad (unión)", f"{100 * len(inter) / (m * n):.2f} %"),
        ("interacciones por usuario (media, unión)", round(len(inter) / inter.UserID.nunique(), 2)),
    ]
    return (
        "12. Cifras clave del dataset",
        "Las cifras que pide el enunciado. La densidad es la fracción de celdas llenas de la "
        "matriz usuario–destino: interacciones / (usuarios × destinos).",
        pd.DataFrame(filas, columns=["medida", "valor"]),
    )


def s13_diccionario_variables(destinos, usuarios, resenas, historial):
    """Cada columna: de qué tabla es, qué tipo de variable es y qué valores toma."""

    def describir(serie):
        if pd.api.types.is_float_dtype(serie):
            return f"continua, {serie.min():.2f}–{serie.max():.2f} (media {serie.mean():.2f})"
        if serie.nunique() > 25:
            return f"identificador, {serie.nunique()} valores distintos"
        conteo = serie.value_counts().sort_index()
        return "; ".join(f"{k}: {v}" for k, v in conteo.items())

    rol = {
        "DestinationID": "ID del ítem", "UserID": "ID del usuario",
        "ReviewID": "ID de fila", "HistoryID": "ID de fila",
        "Name": "contenido (nombre)", "Email": "dato personal falso — no se usa",
        "Rating": "calificación de la interacción", "ExperienceRating": "calificación de la interacción",
        "ReviewText": "texto de la interacción", "VisitDate": "fecha de la interacción",
    }
    filas = []
    for tabla, df in [("Destinos", destinos), ("Usuarios", usuarios), ("Reseñas", resenas), ("Historial", historial)]:
        for col in df.columns:
            papel = rol.get(col, "contenido del destino" if tabla == "Destinos" else "contenido del usuario")
            if tabla == "Usuarios" and col == "Name":
                papel = "dato personal falso — no se usa"
            filas.append({"tabla": tabla, "variable": col, "papel": papel, "valores": describir(df[col])})
    return (
        "13. Diccionario de variables",
        "Todas las columnas de los 4 archivos, qué papel cumplen en un recomendador "
        "(identificador, contenido o dato de la interacción) y qué valores toman.",
        pd.DataFrame(filas),
    )


def s14_lugares_nuevos_y_regresos(destinos, resenas, historial):
    """Tarea 1: ¿conviene excluir los lugares que el usuario ya visitó?"""
    inter = interacciones_unidas(resenas, historial).merge(destinos[["DestinationID", "Name"]], on="DestinationID")
    lugares = inter.drop_duplicates(["UserID", "Name"]).groupby("UserID").size()
    nuevos = 5 - lugares
    filas = [{"lugares_nuevos_disponibles": k, "usuarios": int(v)} for k, v in nuevos.value_counts().sort_index().items()]
    tabla = pd.DataFrame(filas)
    faltan = "; ".join(
        f"K = {k}: {int((nuevos < k).sum())} usuarios ({100 * (nuevos < k).mean():.1f} %)" for k in range(1, 6)
    )
    # ¿La interacción escondida cae en un lugar ya visitado? Se usa una partición
    # exploratoria (1 interacción al azar por usuario con >= 2, semilla 0).
    cuenta = inter.groupby("UserID").size()
    elegibles = inter[inter.UserID.isin(cuenta[cuenta >= 2].index)]
    prueba = elegibles.groupby("UserID").sample(1, random_state=0)
    vistos = inter.drop(prueba.index).groupby("UserID").Name.apply(set)
    regreso = np.mean([n in vistos.get(u, set()) for u, n in zip(prueba.UserID, prueba.Name)])
    return (
        "14. Tarea 1 — ¿Excluir los lugares ya visitados?",
        "Cuántos lugares (de 5) le quedan a cada usuario si se excluyen los que ya visitó.\n\n"
        f"- Usuarios que recibirían menos de K lugares nuevos: {faltan}.\n"
        f"- En una partición exploratoria (1 interacción escondida al azar por usuario con ≥ 2, "
        f"{len(prueba)} usuarios), el **{100 * regreso:.1f} %** de las interacciones escondidas es "
        "a un lugar que el usuario ya había visitado: excluirlos haría fallar por diseño esos casos.",
        tabla,
    )


def particion_dejar_uno_fuera(inter, semilla=0):
    """Por usuario: 1 interacción a prueba (si tiene ≥ 2) y 1 a validación (si tiene ≥ 3)."""
    inter = inter.reset_index(drop=True)
    cuenta = inter.groupby("UserID").size()
    rng = np.random.default_rng(semilla)
    orden = inter.sample(frac=1, random_state=int(rng.integers(1_000_000))).copy()
    orden["pos"] = orden.groupby("UserID").cumcount()
    orden["n"] = orden.UserID.map(cuenta)
    orden["conjunto"] = "entrenamiento"
    orden.loc[(orden.n >= 2) & (orden.pos == 0), "conjunto"] = "prueba"
    orden.loc[(orden.n >= 3) & (orden.pos == 1), "conjunto"] = "validación"
    return orden.sort_index()


def particion_aleatoria(inter, semilla=0):
    """80/10/10 al azar sobre las interacciones, sin mirar al usuario."""
    inter = inter.reset_index(drop=True).copy()
    u = np.random.default_rng(semilla).random(len(inter))
    inter["conjunto"] = np.where(u < 0.8, "entrenamiento", np.where(u < 0.9, "validación", "prueba"))
    return inter


def s15_opciones_particion(destinos, resenas, historial):
    """Tarea 3: compara estrategias de partición con las mismas cifras."""
    inter = interacciones_unidas(resenas, historial).merge(destinos[["DestinationID", "Name"]], on="DestinationID")
    filas = []
    for nombre, part in [
        ("Dejar-uno-fuera por usuario", particion_dejar_uno_fuera(inter)),
        ("Aleatoria 80/10/10", particion_aleatoria(inter)),
    ]:
        ent = part[part.conjunto == "entrenamiento"]
        for conj in ("validación", "prueba"):
            ev = part[part.conjunto == conj]
            usuarios_ev = ev.UserID.unique()
            filas.append({
                "estrategia": nombre,
                "conjunto": conj,
                "interacciones_entrenamiento": len(ent),
                "interacciones_conjunto": len(ev),
                "usuarios_evaluados": len(usuarios_ev),
                "%_usuarios_sin_entrenamiento": round(100 * np.mean(~np.isin(usuarios_ev, ent.UserID)), 1),
                "%_IDs_no_vistos_en_entrenamiento": round(100 * (~ev.DestinationID.isin(ent.DestinationID)).mean(), 1),
                "%_lugares_no_vistos_en_entrenamiento": round(100 * (~ev.Name.isin(ent.Name)).mean(), 1),
            })
    fechas = historial.VisitDate.nunique()
    return (
        "15. Tarea 3 — Comparación de estrategias de partición",
        "Cifras de cada estrategia posible (semilla 0). «Usuarios sin entrenamiento» son usuarios de "
        "validación/prueba que el modelo nunca vio: con ellos el filtrado colaborativo no puede hacer "
        "nada. «IDs no vistos» son destinos de validación/prueba sin ninguna interacción en "
        "entrenamiento: un modelo que solo usa IDs no puede recomendarlos; uno con contenido sí.\n\n"
        f"- **Partición temporal: no es viable.** El historial tiene solo {fechas} fechas distintas, "
        "asignadas en ciclo por el número de fila (tabla 2), y las reseñas no tienen fecha.",
        pd.DataFrame(filas),
    )


# --------------------------------------------------------------------------
# Salida
# --------------------------------------------------------------------------
def _a_markdown(df):
    """Tabla Markdown sin depender de 'tabulate'."""
    cols = [str(c) for c in df.columns]
    lineas = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, fila in df.iterrows():
        lineas.append("| " + " | ".join(str(v) for v in fila.values) + " |")
    return "\n".join(lineas)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    destinos, usuarios, resenas, historial = cargar()

    secciones = [
        s01_tamanos_y_calidad(destinos, usuarios, resenas, historial),
        s02_patrones_ciclicos(destinos, usuarios, resenas, historial),
        s03_lugares_reales(destinos),
        s04_distribucion_notas(resenas, historial),
        s05_nota_vs_popularidad(destinos, resenas, historial),
        s06_preferencias_vs_tipo(destinos, usuarios, historial, resenas),
        s07_texto_vs_tipo(destinos, resenas),
        s08_solapamiento(resenas, historial),
        s09_granularidad(destinos, resenas, historial),
        s10_definiciones_interaccion(resenas, historial),
        s11_cola_larga(resenas, historial),
        s12_cifras_clave(destinos, usuarios, resenas, historial),
        s13_diccionario_variables(destinos, usuarios, resenas, historial),
        s14_lugares_nuevos_y_regresos(destinos, resenas, historial),
        s15_opciones_particion(destinos, resenas, historial),
    ]

    partes = [
        "# Inventario de los datos — Módulo 3\n",
        "_Archivo generado automáticamente por `src/inventario.py`. No editar a mano: "
        "volver a ejecutar el script._\n",
    ]
    for i, (titulo, explicacion, tabla) in enumerate(secciones, start=1):
        tabla.to_csv(SALIDA / f"tabla_{i:02d}.csv", index=False)
        partes += [f"## {titulo}\n", explicacion + "\n", _a_markdown(tabla) + "\n",
                   f"_CSV: `results/inventario/tabla_{i:02d}.csv`_\n"]
        print(f"\n=== {titulo} ===\n{tabla.to_string(index=False)}")

    (SALIDA / "RESUMEN.md").write_text("\n".join(partes), encoding="utf-8")
    print(f"\nTablas y resumen guardados en {SALIDA.relative_to(MODULO.parent)}")


if __name__ == "__main__":
    np.random.seed(0)
    main()
