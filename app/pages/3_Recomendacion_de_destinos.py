"""Módulo 3: recomendación de destinos de viaje.

Usa las funciones de modulo3_recomendacion/src/predict.py (red de dos torres guardada; no reentrena):
recomendar_usuario, recomendar_perfil_nuevo e historial.
"""

import json

import pandas as pd
import streamlit as st

from utils.estilo import IMAGEN_DE_LUGAR, configurar_pagina, creditos, portada, tarjeta
from utils.modulos import RAIZ, estado_modulo

configurar_pagina("Destinos para ti")
portada("Destinos para ti", "Tu próximo viaje a India, elegido según lo que te gusta.", "leh_ladakh")

listo, mensaje = estado_modulo(3)
if not listo:
    st.error(f"No se pudo cargar el módulo 3. {mensaje}")
    st.stop()

from modulo3_recomendacion.src.predict import historial, recomendar_perfil_nuevo, recomendar_usuario  # noqa: E402

PROCESADOS = RAIZ / "modulo3_recomendacion" / "data" / "processed"
TIPO_ES = {"Beach": "Playa", "Historical": "Histórico", "City": "Ciudad", "Nature": "Naturaleza", "Adventure": "Aventura"}
ES_TIPO = {v: k for k, v in TIPO_ES.items()}
GENERO_ES = {"Female": "Mujer", "Male": "Hombre"}
ETIQUETA_A_TIPO = {"Beaches": "Beach", "Historical": "Historical", "City": "City", "Nature": "Nature", "Adventure": "Adventure"}


@st.cache_data
def opciones():
    return json.loads((PROCESADOS / "preferencias.json").read_text(encoding="utf-8"))


@st.cache_data
def usuarios():
    return pd.read_csv(PROCESADOS / "usuarios.csv").set_index("UserID")


def perfil_en_datos(tipos):
    """¿La combinación de preferencias aparece en los datos de entrenamiento?"""
    presentes = [{ETIQUETA_A_TIPO[x.strip()] for x in p.split(",")} for p in opciones()["perfiles_presentes_en_datos"]]
    return set(tipos) in presentes


def mostrar_recomendaciones(recs):
    columnas = st.columns(len(recs), gap="small")
    for col, (_, r) in zip(columnas, recs.iterrows()):
        with col:
            tarjeta(IMAGEN_DE_LUGAR[r.lugar], f"{r.posicion}. {r.lugar}",
                    f"{TIPO_ES.get(r.tipo, r.tipo)} · {r.estado}<br>Mejor época: {r.mejor_epoca}",
                    "🔁 Ya lo visitaste, ¿vuelves?" if r.ya_visitado else None, "vuelve")


st.write(
    "Elige un cliente que ya existe o crea un perfil nuevo con tus preferencias. La app te muestra "
    "los lugares que probablemente te interesen, calculados con el modelo guardado."
)
k = st.slider("¿Cuántos lugares quieres ver? (K)", min_value=1, max_value=5, value=3)

cliente, nuevo = st.tabs(["Cliente existente", "Cliente nuevo"])

with cliente:
    u = usuarios()
    user_id = st.number_input("Número de cliente (1 a 999)", min_value=int(u.index.min()), max_value=int(u.index.max()), value=1, step=1)
    p = u.loc[user_id]
    prefs = " y ".join(TIPO_ES[ETIQUETA_A_TIPO[x.strip()]] for x in p.Preferences.split(","))
    st.markdown(f"**Perfil:** le gusta {prefs} · {GENERO_ES.get(p.Gender, p.Gender)} · "
                f"{p.NumberOfAdults} adulto(s) y {p.NumberOfChildren} niño(s)")
    izq, der = st.columns([1, 2.2], gap="large")
    with izq:
        st.markdown("**Su historial**")
        h = historial(int(user_id))
        if h.empty:
            st.caption("Este cliente todavía no tiene reseñas ni visitas.")
        else:
            st.dataframe(h.assign(tipo=h.tipo.map(TIPO_ES))[["lugar", "tipo", "fuente", "calificacion"]]
                         .rename(columns={"lugar": "Lugar", "tipo": "Tipo", "fuente": "Fuente", "calificacion": "Calificación"}),
                         hide_index=True, width="stretch")
    with der:
        st.markdown(f"**Te recomendamos estos {k} lugares**")
        mostrar_recomendaciones(recomendar_usuario(int(user_id), k=k))

with nuevo:
    st.write("Marca lo que te gusta. No guardamos tu perfil: solo se usa para calcular las recomendaciones.")
    o = opciones()
    elegidas = st.multiselect("¿Qué tipo de destinos te gustan?", [TIPO_ES[t] for t in o["tipos_de_destino"]],
                              default=["Playa", "Histórico"])
    c1, c2, c3 = st.columns(3)
    genero = c1.radio("Género", [GENERO_ES[g] for g in o["generos"]], horizontal=True)
    adultos = c2.selectbox("Adultos que viajan", o["adultos"])
    ninos = c3.selectbox("Niños que viajan", o["ninos"])
    if not elegidas:
        st.warning("Elige al menos un tipo de destino.")
    else:
        tipos = [ES_TIPO[e] for e in elegidas]
        if not perfil_en_datos(tipos):
            st.warning("Esta combinación de preferencias no aparece en nuestros datos, así que el modelo tiene que "
                       "extrapolar y el resultado puede no tener sentido. Las combinaciones que sí conoce son: "
                       + "; ".join(" + ".join(TIPO_ES[ETIQUETA_A_TIPO[x.strip()]] for x in pp.split(","))
                                   for pp in o["perfiles_presentes_en_datos"]) + ".")
        recs = recomendar_perfil_nuevo({"preferencias": tipos, "genero": {v: k_ for k_, v in GENERO_ES.items()}[genero],
                                        "adultos": adultos, "ninos": ninos}, k=k)
        st.markdown(f"**Te recomendamos estos {k} lugares**")
        mostrar_recomendaciones(recs)

creditos(["leh_ladakh", "taj_mahal", "goa", "jaipur", "kerala"])

with st.expander("¿Cómo funciona y qué tan bueno es?"):
    st.markdown(
        "- Las recomendaciones salen de una **red neuronal de dos torres**: una convierte tus preferencias en un "
        "vector, la otra hace lo mismo con cada destino, y se recomiendan los destinos más parecidos a ti.\n"
        "- El modelo solo usa tus **preferencias**: el género y el tamaño del grupo se piden, pero no cambian el "
        "resultado (no ayudaron a predecir).\n"
        "- Los datos del catálogo son **sintéticos** (5 lugares de India). Con ellos, la red acierta el lugar del "
        "próximo viaje entre sus 3 recomendaciones el 61,8 % de las veces, frente al 60 % de recomendar al azar: "
        "la diferencia no es significativa. La app demuestra el método, no la calidad que tendría con datos reales.\n"
        "- Los lugares que ya visitaste no se excluyen: se marcan con «¿vuelves?», porque uno de cada tres viajes "
        "es un regreso."
    )
