"""Módulo 1: pronóstico de demanda y planeación de la operación.

Se activa cuando exista modulo1_demanda/src/predict.py con la función `predecir`
(ver app/README.md y modulo1_demanda/README.md).
"""

import streamlit as st

from utils.modulos import estado_modulo

st.set_page_config(page_title="Demanda y operación", page_icon="🚌", layout="wide")
st.title("1. Pronóstico de demanda y planeación de la operación")

listo, mensaje = estado_modulo(1)
if not listo:
    st.info(f"🛠️ Este módulo está en construcción. {mensaje}")
    st.markdown(
        "Cuando el módulo 1 suba su función de inferencia, esta página mostrará: elegir una ruta, "
        "el pronóstico a 30 días con su banda de incertidumbre, la flota y los conductores "
        "requeridos, y cómo cambian al modificar la ocupación objetivo y el cuantil usado."
    )
    st.stop()

st.warning("El módulo 1 ya tiene su predict.py, pero esta página todavía no muestra sus resultados.")
