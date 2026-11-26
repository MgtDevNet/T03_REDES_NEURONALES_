"""Módulo 2: clasificación de conducción distractiva.

Se activa cuando exista modulo2_conduccion/src/predict.py con `predecir` y `gradcam`
(ver app/README.md y modulo2_conduccion/README.md).
"""

import streamlit as st

from utils.estilo import configurar_pagina, creditos, portada
from utils.modulos import estado_modulo

configurar_pagina("Conducción segura")
portada("Conducción segura", "Detectar a tiempo cuándo un conductor maneja distraído.", "conductor")

listo, mensaje = estado_modulo(2)
if not listo:
    st.info(f"🛠️ Este módulo está en construcción. {mensaje}")
    st.markdown(
        "Cuando el módulo 2 suba su función de inferencia, esta página permitirá subir una imagen "
        "y ver la clase asignada, la probabilidad de cada clase, la decisión de alerta según el "
        "umbral elegido y el mapa de interpretabilidad (Grad-CAM)."
    )
    creditos(["conductor"])
    st.stop()

st.warning("El módulo 2 ya tiene su predict.py, pero esta página todavía no muestra sus resultados.")
