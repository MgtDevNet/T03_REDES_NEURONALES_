"""Página de inicio de la app del Trabajo 3.

Levantar localmente, desde la raíz del repositorio:
    streamlit run app/Inicio.py
"""

import streamlit as st

from utils.modulos import ENLACES, MODULOS, estado_modulo

st.set_page_config(page_title="Trabajo 3 · Sistema inteligente de transporte", page_icon="🚌", layout="wide")

st.title("Sistema inteligente para una empresa de transporte")
st.write(
    "Tres problemas de la empresa resueltos con redes neuronales, en una sola aplicación: "
    "pronosticar la demanda y dimensionar la flota, detectar conductores distraídos y "
    "recomendar destinos de viaje a los clientes. Elige un módulo en el menú de la izquierda."
)

columnas = st.columns(3)
descripciones = {
    1: "Elige una ruta y mira el pronóstico de pasajeros a 30 días, con su banda de incertidumbre, "
       "y la flota y los conductores que se necesitan.",
    2: "Sube una imagen de un conductor y obtén la clase, la probabilidad de cada clase, la decisión "
       "de alerta y el mapa de en qué se fijó el modelo.",
    3: "Elige un cliente existente o crea un perfil con tus preferencias y mira los destinos que te "
       "recomendamos, junto con tu historial.",
}
for col, (n, m) in zip(columnas, MODULOS.items()):
    listo, _ = estado_modulo(n)
    with col:
        st.subheader(f"{n}. {m['titulo']}")
        st.write(descripciones[n])
        st.caption("✅ Disponible" if listo else "🛠️ En construcción")

st.divider()
st.subheader("Enlaces")
for nombre, url in ENLACES.items():
    st.markdown(f"- **{nombre}:** [{url}]({url})" if url else f"- **{nombre}:** pendiente")

st.caption("Trabajo 3 · Redes neuronales y algoritmos bioinspirados · Universidad Nacional de Colombia, 2026-02. "
           "La app usa los modelos guardados en el repositorio; no reentrena nada.")
