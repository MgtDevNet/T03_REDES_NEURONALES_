"""Página de inicio de TravesIA, la app del Trabajo 3.

Levantar localmente, desde la raíz del repositorio:
    streamlit run app/Inicio.py
"""

import streamlit as st

from utils.estilo import LEMA, NOMBRE, NOMBRE_HTML, configurar_pagina, creditos, portada, tarjeta
from utils.modulos import ENLACES, estado_modulo

configurar_pagina("Inicio")

portada(NOMBRE_HTML, f"{LEMA}. Pronosticamos cuántos viajarán, cuidamos a quien conduce "
        "y te recomendamos tu próximo destino.", "kerala")

st.markdown(f"### ¿Qué es {NOMBRE}?")
st.write(
    f"{NOMBRE} es el sistema inteligente de una empresa de transporte intermunicipal de pasajeros. "
    "Reúne en un solo lugar tres herramientas construidas con redes neuronales para resolver tres "
    "retos de la empresa: saber cuántas personas viajarán y con qué buses y conductores atenderlas, "
    "detectar a tiempo cuándo un conductor maneja distraído, y ofrecer a cada cliente destinos de "
    "viaje que le gusten, empezando por una nueva línea de turismo en India."
)

st.markdown("### Explora los módulos")
modulos = [
    (1, "terminal", "Demanda y operación",
     "Elige una ruta y mira cuántos pasajeros se esperan en los próximos 30 días, y cuántos buses y "
     "conductores se necesitan para atenderlos.", "pages/1_Demanda_y_operacion.py"),
    (2, "conductor", "Conducción segura",
     "Sube la foto de un conductor y descubre si maneja distraído, con qué seguridad lo dice el modelo "
     "y en qué parte de la imagen se fijó.", "pages/2_Conduccion_distractiva.py"),
    (3, "taj_mahal", "Destinos para ti",
     "Elige un cliente o cuéntanos qué te gusta, y te recomendamos lugares de India para tu próximo viaje.",
     "pages/3_Recomendacion_de_destinos.py"),
]
columnas = st.columns(3, gap="medium")
for col, (n, img, titulo, texto, pagina) in zip(columnas, modulos):
    listo, _ = estado_modulo(n)
    with col:
        tarjeta(img, titulo, texto, "Disponible" if listo else "En construcción", "lista" if listo else "obra")
        st.page_link(pagina, label=f"Ir a {titulo.lower()}", icon="➡️", disabled=not listo)

st.markdown("### ¿Cómo lo construimos?")
pasos = st.columns(3, gap="medium")
textos = [
    ("Datos públicos", "Cada módulo parte de un conjunto de datos público: la operación de las terminales "
     "de Colombia, fotos de conductores y un catálogo de viajes (este último, sintético)."),
    ("Comparaciones justas", "Cada red neuronal se compara con modelos más simples, con los mismos datos y 30 "
     "repeticiones, para saber si de verdad aporta algo."),
    ("IA verificada", "Usamos asistentes de inteligencia artificial, pero comprobamos cada cifra y cada referencia, "
     "y documentamos los errores que encontramos."),
]
for col, (titulo, texto) in zip(pasos, textos):
    col.markdown(f'<div class="rumbo-paso"><b>{titulo}</b><br>{texto}</div>', unsafe_allow_html=True)

st.markdown("### Sobre el proyecto")
st.write(
    f"{NOMBRE} es el Trabajo 3 del curso *Redes neuronales y algoritmos bioinspirados* de la Universidad Nacional "
    "de Colombia, Facultad de Minas (semestre 2026-02). Todos los modelos están guardados en el repositorio: "
    "la app los usa directamente, sin volver a entrenarlos."
)
for nombre, url in ENLACES.items():
    st.markdown(f"- **{nombre}:** [{url}]({url})" if url else f"- **{nombre}:** próximamente")

st.divider()
creditos(["kerala", "terminal", "conductor", "taj_mahal"])
