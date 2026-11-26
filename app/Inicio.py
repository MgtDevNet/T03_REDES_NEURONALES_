"""Página de inicio de TravesIA, la app del Trabajo 3.

Levantar localmente, desde la raíz del repositorio:
    streamlit run app/Inicio.py

Los elementos «de página de viajes» (buscador, cifras, promoción del mes, destinos destacados) usan
datos y resultados reales del proyecto: no hay precios, descuentos ni reseñas inventados.
"""

from datetime import date

import streamlit as st

from utils.estilo import (DESTINOS, LEMA, MESES_ES, NOMBRE, NOMBRE_HTML, configurar_pagina, creditos, en_temporada,
                          galeria, imagen_base64, mapa_destinos, portada, tarjeta, temporada_es)
from utils.modulos import ENLACES, estado_modulo

configurar_pagina("Inicio")
MES = date.today().month
PAGINA_DESTINOS = "pages/3_Recomendacion_de_destinos.py"

portada(NOMBRE_HTML, f"{LEMA}. Pronosticamos cuántos viajarán, cuidamos a quien conduce "
        "y te recomendamos tu próximo destino.", "kerala")

# ----------------------------------------------------------------------- buscador
st.markdown("### ¿A dónde quieres ir?")
izq, der = st.columns([1, 2], gap="large")
with izq:
    elegido = st.selectbox("Destino", list(DESTINOS), index=None, placeholder="Elige un destino de India",
                           label_visibility="collapsed")
    st.page_link(PAGINA_DESTINOS, label="O deja que te recomendemos", icon="✨")
with der:
    if elegido:
        d = DESTINOS[elegido]
        etiqueta = "En temporada" if en_temporada(d["temporada"], MES) else "Fuera de temporada"
        tarjeta(d["img"], elegido, f'{d["texto"]}<br><b>{d["tipo"]}</b> · {d["estado"]} · '
                f'mejor época: {temporada_es(d["temporada"])}', etiqueta,
                "lista" if etiqueta == "En temporada" else "obra")
    else:
        st.caption("Elige un destino para ver su ficha, o pasa a «Destinos para ti» para recibir recomendaciones "
                   "según lo que te gusta.")

# ----------------------------------------------------------------------- cifras
st.markdown(
    '<div class="rumbo-cifras">'
    '<div class="rumbo-cifra"><b>5</b><span>destinos en India</span></div>'
    '<div class="rumbo-cifra"><b>999</b><span>viajeros en nuestros datos</span></div>'
    '<div class="rumbo-cifra"><b>1998</b><span>reseñas y visitas analizadas</span></div>'
    '<div class="rumbo-cifra"><b>3</b><span>herramientas con inteligencia artificial</span></div>'
    '</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------- galería
st.markdown("### Inspírate")
galeria()
st.caption("Desliza hacia el lado para ver todos los destinos.")

# ----------------------------------------------------------------------- promoción del mes
st.markdown(
    f'<div class="rumbo-promo"><img src="data:image/jpeg;base64,{imagen_base64("kerala")}" alt="Kerala Backwaters">'
    '<div class="cuerpo"><span class="rumbo-etiqueta vuelve">Destino del mes · noviembre</span>'
    '<h2>Kerala Backwaters</h2>'
    '<p>Navega entre palmeras en una casa flotante. Nuestro modelo la recomienda en primer lugar a los viajeros '
    'de «naturaleza y aventura» y de «ciudad e historia»: en las pruebas, uno de cada cuatro de ellos eligió Kerala '
    'para su siguiente viaje.</p>'
    '<span class="rumbo-chip">Naturaleza</span><span class="rumbo-chip">Casas flotantes</span>'
    f'<span class="rumbo-chip">Temporada: {temporada_es("Sep-Mar")}</span>'
    '<p class="rumbo-credito" style="margin-top:.6rem">Elegido con la pregunta de negocio 3 del informe del '
    'módulo 3 (datos sintéticos).</p></div></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------- destinos destacados
st.markdown(f"### Destinos destacados · {MESES_ES[MES - 1]}")
tipos = ["Todos"] + sorted({d["tipo"] for d in DESTINOS.values()})
for pestana, tipo in zip(st.tabs(tipos), tipos):
    with pestana:
        lista = [(n, d) for n, d in DESTINOS.items() if tipo == "Todos" or d["tipo"] == tipo]
        lista.sort(key=lambda x: not en_temporada(x[1]["temporada"], MES))  # primero los que están en temporada
        for fila in range(0, len(lista), 3):
            cols = st.columns(3, gap="medium")
            for col, (nombre, d) in zip(cols, lista[fila:fila + 3]):
                with col:
                    temporada = en_temporada(d["temporada"], MES)
                    tarjeta(d["img"], nombre, f'{d["texto"]}<br><small>Mejor época: {temporada_es(d["temporada"])}'
                            '</small>', "En temporada" if temporada else "Fuera de temporada",
                            "lista" if temporada else "obra")

# ----------------------------------------------------------------------- mapa
st.markdown("### Nuestros destinos en el mapa")
st.pydeck_chart(mapa_destinos(), height=520)
st.caption("Pasa el cursor sobre un punto para ver su tipo y su mejor época. Contorno de India: Natural Earth "
           "(dominio público), límites aproximados.")

# ----------------------------------------------------------------------- viaja seguro
listo2, _ = estado_modulo(2)
st.markdown(
    f'<div class="rumbo-seguro"><img src="data:image/jpeg;base64,{imagen_base64("conductor")}" alt="Puesto del conductor">'
    f'<div class="cuerpo"><span class="rumbo-etiqueta {"lista" if listo2 else "obra"}">'
    f'{"Disponible" if listo2 else "Próximamente"}</span><h2>Viaja seguro</h2>'
    '<p>Tu viaje empieza mucho antes de llegar al destino. Nuestros buses tendrán una cámara que reconoce cuándo un '
    'conductor maneja distraído (si escribe en el celular, habla por teléfono o mira hacia atrás) y genera una alerta '
    'a tiempo.</p><p>Lo construye el módulo 2 de TravesIA con una red neuronal entrenada con fotos de conductores.</p>'
    '</div></div>', unsafe_allow_html=True)
st.page_link("pages/2_Conduccion_distractiva.py", label="Conoce cómo funciona", icon="🛡️", disabled=not listo2)

# ----------------------------------------------------------------------- qué es y módulos
st.markdown(f"### ¿Qué es {NOMBRE}?")
st.write(
    f"{NOMBRE} es el sistema inteligente de una empresa de transporte intermunicipal de pasajeros. "
    "Reúne en un solo lugar tres herramientas construidas con redes neuronales para resolver tres "
    "retos de la empresa: saber cuántas personas viajarán y con qué buses y conductores atenderlas, "
    "detectar a tiempo cuándo un conductor maneja distraído, y ofrecer a cada cliente destinos de "
    "viaje que le gusten, empezando por una nueva línea de turismo en India."
)
modulos = [
    (1, "terminal", "Demanda y operación",
     "Elige una ruta y mira cuántos pasajeros se esperan en los próximos 30 días, y cuántos buses y "
     "conductores se necesitan para atenderlos.", "pages/1_Demanda_y_operacion.py"),
    (2, "conductor", "Conducción segura",
     "Sube la foto de un conductor y descubre si maneja distraído, con qué seguridad lo dice el modelo "
     "y en qué parte de la imagen se fijó.", "pages/2_Conduccion_distractiva.py"),
    (3, "taj_mahal", "Destinos para ti",
     "Elige un cliente o cuéntanos qué te gusta, y te recomendamos lugares de India para tu próximo viaje.",
     PAGINA_DESTINOS),
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

# ----------------------------------------------------------------------- pie de página
enlaces_html = "<br>".join(f'<a href="{u}" target="_blank">{n}</a>' if u else f"{n} (próximamente)"
                           for n, u in ENLACES.items())
st.markdown(
    '<div class="rumbo-pie">'
    f'<div class="col"><b style="font-size:1.25rem">{NOMBRE_HTML}</b><br>{LEMA}.<br><br>'
    'Trabajo 3 · Redes neuronales y algoritmos bioinspirados<br>Universidad Nacional de Colombia · Facultad de Minas'
    ' · 2026-02</div>'
    '<div class="col"><b>Explora</b><br><a href="Demanda_y_operacion" target="_self">Demanda y operación</a><br>'
    '<a href="Conduccion_distractiva" target="_self">Conducción segura</a><br>'
    '<a href="Recomendacion_de_destinos" target="_self">Destinos para ti</a></div>'
    f'<div class="col"><b>El proyecto</b><br>{enlaces_html}</div>'
    '</div>', unsafe_allow_html=True)
st.write("")
creditos(["kerala", "taj_mahal", "goa", "jaipur", "leh_ladakh", "terminal", "conductor"])
