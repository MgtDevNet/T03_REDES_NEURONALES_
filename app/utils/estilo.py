"""Identidad visual de la app «TravesIA»: nombre, logo, imágenes, estilos y créditos.

Las imágenes están en app/assets/img/ (descargadas una vez de Wikimedia Commons, con licencia
libre; ver app/assets/CREDITOS.md), así que la app no depende de internet.
"""

import base64
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parents[1]
ASSETS = APP / "assets"
NOMBRE = "TravesIA"
NOMBRE_HTML = 'Traves<span style="color:#f97366">IA</span>'  # la «IA» resaltada en coral
LEMA = "Viaja con inteligencia"
LOGO = ASSETS / "logo.svg"

IMAGENES = {
    "taj_mahal": ("Taj Mahal Tomb at sunrise", "MarcBW", "CC BY-SA 3.0",
                  "https://commons.wikimedia.org/wiki/File:Taj_Mahal_Tomb_at_sunrise.JPG"),
    "goa": ("Palolem Beach, South Goa", "iMahesh", "CC BY-SA 4.0",
            "https://commons.wikimedia.org/wiki/File:Palolem_Beach,_South_Goa.jpg"),
    "jaipur": ("East facade Hawa Mahal Jaipur (July 2022)", "Chainwit.", "CC BY-SA 4.0",
               "https://commons.wikimedia.org/wiki/File:East_facade_Hawa_Mahal_Jaipur_from_ground_level_(July_2022)_-_img_01.jpg"),
    "kerala": ("Kerala backwaters, Houseboats, India", "Vyacheslav Argenberg", "CC BY 4.0",
               "https://commons.wikimedia.org/wiki/File:Kerala_backwaters,_Houseboats,_India.jpg"),
    "leh_ladakh": ("Pangong Tso 2", "KennyOMG", "CC BY-SA 4.0", "https://commons.wikimedia.org/wiki/File:Pangong_Tso_2.jpg"),
    "terminal": ("Bogotá Terminal de buses entrada 2", "Felipe Restrepo Acosta", "CC BY-SA 3.0",
                 "https://commons.wikimedia.org/wiki/File:Bogot%C3%A1_Terminal_de_buses_entrada_2.JPG"),
    "conductor": ("Buenos Aires - A 1955 Colectivo", "Jorge Royan", "CC BY-SA 3.0",
                  "https://commons.wikimedia.org/wiki/File:Buenos_Aires_-_A_1955_Colectivo_-_6126.jpg"),
}
MESES = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6, "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}
MESES_ES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
            "noviembre", "diciembre"]
# Fichas de los 5 lugares del catálogo (tipo, estado y temporada salen de destinos.csv).
DESTINOS = {
    "Taj Mahal": {"lat": 27.1751, "lon": 78.0421, "img": "taj_mahal", "tipo": "Histórico", "estado": "Uttar Pradesh", "temporada": "Nov-Feb",
                  "texto": "El mausoleo de mármol blanco más famoso del mundo, en la ciudad de Agra."},
    "Goa Beaches": {"lat": 15.01, "lon": 74.0233, "img": "goa", "tipo": "Playa", "estado": "Goa", "temporada": "Nov-Mar",
                    "texto": "Playas de arena dorada y palmeras en la costa occidental de India."},
    "Jaipur City": {"lat": 26.9124, "lon": 75.7873, "img": "jaipur", "tipo": "Ciudad", "estado": "Rajasthan", "temporada": "Oct-Mar",
                    "texto": "La «ciudad rosa», con palacios como el Hawa Mahal y mercados llenos de color."},
    "Kerala Backwaters": {"lat": 9.4981, "lon": 76.3388, "img": "kerala", "tipo": "Naturaleza", "estado": "Kerala", "temporada": "Sep-Mar",
                          "texto": "Canales y lagunas entre palmeras que se recorren en casas flotantes."},
    "Leh Ladakh": {"lat": 34.1526, "lon": 77.5771, "img": "leh_ladakh", "tipo": "Aventura", "estado": "Jammu and Kashmir", "temporada": "Apr-Jun",
                   "texto": "Montañas del Himalaya y lagos de altura como el Pangong, para los más aventureros."},
}


def en_temporada(rango, mes):
    """¿El mes (1–12) está dentro de la mejor época, por ejemplo «Nov-Feb»?"""
    a, b = (MESES[x] for x in rango.split("-"))
    return a <= mes <= b if a <= b else (mes >= a or mes <= b)


def temporada_es(rango):
    a, b = (MESES[x] for x in rango.split("-"))
    return f"{MESES_ES[a - 1]} a {MESES_ES[b - 1]}"


IMAGEN_DE_LUGAR = {"Taj Mahal": "taj_mahal", "Goa Beaches": "goa", "Jaipur City": "jaipur",
                   "Kerala Backwaters": "kerala", "Leh Ladakh": "leh_ladakh"}

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1150px;}
.rumbo-hero {position: relative; border-radius: 18px; overflow: hidden; height: 330px; margin-bottom: 1.6rem;
             background-size: cover; background-position: center;}
.rumbo-hero::after {content: ""; position: absolute; inset: 0;
                    background: linear-gradient(90deg, rgba(8,47,58,.78) 0%, rgba(8,47,58,.35) 55%, rgba(8,47,58,0) 100%);}
.rumbo-hero .texto {position: absolute; z-index: 1; left: 2.2rem; top: 50%; transform: translateY(-50%);
                    color: #fff; max-width: 560px;}
.rumbo-hero h1 {color: #fff; font-size: 2.6rem; margin: 0 0 .4rem 0; line-height: 1.1;}
.rumbo-hero p {font-size: 1.12rem; margin: 0; opacity: .95;}
.rumbo-tarjeta {background: #fff; border: 1px solid #dcebe8; border-radius: 16px; overflow: hidden; height: 100%;
                box-shadow: 0 2px 10px rgba(14,116,144,.06);}
.rumbo-tarjeta img {width: 100%; height: 170px; object-fit: cover; display: block;}
.rumbo-tarjeta .cuerpo {padding: .9rem 1.1rem 1.1rem 1.1rem;}
.rumbo-tarjeta h3 {margin: 0 0 .35rem 0; font-size: 1.15rem;}
.rumbo-tarjeta p {margin: 0; color: #4a5a61; font-size: .95rem;}
.rumbo-etiqueta {display: inline-block; font-size: .78rem; font-weight: 600; padding: 2px 10px; border-radius: 999px;
                 margin-bottom: .45rem;}
.rumbo-etiqueta.lista {background: #d9f3ee; color: #0b7d70;}
.rumbo-etiqueta.obra {background: #fdf0d8; color: #9a6400;}
.rumbo-etiqueta.vuelve {background: #ffe7e3; color: #b8473a;}
.rumbo-paso {background: #eef7f5; border-radius: 14px; padding: 1rem 1.1rem; height: 100%;}
.rumbo-paso b {color: #0e7490;}
.rumbo-credito {font-size: .78rem; color: #6b7a80;}
.rumbo-cifras {display: flex; gap: 1rem; flex-wrap: wrap; margin: .2rem 0 1.4rem 0;}
.rumbo-cifra {flex: 1 1 160px; background: #eef7f5; border-radius: 14px; padding: .8rem 1rem; text-align: center;}
.rumbo-cifra b {display: block; font-size: 1.7rem; color: #0e7490; line-height: 1.1;}
.rumbo-cifra span {font-size: .88rem; color: #4a5a61;}
.rumbo-promo {display: flex; border-radius: 18px; overflow: hidden; background: #fff7f4; border: 1px solid #fbd9d2;
              margin: .4rem 0 1.6rem 0;}
.rumbo-promo img {width: 46%; object-fit: cover; min-height: 250px;}
.rumbo-promo .cuerpo {padding: 1.4rem 1.6rem;}
.rumbo-promo h2 {margin: .2rem 0 .5rem 0; font-size: 1.7rem;}
.rumbo-promo p {color: #4a5a61; margin: 0 0 .5rem 0;}
.rumbo-chip {display: inline-block; font-size: .8rem; padding: 3px 11px; border-radius: 999px; margin: 0 .3rem .3rem 0;
             background: #fff; border: 1px solid #f3c7bd; color: #b8473a;}
.rumbo-pie {background: #0f3d4a; color: #d9eef0; border-radius: 18px; padding: 1.6rem 1.8rem; margin-top: 2rem;}
.rumbo-pie b {color: #fff;}
.rumbo-pie a {color: #8fe3d6; text-decoration: none;}
.rumbo-pie .col {display: inline-block; vertical-align: top; width: 32%; min-width: 200px; margin-bottom: .6rem;}
.rumbo-galeria {display: flex; gap: .9rem; overflow-x: auto; scroll-snap-type: x mandatory; padding-bottom: .5rem;
                margin-bottom: 1.4rem;}
.rumbo-galeria figure {flex: 0 0 360px; height: 230px; margin: 0; position: relative; border-radius: 16px; overflow: hidden;
                       scroll-snap-align: start;}
.rumbo-galeria img {width: 100%; height: 100%; object-fit: cover; transition: transform .4s;}
.rumbo-galeria figure:hover img {transform: scale(1.05);}
.rumbo-galeria figcaption {position: absolute; left: 0; right: 0; bottom: 0; padding: .7rem .9rem; color: #fff;
                           background: linear-gradient(0deg, rgba(8,47,58,.85), rgba(8,47,58,0)); font-weight: 600;}
.rumbo-galeria figcaption span {display: block; font-weight: 400; font-size: .85rem; opacity: .9;}
.rumbo-seguro {display: flex; border-radius: 18px; overflow: hidden; background: #eef7f5; border: 1px solid #cfe7e2;
               margin: .4rem 0 1.4rem 0;}
.rumbo-seguro img {width: 40%; object-fit: cover; min-height: 220px;}
.rumbo-seguro .cuerpo {padding: 1.3rem 1.6rem;}
.rumbo-seguro h2 {margin: .2rem 0 .5rem 0; font-size: 1.55rem;}
.rumbo-seguro p {color: #4a5a61; margin: 0 0 .5rem 0;}
@media (max-width: 700px) {.rumbo-seguro {flex-direction: column;} .rumbo-seguro img {width: 100%;}}
@media (max-width: 700px) {.rumbo-promo {flex-direction: column;} .rumbo-promo img {width: 100%;}
                           .rumbo-hero h1 {font-size: 2rem;}}
</style>
"""


def ruta_imagen(clave):
    return ASSETS / "img" / f"{clave}.jpg"


def imagen_base64(clave):
    return base64.b64encode(ruta_imagen(clave).read_bytes()).decode()


def configurar_pagina(titulo):
    """Configuración común de cada página: título, ícono, logo en el menú y estilos."""
    st.set_page_config(page_title=f"{titulo} · {NOMBRE}", page_icon=str(LOGO), layout="wide")
    st.logo(str(LOGO), size="large")
    st.markdown(CSS, unsafe_allow_html=True)


def portada(titulo, subtitulo, clave_imagen):
    st.markdown(
        f'<div class="rumbo-hero" style="background-image:url(data:image/jpeg;base64,{imagen_base64(clave_imagen)})">'
        f'<div class="texto"><h1>{titulo}</h1><p>{subtitulo}</p></div></div>',
        unsafe_allow_html=True)


def tarjeta(clave_imagen, titulo, texto, etiqueta=None, clase="lista"):
    etq = f'<span class="rumbo-etiqueta {clase}">{etiqueta}</span><br>' if etiqueta else ""
    st.markdown(
        f'<div class="rumbo-tarjeta"><img src="data:image/jpeg;base64,{imagen_base64(clave_imagen)}" alt="{titulo}">'
        f'<div class="cuerpo">{etq}<h3>{titulo}</h3><p>{texto}</p></div></div>',
        unsafe_allow_html=True)


def creditos(claves):
    """Atribución de las fotos usadas en la página (exigida por sus licencias)."""
    partes = [f'<a href="{IMAGENES[c][3]}" target="_blank">{IMAGENES[c][0]}</a> — {IMAGENES[c][1]} ({IMAGENES[c][2]})'
              for c in claves]
    st.markdown('<p class="rumbo-credito">Fotos de Wikimedia Commons: ' + " · ".join(partes) + ".</p>",
                unsafe_allow_html=True)


def mapa_destinos():
    """Mapa de India con los 5 destinos, dibujado sin mapa base (no depende de internet).

    Contorno: Natural Earth 1:50m (dominio público), guardado en app/assets/india.geojson.
    """
    import json

    import pydeck as pdk

    india = json.loads((ASSETS / "india.geojson").read_text(encoding="utf-8"))
    puntos = [{"nombre": n, "lat": d["lat"], "lon": d["lon"], "tipo": d["tipo"],
               "temporada": temporada_es(d["temporada"])} for n, d in DESTINOS.items()]
    capas = [
        pdk.Layer("GeoJsonLayer", india, stroked=True, filled=True, get_fill_color=[222, 241, 237],
                  get_line_color=[14, 116, 144], line_width_min_pixels=1.2),
        pdk.Layer("ScatterplotLayer", puntos, get_position=["lon", "lat"], get_radius=45000,
                  get_fill_color=[249, 115, 102], get_line_color=[255, 255, 255], stroked=True,
                  line_width_min_pixels=2, pickable=True),
        pdk.Layer("TextLayer", puntos, get_position=["lon", "lat"], get_text="nombre", get_size=15,
                  get_color=[15, 61, 74], get_pixel_offset=[0, -22], font_weight=600),
    ]
    vista = pdk.ViewState(latitude=22.5, longitude=79.5, zoom=3.6, pitch=0)
    return pdk.Deck(layers=capas, initial_view_state=vista, map_style=None, map_provider=None,
                    tooltip={"html": "<b>{nombre}</b><br>{tipo}<br>Mejor época: {temporada}"})


def galeria():
    """Franja de fotos que se desliza de lado (los 5 destinos)."""
    figuras = "".join(
        f'<figure><img src="data:image/jpeg;base64,{imagen_base64(d["img"])}" alt="{n}">'
        f'<figcaption>{n}<span>{d["tipo"]} · {d["estado"]}</span></figcaption></figure>'
        for n, d in DESTINOS.items())
    st.markdown(f'<div class="rumbo-galeria">{figuras}</div>', unsafe_allow_html=True)
