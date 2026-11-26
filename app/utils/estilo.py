"""Identidad visual de la app «Rumbo»: nombre, logo, imágenes, estilos y créditos.

Las imágenes están en app/assets/img/ (descargadas una vez de Wikimedia Commons, con licencia
libre; ver app/assets/CREDITOS.md), así que la app no depende de internet.
"""

import base64
from pathlib import Path

import streamlit as st

APP = Path(__file__).resolve().parents[1]
ASSETS = APP / "assets"
NOMBRE = "Rumbo"
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
