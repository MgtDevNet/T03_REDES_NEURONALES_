# Análisis descriptivo — Módulo 3 (tarea 2)

_Generado por `src/descriptivo.py`. No editar a mano: volver a ejecutar el script._

Figuras: `figures/descriptivo_cola_larga.png`, `figures/descriptivo_concentracion.png`

## 1. Interacciones por usuario y por destino (cola larga)

Cuántos usuarios y cuántos destinos tienen 0, 1, 2, … interacciones, comparado con lo que se esperaría si cada interacción se asignara **al azar** (distribución de Poisson con la misma media). En datos reales la cola es mucho más larga que la de Poisson: unos pocos usuarios muy activos y unos pocos destinos muy populares concentran gran parte de la actividad.

- Por usuario: media = 2.00; bondad de ajuste a Poisson χ² = 6.33, p = 0.276 (no se distingue del azar)
- Por destino: media = 2.00; bondad de ajuste a Poisson χ² = 0.27, p = 0.998 (no se distingue del azar)

| por | n_interacciones | observado | esperado_azar_poisson |
|---|---|---|---|
| usuario | 0 | 141 | 135.2 |
| usuario | 1 | 252 | 270.4 |
| usuario | 2 | 280 | 270.4 |
| usuario | 3 | 195 | 180.3 |
| usuario | 4 | 79 | 90.1 |
| usuario | 5 | 31 | 36.1 |
| usuario | 6 | 17 | 12.0 |
| usuario | ≥7 | 4 | 4.5 |
| destino | 0 | 134 | 135.6 |
| destino | 1 | 271 | 270.9 |
| destino | 2 | 270 | 270.7 |
| destino | 3 | 184 | 180.3 |
| destino | 4 | 89 | 90.0 |
| destino | 5 | 37 | 36.0 |
| destino | 6 | 11 | 12.0 |
| destino | ≥7 | 4 | 4.5 |

_CSV: `results/descriptivo/tabla_01.csv`_

## 2. Concentración en los destinos populares

El **coeficiente de Gini** resume la concentración (0 = todos los destinos reciben lo mismo; 1 = un solo destino se lleva todo). Se compara con 2000 simulaciones en las que las mismas 1998 interacciones se reparten **al azar** entre los destinos. Si el Gini observado cae dentro del intervalo del azar, la concentración que vemos es la que el azar produce por sí solo.

- Interacciones por lugar: Leh Ladakh: 422 (21.1 %), Kerala Backwaters: 412 (20.6 %), Taj Mahal: 408 (20.4 %), Goa Beaches: 380 (19.0 %), Jaipur City: 376 (18.8 %). Al azar, cada lugar tendría ≈ 20 %.

| nivel | gini_observado | gini_azar_media | gini_azar_int95 | p_(azar≥observado) | top10%_observado_% | top10%_azar_% |
|---|---|---|---|---|---|---|
| 1000 IDs | 0.383 | 0.386 | 0.369–0.402 | 0.61 | 23.6 | 23.8 |
| 5 lugares | 0.025 | 0.023 | 0.009–0.040 | 0.374 | — | — |

_CSV: `results/descriptivo/tabla_02.csv`_

## 3. Proporción de usuarios con pocas interacciones

Sobre los 999 usuarios del catálogo (unión de reseñas + historial). La mediana es 2 interacciones por usuario y el máximo es 7.

| usuarios_con | usuarios | %_del_total |
|---|---|---|
| 0 (sin actividad) | 141 | 14.1 |
| exactamente 1 | 252 | 25.2 |
| 1 o menos | 393 | 39.3 |
| 2 o menos | 673 | 67.4 |
| 3 o más | 326 | 32.6 |
| 5 o más | 52 | 5.2 |

_CSV: `results/descriptivo/tabla_03.csv`_

## 4. Hipótesis sobre qué explica las preferencias

Hipótesis razonables sobre qué haría que un viajero prefiera un destino, cada una contrastada con una prueba. Como se hacen **varias pruebas a la vez**, aumenta la probabilidad de encontrar un «resultado significativo» por casualidad; por eso se aplica la **corrección de Bonferroni** (multiplicar cada p por el número de pruebas, 6). Una hipótesis se sostiene solo si su p corregido es menor que 0,05.

| hipótesis | enunciado | prueba | p | p_bonferroni | ¿se sostiene? (α = 0,05 tras corregir) | detalle |
|---|---|---|---|---|---|---|
| H1 | Las personas visitan los tipos de destino que dicen preferir | chi-cuadrado perfil × tipo | 0.019 | 0.113 | no | coinciden 42.3 % (azar ≈ 40 %) |
| H2 | Quienes viajan con niños eligen más playa y naturaleza | chi-cuadrado con/sin niños × tipo | 0.19 | 1.0 | no | playa+naturaleza: con niños 38.9 %, sin niños 41.1 % |
| H3 | Viajar solo o con otro adulto cambia el tipo de destino | chi-cuadrado adultos × tipo | 0.159 | 0.952 | no |  |
| H4 | El género cambia el tipo de destino | chi-cuadrado género × tipo | 0.982 | 1.0 | no |  |
| H5 | Los destinos con mayor `Popularity` reciben más interacciones | correlación de Spearman (1000 IDs) | 0.721 | 1.0 | no | ρ = -0.011 |
| H6 | Las visitas ocurren en la mejor temporada del destino | permutación (solo historial) | 0.295 | 1.0 | no | en temporada 73.0 % (azar 72.6 %) |

_CSV: `results/descriptivo/tabla_04.csv`_
