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

**Cómo se construye el intervalo del azar (por ID):**
- Simulación 1, interacciones de los IDs 1 a 10: [2, 1, 0, 0, 3, 4, 2, 3, 2, 4].
- Gini de las simulaciones 1 a 5: 0.374, 0.403, 0.381, 0.386, 0.397.
- Mínimo y máximo de los 2000 Gini: 0.356 y 0.410.
- Ordenados, posición 50 y 1950 (percentiles 2,5 y 97,5): 0.369 y 0.402.
- Simulaciones con Gini ≥ observado (0.383): 1219 de 2000 → p-valor = 0.61.

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

Hipótesis razonables sobre qué haría que un viajero prefiera un destino. Para cada una: hipótesis nula (H0, «no hay relación»), alternativa (H1), prueba y estadístico. Como se hacen **varias pruebas a la vez**, aumenta la probabilidad de encontrar un «resultado significativo» por casualidad; por eso se aplica la **corrección de Bonferroni**: p-valor corregido = mín(1, 6 × p-valor). Se rechaza H0 solo si el p-valor corregido es menor que 0,05 (equivale a p-valor < 0,05/6 = 0.0083).

| hipótesis | enunciado | H0 | H1 | prueba | estadístico | p_valor | p_valor_corregido | decisión (α = 0,05) | detalle |
|---|---|---|---|---|---|---|---|---|---|
| H1 | Las personas visitan los tipos de destino que dicen preferir | el tipo visitado es independiente del perfil de preferencias | el tipo visitado depende del perfil | chi-cuadrado de independencia perfil × tipo | χ² = 18.35, gl = 8 (tabla 3×5; mínimo esperado 123 ≥ 5) | 0.019 | 0.113 | no se rechaza H0: no hay evidencia suficiente | coinciden 42.3 % (azar ≈ 40 %) |
| H2 | Quienes viajan con niños eligen más playa y naturaleza | el tipo visitado es independiente de viajar con niños | el tipo visitado depende de viajar con niños | chi-cuadrado de independencia con/sin niños × tipo | χ² = 6.13, gl = 4 (tabla 2×5; mínimo esperado 129 ≥ 5) | 0.19 | 1.0 | no se rechaza H0: no hay evidencia suficiente | playa+naturaleza: con niños 38.9 %, sin niños 41.1 % |
| H3 | Viajar solo o con otro adulto cambia el tipo de destino | el tipo visitado es independiente del número de adultos | el tipo visitado depende del número de adultos | chi-cuadrado de independencia adultos (1 o 2) × tipo | χ² = 6.60, gl = 4 (tabla 2×5; mínimo esperado 180 ≥ 5) | 0.159 | 0.952 | no se rechaza H0: no hay evidencia suficiente |  |
| H4 | El género cambia el tipo de destino | el tipo visitado es independiente del género | el tipo visitado depende del género | chi-cuadrado de independencia género × tipo | χ² = 0.40, gl = 4 (tabla 2×5; mínimo esperado 186 ≥ 5) | 0.982 | 1.0 | no se rechaza H0: no hay evidencia suficiente |  |
| H5 | Los destinos con mayor `Popularity` reciben más interacciones | ρ = 0 (no hay relación monótona) | ρ ≠ 0 (bilateral) | correlación de Spearman (1000 IDs) | ρ = -0.011 | 0.721 | 1.0 | no se rechaza H0: no hay evidencia suficiente |  |
| H6 | Las visitas ocurren en la mejor temporada del destino | el % de visitas en temporada es el del azar (72.6 %) | el % en temporada es mayor que el del azar (unilateral) | permutación, 2000 repeticiones (solo historial) | en temporada 73.0 % | 0.295 | 1.0 | no se rechaza H0: no hay evidencia suficiente | en temporada 73.0 % (azar 72.6 %) |

_CSV: `results/descriptivo/tabla_04.csv`_
