# Evidencia para la tarea 4 — modelos de referencia

_Generado por `src/evidencia_tarea4.py`. No editar a mano._

## 1. Qué puede lograr cada familia de modelos

Para cada caso evaluado (un usuario y su destino escondido) se mide si el modelo puede, en principio, darle un puntaje mayor que 0 al destino escondido. Si no puede, ese caso es un fallo seguro, entrene como entrene.

| medida | ajuste: entrenamiento → validación | final: entrenamiento+validación → prueba |
|---|---|---|
| casos evaluados | 326.0 | 606.0 |
| HitRate@10 esperado del azar (%) | 1.0 | 1.0 |
| destino escondido nunca visto al aprender (%) | 34.4 | 22.6 |
| techo de un modelo solo de IDs (%) | 65.6 | 77.4 |
| KNN ítem–ítem puede puntuarlo (%) | 0.0 | 0.2 |
| KNN usuario–usuario puede puntuarlo (%) | 0.0 | 0.2 |

## 2. Empates en la popularidad (entrenamiento+validación)

Destinos agrupados por su número de interacciones. Para llenar un top-10 por popularidad hay que desempatar dentro del último grupo que entra.

| interacciones | destinos | destinos acumulados |
|---|---|---|
| 6 | 1 | 1 |
| 5 | 15 | 16 |
| 4 | 24 | 40 |
| 3 | 125 | 165 |
| 2 | 250 | 415 |
| 1 | 340 | 755 |
| 0 | 245 | 1000 |

## 3. Tamaño y densidad de la matriz con la que se aprende


| escenario | interacciones | usuarios con ≥ 1 | destinos con ≥ 1 | densidad (%) |
|---|---|---|---|---|
| ajuste: entrenamiento → validación | 1066 | 858 | 661 | 0.107 |
| final: entrenamiento+validación → prueba | 1392 | 858 | 755 | 0.139 |
