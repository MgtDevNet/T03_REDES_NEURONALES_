# Red de dos torres (tarea 5)

_Generado por `src/ajustar_dos_torres.py`. No editar a mano._

## 1. Ajuste: 24 configuraciones × 3 semillas (72 corridas, 64.3 s)

Valores en %, validación (326 usuarios). Ordenadas por HitRate@3 por lugar.

| d | perdida | negativos | usuario | Lugar3_val | Lugar1_val | HitRate10_val | mediana_mejor_epoca | segundos_total | parametros |
|---|---|---|---|---|---|---|---|---|---|
| 2 | bce | 1 | preferencias | 65.2 | 21.1 | 1.7 | 8.0 | 3.3 | 238 |
| 2 | bpr | 1 | preferencias | 65.2 | 21.1 | 1.8 | 3.0 | 2.4 | 238 |
| 2 | bpr | 4 | preferencias | 65.2 | 22.3 | 2.2 | 3.0 | 2.3 | 238 |
| 8 | bpr | 4 | preferencias | 65.0 | 20.8 | 1.6 | 4.0 | 2.5 | 568 |
| 4 | bce | 1 | preferencias | 64.8 | 23.6 | 2.0 | 7.0 | 2.8 | 348 |
| 8 | bce | 1 | preferencias | 64.7 | 20.2 | 1.4 | 10.0 | 3.6 | 568 |
| 4 | bce | 1 | completo | 64.6 | 20.3 | 1.3 | 5.0 | 2.5 | 460 |
| 4 | bpr | 4 | completo | 64.1 | 20.6 | 1.4 | 6.0 | 2.5 | 460 |
| 4 | bpr | 1 | completo | 64.1 | 20.8 | 1.6 | 6.0 | 2.5 | 460 |
| 8 | bce | 4 | completo | 63.8 | 19.1 | 1.2 | 4.0 | 2.7 | 680 |
| 4 | bce | 4 | preferencias | 63.7 | 23.1 | 1.3 | 8.0 | 3.1 | 348 |
| 4 | bpr | 1 | preferencias | 63.5 | 21.7 | 1.6 | 6.0 | 2.7 | 348 |
| 4 | bce | 4 | completo | 63.3 | 22.4 | 1.1 | 5.0 | 2.5 | 460 |
| 8 | bpr | 1 | preferencias | 62.8 | 20.4 | 1.9 | 2.0 | 2.9 | 568 |
| 8 | bce | 1 | completo | 62.7 | 23.1 | 1.7 | 2.0 | 2.2 | 680 |
| 2 | bce | 1 | completo | 62.5 | 21.8 | 1.7 | 9.0 | 3.0 | 350 |
| 8 | bce | 4 | preferencias | 62.5 | 18.8 | 1.6 | 3.0 | 3.1 | 568 |
| 2 | bpr | 1 | completo | 62.4 | 22.2 | 1.8 | 7.0 | 2.7 | 350 |
| 8 | bpr | 4 | completo | 61.9 | 22.0 | 1.6 | 2.0 | 2.4 | 680 |
| 2 | bpr | 4 | completo | 61.6 | 20.7 | 1.8 | 6.0 | 2.7 | 350 |
| 4 | bpr | 4 | preferencias | 61.5 | 22.0 | 1.7 | 3.0 | 2.3 | 348 |
| 2 | bce | 4 | completo | 61.5 | 19.0 | 0.6 | 3.0 | 2.6 | 350 |
| 2 | bce | 4 | preferencias | 61.0 | 21.5 | 1.4 | 7.0 | 2.6 | 238 |
| 8 | bpr | 1 | completo | 61.0 | 21.5 | 1.6 | 2.0 | 2.3 | 680 |

**Elegida:** {'d': 2, 'perdida': 'bce', 'negativos': 1, 'usuario': 'preferencias', 'epocas': 8, 'motivo': 'mayor HitRate@3 por lugar en validación y supera al azar (213 de 326; p-valor = 0.027)', 'configuraciones': 24, 'corridas': 72, 'segundos_ajuste': 64.3}

## 2. Evaluación final: 30 semillas, entrenamiento + validación → prueba (valores en %)

| K | HitRate | Precision | NDCG | Cobertura | HitRate_lugar@3 | HitRate_lugar@1 |
|---|---|---|---|---|---|---|
| 1.0 | 0.11 | 0.11 | 0.11 | 0.31 | 61.84 | 22.18 |
| 3.0 | 0.43 | 0.14 | 0.29 | 0.81 | 61.84 | 22.18 |
| 5.0 | 0.54 | 0.11 | 0.34 | 1.29 | 61.84 | 22.18 |
| 10.0 | 1.11 | 0.11 | 0.52 | 2.47 | 61.84 | 22.18 |
| 20.0 | 2.35 | 0.12 | 0.83 | 4.74 | 61.84 | 22.18 |

## 3. Comparaciones

| usuarios | lugar@3: aciertos / esperados | lugar@3: p-valor corregido (m = 2) | lugar@1: aciertos / esperados | lugar@1: p-valor corregido (m = 2) | ID@10: aciertos / esperados | ID@10: p-valor vs azar | lugar@3: red − popularidad (puntos) | lugar@3: p-valor vs popularidad (pareada) | lugar@3: red − factorización (puntos) | lugar@3: p-valor vs factorización (pareada) | segundos_por_entrenamiento | parametros |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 606 | 375 / 363.6 | 0.366 | 134 / 121.2 | 0.214 | 7 / 6.1 | 0.405 | 3.73 | 0.005 | 1.39 | 0.2 | 0.1 | 238 |

_Tiempo total: 74 s._
