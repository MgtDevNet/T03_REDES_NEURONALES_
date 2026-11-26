# Experimento «completo» del módulo 3

_Generado por `src/train.py` con `modulo3_recomendacion/configs/completo.yaml`. Tiempo total: 100 s._

## 1. Ajuste en validación

### Ajuste de Factorización (BPR) (9 configuraciones, 27 corridas, 25.2 s)

| d | lam | criterio_media | HitRate@10_media | mediana_mejor_epoca | segundos | parametros |
|---|---|---|---|---|---|---|
| 16.0 | 0.01 | 0.0095 | 0.0194 | 5.0 | 2.8297 | 31984.0 |
| 8.0 | 1.0 | 0.0091 | 0.0164 | 7.0 | 3.0115 | 15992.0 |
| 16.0 | 0.1 | 0.009 | 0.0215 | 4.0 | 2.6744 | 31984.0 |
| 16.0 | 1.0 | 0.0087 | 0.0184 | 2.0 | 2.6116 | 31984.0 |
| 32.0 | 1.0 | 0.0076 | 0.0123 | 4.0 | 2.4721 | 63968.0 |
| 8.0 | 0.1 | 0.0068 | 0.0133 | 3.0 | 2.6397 | 15992.0 |
| 8.0 | 0.01 | 0.0063 | 0.0174 | 5.0 | 3.9234 | 15992.0 |
| 32.0 | 0.01 | 0.0062 | 0.0123 | 3.0 | 3.0025 | 63968.0 |
| 32.0 | 0.1 | 0.0052 | 0.0102 | 2.0 | 1.9951 | 63968.0 |

**Elegida:** {'d': 8, 'lam': 1.0}, 7 épocas — ninguna supera al azar en validación (la mejor acierta 6 de 326 frente a 3.3; p-valor = 0.112): se usa la más simple.

### Ajuste de Red de dos torres (24 configuraciones, 72 corridas, 61.9 s)

| d | perdida | negativos | usuario | criterio_media | HitRate@10_media | mediana_mejor_epoca | segundos | parametros |
|---|---|---|---|---|---|---|---|---|
| 2 | bce | 1 | preferencias | 0.6524 | 0.0174 | 8.0 | 3.0257 | 238 |
| 2 | bpr | 1 | preferencias | 0.6524 | 0.0184 | 3.0 | 2.4406 | 238 |
| 2 | bpr | 4 | preferencias | 0.6524 | 0.0225 | 3.0 | 2.298 | 238 |
| 8 | bpr | 4 | preferencias | 0.6503 | 0.0164 | 4.0 | 2.3826 | 568 |
| 4 | bce | 1 | preferencias | 0.6483 | 0.0204 | 7.0 | 2.8053 | 348 |
| 8 | bce | 1 | preferencias | 0.6472 | 0.0143 | 10.0 | 3.4701 | 568 |
| 4 | bce | 1 | completo | 0.6462 | 0.0133 | 5.0 | 2.4689 | 460 |
| 4 | bpr | 4 | completo | 0.6411 | 0.0143 | 6.0 | 2.4683 | 460 |
| 4 | bpr | 1 | completo | 0.6411 | 0.0164 | 6.0 | 2.5107 | 460 |
| 8 | bce | 4 | completo | 0.638 | 0.0123 | 4.0 | 2.5987 | 680 |
| 4 | bce | 4 | preferencias | 0.637 | 0.0133 | 8.0 | 3.0424 | 348 |
| 4 | bpr | 1 | preferencias | 0.635 | 0.0164 | 6.0 | 2.72 | 348 |
| 4 | bce | 4 | completo | 0.6329 | 0.0112 | 5.0 | 2.558 | 460 |
| 8 | bpr | 1 | preferencias | 0.6278 | 0.0194 | 2.0 | 2.3228 | 568 |
| 8 | bce | 1 | completo | 0.6268 | 0.0174 | 2.0 | 2.0501 | 680 |
| 2 | bce | 1 | completo | 0.6247 | 0.0174 | 9.0 | 3.088 | 350 |
| 8 | bce | 4 | preferencias | 0.6247 | 0.0164 | 3.0 | 2.7932 | 568 |
| 2 | bpr | 1 | completo | 0.6237 | 0.0184 | 7.0 | 2.7064 | 350 |
| 8 | bpr | 4 | completo | 0.6186 | 0.0164 | 2.0 | 2.2355 | 680 |
| 2 | bpr | 4 | completo | 0.6155 | 0.0184 | 6.0 | 2.6528 | 350 |
| 4 | bpr | 4 | preferencias | 0.6145 | 0.0174 | 3.0 | 2.158 | 348 |
| 2 | bce | 4 | completo | 0.6145 | 0.0061 | 3.0 | 2.4858 | 350 |
| 2 | bce | 4 | preferencias | 0.6104 | 0.0143 | 7.0 | 2.6363 | 238 |
| 8 | bpr | 1 | completo | 0.6104 | 0.0164 | 2.0 | 2.0046 | 680 |

**Elegida:** {'d': 2, 'perdida': 'bce', 'negativos': 1, 'usuario': 'preferencias'}, 8 épocas — la mejor en validación supera al azar (213 de 326; p-valor = 0.027).

## 2. Prueba: 30 semilla(s) — media, desviación, mínimo y máximo

| modelo | métrica | media (%) | desv. estándar | mínimo | máximo | parámetros |
|---|---|---|---|---|---|---|
| Azar | HitRate@10 | 0.91 | 0.38 | 0.33 | 1.49 | 0 |
| Azar | Precision@10 | 0.09 | 0.04 | 0.03 | 0.15 | 0 |
| Azar | Recall@10 | 0.91 | 0.38 | 0.33 | 1.49 | 0 |
| Azar | NDCG@10 | 0.42 | 0.2 | 0.1 | 0.74 | 0 |
| Azar | Cobertura@10 | 99.7 | 0.19 | 99.1 | 100.0 | 0 |
| Azar | Lugar@1 | 19.32 | 1.68 | 14.19 | 21.62 | 0 |
| Azar | Lugar@3 | 59.68 | 1.95 | 56.11 | 63.7 | 0 |
| Popularidad | HitRate@10 | 1.16 | 0.22 | 0.5 | 1.65 | 0 |
| Popularidad | Precision@10 | 0.12 | 0.02 | 0.05 | 0.17 | 0 |
| Popularidad | Recall@10 | 1.16 | 0.22 | 0.5 | 1.65 | 0 |
| Popularidad | NDCG@10 | 0.46 | 0.09 | 0.19 | 0.62 | 0 |
| Popularidad | Cobertura@10 | 1.6 | 0.0 | 1.6 | 1.6 | 0 |
| Popularidad | Lugar@1 | 16.14 | 0.12 | 16.01 | 16.34 | 0 |
| Popularidad | Lugar@3 | 58.11 | 1.79 | 54.29 | 61.55 | 0 |
| Factorización (BPR) | HitRate@10 | 0.94 | 0.31 | 0.17 | 1.49 | 15992 |
| Factorización (BPR) | Precision@10 | 0.09 | 0.03 | 0.02 | 0.15 | 15992 |
| Factorización (BPR) | Recall@10 | 0.94 | 0.31 | 0.17 | 1.49 | 15992 |
| Factorización (BPR) | NDCG@10 | 0.43 | 0.16 | 0.05 | 0.73 | 15992 |
| Factorización (BPR) | Cobertura@10 | 17.4 | 0.76 | 15.2 | 18.5 | 15992 |
| Factorización (BPR) | Lugar@1 | 20.35 | 1.34 | 18.15 | 24.59 | 15992 |
| Factorización (BPR) | Lugar@3 | 60.45 | 2.36 | 55.28 | 64.69 | 15992 |
| Red de dos torres | HitRate@10 | 1.11 | 0.27 | 0.66 | 1.82 | 238 |
| Red de dos torres | Precision@10 | 0.11 | 0.03 | 0.07 | 0.18 | 238 |
| Red de dos torres | Recall@10 | 1.11 | 0.27 | 0.66 | 1.82 | 238 |
| Red de dos torres | NDCG@10 | 0.52 | 0.17 | 0.29 | 0.86 | 238 |
| Red de dos torres | Cobertura@10 | 2.47 | 0.59 | 1.1 | 3.3 | 238 |
| Red de dos torres | Lugar@1 | 22.18 | 2.62 | 17.33 | 25.74 | 238 |
| Red de dos torres | Lugar@3 | 61.84 | 2.54 | 55.61 | 66.5 | 238 |

## 3. Contra el azar

| modelo | HitRate@10: aciertos / esperados | HitRate@10: p-valor | Lugar@1: aciertos / esperados | Lugar@1: p-valor corregido (m = 2) | Lugar@3: aciertos / esperados | Lugar@3: p-valor corregido (m = 2) |
|---|---|---|---|---|---|---|
| Azar | 6 / 6.1 | 0.567 | 117 / 121.2 | 1.0 | 362 / 363.6 | 1.0 |
| Popularidad | 7 / 6.1 | 0.405 | 98 / 121.2 | 1.0 | 352 / 363.6 | 1.0 |
| Factorización (BPR) | 6 / 6.1 | 0.567 | 123 / 121.2 | 0.887 | 366 / 363.6 | 0.877 |
| Red de dos torres | 7 / 6.1 | 0.405 | 134 / 121.2 | 0.214 | 375 / 363.6 | 0.366 |

## 4. Red frente a los demás (pareada por usuario)

| comparación | métrica | diferencia (puntos) | p-valor (pareada) |
|---|---|---|---|
| Red de dos torres − Azar | Lugar@3 | 2.16 | 0.045 |
| Red de dos torres − Popularidad | Lugar@3 | 3.73 | 0.005 |
| Red de dos torres − Factorización (BPR) | Lugar@3 | 1.39 | 0.2 |
