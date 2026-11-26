# Experimento «demo» del módulo 3

_Generado por `src/train.py` con `modulo3_recomendacion/configs/demo.yaml`. Tiempo total: 3 s._

## 1. Ajuste en validación

### Factorización (BPR): configuración fija {'d': 8, 'lam': 1.0}, 7 épocas (sin búsqueda)

### Red de dos torres: configuración fija {'d': 2, 'perdida': 'bce', 'negativos': 1, 'usuario': 'preferencias'}, 8 épocas (sin búsqueda)

## 2. Prueba: 5 semilla(s) — media, desviación, mínimo y máximo

| modelo | métrica | media (%) | desv. estándar | mínimo | máximo | parámetros |
|---|---|---|---|---|---|---|
| Azar | HitRate@10 | 0.79 | 0.44 | 0.33 | 1.49 | 0 |
| Azar | Precision@10 | 0.08 | 0.04 | 0.03 | 0.15 | 0 |
| Azar | Recall@10 | 0.79 | 0.44 | 0.33 | 1.49 | 0 |
| Azar | NDCG@10 | 0.34 | 0.24 | 0.1 | 0.73 | 0 |
| Azar | Cobertura@10 | 99.72 | 0.13 | 99.5 | 99.8 | 0 |
| Azar | Lugar@1 | 20.4 | 1.46 | 18.65 | 21.62 | 0 |
| Azar | Lugar@3 | 59.41 | 1.56 | 57.76 | 61.39 | 0 |
| Popularidad | HitRate@10 | 1.09 | 0.09 | 0.99 | 1.16 | 0 |
| Popularidad | Precision@10 | 0.11 | 0.01 | 0.1 | 0.12 | 0 |
| Popularidad | Recall@10 | 1.09 | 0.09 | 0.99 | 1.16 | 0 |
| Popularidad | NDCG@10 | 0.44 | 0.05 | 0.38 | 0.5 | 0 |
| Popularidad | Cobertura@10 | 1.6 | 0.0 | 1.6 | 1.6 | 0 |
| Popularidad | Lugar@1 | 16.17 | 0.12 | 16.01 | 16.34 | 0 |
| Popularidad | Lugar@3 | 58.84 | 2.13 | 55.78 | 61.55 | 0 |
| Factorización (BPR) | HitRate@10 | 1.09 | 0.38 | 0.66 | 1.49 | 15992 |
| Factorización (BPR) | Precision@10 | 0.11 | 0.04 | 0.07 | 0.15 | 15992 |
| Factorización (BPR) | Recall@10 | 1.09 | 0.38 | 0.66 | 1.49 | 15992 |
| Factorización (BPR) | NDCG@10 | 0.49 | 0.1 | 0.36 | 0.63 | 15992 |
| Factorización (BPR) | Cobertura@10 | 17.64 | 0.66 | 16.7 | 18.3 | 15992 |
| Factorización (BPR) | Lugar@1 | 19.64 | 0.5 | 19.14 | 20.46 | 15992 |
| Factorización (BPR) | Lugar@3 | 59.6 | 2.73 | 55.45 | 62.71 | 15992 |
| Red de dos torres | HitRate@10 | 1.19 | 0.32 | 0.66 | 1.49 | 238 |
| Red de dos torres | Precision@10 | 0.12 | 0.03 | 0.07 | 0.15 | 238 |
| Red de dos torres | Recall@10 | 1.19 | 0.32 | 0.66 | 1.49 | 238 |
| Red de dos torres | NDCG@10 | 0.53 | 0.11 | 0.39 | 0.66 | 238 |
| Red de dos torres | Cobertura@10 | 2.58 | 0.47 | 2.2 | 3.3 | 238 |
| Red de dos torres | Lugar@1 | 22.97 | 1.59 | 21.78 | 25.74 | 238 |
| Red de dos torres | Lugar@3 | 61.25 | 2.73 | 57.76 | 65.35 | 238 |

## 3. Contra el azar

| modelo | HitRate@10: aciertos / esperados | HitRate@10: p-valor | Lugar@1: aciertos / esperados | Lugar@1: p-valor corregido (m = 2) | Lugar@3: aciertos / esperados | Lugar@3: p-valor corregido (m = 2) |
|---|---|---|---|---|---|---|
| Azar | 5 / 6.1 | 0.726 | 124 / 121.2 | 0.808 | 360 / 363.6 | 1.0 |
| Popularidad | 7 / 6.1 | 0.405 | 98 / 121.2 | 1.0 | 357 / 363.6 | 1.0 |
| Factorización (BPR) | 7 / 6.1 | 0.405 | 119 / 121.2 | 1.0 | 361 / 363.6 | 1.0 |
| Red de dos torres | 7 / 6.1 | 0.405 | 139 / 121.2 | 0.082 | 371 / 363.6 | 0.569 |

## 4. Red frente a los demás (pareada por usuario)

| comparación | métrica | diferencia (puntos) | p-valor (pareada) |
|---|---|---|---|
| Red de dos torres − Azar | Lugar@3 | 1.85 | 0.262 |
| Red de dos torres − Popularidad | Lugar@3 | 2.41 | 0.148 |
| Red de dos torres − Factorización (BPR) | Lugar@3 | 1.65 | 0.333 |
