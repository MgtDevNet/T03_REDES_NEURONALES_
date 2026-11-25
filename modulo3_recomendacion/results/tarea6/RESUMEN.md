# Tarea 6 — grupos de usuarios y 30 semillas

_Generado por `src/evaluar_tarea6.py`. No editar a mano._

## 1. Medias generales (deben coincidir con las Tablas 16 y 19) y desviación entre semillas (%)

| modelo | Lugar@3 | Lugar@1 | HitRate@10 | Lugar@3 DE | HitRate@10 DE |
|---|---|---|---|---|---|
| Azar | 59.68 | 19.32 | 0.91 | 1.95 | 0.38 |
| Popularidad | 58.11 | 16.14 | 1.16 | 1.79 | 0.22 |
| Factorización (BPR) | 60.45 | 20.35 | 0.94 | 2.36 | 0.31 |
| Red de dos torres | 61.84 | 22.18 | 1.11 | 2.54 | 0.27 |

## 2. Por grupo: usuarios con pocas (1) o muchas (≥ 2) interacciones para aprender

Acierto medio por usuario sobre las 30 semillas. p-valor: binomial unilateral contra el 60 % del azar (sin corregir; es un análisis descriptivo por grupo).

| modelo | grupo | usuarios | Lugar@3 (%) | Lugar@1 (%) | HitRate@10 (%) | Lugar@3: aciertos / esperados | Lugar@3: p-valor vs azar | HitRate@10 esperado del azar (%) |
|---|---|---|---|---|---|---|---|---|
| Azar | muchas (≥ 2) | 326 | 59.2 | 18.8 | 0.77 | 193 / 195.6 | 0.638 | 1.0 |
| Azar | pocas (1) | 280 | 60.2 | 19.9 | 1.08 | 169 / 168.0 | 0.477 | 1.0 |
| Popularidad | muchas (≥ 2) | 326 | 58.6 | 17.7 | 1.32 | 191 / 195.6 | 0.719 | 1.0 |
| Popularidad | pocas (1) | 280 | 57.5 | 14.3 | 0.96 | 161 / 168.0 | 0.82 | 1.0 |
| Factorización (BPR) | muchas (≥ 2) | 326 | 61.1 | 20.6 | 1.05 | 199 / 195.6 | 0.373 | 1.0 |
| Factorización (BPR) | pocas (1) | 280 | 59.7 | 20.0 | 0.81 | 167 / 168.0 | 0.574 | 1.0 |
| Red de dos torres | muchas (≥ 2) | 326 | 60.5 | 20.8 | 1.22 | 197 / 195.6 | 0.461 | 1.0 |
| Red de dos torres | pocas (1) | 280 | 63.4 | 23.8 | 0.99 | 178 / 168.0 | 0.123 | 1.0 |

Figura: `figures/tarea6_semillas.png`
