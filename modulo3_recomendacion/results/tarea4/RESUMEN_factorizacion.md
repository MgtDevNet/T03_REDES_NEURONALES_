# Factorización de matrices con BPR (tarea 4)

_Generado por `src/ajustar_factorizacion.py`. No editar a mano._

## 1. Ajuste: 9 configuraciones × 3 semillas (27 corridas, 23.4 s)

Valores en %. Ordenadas por NDCG@10 medio en validación (326 usuarios).

| d | lambda | NDCG_val_media | HitRate_val_media | HitRate_val_de | mediana_mejor_epoca | segundos_total | parametros |
|---|---|---|---|---|---|---|---|
| 16.0 | 0.01 | 0.95 | 1.94 | 0.77 | 5.0 | 2.5 | 31984.0 |
| 8.0 | 1.0 | 0.91 | 1.64 | 0.64 | 7.0 | 3.0 | 15992.0 |
| 16.0 | 0.1 | 0.9 | 2.15 | 0.31 | 4.0 | 2.2 | 31984.0 |
| 16.0 | 1.0 | 0.87 | 1.84 | 0.53 | 2.0 | 2.2 | 31984.0 |
| 32.0 | 1.0 | 0.76 | 1.23 | 0.31 | 4.0 | 2.4 | 63968.0 |
| 8.0 | 0.1 | 0.68 | 1.33 | 0.35 | 3.0 | 2.3 | 15992.0 |
| 8.0 | 0.01 | 0.63 | 1.74 | 0.64 | 5.0 | 3.8 | 15992.0 |
| 32.0 | 0.01 | 0.62 | 1.23 | 0.0 | 3.0 | 2.9 | 63968.0 |
| 32.0 | 0.1 | 0.52 | 1.02 | 0.18 | 2.0 | 2.0 | 63968.0 |

**Elegida:** d = 8, λ = 1, 7 épocas. Motivo: ninguna configuración supera al azar en validación (la mejor, d = 16, λ = 0.01, acierta 6 de 326 frente a 3.3 esperados; p-valor = 0.112), así que se elige la más simple.

## 2. Evaluación final: 30 semillas, entrenamiento + validación → prueba

| K | HitRate (%) | IC 95 % entre semillas | Precision (%) | NDCG (%) | Cobertura (%) | HitRate_lugar@3 (%) | HitRate_lugar@1 (%) |
|---|---|---|---|---|---|---|---|
| 1 | 0.08 | 0.04–0.13 | 0.08 | 0.08 | 3.98 | 60.45 | 20.35 |
| 3 | 0.3 | 0.22–0.37 | 0.1 | 0.2 | 7.97 | 60.45 | 20.35 |
| 5 | 0.48 | 0.38–0.59 | 0.1 | 0.28 | 10.95 | 60.45 | 20.35 |
| 10 | 0.94 | 0.83–1.06 | 0.09 | 0.43 | 17.4 | 60.45 | 20.35 |
| 20 | 2.06 | 1.87–2.24 | 0.1 | 0.71 | 28.23 | 60.45 | 20.35 |

## 3. Comparaciones con K = 10 (los usuarios como muestra)

| usuarios | aciertos_fm_media | aciertos_popularidad_media | esperados_por_azar | p_valor_vs_azar | usuarios_donde_fm_acierta_mas | usuarios_donde_popularidad_acierta_mas | p_valor_vs_popularidad_pareada | segundos_por_entrenamiento | lugar@3: aciertos / esperados | lugar@3: p-valor corregido (m = 2) | lugar@1: aciertos / esperados | lugar@1: p-valor corregido (m = 2) | lugar@3: p-valor vs popularidad (pareada) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 606 | 5.7 | 7.0 | 6.1 | 0.567 | 133 | 11 | 0.565 | 0.03 | 366 / 363.6 | 0.877 | 123 / 121.2 | 0.887 | 0.005 |

_Tiempo total: 28 s. Modelo guardado (semilla 0): `models/factorizacion_bpr.npz`._
