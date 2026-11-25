# Modelos de referencia: azar y popularidad (tarea 4)

_Generado por `src/referencias.py`. No editar a mano._

Media de 30 semillas. IC 95 % = media ± t(0,975; 29) · DE / √30. Recall@K = HitRate@K porque cada usuario tiene un solo destino escondido.

## 1. Resumen con K = 10

| escenario | modelo | HitRate@10 (%) | IC 95 % | NDCG@10 (%) | Precision@10 (%) | Cobertura@10 (%) | HitRate_lugar@3 (%) | HitRate_lugar@1 (%) |
|---|---|---|---|---|---|---|---|---|
| ajuste | Azar | 1.06 | 0.81–1.32 | 0.48 | 0.11 | 96.3 | 60.11 | 19.73 |
| ajuste | Popularidad | 0.99 | 0.83–1.15 | 0.33 | 0.1 | 1.8 | 62.31 | 20.9 |
| final | Azar | 0.91 | 0.77–1.06 | 0.42 | 0.09 | 99.7 | 59.68 | 19.32 |
| final | Popularidad | 1.16 | 1.07–1.24 | 0.46 | 0.12 | 1.6 | 58.11 | 16.14 |

## 2. HitRate según K

| escenario | modelo | HitRate@1 (%) | HitRate@3 (%) | HitRate@5 (%) | HitRate@10 (%) | HitRate@20 (%) |
|---|---|---|---|---|---|---|
| ajuste | Azar | 0.09 | 0.32 | 0.56 | 1.06 | 2.21 |
| ajuste | Popularidad | 0.0 | 0.0 | 0.15 | 0.99 | 2.61 |
| final | Azar | 0.1 | 0.29 | 0.47 | 0.91 | 1.98 |
| final | Popularidad | 0.01 | 0.26 | 0.51 | 1.16 | 2.28 |

## 3. ¿Acierta más que el azar? (K = 10, prueba binomial unilateral)

El IC 95 % de la sección 1 solo mide la variación entre semillas, no que los usuarios evaluados sean una muestra. Esta prueba sí lo tiene en cuenta.

| escenario | modelo | usuarios | aciertos (media de 30 semillas) | esperados por azar | error estándar por muestreo (pp) | p-valor (H1: acierta más que el azar) | lugar@3: aciertos / esperados | lugar@3: p-valor corregido (m = 2) | lugar@1: aciertos / esperados | lugar@1: p-valor corregido (m = 2) |
|---|---|---|---|---|---|---|---|---|---|---|
| ajuste | Azar | 326 | 3 | 3.3 | 0.55 | 0.635 | 196 / 195.6 | 1.0 | 64 / 65.2 | 1.0 |
| ajuste | Popularidad | 326 | 3 | 3.3 | 0.55 | 0.635 | 203 / 195.6 | 0.436 | 68 / 65.2 | 0.741 |
| final | Azar | 606 | 6 | 6.1 | 0.4 | 0.567 | 362 / 363.6 | 1.0 | 117 / 121.2 | 1.0 |
| final | Popularidad | 606 | 7 | 6.1 | 0.4 | 0.405 | 352 / 363.6 | 1.0 | 98 / 121.2 | 1.0 |

_CSV: `results/tarea4/referencias_resumen.csv` (todas las métricas y K) y `referencias_por_semilla.csv`._
