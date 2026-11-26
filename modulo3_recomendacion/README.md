# Módulo 3 — Recomendación de destinos de viaje

Datos: *Travel Recommendation Dataset* (India Travel Recommender System Dataset),
Aman Mehra, Kaggle: https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset

Ver el enunciado completo en `docs/instrucciones_trabajo_3.pdf` (páginas 5–7).

## Datos

Los 4 CSV originales (≈ 178 KB) están versionados en `data/raw/`, copiados sin cambios del zip
de Kaggle (Mehra, 2024, versión 1 del 23-ene-2024), para que todo corra sin internet. Se agregaron
con `git add -f` porque el `.gitignore` del equipo excluye `data/raw/` en general.
`src/preprocesar.py` genera `data/processed/` a partir de ellos.

## Entorno

Python **3.12** con las mismas versiones que el módulo 2. Desde la raíz del repo `T03`:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r modulo3_recomendacion/requirements.txt
```

Para el notebook de experimentos (`notebooks/experimentos_modulo_3.ipynb`) hace falta además un
kernel de Jupyter: `.venv/bin/python -m pip install ipykernel` (o abrirlo con VS Code eligiendo el
intérprete de `.venv`). Ahí se cambian los hiperparámetros y se comparan los 4 modelos.

Orden de los scripts (cada uno regenera sus tablas en `results/`):
`inventario.py` → `preprocesar.py` → `descriptivo.py` → `evidencia_tarea4.py` →
`probar_evaluacion.py` → `referencias.py`.

## Lo que este módulo debe entregar para la aplicación web

La app debe permitir: **elegir un usuario existente o crear un perfil nuevo (con un listado de
preferencias para elegir) → ver sus K destinos recomendados junto con su historial**.

### 1. Modelos guardados (`models/`)
- [ ] Recomendador neuronal final (NCF o dos torres, con al menos una variable de contenido).
- [ ] Referencias: popularidad y filtrado colaborativo clásico (KNN o factorización de matrices).
- [ ] Codificadores de IDs de usuarios/destinos y de variables de contenido.

### 2. Funciones de inferencia (`src/predict.py`)
✅ Implementadas y probadas (`src/probar_predict.py`). Usan la red guardada, no reentrenan.

```python
from modulo3_recomendacion.src.predict import recomendar_usuario, recomendar_perfil_nuevo, historial

recomendar_usuario(user_id, k=3)            # k = número de LUGARES (1–5); por defecto 3 (caso de uso)
recomendar_perfil_nuevo({"preferencias": ["Beach", "Historical"],   # obligatorio
                         "genero": "Female", "adultos": 2, "ninos": 0}, k=3)  # opcionales
historial(user_id)
```
- Devuelven lugares (no IDs): columnas `posicion, lugar, tipo, estado, mejor_epoca, destination_id,
  puntaje, ya_visitado, mensaje` («Ya lo visitaste, ¿vuelves?»).
- Las opciones del formulario están en `data/processed/preferencias.json`.
- Con combinaciones de preferencias que no están en los datos (p. ej. Beach + Nature) el modelo
  extrapola y el resultado puede no tener sentido: conviene avisarlo en la app.

### 3. Datos procesados (`data/processed/`)
- [ ] Usuarios, destinos (con sus atributos para mostrar en la app) e interacciones.
- [ ] Particiones de entrenamiento, validación y prueba.
- [ ] Lista de valores posibles de preferencias (para el formulario de perfil nuevo).

### 4. Resultados para el reporte (`results/`, `figures/`)
- [ ] Precision, Recall y NDCG@K, cobertura, desempeño por grupo de usuarios (pocas/muchas interacciones).
- [ ] Resultados de las 30 semillas; recomendaciones para ≥ 5 usuarios con perfiles distintos.
- [ ] Figuras mínimas del módulo + figuras comunes; t-SNE/UMAP de los embeddings de destinos.
- [ ] Respuestas a las 3 preguntas de negocio.
- [ ] Al menos un hallazgo de «Cacería de la alucinación» documentado.
