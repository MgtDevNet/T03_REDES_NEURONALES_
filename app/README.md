# Aplicación web

Integra los tres módulos usando los **modelos guardados, sin reentrenar**, y debe poder
levantarse localmente con un solo comando y estar publicada (link requerido).

```
app/
├── pages/    # Una página por módulo (demanda, conducción, recomendación)
├── utils/    # Carga de modelos y funciones compartidas
└── assets/   # Logos, imágenes de ejemplo, estilos
```

La app solo importa las funciones de inferencia de cada módulo:

| Módulo | Importa de                              | Funciones                                   |
|--------|-----------------------------------------|---------------------------------------------|
| 1      | `modulo1_demanda/src/predict.py`, `flota.py` | `predecir`, `dimensionar`                   |
| 2      | `modulo2_conduccion/src/predict.py`     | `predecir`, `gradcam`                       |
| 3      | `modulo3_recomendacion/src/predict.py`  | `recomendar_usuario`, `recomendar_perfil_nuevo`, `historial` |

Debe incluir enlaces al reporte técnico, al video promocional y a los videos de
contribución individual.

## Cómo levantarla (un solo comando)

Desde la raíz del repositorio, con el entorno activado:

```bash
python -m pip install -r app/requirements.txt   # solo la primera vez
streamlit run app/Inicio.py
```

Pruebas sin navegador: `python app/probar_app.py`.

## Estado (2026-11-26)

| Página | Estado |
|---|---|
| `Inicio.py` | ✅ Descripción de los tres módulos y enlaces (repositorio; faltan reporte y videos). |
| `pages/1_Demanda_y_operacion.py` | 🛠️ Muestra «en construcción» hasta que exista `modulo1_demanda/src/predict.py`. |
| `pages/2_Conduccion_distractiva.py` | 🛠️ Muestra «en construcción» hasta que exista `modulo2_conduccion/src/predict.py`. |
| `pages/3_Recomendacion_de_destinos.py` | ✅ Cliente existente (historial + K lugares) y cliente nuevo (formulario de preferencias), con el modelo guardado. |

Las páginas de los módulos 1 y 2 detectan solas cuándo existe su `predict.py` con las funciones
del contrato (`utils/modulos.py`); falta entonces escribir su interfaz. Cada módulo agrega sus
dependencias a `app/requirements.txt`.
