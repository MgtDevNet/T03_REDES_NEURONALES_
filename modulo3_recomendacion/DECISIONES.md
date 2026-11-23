# Bitácora de decisiones — Módulo 3 (recomendación)

Cada paso registra **qué se hizo, por qué, qué alternativas se descartaron y qué falta decidir**.
Sirve para dos cosas: aprender, y redactar después las secciones «Datos», «Partición» y
«Cacería de la alucinación» del reporte.

La teoría de apoyo está en [`teoria/01_fundamentos_sistemas_recomendacion.md`](teoria/01_fundamentos_sistemas_recomendacion.md).

---

## Paso 1 — Obtener los datos e inventariarlos (2026-11-23)

### Qué se hizo
1. El zip de Kaggle (*Travel Recommendation Dataset*, Aman Mehra) se descargó a mano en
   `data/raw/archive.zip` y se descomprimió en esa misma carpeta. `data/raw/` está en
   `.gitignore`: **los datos crudos no se suben al repo**. Cada integrante los descarga desde el
   enlace del README.
2. Se creó un entorno virtual `T03/.venv` (también ignorado por git) con `pandas` y `numpy`.
3. Se revisó cada tabla: tamaño, valores únicos, nulos, distribución de cada columna y
   coherencia entre tablas.

### Archivos del dataset

| Archivo | Filas | Columnas | Qué representa |
|---|---|---|---|
| `Expanded_Destinations.csv` | 1000 | DestinationID, Name, State, Type, Popularity, BestTimeToVisit | catálogo de destinos |
| `Final_Updated_Expanded_Users.csv` | 999 | UserID, Name, Email, Preferences, Gender, NumberOfAdults, NumberOfChildren | usuarios |
| `Final_Updated_Expanded_Reviews.csv` | 999 | ReviewID, DestinationID, UserID, Rating (1–5), ReviewText | reseñas |
| `Final_Updated_Expanded_UserHistory.csv` | 999 | HistoryID, UserID, DestinationID, VisitDate, ExperienceRating (1–5) | visitas |

No hay nulos, ni pares usuario–destino duplicados dentro de cada tabla, ni IDs que apunten a
usuarios o destinos inexistentes.

### Hallazgo principal: los datos son sintéticos y casi aleatorios

El enunciado pide decir si los datos son reales o sintéticos «según la documentación de la
fuente».

**Lo que dice la fuente** (página de Kaggle, consultada el 2026-11-23 a través de la API pública
de Kaggle, porque la página web no carga sin JavaScript):
- el autor dice que el dataset «was generated for educational and research purposes» (para
  hackatones y proyectos académicos);
- que los perfiles de usuario se «enriquecieron» con diversidad de género;
- **no** describe ninguna fuente primaria ni ningún método de recolección; hay una sola versión
  (2024-01-23);
- las licencias se contradicen: los metadatos dicen MIT; la descripción, CC0 y «uso no comercial».

**Lo que muestran los datos** confirma que son **sintéticos**, generados repitiendo unos pocos
valores de forma cíclica:

| Evidencia | Detalle |
|---|---|
| **Solo 5 destinos reales** | Los 1000 `DestinationID` son 5 lugares (Taj Mahal, Goa Beaches, Jaipur City, Kerala Backwaters, Leh Ladakh) repetidos 200 veces cada uno. El nombre lo determina `(DestinationID−1) mod 5`. `Name`, `State`, `Type` y `BestTimeToVisit` son la misma información (5 combinaciones). Solo `Popularity` cambia entre copias, y es un número aleatorio entre 7,5 y 9,5. |
| **Usuarios cíclicos** | `Preferences` toma solo 3 valores y la determina `(UserID−1) mod 3`. Hay solo 20 nombres y 20 correos distintos para 999 usuarios. |
| **Fechas cíclicas** | `VisitDate` toma solo 3 fechas (2024-01-01, 2024-02-15, 2024-03-20), determinadas por `(HistoryID−1) mod 3`. |
| **Texto cíclico** | `ReviewText` toma 3 frases, determinadas por `(ReviewID−1) mod 3`, y **no** tienen relación con el destino: «Loved the beaches!» aparece igual de seguido en el Taj Mahal que en Leh Ladakh. |
| **Calificaciones uniformes** | `Rating` y `ExperienceRating` se reparten casi por igual entre 1 y 5. La correlación entre `ExperienceRating` y `Popularity` es 0,03. |
| **Preferencias sin relación con las visitas** | Solo el 41,7 % de las visitas va a un destino cuyo tipo coincide con las preferencias del usuario. Cada perfil tiene 2 de los 5 tipos, así que **al azar se esperaría un 40 %**. |
| **Reseñas e historial sin solapamiento** | No hay ni un par (usuario, destino) que aparezca en las dos tablas. |

**Qué implica (para el reporte):** si las interacciones se generaron al azar, **no hay una
señal de preferencia que aprender**. Es de esperar que ningún modelo, ni siquiera la red, supere
con claridad a la popularidad o al azar, y que las métricas sean muy bajas. Eso **no** es un error
nuestro: es una conclusión válida, y el trabajo debe mostrarlo con evidencia (comparar contra un
recomendador aleatorio y contra las 30 semillas). Es además un candidato natural para la sección
**«Cacería de la alucinación»**: la página de Kaggle presenta los datos como un recomendador de
viajes de India, pero los datos no lo sostienen.

### Cifras descriptivas (por tabla, contando los IDs tal como vienen)

| | Reseñas | Historial | Unión |
|---|---|---|---|
| Interacciones | 999 | 999 | 1998 |
| Usuarios con ≥ 1 interacción | 624 | 642 | 858 |
| Destinos (ID) con ≥ 1 interacción | 651 | 638 | 866 |
| Densidad (999 usuarios × 1000 destinos) | 0,10 % | 0,10 % | 0,20 % |
| Usuarios con exactamente 1 interacción | 355 (57 %) | 384 (60 %) | — |

Si se agrupa por **lugar real** (5 destinos), la unión tiene 1644 pares usuario–lugar distintos,
con entre 376 y 422 interacciones por lugar: casi uniforme.

### Decisión pendiente (para el próximo paso): ¿qué es un «destino»?

Teoría de apoyo: sección 2.1 (qué es una interacción) y 2.2 (granularidad del ítem).

**Cifras de cada opción** (usando la unión de reseñas + historial, 1998 interacciones):

| Aspecto | A. 1000 IDs | B. 5 lugares |
|---|---|---|
| Ítems en el catálogo ($n$) | 1000 | 5 |
| Ítems sin ninguna interacción | 134 | 0 |
| Ítems con 1 sola interacción | 271 | 0 |
| Interacciones del ítem más popular | 7 | 422 |
| Lugares distintos por usuario | — | 1: 322 usuarios, 2: 339, 3: 148, 4: 45, 5: 4 |
| Usuarios con interacciones repetidas al agrupar | 0 | 266 (visitaron «el mismo lugar» con dos IDs) |
| K máximo con sentido | 10 o más sin problema | ≤ 4 (y 0 candidatos para quien ya visitó los 5) |
| HitRate@10 de un recomendador **al azar** | ≈ 1 % | 100 % con K = 5 (trivial) |
| Si se oculta 1 interacción por usuario para prueba, ¿el ítem oculto aparece en entrenamiento? | sí, en el 73 % de los casos | siempre |

Cómo leer estas cifras:
- En **A**, el problema es un ranking «de verdad» (1000 candidatos, el azar casi nunca acierta),
  pero cada ítem tiene muy pocos datos (máximo 7 interacciones), así que el CF tiene poco que
  aprender. Si un modelo supera al azar, se puede medir.
- En **B**, cada lugar tiene muchos datos, pero solo hay 5 ítems: casi no queda nada que
  recomendar a cada usuario, las métricas top-K son triviales y un t-SNE de 5 puntos no dice nada.

| Opción | Ventaja | Desventaja |
|---|---|---|
| **A. Los 1000 `DestinationID` como ítems distintos** | Respeta el dataset tal como se publicó; hay suficientes ítems para que Precision/Recall@K, la cobertura y la cola larga tengan sentido. | Recomendar «Taj Mahal #178» frente a «Taj Mahal #183» no tiene sentido de negocio; casi todos los ítems tienen 0–2 interacciones. |
| **B. Colapsar a los 5 lugares reales** | Tiene sentido de negocio. | Con 5 ítems, un top-K con K ≥ 5 recomienda todo; las métricas, la cobertura y los embeddings quedan degenerados. |
| **C. Modelar con los 1000 IDs, pero analizar y reportar también por lugar** | Cumple el enunciado y a la vez documenta la limitación. | Algo más de trabajo de reporte. |

Recomendación: **C**. Se usan los IDs para los modelos y las métricas, y en el reporte se explica
que equivalen a 5 lugares con atributos repetidos. Por ejemplo, se puede mostrar a qué lugar
corresponden las recomendaciones de cada usuario y medir la concentración también por lugar
(pregunta de negocio 2).

Otras decisiones que vienen después: qué tabla(s) cuentan como interacción (reseñas, historial
o la unión), si la calificación se usa como umbral de relevancia, el valor de K y la estrategia de
partición (las 3 fechas no alcanzan para una partición temporal útil).
