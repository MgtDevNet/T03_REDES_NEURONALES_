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
| **Calificaciones uniformes** | `Rating` y `ExperienceRating` se reparten casi por igual entre 1 y 5 (chi-cuadrado de uniformidad: p = 0,66 y p = 0,12). Su correlación con `Popularity` es 0,04 (reseñas) y 0,03 (historial), ambas no significativas. |
| **Preferencias casi sin relación con las visitas** ⚠️ *corregido* | En la unión, el 42,3 % de las interacciones va a un tipo de destino que el usuario dice preferir, frente al **40 % esperado por azar**. La diferencia es **pequeña pero estadísticamente significativa** (permutación p = 0,019; chi-cuadrado p = 0,019). Por perfil: Nature/Adventure 45,4 %, Beaches/Historical 43,0 %, City/Historical 38,6 % (por debajo del azar). |
| **Reseñas e historial sin solapamiento** | No hay ni un par (usuario, destino) que aparezca en las dos tablas. |

> ⚠️ **Corrección (2026-11-23).** La primera versión de esta tabla decía que las preferencias
> **no** tenían relación con las visitas. Esa afirmación se basaba solo en el historial
> (41,7 %, p = 0,14). Al rehacer el análisis con la unión y con tres pruebas en el script, se
> encontró una relación **débil pero real** (+2,3 puntos sobre el azar). La conclusión correcta
> es «la señal de preferencia es muy débil», no «no hay señal». Esto también vale para el
> informe: una afirmación descriptiva necesita su prueba.

**Qué implica (para el reporte):** las interacciones son casi aleatorias, así que **la señal de
preferencia que se puede aprender es muy débil**. Es de esperar que ningún modelo supere por
mucho a la popularidad o al azar, y que las métricas sean muy bajas. Si la red supera un poco a
la popularidad, una explicación posible es que aprovecha esa débil relación entre preferencias
y tipo de destino. Eso **no** es un error
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

### ✅ Decisión tomada (2026-11-23): opción C

- **Para los modelos y las métricas**, cada `DestinationID` (1000) es un ítem distinto.
- **Para el análisis y el reporte**, cada recomendación se traduce también a su lugar real (5),
  y la concentración se mide en ambos niveles.
- **Por qué:** es la única opción con la que las métricas top-K, la cobertura y los embeddings
  tienen sentido (con 5 ítems todo es trivial), sin esconder la limitación del dataset, que se
  documenta y se discute.
- **Descartadas:** A pura (oculta que hay solo 5 lugares) y B (métricas degeneradas, K ≤ 4).

---

## Paso 2 — ¿Qué cuenta como interacción? (2026-11-23)

Teoría de apoyo: sección 2.1 (qué es una interacción) y sección 3 (explícita frente a implícita).

### Las preguntas
1. ¿Qué tabla se usa: reseñas, historial o la unión de ambas?
2. ¿Qué se hace con la calificación (1–5)? ¿Se ignora (toda interacción es positiva) o se usa como
   **umbral** (solo es positiva si r ≥ 4, o si r ≥ 3)?

### Dato clave para decidir
Para **evaluar** a un usuario se necesitan **al menos 2 interacciones**: una para entrenar (que el
modelo lo conozca) y otra que se esconde para prueba. Los usuarios con 1 sola interacción solo sirven
para entrenar, o para simular usuarios nuevos.

| Definición | Interacciones | Usuarios | Usuarios con 1 | **Usuarios con ≥ 2 (evaluables)** |
|---|---|---|---|---|
| Solo reseñas | 999 | 624 | 355 | 269 |
| Solo historial | 999 | 642 | 384 | 258 |
| **Unión** | **1998** | **858** | 252 | **606** |
| Unión, solo r ≥ 3 | 1167 | 675 | 351 | 324 |
| Unión, solo r ≥ 4 | 781 | 534 | 353 | 181 |

Otros hechos:
- La unión no genera duplicados: ningún par (usuario, destino) aparece en las dos tablas.
- 408 usuarios tienen interacciones en las dos tablas, así que al unirlas sus historiales se
  vuelven más completos.
- Las calificaciones son casi uniformes (paso 1) y no se relacionan con nada, así que filtrar por
  calificación **no limpia ruido: solo quita datos**.

### Recomendación
- **Unión de reseñas + historial**, y **toda interacción cuenta como positiva** ($y_{ui}=1$), sin
  umbral.
  - Así hay más del doble de usuarios evaluables (606 frente a ~260 con una sola tabla, y 181 con
    r ≥ 4).
  - La interpretación es «el usuario mostró interés en el destino: lo visitó o lo reseñó». Que
    haya visitado un lugar con mala experiencia sigue siendo una señal de interés en ese
    destino.
- Se guarda una columna `fuente` (`reseña`/`historial`) y la calificación, por si después se
  quieren usar como peso o para un NDCG graduado, o para comparar resultados por fuente.
- **Lo que se pierde:** no se distingue «le gustó» de «no le gustó». Con datos reales eso sería
  una pérdida importante; aquí la calificación es ruido. Se discute en el informe.

---

## Resumen de decisiones tomadas (pasos 1 y 2)

> Redacción de las decisiones acordada con el equipo. Cada una cita la tabla que la respalda.
> Todas las tablas se regeneran con `src/inventario.py` y quedan en
> [`results/inventario/RESUMEN.md`](results/inventario/RESUMEN.md).

### Decisión 1 — Destinos: 1000 IDs para el modelo, 5 lugares para el cliente (opción C)
Aunque el dataset trae 1000 `DestinationID`, en realidad son solo 5 lugares (Taj Mahal, Goa
Beaches, Jaipur City, Kerala Backwaters y Leh Ladakh), cada uno repetido 200 veces
(**tablas 2 y 3**). Si trabajáramos solo con los 5 lugares, casi todo lo que pide el enunciado
dejaría de tener sentido: un top-K solo podría tener K entre 1 y 4, la cobertura saldría siempre en
100 %, no habría cola larga y la proyección de embeddings tendría apenas 5 puntos (**tabla 9**). Por
eso el modelo y todas las métricas tratan cada ID como un ítem distinto: así hay 1000 candidatos y
se puede medir de verdad si un modelo ordena mejor que el azar. Como para el negocio lo que importa
es el lugar, al cliente se le muestran lugares (tomando el mejor puntaje de cada uno), y en el
informe se analiza también la concentración por lugar y se explica que los 1000 IDs equivalen a 5
lugares repetidos.

> *Nota de implementación:* «mostrar lugares tomando el mejor puntaje de cada uno» significa que
> la app calcula el puntaje de los 1000 IDs, agrupa por lugar, se queda con el máximo de cada lugar
> y ordena los lugares. Como son solo 5, **la lista que ve el cliente tiene como mucho 5 lugares**
> (menos los que ya visitó). Las métricas del informe se calculan sobre los IDs, no sobre esa lista.

### Decisión 2 — Interacciones: unir las reseñas y el historial de visitas
Para evaluar a un usuario necesitamos al menos 2 interacciones suyas: una para que el modelo lo
conozca en entrenamiento y otra que se esconde para ver si el modelo la adivina. Usando una sola
tabla, solo unos 260 usuarios cumplen esa condición; al unir las dos tablas son 606, porque muchos
usuarios tienen una reseña y una visita que por separado no alcanzan (**tabla 10**). La unión no
genera duplicados ni contradicciones, ya que ningún par usuario–destino aparece en ambas tablas
(**tabla 8**). El costo es que se mezclan dos tipos de evento (escribir una reseña y visitar), pero
en este dataset ambas tablas se generaron de la misma forma, y guardamos una columna `fuente` por
si después queremos separarlas.

### Decisión 3 — Calificaciones: toda interacción cuenta como positiva
La alternativa era quedarse solo con las interacciones de nota 4 o 5 («le gustó»), pero eso solo
tiene sentido si la nota contiene información. En este dataset las notas se reparten casi por igual
entre 1 y 5 (chi-cuadrado de uniformidad p = 0,66 y p = 0,12, **tabla 4**) y no se relacionan con la
popularidad del destino (correlación de 0,03–0,04, no significativa, **tabla 5**), así que filtrar
por ellas eliminaría interacciones al azar sin dejar datos más limpios. Además, los usuarios
evaluables bajarían de 606 a 181 (**tabla 10**). Por eso cada interacción se interpreta como «el
usuario mostró interés en ese destino». La limitación es que el modelo no distingue una visita que
gustó de una que no; lo reportamos en el informe y conservamos la calificación en los datos por si
queremos compararlo después.

---

## Paso 3 — Tarea 1: caso de uso y K (2026-11-23) ✅ decidido

Evidencia: **tablas 9 y 14** del inventario.

| Decisión | Qué se decidió | Por qué |
|---|---|---|
| **Canal y momento** | Sección «Destinos para ti» en la página de inicio de la app o la web, al iniciar sesión. Para un cliente nuevo aparece justo después de registrarse y llenar el formulario de preferencias. | Es el momento en que tenemos toda la información del usuario, y coincide con lo que pide la app (usuario existente o perfil nuevo). Recomendar al comprar un tiquete compite con la compra; queda como canal secundario. El correo se reserva para la campaña (pregunta 3). |
| **¿Excluir los lugares ya visitados?** | **No.** Se muestran los mejores lugares y, si el cliente ya fue, se marcan con «Ya lo visitaste, ¿vuelves?». A nivel de **ID** sí se excluyen los IDs ya vistos en entrenamiento (práctica estándar de evaluación). | El 33,8 % de las interacciones escondidas en una partición exploratoria es un regreso a un lugar ya visitado (tabla 14): excluirlos haría fallar por diseño uno de cada tres casos. Además, con K = 3, 197 usuarios (23 %) recibirían menos de 3 lugares. |
| **K del cliente** | **3 lugares.** | Una lista corta es una recomendación y no un catálogo; 1 es muy arriesgado y 4–5 es casi mostrar todo. *Limitación:* 3 de 5 es el 60 % del catálogo, por los datos sintéticos. |
| **K de evaluación** | **K = 10** como principal; curvas con K = 1, 3, 5, 10 y 20. | Es el estándar de la literatura (por ejemplo, HR@10 y NDCG@10 en NCF). Con 1000 IDs el azar acierta ≈ 1 %, así que hay margen. *Ojo:* con 1 interacción escondida por usuario, la Precision@10 vale como mucho 0,1; por eso se destacan Recall, HitRate y NDCG@10. |
| **Información disponible** | Usuario existente: perfil (preferencias, género, adultos, niños) + historial. Usuario nuevo: solo lo del formulario. | Define qué entradas puede usar el modelo en cada caso (y la pregunta 1, sobre usuarios nuevos). |

---

## Paso 4 — Tarea 3: partición (2026-11-23) ✅ decidido

Teoría de apoyo: sección 14 de `teoria/01_fundamentos_sistemas_recomendacion.md`.
Evidencia: **tabla 15** del inventario.

**Reglas del enunciado:** ninguna interacción de prueba se usa para entrenar, y todos los
modelos usan las mismas particiones.

### Opciones evaluadas

| Estrategia | Entrenamiento | Validación | Prueba | Usuarios de prueba que el modelo nunca vio | IDs de prueba sin ninguna interacción en entrenamiento |
|---|---|---|---|---|---|
| **Temporal** | — | — | — | — | **No viable:** solo hay 3 fechas, asignadas en ciclo por número de fila, y las reseñas no tienen fecha. |
| **Aleatoria 80/10/10** | 1586 | 213 | 199 | **18,8 %** | 21,1 % |
| **Dejar uno fuera por usuario** | 1066 | 326 | 606 | **0 %** | 33,7 % |

En todas las opciones, el 100 % de los **lugares** de prueba aparece en entrenamiento: con solo
5 lugares, todos se ven.

### Hallazgo importante: un tercio de los destinos de prueba son «IDs fríos»
Con dejar-uno-fuera, el 33,7 % de los IDs escondidos para prueba **no tiene ninguna interacción en
entrenamiento**. Un modelo que solo conoce IDs (popularidad, factorización de matrices) nunca
podrá recomendarlos, porque no sabe nada de ellos: **su HitRate tiene un techo cercano al 66 %**.
Un modelo con contenido (el lugar, el tipo) sí puede, porque sabe que ese ID «es un Taj Mahal».
Es un argumento concreto a favor de la red con variables de contenido, y hay que reportarlo.

### ✅ Decisión tomada: dejar uno fuera por usuario

1. **Cómo se reparte.** A cada usuario con ≥ 2 interacciones se le esconde 1 interacción al azar
   para **prueba** (606 usuarios). A los que tienen ≥ 3, otra para **validación** (326 usuarios).
   El resto va a **entrenamiento** (1066 interacciones). La semilla es fija (0) para que la
   partición sea reproducible.
2. **Los 252 usuarios con 1 sola interacción** quedan enteros en entrenamiento: no se evalúan,
   pero enseñan al modelo sobre los destinos. Aportan 252 de las 1066 interacciones de
   entrenamiento, y para algunos destinos son la única información que hay (por ejemplo, el
   destino 963 solo lo tiene el usuario 6).
3. **Flujo:** se buscan los hiperparámetros con entrenamiento → validación; luego se reentrena la
   configuración ganadora con entrenamiento + validación (1392 interacciones) y se mide **una sola
   vez** en prueba.
4. **La misma partición** para todos los modelos y las 30 semillas (las semillas cambian la
   inicialización y los negativos, no la partición).
5. **Arranque en frío (pregunta 1):** no requiere otra partición; se simula con los mismos
   usuarios de prueba, ocultándoles el historial y usando solo su perfil.

**Por qué no la aleatoria 80/10/10:** el 18,8 % de sus usuarios de prueba no tendría nada en
entrenamiento (sería evaluar arranque en frío y no el caso de uso), y solo evaluaría 181 usuarios
frente a 606. **Por qué no la temporal:** solo hay 3 fechas cíclicas y las reseñas no tienen fecha.
**Costo:** el entrenamiento inicial usa la mitad de los datos; el reentrenamiento con validación
lo compensa en parte.

**Ejemplos reales** (para el informe):

| Usuario | Interacciones | Entrenamiento | Validación | Prueba |
|---|---|---|---|---|
| 6 | 1 | Jaipur City (963) | — | — |
| 1 | 2 | Goa Beaches (467) | — | Taj Mahal (941) |
| 16 | 4 | Leh Ladakh (970), Taj Mahal (176) | Leh Ladakh (40) | Jaipur City (113) |

---

## Paso 5 — Preprocesamiento: `src/preprocesar.py` (2026-11-23) ✅ hecho

**Qué hace:** aplica las decisiones de los pasos 1 a 4 y genera los archivos que leen **todos**
los modelos y la app (mismos datos = comparación justa). Reutiliza la función de partición del
inventario, así que la partición es idéntica a la analizada (mismos ejemplos de los usuarios
6, 1 y 16).

| Archivo en `data/processed/` | Contenido |
|---|---|
| `interacciones.csv` | 1998 filas: `UserID, DestinationID, lugar, fuente, calificacion, conjunto` (1066 entrenamiento / 326 validación / 606 prueba) |
| `usuarios.csv` | 999 perfiles **sin nombre ni correo** + preferencias en 5 columnas 0/1 (`pref_Beach`, …) + `n_interacciones`, `n_entrenamiento` |
| `destinos.csv` | el catálogo de 1000 destinos, sin cambios |
| `preferencias.json` | los valores posibles para el formulario de la app |

**Decisiones menores tomadas aquí:**
- **Se eliminan `Name` y `Email` de los usuarios:** son ficticios (20 valores repetidos) y no
  aportan nada al modelo.
- **Preferencias en formato multi-hot** (una columna 0/1 por tipo, en lugar del texto «Beaches,
  Historical»). Así el modelo acepta **cualquier combinación** de preferencias. Importa para la
  app: un cliente nuevo podría marcar «Beach + Nature», una combinación que no existe en los
  datos (solo hay 3 perfiles). Con multi-hot el modelo puede procesarla igual, aunque nunca la
  haya visto (hay que advertirlo en el informe).
- **No se necesitan codificadores de IDs:** se verificó que los IDs de usuarios (1–999) y de
  destinos (1–1000) son consecutivos, así que el índice es simplemente `ID − 1`.

**Verificaciones automáticas** (el script se detiene si alguna falla): ningún par de
validación/prueba aparece en entrenamiento; todo usuario evaluado tiene entrenamiento; una sola
interacción por usuario en validación y en prueba; no se pierde ni se duplica ninguna
interacción.

**Dato para más adelante (grupos de pocas/muchas interacciones, tarea 6):** en entrenamiento,
727 usuarios tienen 1 interacción, 79 tienen 2, 31 tienen 3, 17 tienen 4, 4 tienen 5 y 141 no
tienen ninguna (usuarios sin actividad). Habrá que fijar el umbral de «pocas» frente a «muchas»
con esta distribución.

---

## Paso 6 — Tarea 2: análisis descriptivo, `src/descriptivo.py` (2026-11-23) ✅ hecho

**Qué hace:** compara cada patrón de los datos con lo que produciría **el azar** (las mismas 1998
interacciones repartidas sin preferencias), y genera las dos figuras mínimas del módulo sobre la
cola larga y la concentración, más cuatro tablas en `results/descriptivo/`.

**Decisiones de método:**
- **Se usan las 1998 interacciones**, no solo las de entrenamiento: es una descripción del
  dataset, no una evaluación de modelos, así que no hay riesgo de fuga.
- **Referencia de azar:** Poisson para las distribuciones por usuario y por destino (es la
  distribución que resulta de repartir interacciones al azar); 2000 simulaciones multinomiales
  para la concentración (Gini y participación del top 10 %); permutaciones para la temporada.
- **Corrección de Bonferroni** para las 6 hipótesis, porque se prueban juntas.

**Resultados:**
- Las distribuciones por usuario y por destino no se distinguen de Poisson (p = 0,28 y 0,998).
- La concentración es la del azar (Gini por ID 0,383, rango del azar 0,369–0,402; por lugar,
  18,8–21,1 % cada uno).
- El 39,3 % de los usuarios tiene 0 o 1 interacción.
- **Ninguna de las 6 hipótesis se sostiene tras la corrección.**

> ⚠️ **Matiz sobre H1 (preferencias → tipo visitado).** En el paso 1 se reportó como «débil pero
> significativa» (p = 0,019). Como prueba aislada lo es; pero dentro de un conjunto de seis
> hipótesis, el p corregido es 0,11 y deja de ser concluyente. La redacción del informe debe
> decir «una diferencia pequeña (42,3 % frente a 40 %) que no resiste la corrección por pruebas
> múltiples». Se actualizaron las secciones 1.3 y 1.4 del borrador en `IDEAS_INFORME.md`.

**Detalle gráfico:** las figuras siguen la guía de visualización (un color para lo observado,
gris discontinuo para la referencia del azar, paneles separados en lugar de dos ejes, etiquetas
directas) y se revisaron visualmente para que no haya etiquetas superpuestas.

---

## Requisito de la app — sin registro ni base de datos (2026-11-24) ✅ decidido

**Qué se decidió:** la app del módulo 3 **no tiene registro ni base de datos**.
- «Usuario existente» = un selector con los IDs de `data/processed/usuarios.csv` (simula «iniciar
  sesión») que muestra su historial y sus recomendaciones.
- «Perfil nuevo» = un formulario (preferencias, adultos, niños, género) con las opciones de
  `data/processed/preferencias.json`. Las recomendaciones se calculan al instante con el modelo
  guardado y el perfil **vive solo en la sesión**: no se escribe en ningún archivo.

**Por qué:**
1. El enunciado no lo pide: exige «elegir un usuario existente o crear un perfil nuevo y ver sus K
   destinos recomendados», usando «los modelos guardados, sin reentrenar».
2. Guardar perfiles no aportaría nada al modelo: sin reentrenar, un usuario nuevo siempre se
   recomienda solo con su perfil (arranque en frío, pregunta 1).
3. Menos piezas que puedan fallar: la app debe levantarse «con un solo comando» en un clon limpio.
4. Privacidad: la app no almacena datos personales (argumento para la sección de ética).

**Consecuencia para el caso de uso:** el texto distingue dos niveles: el escenario de negocio
(la empresa mostraría «Destinos para ti» al entrar a la cuenta) y el prototipo (la app lo simula
sin registro). Se corrigieron `IDEAS_INFORME.md` (sección 2) e `informe/index.html`.

**Consecuencia para el código (paso 11):** `predict.py` debe ofrecer
`recomendar_perfil_nuevo(preferencias: dict, k)` que funcione **sin ID de usuario**. Esto favorece
una red cuya torre de usuario pueda funcionar solo con variables de contenido (dos torres).
