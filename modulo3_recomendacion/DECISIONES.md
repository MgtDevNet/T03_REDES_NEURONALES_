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

> ⚠️ **Corrección (2026-11-24, paso 7).** El 33,7 % compara prueba contra **entrenamiento solo**.
> El modelo final aprende con **entrenamiento + validación**, y entonces solo el **22,6 %** de los
> destinos de prueba es frío (techo **77,4 %**). Al ajustar (entrenamiento → validación) es 34,4 %
> (techo 65,6 %). Evidencia: `results/tarea4_evidencia/tabla_01.csv`. Además, «nunca podrá
> recomendarlos» es demasiado fuerte: no tiene información sobre ellos y solo los acertaría por
> casualidad.

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

---

## Paso 7 — Tarea 4: modelos de referencia (2026-11-24) ✅ decidido

**Evidencia previa:** `src/evidencia_tarea4.py` → `results/tarea4_evidencia/` (T4-1 a T4-3). Antes
de programar, se midió qué puede lograr cada familia de modelos con estos datos.

| Medida (T4-1) | Ajuste (ent → val) | Final (ent+val → prueba) |
|---|---|---|
| Casos evaluados | 326 | 606 |
| HitRate@10 esperado del azar | 1,0 % | 1,0 % |
| Destino escondido nunca visto al aprender | 34,4 % | 22,6 % |
| Techo de un modelo solo de IDs | 65,6 % | 77,4 % |
| KNN ítem–ítem puede puntuarlo | 0,0 % | 0,2 % |
| KNN usuario–usuario puede puntuarlo | 0,0 % | 0,2 % |

### Decisión 7.1 — Modelos: azar + popularidad + factorización de matrices (sin KNN)
- **Por qué no KNN:** KNN solo puntúa un destino si otro usuario lo comparte con alguno de los
  destinos del usuario evaluado; aquí eso ocurre en el 0–0,2 % de los casos (T4-1). Fallaría casi
  siempre, sin importar sus hiperparámetros.
- **Por qué factorización de matrices:** puntúa cualquier destino visto al aprender (techo 77 %).
- **Por qué el azar:** el enunciado no lo pide, pero con datos que se comportan como aleatorios
  (tarea 2) es el piso obligado: HitRate@10 ≈ 1 %.

### Decisión 7.2 — Popularidad = conteo de interacciones de entrenamiento, desempate al azar
- Para llenar el top-10 se toma el destino con 6 interacciones y 9 de los 15 empatados con 5
  (T4-2): el orden dentro del empate es arbitrario, así que se desempata al azar con semilla y se
  repite en las 30 semillas.
- **Descartado:** la columna `Popularity`, porque H5 mostró que no se relaciona con las
  interacciones (ρ = −0,011; D-4).

### Decisión 7.3 — Factorización de matrices con pérdida BPR, en PyTorch
- **BPR** premia que el destino visitado quede por encima de uno no visitado: es exactamente lo
  que mide un top-K, y sirve con retroalimentación implícita (decisión 3). BCE con negativos queda
  para comparar en la tarea 5.
- Dimensión $d \in \{8, 16, 32\}$ y regularización elegidas en validación; se reportan el número de
  configuraciones probadas y el tiempo de cómputo.
- **PyTorch** porque lo usa el módulo 2 (torch 2.8.0): la app carga los modelos igual.

### Decisión 7.4 — Protocolo de evaluación común
- Se ordenan los **1000 destinos completos** (no una muestra de negativos), quitando los IDs que el
  usuario ya tiene en lo aprendido. Como el destino escondido nunca está en lo aprendido (verificado
  en `preprocesar.py`), quitarlos no le resta ningún acierto posible. Se quitan **IDs**, no
  lugares: un usuario puede recibir otro ID del mismo lugar («¿vuelves?», paso 3).
- Métricas: HitRate, Recall, Precision y NDCG con K ∈ {1, 3, 5, 10, 20}, y cobertura.
- 30 semillas para todo lo que tenga azar (desempates, inicialización, muestreo de negativos).

**Expectativa honesta:** T4-1 sugiere que la factorización de matrices tendrá muy poca señal
colaborativa (casi no hay caminos entre usuarios), así que probablemente quede cerca del azar. Es
lo que predice la tarea 2 y se documentará como resultado, no como fallo.

### Resultado 7.a — Azar y popularidad (`src/referencias.py`, 2026-11-24) ✅ hecho
- Final (606 usuarios, 30 semillas): HitRate@10 azar 0,91 %, popularidad 1,16 %; cobertura@10
  99,7 % frente a 1,6 %. Detalle: `results/tarea4/RESUMEN.md`.
- **La popularidad no supera al azar:** 7 aciertos frente a 6,1 esperados, p-valor = 0,41 (binomial
  unilateral). Los IC 95 % entre semillas no bastan para comparar: solo miden la variación por
  semilla, no el muestreo de usuarios (error estándar ≈ 0,40 puntos). **Regla para el resto del
  módulo:** toda comparación entre modelos se acompaña de una prueba que considere a los usuarios
  como muestra.
- Evaluador común: `src/evaluacion.py`, validado con `src/probar_evaluacion.py` (ejemplo a mano,
  empates, reproducibilidad, azar ≈ 1 %).

### Cambio de entorno: Python 3.14 → 3.12 (2026-11-24) ✅ hecho
- **Por qué:** el módulo 2 usa Python 3.12 + torch 2.8.0, y torch 2.8.0 no existe para Python 3.14.
  La app carga los tres módulos en un solo entorno, así que conviene un entorno común.
- **Qué cambió:** `T03/.venv` ahora es Python 3.12.15 con pandas 3.1.1, numpy 2.3.5, scipy 1.17.0,
  matplotlib 3.10.8 y torch 2.8.0+cpu (versiones del módulo 2). El entorno viejo (Python 3.14.4,
  numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2) quedó como respaldo en `T03/.venv314`.
- **Verificación:** se volvieron a correr los 6 scripts. Todas las tablas CSV, los RESUMEN.md y los
  datos procesados (incluida la partición) salieron **idénticos byte a byte**. Las dos figuras
  cambian solo en el renderizado de matplotlib; revisadas a ojo, son iguales.

### Resultado 7.b — Prueba de la factorización con datos de juguete (`src/probar_factorizacion.py`) ✅ pasa
- **Para qué:** distinguir «los datos no tienen señal» de «el código tiene un error». Se fabricaron
  datos del mismo tamaño CON señal: 50 grupos de 20 destinos; cada usuario visita 4 destinos de
  su grupo. Techo teórico de HitRate@10: 10/17 ≈ 59 %; azar ≈ 1 %.
- **Resultado (d = 16, λ = 0,1):** HitRate@10 = **58,4 %** (azar 0,9 %, popularidad 0,5 %); la
  pérdida baja; parámetros = 31 984 = (999 + 1000) × 16. **El código aprende.**
- **Hallazgo:** con la regularización planeada el modelo memorizaba:

  | λ | Pérdida final | HitRate@10 (prueba de juguete) |
  |---|---|---|
  | 0,0001 | 0,007 | 12,8 % |
  | 0,001 | 0,029 | 15,2 % |
  | 0,01 | 0,148 | 48,4 % |
  | 0,1 | 0,602 | 58,4 % |

  Con tan pocas interacciones por usuario (2–3), hace falta una regularización mucho más fuerte de
  la que se planeó.

### ⚠️ Cambio en la decisión 7.3 (por el resultado 7.b)
- **λ ∈ {0,01; 0,1; 1}** en lugar de {0,0001; 0,001; 0,01}: la rejilla original quedaba entera en
  la zona donde el modelo memoriza. Se incluye 1 para no dejar el mejor valor en el borde.
- **Hasta 200 épocas** (antes 100): con λ = 0,01 la mejor época fue la 111.
- Sigue: d ∈ {8, 16, 32} → 9 configuraciones × 3 semillas en validación.

### Resultado 7.c — Factorización con los datos reales (`src/ajustar_factorizacion.py`) ✅ hecho
- **Ajuste:** 9 configuraciones × 3 semillas = 27 corridas, 25 s en total (incluida la fase final).
  HitRate@10 medio en validación entre 1,0 % y 2,2 %; las mejores épocas fueron muy tempranas
  (1 a 24), señal de que la parada temprana sigue ruido. La mejor (d = 16, λ = 0,01) acierta 6 de
  326 frente a 3,3 esperados por azar: p-valor = 0,112 (y ese p-valor es optimista, porque la
  validación se usó también para escoger la época). **Ninguna supera al azar** → por la regla
  acordada se eligió la más simple: **d = 8, λ = 1, 7 épocas** (15 992 parámetros).
- **Final (606 usuarios, 30 semillas):** HitRate@10 = 0,94 % (IC entre semillas 0,83–1,06),
  NDCG@10 = 0,43 %, cobertura@10 = 17,4 %. 0,04 s por entrenamiento.
- **Contra el azar:** 5,7 aciertos frente a 6,1 esperados, p-valor = 0,57. **No lo supera.**
- **Contra la popularidad (pareada por usuario):** p-valor = 0,57. **No hay diferencia.** (La
  factorización acierta «un poco» a 133 usuarios, en algunas semillas; la popularidad acierta
  siempre a los mismos 7. En total, empatan.)
- **Conclusión de la tarea 4:** ninguna referencia supera al azar. Como la prueba de juguete
  (7.b) mostró que el código aprende cuando hay señal, la causa son los datos, coherente con la
  tarea 2. Diferencia útil entre ellas: la cobertura (azar 99,7 %, factorización 17,4 %,
  popularidad 1,6 %).
- **Modelos guardados** (`models/`): `factorizacion_bpr.npz` (+ `.json`, semilla 0) y
  `popularidad.npz` (conteos). Se guardan como numpy: la app no depende de la versión de torch.

---

## Decisión 7.5 — Métricas por lugar, fijadas ANTES de entrenar la red (2026-11-25) ✅ decidido

**Qué:** además de las métricas por ID (Precision, Recall, NDCG@K, HitRate@K, cobertura), todos los
modelos se evalúan por **lugar**, que es lo que ve el cliente (decisión 1, K = 3 lugares):
1. Con los puntajes del modelo (sin los IDs ya aprendidos), cada lugar recibe el **mejor puntaje**
   de sus IDs; se ordenan los 5 lugares (empates al azar con semilla).
2. **HitRate@3 por lugar (métrica PRINCIPAL):** el lugar del destino escondido está entre los 3
   que ve el cliente. Azar: 3/5 = 60 %.
3. **HitRate@1 por lugar (secundaria):** el lugar escondido queda primero. Azar: 1/5 = 20 %.

**Por qué:** las preferencias predicen el lugar fuera de muestra (23,9 % frente a 20 %, p-valor =
0,010; T5-4), pero por ID ese efecto es invisible (cada lugar tiene 200 IDs; un oráculo que
siempre acierta el lugar solo llega a ≈ 5 % de HitRate@10; T5-3).

**Salvaguardas contra «buscar la métrica que salga bien»** (cf. hallazgo 1):
- Se fijan aquí, antes de ver ningún resultado de la red, y se aplican igual a los 4 modelos.
- Contra el azar, las dos métricas por lugar se prueban con corrección de Bonferroni (m = 2).
- Elección de la configuración de la red: mayor HitRate@3 por lugar medio en validación (3
  semillas). Si ninguna supera al azar en validación, se elige la más simple.
- Las métricas por ID se siguen reportando: las pide el enunciado.

## Paso 8 — Tarea 5: red neuronal de dos torres (2026-11-25) ✅ diseño decidido

**Evidencia:** `src/evidencia_tarea5.py` → `results/tarea5_evidencia/` (T5-1 a T5-4).
- T5-1: `Type`, `State` y `BestTimeToVisit` quedan fijados por el lugar (redundantes); solo
  `Popularity` cambia entre IDs del mismo lugar (y no se relaciona con las interacciones, H5).
- T5-2: 3 perfiles de preferencias; 36 si se agregan género, adultos y niños.
- T5-3: un oráculo que siempre sabe el lugar llega a ≈ 5 % de HitRate@10 (máximo del contenido).
- T5-4 (prueba, 606): `Preferences` → lugar más frecuente del perfil acierta 23,9 % (azar 20 %,
  p-valor = 0,010); el perfil completo, 20,0 % (p-valor = 0,52, sobreajuste); volver a un lugar ya
  visitado, 32,2 % frente a 32,4 % esperado (p-valor = 0,56).

**Decisiones:**
1. **Dos torres:** la app recomienda a perfiles nuevos sin ID, y no depende de caminos entre
   usuarios (lo que hizo fallar a KNN y a la factorización).
2. **Torre del usuario:** las 5 casillas de preferencias. Variante en el ajuste: perfil completo
   (para mostrar con datos si ayuda). Sin embedding de usuario ni de ID de destino.
3. **Torre del destino:** embedding del lugar (5 valores) + `Popularity` estandarizada.
4. **Dimensión d ∈ {2, 4, 8}:** la tabla perfil × lugar es 3 × 5 (rango ≤ 3); más dimensiones solo
   darían capacidad para sobreajustar.
5. **Negativos:** uniformes entre los 1000 IDs (sin los positivos del usuario), 1 o 4 por positivo;
   muestrear por popularidad sería casi igual porque el catálogo es parejo (Gini del azar).
6. **Pérdida:** BCE con negativos y BPR, ambas en el ajuste; derivación en la fundamentación.
7. **Prueba previa con datos simulados** en los que las preferencias sí determinen el lugar.

### Resultado 7.d — Referencias con métricas por lugar (2026-11-25) ✅ hecho
- Evaluador ampliado (`evaluacion.py`) y probado (`probar_evaluacion.py`: ejemplo a mano por lugar;
  azar ≈ 20 % y 60 %). Al volver a correr, **todas las métricas por ID salieron idénticas**.
- Prueba (606 usuarios, 30 semillas): Lugar@3 / Lugar@1 = azar 59,7 % / 19,3 %; popularidad
  58,1 % / 16,1 %; factorización 60,4 % / 20,4 %. Ninguno supera al azar (binomial, Bonferroni
  m = 2; p-valores corregidos ≥ 0,88).
- ⚠️ La popularidad por lugar ordena según el **mejor ID** de cada lugar (un ID de Jaipur City con
  6 interacciones), no según el total del lugar: pone primero a Jaipur City, el lugar menos
  frecuente entre los escondidos de prueba (16,2 %). Se había anticipado «Leh Ladakh, ≈ 21 %»
  sin calcularlo; era incorrecto.
- Exploratorio (no estaba fijado de antemano): la factorización supera a la popularidad en
  Lugar@3 en la prueba pareada (p-valor = 0,005), pero ninguna de las dos supera al azar; la
  diferencia se explica porque la popularidad fija siempre los mismos 3 lugares y estos son algo
  menos frecuentes en prueba. No se reporta como ventaja de la factorización.

### Detalle del ajuste de la red, fijado ANTES de entrenar (2026-11-25)
- Rejilla: d ∈ {2, 4, 8} × pérdida ∈ {BCE, BPR} × negativos por positivo ∈ {1, 4} × torre del
  usuario ∈ {solo preferencias, perfil completo} = 24 configuraciones × 3 semillas.
- Fijos: capa oculta de 16 neuronas (ReLU) en cada torre, Adam (tasa 0,01, decaimiento de pesos
  10⁻⁴), lotes de 256, hasta 200 épocas, parada si HitRate@3 por lugar en validación no mejora en
  10 épocas.
- Criterio: mayor HitRate@3 por lugar medio en validación. Si la mejor no supera al azar en
  validación (binomial contra 60 %), se elige la **más simple: d = 2, solo preferencias,
  1 negativo, BPR** (BPR por coherencia con la factorización).

### Resultado 8.a — Prueba de la red con datos simulados (`src/probar_dos_torres.py`) ✅ pasa
- Datos simulados con los usuarios reales: cada interacción va con probabilidad 0,9 a un lugar de
  sus 2 tipos preferidos. Máximo posible: Lugar@3 ≈ 96 %, Lugar@1 ≈ 47 %.
- d = 4, 1 negativo: BPR y BCE logran Lugar@3 = 95,5 % y Lugar@1 = 48,2 %; la pérdida baja.
  **El código aprende la relación cuando existe.**

### Resultado 8.b — Red con los datos reales (`src/ajustar_dos_torres.py`) ✅ hecho
- **Ajuste:** 24 configuraciones × 3 semillas = 72 corridas (64 s). Lugar@3 en validación entre
  61,0 % y 65,2 %. Tres configuraciones empatan en 65,2 % (todas d = 2 y solo preferencias); por el
  orden de la tabla se eligió **d = 2, BCE, 1 negativo, solo preferencias, 8 épocas** (238
  parámetros). En validación superaba al azar: 213 de 326 frente a 195,6 (p-valor = 0,027).
- Patrón del ajuste: «solo preferencias» queda por encima de «perfil completo» en la mayoría de
  combinaciones, coherente con T5-4 (el perfil completo sobreajusta).
- **Final (606 usuarios, 30 semillas):**
  - PRINCIPAL — Lugar@3: 61,8 % (375 frente a 363,6 esperados), p-valor corregido = 0,37.
  - Lugar@1: 22,2 % (134 frente a 121,2), p-valor corregido = 0,21.
  - HitRate@10 por ID: 1,11 % (7 frente a 6,1), p-valor = 0,41.
  - Pareadas en Lugar@3: +3,7 puntos frente a la popularidad (p-valor = 0,005; pero la popularidad
    queda por debajo del azar, resultado 7.d) y +1,4 frente a la factorización (p-valor = 0,20).
- **Conclusión:** la red va en la dirección esperada (por encima del azar en todas las métricas por
  lugar), pero **no hay evidencia suficiente para afirmar que supera al azar** en la métrica
  principal fijada de antemano.
- ⚠️ **Lección (maldición del ganador):** en validación la elegida «superaba al azar»
  (p-valor = 0,027), pero era la mejor de 24 configuraciones evaluadas en los mismos 326 usuarios
  y con parada temprana en esa misma validación: ese p-valor es optimista. La prueba, que se miró
  una sola vez, no lo confirma. Es exactamente para esto que existe el conjunto de prueba.
- Modelo guardado: `models/dos_torres.pt` + `dos_torres.json` (semilla 0).

> ⚠️ **Corrección (2026-11-25), decisión 7.2:** decía «escoger 10 entre los 24 destinos empatados
> con 4 interacciones». Era una lectura equivocada de T4-2: el acumulado con ≥ 5 interacciones ya
> es 16, así que el empate en el corte del top-10 es entre los 15 destinos con 5 interacciones
> (se toman 9). Corregido en el informe (6.2), aquí y en IDEAS_INFORME.md.

---

## Paso 9 — Tarea 6: evaluación (2026-11-25) 🔄 en curso

### Resultado 9.a — Grupos de usuarios y 30 semillas (`src/evaluar_tarea6.py`) ✅ hecho
- **Grupos** (definidos por las interacciones que el usuario tiene para aprender en el momento
  final): «pocas» = 1 (280 usuarios), «muchas» = 2 o más (326). Quedan equilibrados.
- Las medias generales coinciden con las Tablas 16 y 19 (reproducibilidad comprobada).
- Lugar@3 por grupo: azar 60,2 / 59,2 %; popularidad 57,5 / 58,6 %; factorización 59,7 / 61,1 %;
  red **63,4 % (pocas)** / 60,5 % (muchas). Ningún grupo de ningún modelo supera al azar
  (p-valores ≥ 0,12, sin corregir).
- La red rinde algo mejor con «pocas» que con «muchas», pero la diferencia (2,9 puntos) no es
  significativa (z = 0,73; p-valor = 0,46). Es coherente con su diseño: solo usa preferencias, no
  el historial, así que tener más interacciones no le da más información.
- Figura `figures/tarea6_semillas.png`: diagramas de caja de Lugar@3 y HitRate@10 en las 30
  semillas, con la advertencia de que solo muestran la variación entre semillas (hallazgo 7).

### Resultado 9.b — Usuarios de ejemplo y embeddings (`src/ejemplos_embeddings.py`) ✅ hecho
- Se instaló `scikit-learn==1.8.0` (misma versión del módulo 2) para t-SNE; numpy y scipy no cambian.
- **5 usuarios de ejemplo** con reglas fijas (el menor UserID que cumple cada una): uno por perfil
  de preferencias sin niños (usuarios 1, 3 y 20), uno con niños (2) y el que más interacciones
  tiene para aprender (234, con 6). Con la red guardada, el lugar escondido está entre los 3
  recomendados en 3 de los 5. Se carga `models/dos_torres.pt` igual que lo hará la app.
- **Qué aprendió la red** (puntaje medio por lugar y perfil): «Beaches, Historical» → Goa y Taj
  Mahal arriba; «Nature, Adventure» → Kerala y Leh Ladakh arriba (coinciden con lo declarado).
  «City, Historical» → Kerala arriba y Taj Mahal abajo, **porque así está en los datos
  aprendidos** (Kerala 21,6 %, Taj Mahal 18,5 % para ese perfil): la red sigue a los datos, no a
  lo que el usuario dice preferir.
- **Embeddings** (`figures/tarea6_embeddings.png`): la red (d = 2) se grafica sin proyectar; cada
  lugar forma una curva (por diseño: la torre del destino solo recibe el lugar y Popularity, que
  mueve el punto a lo largo de la curva). La factorización (d = 8, t-SNE) no muestra ninguna
  estructura por lugar, coherente con que no aprendió nada útil.

---

## Paso 10 — Preguntas de negocio (`src/preguntas_negocio.py`, 2026-11-25) ✅ calculadas

### P1. Usuarios nuevos (606 de prueba simulados como nuevos; 5 pliegues; 30 semillas)
- Para cada pliegue se borra TODO el historial de esos usuarios del entrenamiento (verificado:
  0 interacciones aprendidas) y se les recomienda solo con su perfil.
- Lugar@3 / Lugar@1: azar 60,1 / 19,8 %; popularidad 58,9 / 17,7 %; **regla «lo que más visita tu
  perfil» 60,7 / 24,6 %**; red 61,5 / 21,9 %.
- En la métrica principal (Lugar@3) ninguna supera al azar. En Lugar@1, la regla del perfil sí
  (149 frente a 121,2 esperados; p-valor corregido = 0,007; sigue < 0,05 aun corrigiendo por las
  3 estrategias). Red − popularidad en Lugar@3: +2,6 puntos, p-valor pareado = 0,12.
- Respuesta: a un cliente nuevo, recomendarle según sus preferencias (lo que más visitan los de
  su perfil), no lo más popular; la ventaja es pequeña.

### P2. Concentración (modelos finales, 606 usuarios, 30 semillas)
- % de clientes que recibirían cada lugar como 1.ª recomendación: popularidad → Jaipur City
  99,5 % (×5,0 la demanda actual); red → Kerala 34,1 % (×1,7) y Leh Ladakh 28,6 % (×1,4), Jaipur
  solo 5,6 %; factorización y azar ≈ 20 % cada uno (×1,0–1,1).
- Gini de las recomendaciones top-10 por ID: azar 0,04; factorización 0,58; red 0,92; popularidad
  0,99. Figura `figures/negocio_concentracion.png`.
- ⚠️ **La cobertura de la factorización no significa que «reparta mejor».** Con λ = 1, cada
  aparición de un destino en el entrenamiento encoge su vector: norma media 0,06 en los destinos
  sin interacciones frente a 0,009 en el de 6. Resultado: el 71 % de su top-10 son destinos fríos
  (24,5 % del catálogo). Recomienda justo lo que menos conoce. Corregir la interpretación de la
  cobertura en la sección 6.5 del informe.

### P3. Campaña de un solo destino en noviembre (red guardada; 999 clientes)
- En temporada en noviembre: Taj Mahal, Goa, Jaipur y Kerala (Leh Ladakh no: abril–junio).
- La red pone primero a Kerala para «City, Historical» y «Nature, Adventure» (666 clientes) y a
  Goa para «Beaches, Historical» (333). A Taj Mahal y Jaipur no los pone primero para nadie.
- **Kerala a esos 666 clientes:** tasa en entrenamiento 21,9 % → ≈ 146 conversiones esperadas,
  frente a ≈ 133 si se mostrara a 666 clientes al azar (+13). Comprobación en prueba: 103 de 399
  = 25,8 % (IC 95 %: 21,8–30,3 %), por encima del 20 %.
- Goa a «Beaches, Historical»: 20,8 % en entrenamiento, pero 16,9 % en prueba (12,4–22,6 %): no
  se confirma.

---

## Paso 11 — Funciones para la app (`src/predict.py`, 2026-11-25) ✅ hecho
- `recomendar_usuario`, `recomendar_perfil_nuevo` e `historial` con la red guardada; solo leen
  `data/processed/` y `models/` (funcionan en un clon limpio). Probadas con `src/probar_predict.py`,
  importando desde la raíz del repo como lo hará la app (2,4 s con la carga del modelo).
- **Cambio de contrato:** `k` pasa de 10 a **3 por defecto** y cuenta **lugares** (1–5), según la
  decisión 1 y el caso de uso (K cliente = 3). Se actualizó el README.
- No se repite el mismo ID del historial; los lugares ya visitados se marcan «¿vuelves?».
- ⚠️ **Extrapolación:** con Beach + Nature (combinación que no existe en los datos) la red pone
  Goa Beaches de último. Es la limitación de la sección 5 del informe; conviene avisarlo en la app.

### Notebook de experimentos (`notebooks/experimentos_modulo_3.ipynb`, 2026-11-25) ✅ hecho
- Pedido: un notebook para cambiar los hiperparámetros y comparar las métricas de todos los modelos.
- Lógica en `src/experimentos.py` (mismo evaluador y partición que el informe); el notebook solo
  tiene la configuración y las llamadas: tabla comparativa con p-valores contra el azar,
  comparación pareada entre dos modelos, gráfico de métricas en función de K, barrido de un
  hiperparámetro, curvas de aprendizaje y recomendaciones de un usuario.
- Por defecto usa el escenario «ajuste» (validación), con la advertencia de no elegir
  hiperparámetros mirando la prueba.
- Se ejecutó completo (16 s, sin errores) y se guardó con sus salidas para verlo en GitHub.
- Detalle técnico: `descriptivo.py` fija el backend «Agg» de matplotlib (apaga los gráficos en
  un notebook), así que `experimentos.py` copia el estilo en lugar de importarlo.
- La curva de aprendizaje de la red muestra la pérdida casi fija en 0,693 = ln 2 (la de una
  moneda al aire): otra forma de ver que casi no hay nada que aprender.

### Figuras comunes (`src/figuras_comunes.py`, 2026-11-25) ✅ hechas
- Arquitectura (Figura 4 del informe): la suma de parámetros por capa se comprueba contra el modelo
  guardado (238). Búsqueda de hiperparámetros (Figura 5): factorización (9) y red (24).
  Curvas de aprendizaje (Figura 6): 60 épocas sin parada temprana, 3 semillas.
- Hallazgo de las curvas: la pérdida de la factorización con λ = 1 se estanca exactamente en
  ln 2: los vectores se encogen casi a cero y todos los puntajes quedan ≈ 0 (moneda al aire).
  La de la red casi no baja de ln 2. Coherente con el hallazgo 8.
- Los diagramas de caja de las 30 semillas ya estaban (Figura 8).
