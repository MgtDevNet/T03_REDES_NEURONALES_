# Ideas para el informe — Módulo 3 (recomendación)

Archivo vivo: en cada paso se añaden ideas, cifras, frases y figuras para el reporte. Está
organizado según lo que pide el enunciado (`docs/instrucciones_trabajo_3.pdf`, pp. 5–7).
Marcas: ✅ ya tenemos la evidencia · ⏳ falta calcularlo o hacerlo · 💡 idea de redacción.

### Estilo acordado para la sección del módulo 3
- **Lenguaje sencillo**: que lo entienda alguien sin formación en recomendadores. Cada
  decisión se cuenta como *qué hicimos → por qué → qué alternativa descartamos → qué costo tiene*
  (ver el modelo en `DECISIONES.md`, «Resumen de decisiones tomadas»).
- **Toda afirmación lleva su número**: una tabla o una prueba de `results/inventario/` (o de los
  scripts posteriores). Nada de «se ve parejo» sin el p-valor.
- Las pruebas se explican en una frase («la prueba chi-cuadrado compara lo observado con lo que
  pasaría si todo fuera al azar») y la teoría detallada va en la fundamentación matemática.

---

## 1. Datos — ✅ BORRADOR LISTO

> Borrador en lenguaje sencillo para la sección «Datos» del informe. Responde una por una las
> preguntas del enunciado. Entre corchetes va la tabla de `results/inventario/RESUMEN.md` que
> respalda cada cifra (se regeneran con `src/inventario.py`).

### 1.1 Fuente y archivos

Usamos el *Travel Recommendation Dataset* («India Travel Recommender System Dataset»), publicado
en Kaggle por Aman Mehra (versión única, del 23 de enero de 2024). Son cuatro archivos CSV:

| Archivo | Qué contiene | Filas |
|---|---|---|
| `Expanded_Destinations.csv` | El **catálogo de destinos**: un ID, el nombre del lugar, el estado de India donde está, el tipo de destino (playa, histórico, ciudad, naturaleza o aventura), una puntuación de popularidad y la mejor época para visitarlo. | 1000 |
| `Final_Updated_Expanded_Users.csv` | Los **usuarios**: un ID, nombre y correo (ficticios), las preferencias de viaje que declararon, el género y cuántos adultos y niños viajan. | 999 |
| `Final_Updated_Expanded_Reviews.csv` | Las **reseñas**: qué usuario reseñó qué destino, con una calificación de 1 a 5 y un texto corto. | 999 |
| `Final_Updated_Expanded_UserHistory.csv` | El **historial de visitas**: qué usuario visitó qué destino, en qué fecha y con qué calificación de la experiencia (1 a 5). | 999 |

Los dos primeros describen **quién** es cada usuario y **qué** es cada destino. Los dos últimos
registran **interacciones**, es decir, momentos en que un usuario tuvo contacto con un destino.
Son la materia prima del recomendador. Ninguna tabla tiene datos faltantes, IDs repetidos ni
interacciones que apunten a usuarios o destinos inexistentes [tabla 1].

### 1.2 Lo que pide el enunciado, punto por punto

**Número de usuarios, destinos e interacciones** [tabla 12]
- **999 usuarios**, de los cuales 858 tienen al menos una interacción.
- **1000 destinos** (IDs), de los cuales 866 tienen al menos una interacción. *Pero* esos 1000 IDs
  son en realidad **5 lugares** (Taj Mahal, Goa Beaches, Jaipur City, Kerala Backwaters y Leh
  Ladakh), cada uno repetido 200 veces [tabla 3].
- **1998 interacciones**: 999 reseñas + 999 visitas. En promedio, 2,3 interacciones por usuario
  activo.

**Escala de las calificaciones** [tablas 4 y 12]
Ambas tablas de interacciones califican con **enteros de 1 a 5**. Las cinco notas aparecen casi
con la misma frecuencia (≈ 20 % cada una). Una prueba chi-cuadrado no rechaza que la distribución
sea uniforme (p = 0,66 en reseñas y p = 0,12 en historial), algo poco habitual en datos reales,
donde suelen predominar las notas altas.

**Densidad de la matriz usuario–destino** [tabla 12]
Si se arma una tabla con un usuario por fila y un destino por columna, solo el **0,20 %** de las
celdas tiene información: 1998 / (999 × 1000). Usando una sola de las tablas de interacciones
sería el 0,10 %. Es una matriz extremadamente dispersa: casi todo lo que el modelo debe predecir
es desconocido. Además, 252 usuarios tienen una sola interacción y 134 destinos no tienen
ninguna [tabla 11].

**Variables disponibles de usuarios y destinos** [tabla 13]

| De los destinos | Valores | De los usuarios | Valores |
|---|---|---|---|
| `Type` (tipo) | Beach, Historical, City, Nature, Adventure | `Preferences` | 3 perfiles: «Beaches, Historical», «City, Historical», «Nature, Adventure» |
| `State` (estado) | 5 estados de India | `Gender` | Female 496, Male 503 |
| `Popularity` | continua, de 7,5 a 9,5 | `NumberOfAdults` | 1 o 2 |
| `BestTimeToVisit` | 5 temporadas | `NumberOfChildren` | 0, 1 o 2 |
| `Name` | 5 nombres de lugar | `Name`, `Email` | ficticios (20 nombres repetidos); no se usan |

Hay que tener en cuenta que `Type`, `State`, `BestTimeToVisit` y `Name` dicen **lo mismo**:
cada lugar tiene un único tipo, estado y temporada. En la práctica, el destino tiene **una
variable categórica con 5 niveles** (el lugar) más una numérica (`Popularity`).

### 1.3 ¿Reales o sintéticos?

**Son sintéticos.** Lo sostienen dos tipos de evidencia:

1. **Lo que dice la fuente.** El autor escribe que el conjunto «was generated for educational and
   research purposes» (fue *generado* con fines educativos) y que los perfiles se «enriquecieron»
   con diversidad de género. No describe de dónde salieron los datos ni cómo se recolectaron.
2. **Lo que muestran los datos.** Hay patrones que un registro real de turistas no tendría
   [tablas 2, 3, 7 y 8]:
   - el lugar de cada destino lo determina una fórmula: (ID − 1) mod 5;
   - las preferencias del usuario, la fecha de visita y el texto de la reseña se repiten en ciclos
     exactos de 3 (por ejemplo, 333 visitas en cada una de solo 3 fechas);
   - el texto «Loved the beaches!» aparece igual de seguido en el Taj Mahal que en las montañas
     (chi-cuadrado p = 0,88);
   - ninguna persona reseñó un destino que aparezca en su historial de visitas.

   Lo que **sí** aparece es una relación débil entre lo que el usuario dice preferir y lo que
   visita: el 42,3 % de sus interacciones van a un tipo de destino preferido, frente al 40 % que se
   esperaría por azar (permutación p = 0,019) [tabla 6].

### 1.4 ¿Qué implica para nuestras conclusiones?

- **No hablamos de turistas reales.** Ningún resultado se puede usar para afirmar algo sobre los
  viajeros reales de India ni sobre los clientes de la empresa. El módulo demuestra el **método**
  (cómo se construye, evalúa y compara un recomendador), no un hallazgo de negocio.
- **Hay muy poca señal que aprender.** Si las interacciones son casi aleatorias, ningún modelo
  puede acertar mucho. Esperamos métricas bajas y diferencias pequeñas entre modelos. Por eso
  incluiremos un recomendador **aleatorio** como piso y compararemos con 30 semillas antes de
  afirmar que un modelo es mejor que otro.
- **Si la red supera a la popularidad, ya sabemos dónde buscar la causa:** la única señal
  detectada es la débil relación entre preferencias y tipo de destino, que solo un modelo con
  variables de contenido puede aprovechar.
- **El catálogo real tiene 5 lugares.** Esto obliga a la decisión de modelado de la sección
  siguiente (1000 IDs para el modelo, 5 lugares para el cliente).

### 1.5 Decisiones sobre los datos (resumen; detalle en `DECISIONES.md`)
- **Destino:** 1000 IDs para el modelo y las métricas; 5 lugares para el cliente y el análisis
  de concentración [tabla 9].
- **Interacciones:** la unión de reseñas + historial (606 usuarios evaluables frente a ~260 con
  una sola tabla) [tablas 8 y 10].
- **Calificaciones:** toda interacción cuenta como positiva («mostró interés»); no se filtra por
  nota porque la nota es ruido y filtrar dejaría 181 usuarios evaluables [tablas 4, 5 y 10].

## 2. Caso de uso (tarea 1) — ✅ BORRADOR LISTO

> Borrador en lenguaje sencillo. Evidencia: tablas 9 y 14 del inventario.

**Dónde y cuándo.** La recomendación aparece en la página de inicio de la app o la web de la
empresa, en una sección «Destinos para ti», cada vez que el cliente inicia sesión. Si el cliente
es nuevo, aparece justo después de registrarse, cuando acaba de indicar sus preferencias de viaje.

**Cuántos destinos.** Al cliente se le muestran **3 lugares**. Una lista corta se lee como una
recomendación y no como un catálogo. Como el catálogo real tiene solo 5 lugares, 3 ya es el
60 % de la oferta; esta es una limitación del dataset que discutimos en la sección de datos.
Para medir la calidad del modelo usamos una lista más larga, de **10 destinos (K = 10)**, sobre
los 1000 IDs: es el valor estándar en la literatura y permite distinguir un buen modelo del azar
(que acertaría ≈ 1 % de las veces).

**¿Se recomiendan lugares ya visitados?** Sí, marcados como «Ya lo visitaste, ¿vuelves?». En
los datos, uno de cada tres «próximos viajes» es un regreso a un lugar conocido (33,8 %,
tabla 14): excluirlos haría perder esos aciertos, y además dejaría al 23 % de los usuarios con
menos de 3 lugares para mostrar.

**Qué sabemos del usuario en ese momento.**
- Cliente con historial: sus preferencias declaradas, género, cuántos adultos y niños viajan, y
  qué destinos reseñó o visitó antes.
- Cliente nuevo: solo lo que indicó en el formulario (preferencias y tamaño del grupo). Ese caso
  se trata en la pregunta de negocio 1.

## 3. Análisis descriptivo (tarea 2)
- ✅ Usuarios por número de interacciones (unión): 1 → 252, 2 → 280, 3 → 195, 4 → 79, 5 → 31,
  6 → 17, 7 → 4.
- ✅ 134 destinos sin interacciones y 271 con una sola. El más popular tiene solo 7.
- ⏳ Figura de cola larga (por usuario y por destino).
- 💡 Hipótesis sobre las preferencias: el enunciado pide formularlas. Hay que plantearlas
  (familia → destinos de playa, etc.) **y contrastarlas con datos**. El cruce preferencias ×
  tipo visitado muestra una relación **débil pero significativa** (tabla 6 del inventario):
  Nature/Adventure coincide el 45,4 %, Beaches/Historical el 43,0 % y City/Historical el 38,6 %
  (este último, por debajo del azar). 💡 Es una buena historia para el informe: la primera
  impresión fue «no hay relación», y la prueba formal mostró que sí la hay, aunque pequeña.

## 4. Partición (tarea 3) — ✅ BORRADOR LISTO

> Borrador en lenguaje sencillo. Evidencia: tabla 15 del inventario; detalle en `DECISIONES.md`, paso 4.

**Para qué se parten los datos.** Para saber si un recomendador funciona hay que preguntarle por
algo que **no vio** al aprender, como en un examen. Dividimos las interacciones en tres grupos:
*entrenamiento* (lo que el modelo estudia), *validación* (un examen de práctica para elegir la
configuración del modelo) y *prueba* (el examen final, que se mira una sola vez). Ninguna
interacción de prueba se usa para entrenar.

**Cómo las partimos: «dejar uno fuera» por usuario.** A cada usuario con al menos 2
interacciones le escondemos una, elegida al azar, para prueba; si tiene al menos 3, le escondemos
otra para validación; todo lo demás va a entrenamiento. Por ejemplo, la usuaria 1 visitó Goa
Beaches y el Taj Mahal: el modelo aprende de Goa, y al final comprobamos si entre sus 10
recomendaciones aparece el Taj Mahal que le escondimos. El resultado: 1066 interacciones de
entrenamiento, 326 de validación y 606 de prueba (una por usuario evaluado).

**Qué pasa con los usuarios que tienen una sola interacción.** Son 252. Su interacción se usa para
entrenar, porque le enseña al modelo sobre ese destino (para algunos destinos es la única
información disponible), pero no se pueden evaluar: no queda nada que esconderles.

**Por qué esta forma y no otra.**
- *Por fechas* (lo más realista en general): no es posible, porque el historial tiene solo 3
  fechas repetidas en ciclo y las reseñas no tienen fecha.
- *Al azar, sin mirar al usuario (80/10/10):* casi uno de cada cinco usuarios de prueba (18,8 %)
  quedaría sin nada en entrenamiento, y el modelo tendría que adivinar a ciegas. Además, solo
  evaluaría 181 usuarios frente a 606.
- *Dejar uno fuera* garantiza que todo usuario evaluado tiene historial, que es la situación del
  caso de uso (un cliente que inicia sesión y ya ha viajado), y es el esquema habitual en la
  literatura de recomendadores.

**Un hallazgo que sale de la partición.** Uno de cada tres destinos de prueba (33,7 %) es un ID
que nadie tocó en entrenamiento. Un modelo que solo conoce IDs, como la popularidad o la
factorización de matrices, no puede recomendar algo que nunca vio, así que su tasa de aciertos
tiene un techo cercano al 66 %. Un modelo que conoce el contenido («este ID es un Taj Mahal»)
no tiene ese techo. Es una razón concreta para usar variables de contenido en la red.

**Cómo se usan los conjuntos.** Elegimos la configuración de cada modelo entrenando con
entrenamiento y midiendo en validación; con la configuración ganadora volvemos a entrenar con
entrenamiento + validación (1392 interacciones) y medimos una sola vez en prueba. Todos los
modelos y las 30 repeticiones con semillas distintas usan exactamente la misma partición.

## 4b. Preprocesamiento (para la sección de metodología)
- ✅ `src/preprocesar.py` genera `data/processed/`; todos los modelos leen los mismos archivos.
- 💡 Explicar en una frase cada transformación: unión de tablas; partición; se eliminan nombre y
  correo (ficticios); preferencias en multi-hot («una casilla por tipo de destino, marcada o no»),
  que permite aceptar combinaciones nuevas desde la app.
- 💡 Mencionar las verificaciones automáticas contra fugas: refuerzan la credibilidad del
  resultado.
- ⚠️ Limitación: en los datos solo hay 3 combinaciones de preferencias. Un perfil nuevo con otra
  combinación (por ejemplo, Beach + Nature) es una extrapolación del modelo.

## 5–6. Modelos y evaluación
- ⏳ Añadir un recomendador **aleatorio** como piso de comparación, además de la popularidad y
  la factorización de matrices.
- 💡 Decisión sobre la granularidad del ítem (ver `DECISIONES.md`, paso 1): explicarla en el
  informe como una decisión de modelado.

## Preguntas de negocio
- ⏳ P1 (usuarios nuevos): 💡 explicar el arranque en frío con la analogía del bibliotecario
  (teoría, sección 13). Simulación: a los 606 usuarios de prueba se les oculta el historial y se
  recomienda solo con el perfil; el ejemplo es la usuaria 1. Estrategias para comparar:
  popularidad global, popularidad por perfil, contenido (tipo = preferencia) y torre de usuario
  sin ID.
- ⏳ P2 (concentración), P3 (campaña).

## Cacería de la alucinación
- 💡 **Candidato fuerte:** el título y la descripción del dataset («India Travel Recommender
  System», reseñas «generadas por usuarios», un historial de viajes) sugieren datos de
  comportamiento real con una señal aprovechable, pero el análisis muestra 5 destinos clonados e
  interacciones sin relación con las preferencias. Se puede contar como: «qué dice la fuente →
  qué asumiría un análisis apresurado → qué muestran los datos → cómo lo detectamos».

## Fundamentación matemática
- ✅ Borrador en `teoria/01_fundamentos_sistemas_recomendacion.md` (factorización de matrices,
  derivación de BCE y BPR, NCF y dos torres, métricas). Falta adaptarlo al formato del informe una
  vez elegido el modelo final.
