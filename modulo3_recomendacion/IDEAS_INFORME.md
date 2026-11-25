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
sea uniforme (p-valor = 0,66 en reseñas y p-valor = 0,12 en historial), algo poco habitual en datos reales,
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
     (chi-cuadrado p-valor = 0,88);
   - ninguna persona reseñó un destino que aparezca en su historial de visitas.

   Lo que **sí** aparece es una relación débil entre lo que el usuario dice preferir y lo que
   visita: el 42,3 % de sus interacciones van a un tipo de destino preferido, frente al 40 % que se
   esperaría por azar (permutación p-valor = 0,019) [tabla 6]. Sin embargo, esta diferencia **no se
   sostiene** cuando se corrige por hacer varias pruebas a la vez (p-valor corregido = 0,11; ver
   análisis descriptivo, sección 3.4): es una señal, en el mejor de los casos, débil e incierta.

### 1.4 ¿Qué implica para nuestras conclusiones?

- **No hablamos de turistas reales.** Ningún resultado se puede usar para afirmar algo sobre los
  viajeros reales de India ni sobre los clientes de la empresa. El módulo demuestra el **método**
  (cómo se construye, evalúa y compara un recomendador), no un hallazgo de negocio.
- **Hay muy poca señal que aprender.** Si las interacciones son casi aleatorias, ningún modelo
  puede acertar mucho. Esperamos métricas bajas y diferencias pequeñas entre modelos. Por eso
  incluiremos un recomendador **aleatorio** como piso y antes de afirmar que un modelo es mejor
  que otro lo compararemos con 30 semillas y con pruebas que tratan a los usuarios como muestra.
- **Si la red supera a la popularidad, ya sabemos dónde buscar la causa:** la única pista de
  señal es la débil relación entre preferencias y tipo de destino (que además no sobrevive a la
  corrección por pruebas múltiples), y solo un modelo con variables de contenido podría
  aprovecharla. Lo más probable es que ningún modelo supere claramente al azar.
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

**Dónde y cuándo (en el negocio).** La recomendación aparecería en la página de inicio de la app
o la web de la empresa, en una sección «Destinos para ti», cuando el cliente entra a su cuenta.
Para un cliente sin historial, se mostraría en cuanto indique sus preferencias de viaje.

**En nuestro prototipo.** La aplicación web simula ese momento sin registro ni base de datos. Se
elige un usuario existente del conjunto de datos (el equivalente a «iniciar sesión») o se llena un
formulario de preferencias para un cliente nuevo. Las recomendaciones se calculan al instante con
el modelo guardado, y el perfil nuevo no se almacena.

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

## 3. Análisis descriptivo (tarea 2) — ✅ BORRADOR LISTO

> Borrador en lenguaje sencillo. Evidencia: `results/descriptivo/RESUMEN.md` (tablas 1–4) y las
> figuras `figures/descriptivo_cola_larga.png` y `figures/descriptivo_concentracion.png`.
> Se usan las 1998 interacciones (describe el dataset; la partición no interviene).

**Idea que guía el análisis.** Para saber si un patrón dice algo sobre las preferencias de los
viajeros, lo comparamos con lo que produciría **el puro azar**: las mismas 1998 interacciones
repartidas sin ninguna preferencia. Si los datos se parecen al azar, el patrón no revela nada.

### 3.1 Interacciones por usuario y por destino (figura: cola larga)
Contando los 999 usuarios (incluidos los 141 sin interacciones), cada usuario tiene en promedio 2 interacciones, y cada destino (ID) también 2. Para saber si esa distribución dice algo, la comparamos con la que produciría el azar. Si las interacciones se hubieran sorteado al azar, cada una de las $N = 1998$ caería en cualquiera de los 1000 destinos con la misma probabilidad, $p = 1/1000 = 0{,}001$. Así, el número de interacciones de un destino concreto es el número de «aciertos» en 1998 intentos independientes con probabilidad 0,001: una variable binomial, $B(N = 1998,\ p = 0{,}001)$.

Cuando $N$ es grande y $p$ es pequeño, la binomial se aproxima muy bien por una distribución de Poisson con media $\lambda = N \cdot p$, cuya fórmula es más simple:

$$P(k) = \frac{e^{-\lambda}\,\lambda^k}{k!}, \qquad \lambda = 1998 \times 0{,}001 = 1{,}998 \approx 2.$$

Con los usuarios pasa lo mismo: cada interacción pertenece a cualquiera de los 999 usuarios con $p = 1/999$, así que $\lambda = 1998/999 = 2$. Multiplicando $P(k)$ por el número de destinos (1000) o de usuarios (999) se obtienen los valores esperados de la Tabla 7. Por ejemplo, se esperarían $1000 \times e^{-1{,}998} \times 1{,}998 \approx 270{,}9$ destinos con una sola interacción, y hay 271.

| k | Usuarios esperados (λ = 2) | Usuarios observados | Destinos esperados (λ = 1,998) | Destinos observados |
|---|---|---|---|---|
| 0 | 135,2 | 141 | 135,6 | 134 |
| 1 | 270,4 | 252 | 270,9 | 271 |
| 2 | 270,4 | 280 | 270,7 | 270 |
| 3 | 180,3 | 195 | 180,3 | 184 |
| 4 | 90,1 | 79 | 90,0 | 89 |
| 5 | 36,1 | 31 | 36,0 | 37 |
| 6 | 12,0 | 17 | 12,0 | 11 |
| ≥ 7 | 4,5 | 4 | 4,5 | 4 |

La Figura 1 muestra que la forma de ambas distribuciones es muy parecida a la del azar, y las pruebas de bondad de ajuste (chi-cuadrado) no permiten rechazar que provengan de una Poisson (p-valor = 0,28 por usuario y p-valor = 0,998 por destino). Es decir, no hay destinos más populares ni usuarios más activos de lo que produciría el azar, y por eso no hay cola larga. Esto no dice si cada usuario prefiere ciertos tipos de destino; esa relación se analiza aparte. [tabla 1]
💡 En datos reales de turismo, unos pocos destinos famosos y unos pocos viajeros frecuentes
concentran gran parte de la actividad, lo que produce una cola larga: muchos destinos y usuarios con
muy poca actividad y unos pocos con muchísima.

### 3.2 Concentración en los destinos populares (figura: concentración)
Pregunta: ¿unos pocos destinos se llevan la mayoría de las interacciones? Tres medidas, las tres
comparadas con el **azar** (repartir las mismas 1998 interacciones entre los 1000 destinos al azar,
2000 veces):
- **Curva de Lorenz:** destinos ordenados de menos a más interacciones; cada punto dice «el x % de
  los destinos con menos interacciones acumula el y % de todas». Diagonal = reparto parejo; cuanto
  más se hunde, más concentración.
- **Gini:** el doble del área entre la diagonal y la curva (0 = parejo, 1 = todo en un destino).
- **Top 10 %:** porcentaje que se llevan los 100 destinos con más interacciones.

Cómo se obtiene el rango del azar: mismas 1998 interacciones repartidas entre 1000 destinos con
un «dado de 1000 caras» (multinomial), Gini de cada reparto, 2000 veces; se ordenan y se quita el
2,5 % de cada extremo → 0,369–0,402 (percentiles 2,5 y 97,5). p-valor = 1219/2000 = 0,61. Lo
*observado* (% por lugar, top 10 %, Gini, curva y barras azules) sale de contar los datos; solo la
*referencia* (rango, banda gris, top 10 % del azar, p-valores) sale de la simulación. 20 % = 1/5.
Evidencia: `results/descriptivo/RESUMEN.md` §2; teoría 22.9.

Interpretación (un solo párrafo):
La curva de Lorenz parece mostrar concentración: el 13,4 % de los destinos no tiene interacciones y el 10 % con más se lleva el 23,6 %. Pero la referencia correcta no es la diagonal, sino el azar, porque con 2 interacciones por destino en promedio la suerte ya deja destinos con 0 y otros con 7 (la Poisson de 3.1). La curva queda dentro de la banda del azar, y la Tabla 8 lo confirma: el Gini (0,383) cae en el rango del azar (0,369–0,402, p-valor = 0,61) y el top 10 % es casi idéntico (23,6 % frente a 23,8 %). Por lugar real ocurre lo mismo: los 5 lugares reciben entre el 18,8 % y el 21,1 %, cerca del 20 % parejo (p-valor = 0,37). En conclusión, no hay destinos «estrella»: la popularidad tendrá poco que aprovechar como recomendador, y esta es la base de la pregunta de negocio 2 (concentración de la demanda).

### 3.3 Usuarios con pocas interacciones
De los 999 usuarios, 141 (14,1 %) no tienen ninguna interacción, 252 (25,2 %) tienen exactamente
una y, en total, el 67,4 % tiene 2 o menos. Solo 52 (5,2 %) tienen 5 o más. La mediana es 2 y el
máximo, 7 [tabla 3]. 💡 Esto explica por qué el recomendador tiene tan poca información por
usuario, y por qué conviene evaluar por separado a los usuarios con pocas y muchas interacciones
(tarea 6).

### 3.4 Hipótesis sobre qué explica las preferencias
Seis hipótesis razonables sobre qué haría que un viajero prefiera un destino. En cada una, H₀ dice
que **no** hay relación y H₁ que sí; α = 0,05 [tabla 4]:

| Hipótesis | H₀ y H₁ | Prueba y estadístico | Resultado | p-valor | p-valor corregido | Decisión |
|---|---|---|---|---|---|---|
| H1. Visitan los tipos de destino que dicen preferir | H₀: el tipo visitado es independiente del perfil de preferencias. H₁: el tipo visitado depende del perfil | Chi-cuadrado de independencia, perfil × tipo: χ² = 18,35; gl = 8 | coinciden 42,3 % frente a ≈ 40 % del azar | 0,019 | 0,113 | no se rechaza H₀ |
| H2. Con niños eligen más playa y naturaleza | H₀: el tipo visitado es independiente de viajar con niños. H₁: depende de viajar con niños | Chi-cuadrado de independencia, con/sin niños × tipo: χ² = 6,13; gl = 4 | playa y naturaleza: 38,9 % con niños frente a 41,1 % sin niños | 0,190 | 1,000 | no se rechaza H₀ |
| H3. Viajar solo o con otro adulto cambia el destino | H₀: el tipo visitado es independiente del número de adultos. H₁: depende del número de adultos | Chi-cuadrado de independencia, adultos (1 o 2) × tipo: χ² = 6,60; gl = 4 | — | 0,159 | 0,952 | no se rechaza H₀ |
| H4. El género cambia el destino | H₀: el tipo visitado es independiente del género. H₁: depende del género | Chi-cuadrado de independencia, género × tipo: χ² = 0,40; gl = 4 | — | 0,982 | 1,000 | no se rechaza H₀ |
| H5. Mayor `Popularity` atrae más interacciones | H₀: ρ = 0 (no hay relación). H₁: ρ ≠ 0 (bilateral) | Correlación de Spearman entre `Popularity` e interacciones (1000 IDs) | ρ = −0,011 | 0,721 | 1,000 | no se rechaza H₀ |
| H6. Visitan en la mejor temporada del destino | H₀: el % de visitas en temporada es el del azar (72,6 %). H₁: es mayor que el del azar (unilateral) | Permutación, 2000 repeticiones (solo historial) | 73,0 % en temporada | 0,295 | 1,000 | no se rechaza H₀ |

Validez del chi-cuadrado: la menor frecuencia esperada es 123 (mínimo exigido: 5).

**Qué es el p-valor corregido.** Con 6 pruebas y α = 0,05, la probabilidad de al menos un falso
positivo es 1 − 0,95⁶ ≈ 26 %. Bonferroni: p-valor corregido = mín(1; 6 × p-valor), y se rechaza H₀
solo si es < 0,05 (equivale a p-valor < 0,0083). H1: 6 × 0,019 = 0,113 > 0,05.

**Decisión:** en las seis hipótesis **no hay evidencia suficiente para rechazar H₀**. No demuestra
que la relación no exista; dice que estos datos no alcanzan para afirmarla.

💡 **Conclusión de la tarea 2:** ninguna de las explicaciones razonables de las preferencias se
sostiene en estos datos. Las interacciones se comportan como si se hubieran asignado al azar, lo
que es coherente con que el dataset sea sintético. Para el modelado, esto significa que
**esperamos que ningún modelo supere con claridad al azar**, y que cualquier diferencia pequeña
debe confirmarse con 30 semillas y con pruebas que tratan a los usuarios como muestra antes de
afirmarse (ver hallazgo 7).

💡 *Historia para el informe (también para la «Cacería de la alucinación»):* el análisis de H1 se
hizo en tres momentos: (1) mirando solo el historial, «no hay relación» (p-valor = 0,14); (2) con la
unión de tablas, «hay una relación débil pero significativa» (p-valor = 0,019); (3) dentro de un
conjunto de seis hipótesis, «no es concluyente» (p-valor corregido = 0,11). La lección es que la
conclusión depende de cómo se plantea la prueba, y por eso hay que reportar todo el proceso.

## 4. Partición (tarea 3) — ✅ BORRADOR LISTO

> Borrador en lenguaje sencillo. Evidencia: tabla 15 del inventario; detalle en `DECISIONES.md`, paso 4.

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

**Un hallazgo que sale de la partición: los «IDs fríos».** Muchos destinos escondidos son IDs
que nadie tocó en los datos con los que aprende el modelo. Un modelo que solo conoce IDs (popularidad,
factorización de matrices) no tiene información sobre ellos y solo los acertaría por casualidad, así
que su tasa de aciertos tiene un techo que depende del momento [T4-1]:
- al ajustar (entrenamiento → validación): 34,4 % fríos, techo 65,6 %;
- en la evaluación final (entrenamiento + validación → prueba): 22,6 % fríos, techo 77,4 %.

⚠️ Corrección (24-nov): antes decía «33,7 %, techo ≈ 66 %». Ese número compara prueba contra
entrenamiento solo, pero el modelo final aprende con entrenamiento + validación. El 33,7 % sigue
siendo válido en la Tabla 12, que compara estrategias de partición contra entrenamiento. Un modelo que conoce el contenido («este ID es un Taj Mahal»)
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

### 5.1 Modelos de referencia (tarea 4) — decisiones tomadas, en desarrollo
> Evidencia: `results/tarea4_evidencia/` (T4-1 a T4-3); detalle en `DECISIONES.md`, paso 7.

**Qué puede lograr cada modelo, antes de entrenarlo.** Para cada usuario evaluado nos preguntamos
si el modelo puede siquiera darle puntaje al destino escondido [T4-1]:

| Medida | Ajuste (ent → val) | Final (ent+val → prueba) |
|---|---|---|
| HitRate@10 esperado del azar | 1,0 % | 1,0 % |
| Techo de un modelo solo de IDs | 65,6 % | 77,4 % |
| KNN puede puntuarlo | 0,0 % | 0,2 % |

- **Tres modelos:** azar (piso, ≈ 1 %), popularidad y factorización de matrices. **Sin KNN:** solo
  puntúa un destino si otro usuario lo comparte con alguno de los del usuario, y aquí eso casi
  nunca pasa (0–0,2 %).
- **Popularidad = conteo de interacciones**, con desempate al azar: el top-10 es el destino con 6
  interacciones más 9 de los 15 empatados con 5 [T4-2]. La columna `Popularity` se descarta (H5: ρ = −0,011).
- **Factorización de matrices con BPR** («el destino visitado debe quedar por encima de uno no
  visitado»), en PyTorch como el módulo 2; *d* y regularización elegidas en validación.
- **Evaluación común:** se ordenan los 1000 destinos (sin muestrear), quitando los IDs ya vistos
  por el usuario; HitRate, Recall, Precision y NDCG con K = 1, 3, 5, 10, 20, y cobertura; 30
  semillas.
- 💡 Expectativa para el informe: la factorización de matrices quedará probablemente cerca del
  azar, porque casi no hay caminos entre usuarios. Es coherente con la tarea 2.

**Resultados de azar y popularidad** (`src/referencias.py`, 30 semillas; `results/tarea4/RESUMEN.md`):

| Momento | Modelo | HitRate@10 | NDCG@10 | Cobertura@10 | Aciertos / esperados por azar | p-valor |
|---|---|---|---|---|---|---|
| Ajuste (326) | Azar | 1,06 % | 0,48 % | 96,3 % | 3 / 3,3 | 0,64 |
| Ajuste (326) | Popularidad | 0,99 % | 0,33 % | 1,8 % | 3 / 3,3 | 0,64 |
| Final (606) | Azar | 0,91 % | 0,42 % | 99,7 % | 6 / 6,1 | 0,57 |
| Final (606) | Popularidad | 1,16 % | 0,46 % | 1,6 % | 7 / 6,1 | 0,41 |

- 💡 **La popularidad no supera al azar** (prueba binomial unilateral, p-valor = 0,41): acierta 7
  de 606 cuando el azar acertaría 6. Es lo que anticipó la tarea 2 (no hay destinos «estrella»).
- ⚠️ **Trampa a explicar en el informe:** en el momento final los IC 95 % entre semillas no se
  solapan (azar 0,77–1,06; popularidad 1,07–1,24) y parecería que la popularidad gana. Pero esos
  intervalos solo miden la variación por semilla (desempates), no que los 606 usuarios son una
  muestra: el error estándar por muestreo es ≈ 0,40 puntos, mayor que la diferencia (0,25). Por
  eso se agregó la prueba binomial contra el azar.
- 💡 **Prueba con datos de juguete** (`src/probar_factorizacion.py`): antes de usar la
  factorización con los datos reales, se probó con datos fabricados CON señal (cada usuario visita
  destinos de «su» grupo). Acertó 58,4 % frente a un techo de 59 % y 1 % del azar: el código
  aprende. Así, si con los datos reales queda igual al azar, la causa son los datos y no el código.
  Además mostró que hacía falta regularizar mucho más de lo planeado (con λ = 0,0001 memorizaba y
  acertaba 12,8 %). Buen ejemplo de «verificar con un caso de respuesta conocida».
- 💡 **Factorización con los datos reales** (`results/tarea4/RESUMEN_factorizacion.md`):
  9 configuraciones × 3 semillas en validación; ninguna supera al azar (la mejor: 6 de 326 frente
  a 3,3 esperados, p-valor = 0,11), así que se eligió la más simple (d = 8, λ = 1, 7 épocas).
  En prueba: HitRate@10 = 0,94 %, 5,7 aciertos frente a 6,1 del azar (p-valor = 0,57) y sin
  diferencia con la popularidad (prueba pareada, p-valor = 0,57). Cobertura@10 = 17,4 %.

**Tabla para el informe (prueba, 606 usuarios, 30 semillas, K = 10):**

| Modelo | HitRate@10 | NDCG@10 | Cobertura@10 | Aciertos / esperados por azar | p-valor vs azar | Parámetros entrenables |
|---|---|---|---|---|---|---|
| Azar | 0,91 % | 0,42 % | 99,7 % | 6 / 6,1 | 0,57 | 0 |
| Popularidad | 1,16 % | 0,46 % | 1,6 % | 7 / 6,1 | 0,41 | 0 |
| Factorización (BPR) | 0,94 % | 0,43 % | 17,4 % | 5,7 / 6,1 | 0,57 | 15 992 |

💡 **Conclusión de la tarea 4:** ninguna referencia supera al azar, y como la prueba de juguete
demostró que el código aprende cuando hay señal, la causa son los datos (coherente con la
tarea 2). Lo que sí las distingue es la **cobertura**: la popularidad concentra todo en ~16
destinos; la factorización reparte más; el azar cubre todo.
- 💡 **Cobertura:** la popularidad recomienda a todos casi los mismos ~16 destinos (cobertura
  1,6 %), el azar cubre casi todo el catálogo. Dato para la pregunta de negocio 2: la popularidad
  concentraría la demanda en pocas rutas sin acertar más.
- 💡 Decisión sobre la granularidad del ítem (ver `DECISIONES.md`, paso 1): explicarla en el
  informe como una decisión de modelado.

### 5.2 Red de dos torres (tarea 5) — resultados, falta redactar en el informe
> Evidencia: `results/tarea5_evidencia/` (T5-1 a T5-4), `results/tarea5/RESUMEN.md`; detalle en
> `DECISIONES.md`, decisión 7.5 y paso 8.

- 💡 **Por qué dos torres y qué entra:** la app debe recomendar a perfiles nuevos sin ID; torre del
  usuario = 5 casillas de preferencias (lo único que predice el lugar: 23,9 % frente a 20 %,
  p-valor = 0,010, T5-4); torre del destino = embedding del lugar + `Popularity`. `Type`, `State` y
  `BestTimeToVisit` son redundantes con el lugar (T5-1).
- 💡 **Por qué métricas por lugar:** un oráculo que siempre sabe el lugar solo llega a ≈ 5 % de
  HitRate@10 por ID (200 IDs por lugar, T5-3). Se fijaron ANTES de entrenar la red (principal:
  Lugar@3, azar 60 %).
- 💡 **Dimensión d ∈ {2, 4, 8}:** tabla perfil × lugar de 3 × 5 (rango ≤ 3).
- **Prueba con datos simulados:** Lugar@3 = 95,5 % (máximo ≈ 96 %): el código aprende.
- **Ajuste:** 24 configuraciones × 3 semillas; elegida d = 2, BCE, 1 negativo, solo preferencias
  (238 parámetros). «Solo preferencias» le gana casi siempre a «perfil completo».
- **Prueba (606, 30 semillas):** Lugar@3 = 61,8 % (azar 60 %; p-valor corregido = 0,37);
  Lugar@1 = 22,2 % (azar 20 %; 0,21); HitRate@10 = 1,11 % (0,41). Va en la dirección esperada,
  pero no hay evidencia suficiente para afirmar que supera al azar.
- ⚠️ **Maldición del ganador (para el informe):** en validación la elegida «superaba al azar»
  (p-valor = 0,027), pero era la mejor de 24 configuraciones medidas en los mismos 326 usuarios;
  en prueba no se confirma. Buen ejemplo de por qué la prueba se mira una sola vez.

### 6. Evaluación (tarea 6) — ✅ borrador en el informe (sección 8)
- Grupos: pocas (1 interacción para aprender, 280) y muchas (≥ 2, 326). Ningún modelo supera al
  azar en ningún grupo; la red rinde algo mejor con pocas (63,4 % frente a 60,5 % en Lugar@3), sin
  significancia (p-valor = 0,46): solo usa preferencias, no el historial.
- Diagramas de caja de las 30 semillas, con la advertencia del hallazgo 7.
- 5 usuarios de ejemplo (red guardada): acierta 3 de 5. «City, Historical» recibe Kerala primero
  porque así está en los datos: la red sigue a los datos, no a lo declarado.
- Embeddings: red (d = 2) sin proyectar, una curva por lugar (por diseño); factorización con t-SNE,
  sin estructura.

## Preguntas de negocio
- ⏳ P1 (usuarios nuevos): 💡 explicar el arranque en frío con la analogía del bibliotecario
  (teoría, sección 13). Simulación: a los 606 usuarios de prueba se les oculta el historial y se
  recomienda solo con el perfil; el ejemplo es la usuaria 1. Estrategias para comparar:
  popularidad global, popularidad por perfil, contenido (tipo = preferencia) y torre de usuario
  sin ID.
- ⏳ P2 (concentración), P3 (campaña).

## Cacería de la alucinación — ✅ BORRADOR (7 hallazgos del módulo 3)

> ⚠️ **Definición correcta (enunciado, pp. 9–10):** errores que generó **una IA** durante el
> trabajo, no errores de los datos. Cada hallazgo debe incluir: prompt literal, respuesta de la IA,
> cómo se sospechó, evidencia (derivación, experimento o fuente primaria), corrección y lección.
> Se exigen ≥ 5 en todo el trabajo, ≥ 1 por módulo, cubriendo ≥ 3 categorías (Matemáticas,
> Código, Datos del mundo real, Bibliografía, Conceptos). **Un hallazgo inventado se penaliza más
> que uno faltante.**
>
> La propuesta anterior de este archivo («la descripción de Kaggle no coincide con los datos») **no
> cumplía la definición** y se retiró; ese malentendido de la IA es ahora el hallazgo 3.

Redacción completa, con los prompts literales, en `informe/index.html`, sección 6.

1. **Conclusión estadística prematura (Datos, gravedad alta):** la IA afirmó «las visitas no
   tienen relación con las preferencias» (solo con el historial, sin prueba) y luego «relación
   significativa» (p-valor = 0,019). La evidencia es la tabla I-6 y la corrección de Bonferroni en D-4
   (p-valor corregido = 0,11).
2. **Correlación incompleta (Datos, baja):** reportó «0,03», omitiendo la de reseñas (0,044) y
   los p-valores (I-5).
3. **Hallazgo mal definido (interpretación del enunciado, media):** propuso como «candidato
   fuerte» un error de los datos y no de una IA.
4. **Dataset «por escoger» (Datos, baja):** propuso escoger el dataset cuando el enunciado (p. 5)
   ya lo fija.
5. **Hipótesis que no coincidían con la prueba (Matemáticas, media):** H2 («con niños eligen
   **más** playa y naturaleza») y H5 («mayor `Popularity` → **más** interacciones») tienen
   dirección, pero se contrastaron con un chi-cuadrado de independencia (cualquier diferencia) y
   un Spearman bilateral. La tabla no mostraba H₀ ni H₁, así que no se veía. Se detectó al pedir
   H₀, H₁ y la prueba de cada fila. Corrección: H₀/H₁ explícitas según la prueba (D-4); las
   conclusiones no cambian. Prompt literal: «sigamos con el análisi descriptivo».
6. **Afirmaciones del informe que no coincidían con sus tablas (Datos/resultados, media):** «la
   red es el único modelo por encima del azar en las dos métricas por lugar» (la factorización
   también lo está: 60,4 % y 20,4 %), «la única que puede recomendar a un perfil nuevo», 25 s
   escrito frente a 23 s en el archivo, «igual de seguido» con conteos 63 y 80, y un p-valor sin
   corregir. Detectado al pedir verificar todo el informe (`src/verificar_informe.py`).
7. **«Confirmar con 30 semillas» (Conceptos, alta):** la conclusión de la tarea 2 decía que las
   diferencias pequeñas se confirmaban con 30 semillas; las semillas no miden el muestreo de
   usuarios (error ≈ 0,40 puntos frente a una diferencia de 0,25 entre popularidad y azar;
   binomial p-valor = 0,41). Corrección: semillas + pruebas que tratan a los usuarios como muestra.

- ✅ Verificaciones sin error: el 40 % esperado (recalculado: 40,1 %), Poisson como referencia,
  partición sin fugas.
- ⏳ **Pendiente (Bibliografía):** verificar en la fuente primaria las referencias que propuso la
  IA en `teoria/` (DOI, volumen, páginas). Si alguna falla, es un hallazgo de esa categoría.
- 💡 Hasta ahora el módulo 3 cubre sobre todo la categoría «Datos» (y una de «Matemáticas», el hallazgo 5). Conviene buscar activamente
  errores de **Código** y **Matemáticas** cuando se programen los modelos y las métricas (por
  ejemplo, revisar a mano el cálculo de NDCG y la derivación de BPR o BCE).

## Ética (aporte del módulo 3 a la sección común)
- 💡 **Privacidad por diseño en la app:** no hay registro ni base de datos. Los perfiles nuevos
  existen solo durante la sesión del navegador y no se guardan, así que la app **no almacena datos
  personales**. Esto también evita la tentación de reutilizarlos sin consentimiento.
- 💡 En el preprocesamiento se eliminaron `Name` y `Email` (aunque sean ficticios): el modelo no
  necesita identificadores personales, solo un ID anónimo y variables de preferencia.
- 💡 Sesgos: los datos son sintéticos (3 perfiles de preferencias, género binario, 5 lugares), así
  que el modelo no representa a la población real; si se implementara, habría que reentrenar con
  datos reales y auditar los sesgos (por ejemplo, la concentración en pocos destinos, P2).
- 💡 Riesgo si se implementara: recomendar siempre los mismos destinos podría saturar rutas
  (conexión con el módulo 1) y reducir la diversidad de lo que ven los clientes.

## Fundamentación matemática
- ✅ Borrador en `teoria/01_fundamentos_sistemas_recomendacion.md` (factorización de matrices,
  derivación de BCE y BPR, NCF y dos torres, métricas). Falta adaptarlo al formato del informe una
  vez elegido el modelo final.
