# Fundamentos de los sistemas de recomendación

> Documento de estudio del módulo 3. Está escrito para alguien que nunca ha construido un
> sistema de recomendación. Cada sección termina con un recuadro **«En nuestro trabajo»** que
> conecta la teoría con el dataset de destinos de India y con lo que pide el enunciado
> (`docs/instrucciones_trabajo_3.pdf`, páginas 5–7).
>
> Las fórmulas usan LaTeX; GitHub y VS Code (con vista previa de Markdown) las muestran bien.

## Índice

1. [¿Qué problema resuelve un recomendador?](#1-qué-problema-resuelve-un-recomendador)
2. [Notación: la matriz usuario–ítem](#2-notación-la-matriz-usuarioítem)
3. [Retroalimentación explícita e implícita](#3-retroalimentación-explícita-e-implícita)
4. [Familias de recomendadores](#4-familias-de-recomendadores)
5. [Referencia 1: popularidad](#5-referencia-1-popularidad)
6. [Referencia 2a: vecinos más cercanos (KNN)](#6-referencia-2a-vecinos-más-cercanos-knn)
7. [Referencia 2b: factorización de matrices](#7-referencia-2b-factorización-de-matrices)
8. [De predecir calificaciones a ordenar: el problema top-K](#8-de-predecir-calificaciones-a-ordenar-el-problema-top-k)
9. [Muestreo de negativos](#9-muestreo-de-negativos)
10. [Funciones de pérdida derivadas de un modelo probabilístico](#10-funciones-de-pérdida-derivadas-de-un-modelo-probabilístico)
11. [Embeddings](#11-embeddings)
12. [Recomendadores neuronales: NCF y dos torres](#12-recomendadores-neuronales-ncf-y-dos-torres)
13. [Arranque en frío (usuarios nuevos)](#13-arranque-en-frío-usuarios-nuevos)
14. [Partición de los datos sin fugas](#14-partición-de-los-datos-sin-fugas)
15. [Métricas de evaluación](#15-métricas-de-evaluación)
16. [Concentración y sesgo de popularidad](#16-concentración-y-sesgo-de-popularidad)
17. [Visualizar embeddings: t-SNE y UMAP](#17-visualizar-embeddings-t-sne-y-umap)
18. [Comparación justa entre modelos](#18-comparación-justa-entre-modelos)
19. [Mapa: del enunciado a la teoría](#19-mapa-del-enunciado-a-la-teoría)
20. [Glosario](#20-glosario)
21. [Referencias](#21-referencias)

---

## 1. ¿Qué problema resuelve un recomendador?

Una empresa tiene **muchos ítems** (productos, películas, destinos) y **muchos usuarios**, y cada
usuario solo puede ver unos pocos. El recomendador decide **qué K ítems mostrarle a cada usuario**
para maximizar la probabilidad de que alguno le interese.

Fíjate en que el problema **no** es «predecir un número» (eso es regresión) ni «asignar una
clase» (eso es clasificación). Es un problema de **ordenamiento (ranking)**: entre todos los
ítems que el usuario aún no conoce, ¿cuáles van primero? Lo que importa es el **orden** de la
lista, y sobre todo sus primeras posiciones, porque son las únicas que el usuario ve.

> **En nuestro trabajo.** La empresa de transporte quiere sugerir destinos turísticos de India
> a sus clientes. Una buena sugerencia genera un viaje; una mala hace que el cliente ignore las
> siguientes. Por eso las métricas que pide el enunciado (Precision, Recall, NDCG@K) miden la
> calidad de una **lista corta**, no el error de un número.

---

## 2. Notación: la matriz usuario–ítem

Se usa esta notación durante todo el documento:

| Símbolo | Significado |
|---|---|
| $\mathcal{U}$, $m = \lvert\mathcal{U}\rvert$ | conjunto de usuarios y su número |
| $\mathcal{I}$, $n = \lvert\mathcal{I}\rvert$ | conjunto de ítems (destinos) y su número |
| $r_{ui}$ | interacción observada del usuario $u$ con el ítem $i$ (calificación, visita) |
| $\hat r_{ui}$ o $\hat y_{ui}$ | lo que el modelo predice para el par $(u,i)$ |
| $\mathcal{I}_u$ | ítems con los que $u$ ya interactuó (su **historial**) |
| $\mathcal{U}_i$ | usuarios que interactuaron con $i$ |
| $\Omega$ | conjunto de pares $(u,i)$ observados |
| $K$ | longitud de la lista recomendada |

Las interacciones se organizan en una **matriz** $R \in \mathbb{R}^{m\times n}$ con un usuario por
fila y un ítem por columna. Casi todas las celdas están **vacías**: nadie ha visitado todos los
destinos. La **densidad** mide qué fracción está llena:

```math
\text{densidad} = \frac{\lvert\Omega\rvert}{m \cdot n}
```

Una celda vacía **no** significa «no le gusta»; significa «no sabemos». Esta es la dificultad
central de los recomendadores y vuelve a aparecer en las secciones 8 y 9.

> **En nuestro trabajo.** El enunciado pide reportar el número de usuarios, destinos e
> interacciones, la escala de las calificaciones y la densidad. Con los archivos crudos
> tenemos $m \approx 1000$ usuarios, $n = 1000$ IDs de destino y unas 1000 interacciones en cada
> tabla, así que la densidad es $1000/(1000\cdot1000)=0{,}001$ (0,1 %). Es una matriz
> **extremadamente dispersa**. Ver `DECISIONES.md` para el análisis de qué es un «destino» en
> este dataset.

### 2.1 ¿Qué es exactamente una interacción?

Una **interacción** es **un registro de que un usuario concreto tuvo contacto con un ítem
concreto**. Es la materia prima del filtrado colaborativo: cada interacción llena una celda
$(u,i)$ de la matriz $R$.

Mínimo, una interacción tiene la forma `(usuario, ítem)`. Muchas veces trae datos extra:

| Componente | Para qué sirve | Ejemplo |
|---|---|---|
| `usuario`, `ítem` | ubica la celda en $R$ | `UserID=525`, `DestinationID=760` |
| **valor** (opcional) | qué tan fuerte o positiva fue | `ExperienceRating=3` |
| **momento** (opcional) | ordenar en el tiempo y particionar | `VisitDate=2024-01-01` |
| **tipo de evento** | distinguir señales de distinta fuerza | «visitó» frente a «reseñó» frente a «hizo clic» |

**Lo que NO es una interacción:** los atributos fijos del usuario (`Gender`, `Preferences`) o del
ítem (`Type`, `State`). Esos son **contenido**. Describen a una de las dos partes, no la relación
entre ambas. La distinción importa porque:
- el **filtrado colaborativo** solo usa interacciones;
- los modelos **basados en contenido** y los **híbridos** (nuestra red) usan ambos.

**Decisiones que hay que tomar al definir las interacciones:**

1. **Qué eventos cuentan.** Si hay varias fuentes (en nuestro caso, reseñas e historial de
   visitas), se puede usar una, otra o la unión. Unirlas da más datos, pero mezcla señales que
   pueden significar cosas distintas.
2. **Qué hacer con el valor.** Hay tres opciones:
   - ignorarlo (toda interacción es un positivo, $y_{ui}=1$);
   - usarlo como **umbral** (solo es positivo si $r_{ui}\ge 4$, y una calificación baja se descarta o
     se trata como negativo explícito);
   - usarlo como **peso** (la confianza $c_{ui}$ en Hu et al., 2008).

   Con un umbral se pierden datos; sin él, un viaje que el usuario odió cuenta como «le gustó».
3. **Repeticiones.** Si el mismo usuario interactúa varias veces con el mismo ítem, ¿cuenta una vez
   o varias? En la matriz binaria cuenta una sola vez.

**Cómo se ve en nuestros datos.** Una fila de `Final_Updated_Expanded_UserHistory.csv`:

```
HistoryID  UserID  DestinationID  VisitDate   ExperienceRating
        1     525            760  2024-01-01                 3
```

se lee: «el usuario 525 visitó el destino 760 el 1 de enero de 2024 y calificó la experiencia
con 3». Es una interacción $(u=525,\ i=760)$ con valor 3. Una fila de reseñas, `(UserID=327,
DestinationID=178, Rating=2)`, es otra interacción de otro tipo de evento. En cambio, la fila de
`Users.csv` que dice que el usuario 1 prefiere «Beaches, Historical» **no** es una interacción:
es contenido del usuario.

### 2.2 ¿Qué es exactamente un ítem? (granularidad)

Antes de construir $R$ hay que decidir **qué cuenta como un ítem distinto**: qué va en cada
columna. Esto se llama la **granularidad** del ítem, y casi nunca viene dada: es una decisión de
modelado. Ejemplos de otros dominios:

| Dominio | Granularidad fina | Granularidad gruesa |
|---|---|---|
| Ropa | camiseta talla M color rojo (SKU) | modelo de camiseta |
| Música | una grabación concreta (en vivo, remasterizada) | la canción |
| Viajes | paquete «Taj Mahal, 3 noches, hotel X» | el lugar «Taj Mahal» |

**¿Qué cambia según la granularidad?**

1. **Tamaño del catálogo $n$** y, con él, la **densidad**. Agrupar ítems sube la densidad: hay
   menos columnas y las interacciones se concentran.
2. **Qué significa recomendar.** El ítem es lo que aparece en la lista. Si dos ítems son
   indistinguibles para el cliente, recomendar ambos en la misma lista es un desperdicio.
3. **Qué tan bien funcionan las métricas.** Precision@K, Recall@K y NDCG@K solo discriminan si
   hay **bastantes más ítems candidatos que K**. Si el catálogo tiene 5 ítems y $K=5$, cualquier
   modelo «recomienda todo» y acierta siempre. La **cobertura** tampoco dice nada (siempre 100 %).
4. **Cuánto puede aprender el CF.** Para aprender el embedding de un ítem, este debe tener
   interacciones en entrenamiento. Un ítem con 0–1 interacciones queda prácticamente sin
   aprender (es un **ítem frío**). Agrupar le da más datos a cada ítem.
5. **Qué pasa con las repeticiones.** Si se agrupan ítems, dos interacciones que eran distintas
   (usuario 7 con el destino 3 y usuario 7 con el destino 8) pueden convertirse en **la misma**
   (usuario 7 con «Jaipur»).
6. **El contenido.** Si varios ítems comparten todos sus atributos, el contenido no los distingue
   entre sí. Solo el ID (o un atributo que varíe, como `Popularity`) los separa.

**Criterio general:** el ítem debe ser **la unidad que el negocio puede ofrecer y que el cliente
percibe como una opción distinta**, siempre que haya **suficientes datos por ítem** para aprender
y **suficientes ítems** para que el ranking tenga sentido. Cuando estos criterios chocan, se
documenta la tensión y se elige la opción que permite responder las preguntas del estudio.

> **En nuestro trabajo.** Los 1000 `DestinationID` corresponden a solo 5 lugares reales, así que
> los criterios chocan: la unidad con sentido de negocio (el lugar, 5 ítems) deja el ranking sin
> sentido, y la unidad del dataset (el ID, 1000 ítems) tiene sentido para el ranking pero no
> para el negocio. Las cifras concretas de cada opción están en `DECISIONES.md`, Paso 1.

---

## 3. Retroalimentación explícita e implícita

- **Explícita:** el usuario dice cuánto le gustó algo (estrellas de 1 a 5). Es precisa, pero escasa,
  y tiene sesgo: la gente califica sobre todo lo que eligió (y le gustó).
- **Implícita:** se infiere de la conducta (visitó, compró, hizo clic). Es abundante, pero solo hay
  **positivos**: una celda vacía mezcla «no le interesa» con «no lo conoce».

Hay dos formas habituales de usar una calificación:

1. **Como regresión:** predecir $r_{ui}$ y minimizar el error cuadrático (la idea clásica del
   Premio Netflix).
2. **Como señal implícita:** convertirla en $y_{ui}\in\{0,1\}$ («interactuó» o «interactuó y le
   gustó», por ejemplo $r_{ui}\ge 4$) y aprender a **ordenar**.

Para recomendar top-K se usa sobre todo la segunda. Un modelo con poco error cuadrático no
necesariamente ordena bien.

> **En nuestro trabajo.** Hay dos fuentes:
> `Final_Updated_Expanded_Reviews.csv` (`Rating` de 1 a 5) y
> `Final_Updated_Expanded_UserHistory.csv` (`ExperienceRating` de 1 a 5 + `VisitDate`).
> Ambas son calificaciones explícitas, pero lo más natural es tratarlas como **interacción
> implícita** («el usuario visitó o reseñó el destino»). La calificación se puede usar como peso
> o como umbral de relevancia. Es una decisión que tomaremos y justificaremos en `DECISIONES.md`.

---

## 4. Familias de recomendadores

| Familia | Idea | Necesita | Problema típico |
|---|---|---|---|
| **No personalizado** (popularidad) | a todos lo mismo: lo más popular | solo conteos | nada personal, concentra la demanda |
| **Basado en contenido** | ítems parecidos a los que te gustaron, según sus atributos | atributos de ítems/usuarios | se queda en «más de lo mismo» |
| **Filtrado colaborativo (CF)** | usuarios parecidos a ti disfrutaron esto | solo la matriz $R$ | no sirve sin historial (arranque en frío) |
| **Híbrido** | combina CF + contenido | ambos | más complejo |

El filtrado colaborativo se divide en dos:
- **Basado en memoria** (KNN): calcula similitudes directamente sobre $R$.
- **Basado en modelos** (factorización de matrices, redes): aprende parámetros que resumen $R$.

> **En nuestro trabajo.** El enunciado pide exactamente: popularidad (referencia 1), CF clásico
> con KNN **o** factorización de matrices (referencia 2) y una red neuronal que use las
> interacciones **más al menos una variable de contenido**, es decir, un **híbrido**.
> Variables de contenido disponibles: del destino, `Type`, `State`, `Popularity`, `BestTimeToVisit`;
> del usuario, `Preferences`, `Gender`, `NumberOfAdults`, `NumberOfChildren`.

---

## 5. Referencia 1: popularidad

Es el recomendador más simple. Se calcula un puntaje por ítem, igual para todos los usuarios:

```math
s_i = \lvert \mathcal{U}_i \rvert \quad\text{(número de interacciones en entrenamiento)}
```

y a cada usuario se le recomiendan los $K$ ítems de mayor $s_i$ que **todavía no estén en su
historial** $\mathcal{I}_u$.

Hay variantes: la calificación media (con un suavizado bayesiano para que un ítem con una sola
reseña de 5 no quede de primero) o la popularidad reciente.

**¿Por qué es obligatorio?** Porque es sorprendentemente difícil de superar. Si una red neuronal
no le gana a «recomendar lo más popular», la red no está aprendiendo nada personal. También es
la respuesta natural para usuarios nuevos (sección 13).

> **En nuestro trabajo.** El dataset trae una columna `Popularity` (7,5–9,5) en los destinos.
> **Ojo:** esa columna es un atributo que viene dado, no algo que nosotros contamos en
> entrenamiento. Podemos comparar ambas versiones: popularidad = conteo de interacciones de
> entrenamiento frente a popularidad = columna `Popularity`.

---

## 6. Referencia 2a: vecinos más cercanos (KNN)

**KNN usuario–usuario:** para recomendarle a $u$, se buscan los $k$ usuarios más parecidos y se
puntúan los ítems que ellos consumieron:

```math
\hat r_{ui} = \frac{\sum_{v\in N_k(u)} \text{sim}(u,v)\, r_{vi}}{\sum_{v\in N_k(u)} \lvert\text{sim}(u,v)\rvert}
```

**KNN ítem–ítem** (suele funcionar mejor): un ítem $i$ es bueno para $u$ si se parece a los ítems
que $u$ ya consumió:

```math
\hat r_{ui} = \sum_{j \in \mathcal{I}_u} \text{sim}(i,j)\, r_{uj}
```

La similitud más común es el **coseno** entre las columnas (o filas) de $R$:

```math
\text{sim}(i,j) = \frac{\mathbf r_{\cdot i}\cdot \mathbf r_{\cdot j}}{\lVert\mathbf r_{\cdot i}\rVert\,\lVert \mathbf r_{\cdot j}\rVert}
```

**Limitación clave:** si dos ítems no comparten **ningún** usuario, su similitud es 0. En matrices
muy dispersas casi todas las similitudes son 0, y KNN apenas puede recomendar.

---

## 7. Referencia 2b: factorización de matrices

### 7.1 La idea

Se supone que los gustos se explican con **pocos factores latentes** (por ejemplo «le gusta la
playa», «viaja con niños», «busca historia»). Cada usuario $u$ se representa con un vector
$\mathbf p_u\in\mathbb R^d$ y cada ítem $i$ con un vector $\mathbf q_i\in\mathbb R^d$, con $d$ pequeño
(entre 8 y 64). La afinidad es su producto punto:

```math
\hat r_{ui} = \mathbf p_u^\top \mathbf q_i = \sum_{f=1}^{d} p_{uf}\, q_{if}
```

En forma matricial, $R \approx P Q^\top$, con $P\in\mathbb R^{m\times d}$ y $Q\in\mathbb R^{n\times d}$. Es
como una descomposición de bajo rango, pero **solo se ajusta sobre las celdas observadas**.

Se suelen añadir **sesgos**: una media global $\mu$, un sesgo por usuario (unos califican más alto)
y uno por ítem (unos ítems son mejores en general):

```math
\hat r_{ui} = \mu + b_u + b_i + \mathbf p_u^\top \mathbf q_i
```

### 7.2 Función objetivo (versión explícita)

```math
\min_{P,Q,b}\; \sum_{(u,i)\in\Omega}\big(r_{ui}-\hat r_{ui}\big)^2 \;+\; \lambda\Big(\lVert\mathbf p_u\rVert^2+\lVert\mathbf q_i\rVert^2+b_u^2+b_i^2\Big)
```

El término $\lambda$ es **regularización L2**. Sin ella, con tan pocos datos, el modelo memoriza.

**Interpretación probabilística** (útil para la derivación que pide el enunciado): si
$r_{ui} \sim \mathcal N(\mathbf p_u^\top\mathbf q_i, \sigma^2)$ y se ponen *priors*
$\mathbf p_u,\mathbf q_i \sim \mathcal N(\mathbf 0, \sigma_0^2 I)$, el estimador **MAP** (máximo a
posteriori) es exactamente el problema anterior con $\lambda = \sigma^2/\sigma_0^2$. Es decir:
**error cuadrático = verosimilitud gaussiana** y **L2 = prior gaussiano**.

### 7.3 Cómo se optimiza

**Descenso de gradiente estocástico (SGD).** Para cada $(u,i)\in\Omega$, con $e_{ui} = r_{ui}-\hat r_{ui}$:

```math
\mathbf p_u \leftarrow \mathbf p_u + \eta\,(e_{ui}\,\mathbf q_i - \lambda\,\mathbf p_u), \qquad
\mathbf q_i \leftarrow \mathbf q_i + \eta\,(e_{ui}\,\mathbf p_u - \lambda\,\mathbf q_i)
```

**ALS (mínimos cuadrados alternados).** Si se fija $Q$, el problema en $\mathbf p_u$ es una regresión
ridge con solución cerrada, y lo mismo al revés. Se alterna hasta converger. Es la base de la
versión implícita de Hu, Koren y Volinsky (2008).

### 7.4 Versión implícita

Con datos implícitos se usa $y_{ui}\in\{0,1\}$ y se aprende a **ordenar** con las pérdidas de la
sección 10 (BCE con negativos o BPR) en lugar del error cuadrático.

> **En nuestro trabajo.** La factorización de matrices será probablemente nuestra referencia de
> CF, porque KNN casi no funciona con una densidad de 0,1 %. Hay que reportar su número de
> parámetros: $(m+n)\,d$ (+ $m+n+1$ si lleva sesgos).

---

## 8. De predecir calificaciones a ordenar: el problema top-K

En producción, para cada usuario:

1. Se calcula un puntaje $\hat y_{ui}$ para **todos** los ítems candidatos.
2. Se **excluyen** los ítems de su historial de entrenamiento (no se recomienda algo que ya hizo,
   salvo que la recompra tenga sentido).
3. Se ordena de mayor a menor y se toman los $K$ primeros.

Por eso el entrenamiento debe premiar que los ítems relevantes queden **por encima** de los
irrelevantes. Esa es la motivación de las pérdidas de la sección 10.

---

## 9. Muestreo de negativos

Con datos implícitos solo hay positivos ($y_{ui}=1$). Si el modelo solo ve positivos, aprende a
decir «1» siempre. La solución es **fabricar negativos**: para cada positivo $(u,i)$ se sortean
$N$ ítems $j\notin\mathcal I_u$ y se tratan como $y_{uj}=0$.

Decisiones que el enunciado pide justificar:

- **Cuántos negativos por positivo ($N$).** Habitualmente entre 1 y 10. Más negativos dan una
  señal más fuerte pero cuestan más cómputo y aumentan el riesgo de «falsos negativos» (ítems que
  el usuario sí querría). Es un hiperparámetro que se ajusta en validación.
- **De qué distribución se sortean.**
  - *Uniforme:* simple y estándar.
  - *Proporcional a la popularidad:* produce negativos «más difíciles» y corrige el sesgo de
    popularidad, porque sin él el modelo aprende a castigar los ítems raros.
- **Re-muestrear en cada época.** Se recomienda sortear negativos nuevos en cada época, para no
  fijar siempre los mismos falsos negativos.

---

## 10. Funciones de pérdida derivadas de un modelo probabilístico

El enunciado pide **derivar** la pérdida a partir de un modelo probabilístico. Estas son las dos
más usadas.

### 10.1 Entropía cruzada binaria (BCE) con negativos

**Modelo.** La interacción es una variable de Bernoulli cuya probabilidad la da el modelo a
través de la sigmoide $\sigma(z)=1/(1+e^{-z})$:

```math
Y_{ui}\sim\text{Bernoulli}(\hat y_{ui}),\qquad \hat y_{ui} = \sigma\big(f_\theta(u,i)\big)
```

**Verosimilitud** sobre los positivos observados $\Omega^+$ y los negativos muestreados $\Omega^-$,
suponiendo independencia:

```math
L(\theta) = \prod_{(u,i)\in\Omega^+}\hat y_{ui}\;\prod_{(u,j)\in\Omega^-}(1-\hat y_{uj})
```

**Log-verosimilitud negativa** (lo que se minimiza):

```math
\mathcal L_{\text{BCE}}(\theta) = -\sum_{(u,i)\in\Omega^+}\log \hat y_{ui} \;-\; \sum_{(u,j)\in\Omega^-}\log(1-\hat y_{uj}) \;+\; \lambda\lVert\theta\rVert^2
```

Esta es la **entropía cruzada binaria**. El término $\lambda\lVert\theta\rVert^2$ vuelve a ser
un prior gaussiano sobre los parámetros (MAP).

**Gradiente útil** (para entender qué hace): si $z = f_\theta(u,i)$, entonces
$\partial \mathcal L/\partial z = \hat y_{ui} - y_{ui}$. Cuando el modelo se equivoca mucho, el
empujón es grande; cuando acierta, es casi nulo.

### 10.2 BPR (*Bayesian Personalized Ranking*)

**Modelo.** En lugar de modelar cada celda, BPR modela **preferencias por pares**: «$u$ prefiere
el ítem visto $i$ al no visto $j$», que se escribe $i >_u j$:

```math
P(i >_u j \mid \theta) = \sigma\big(\hat x_{ui} - \hat x_{uj}\big)
```

donde $\hat x_{ui}$ es el puntaje del modelo (por ejemplo $\mathbf p_u^\top \mathbf q_i$).

**Datos de entrenamiento:** tripletas $D_S=\{(u,i,j): i\in\mathcal I_u,\ j\notin\mathcal I_u\}$
(otra vez con muestreo de negativos).

**Criterio MAP** con prior $\theta\sim\mathcal N(0,\lambda^{-1}I)$:

```math
\max_\theta\; \log P(\theta\mid >_u) = \sum_{(u,i,j)\in D_S}\log\sigma(\hat x_{ui}-\hat x_{uj}) - \lambda\lVert\theta\rVert^2 + \text{cte.}
```

y la pérdida es su negativo:

```math
\mathcal L_{\text{BPR}} = -\sum_{(u,i,j)\in D_S}\log\sigma(\hat x_{ui}-\hat x_{uj}) + \lambda\lVert\theta\rVert^2
```

**¿Por qué es atractiva?** Porque optimiza directamente el **orden** (que $i$ quede encima de $j$),
y eso es lo que miden las métricas top-K. Rendle et al. (2009) muestran que maximiza una
aproximación suave del AUC.

### 10.3 ¿Cuál elegir?

| | BCE + negativos | BPR |
|---|---|---|
| Modela | probabilidad absoluta de interacción | orden relativo entre pares |
| Salida interpretable como probabilidad | sí | no (solo el orden) |
| Uso típico | NCF (He et al., 2017) | factorización de matrices para ranking |

Para la **pregunta de negocio 3** (estimar conversiones de una campaña) conviene la BCE, porque
da probabilidades $\hat y_{ui}$ que se pueden sumar: conversiones esperadas
$\approx\sum_{u\in\text{grupo}}\hat y_{u,\text{destino}}$, siempre que estén **calibradas**.

---

## 11. Embeddings

Un **embedding** es una tabla de parámetros $E\in\mathbb R^{n\times d}$: el ítem con índice $i$ se
representa por la fila $\mathbf e_i$. Equivale a multiplicar un vector *one-hot* por una matriz,
pero sin calcular la multiplicación. Los vectores $\mathbf p_u$ y $\mathbf q_i$ de la factorización
de matrices **ya son embeddings**.

Las variables categóricas de contenido (`Type`, `State`, cada preferencia del usuario) también
se representan con embeddings pequeños, y las numéricas (`Popularity`, `NumberOfAdults`) se
**estandarizan** y se concatenan.

**¿Cómo justificar la dimensión $d$?**
- **Capacidad frente a datos:** los parámetros crecen como $(m+n)\,d$. Con unas 2000
  interacciones, un $d$ grande (como 128) tiene muchos más parámetros que datos y se sobreajusta.
- **Regla empírica:** $d \approx \lceil \text{(número de categorías)}^{1/4}\rceil$ para variables
  categóricas, que es una heurística conocida de Google. Para IDs, explorar $d\in\{4,8,16,32\}$ en
  validación.
- **Criterio final:** la curva de NDCG@K en validación frente a $d$, eligiendo el $d$ más pequeño
  que esté cerca del mejor (principio de parsimonia).

---

## 12. Recomendadores neuronales: NCF y dos torres

### 12.1 NCF — *Neural Collaborative Filtering* (He et al., 2017)

Reemplaza el producto punto por una red que aprende la interacción:

- **GMF** (factorización generalizada): $\phi^{GMF} = \mathbf p_u \odot \mathbf q_i$ (producto elemento
  a elemento). Si la capa de salida tiene pesos todos iguales a 1, se recupera la factorización de
  matrices clásica.
- **MLP:** $\phi^{MLP} = \text{MLP}([\mathbf p_u ; \mathbf q_i])$, la concatenación pasada por capas
  densas con ReLU.
- **NeuMF:** concatena ambas y aplica una salida sigmoide:

```math
\hat y_{ui} = \sigma\!\left(\mathbf h^\top\big[\phi^{GMF};\ \phi^{MLP}\big]\right)
```

Se entrena con **BCE + muestreo de negativos** (sección 10.1).

**Agregar contenido:** se concatenan, a la entrada del MLP, los embeddings de los atributos del
destino (`Type`, `State`) y del usuario (`Preferences`, etc.).

### 12.2 Modelo de dos torres (*two-tower*)

Usa dos redes independientes:
- **Torre de usuario:** $\mathbf u = g_\theta(\text{ID usuario}, \text{preferencias}, \text{género}, \text{adultos}, \text{niños})\in\mathbb R^d$
- **Torre de ítem:** $\mathbf v = h_\phi(\text{ID destino}, \text{tipo}, \text{estado}, \text{popularidad}, \text{temporada})\in\mathbb R^d$
- **Puntaje:** $\hat x_{ui} = \mathbf u^\top\mathbf v$ (o un coseno), seguido de una sigmoide o de BPR.

**Ventajas para nosotros:**
1. **Arranque en frío natural.** Un usuario nuevo sin ID conocido se representa solo con sus
   preferencias: se pasa por la torre de usuario con el ID «desconocido» y se obtiene un
   vector $\mathbf u$. Es justo lo que pide la app («crear un perfil nuevo con un listado de
   preferencias»).
2. **Escalable.** Los vectores de ítem se precalculan, y recomendar es buscar los productos
   punto más altos.
3. Los vectores $\mathbf v$ son los **embeddings de destinos** que se proyectan con t-SNE/UMAP.

### 12.3 Optimización

Se usa Adam, mini-lotes, *early stopping* sobre NDCG@K de validación y regularización (L2 o
*dropout*). Los hiperparámetros a buscar son $d$, el número y ancho de las capas, $N$ negativos,
la tasa de aprendizaje y $\lambda$.

---

## 13. Arranque en frío (usuarios nuevos)

Un usuario sin historial no tiene fila en $R$, así que el CF puro no puede recomendarle nada.
Estrategias para comparar (la **pregunta de negocio 1** pide al menos dos):

1. **Popularidad global:** lo más popular para todos.
2. **Popularidad por segmento:** lo más popular entre los usuarios con las mismas preferencias o
   el mismo perfil (por ejemplo «viaja con niños»).
3. **Basado en contenido:** destinos cuyo `Type` coincide con las preferencias declaradas.
4. **Torre de usuario con solo contenido:** el modelo de dos torres sin ID (sección 12.2).

**¿Cómo simularlo?** Se toma un grupo de usuarios de prueba, se **oculta todo su historial** en
entrenamiento (como si fueran nuevos) y se mide qué tan bien cada estrategia predice los destinos
que en realidad visitaron.

**Analogía.** Un bibliotecario recomienda bien a un lector habitual: sabe qué ha leído y qué
leen lectores parecidos. Cuando entra alguien que nunca ha venido, no tiene historial con qué
compararlo, y solo puede preguntarle qué le gusta (contenido) o darle lo más leído (popularidad).

**Hay dos arranques en frío:**
- **Usuario frío:** un cliente sin historial. El filtrado colaborativo no tiene vector $\mathbf p_u$
  para él.
- **Ítem frío:** un destino sin interacciones en entrenamiento. No tiene vector $\mathbf q_i$, así
  que un modelo que solo usa IDs **nunca** lo recomienda. Un modelo con contenido sí puede, porque
  conoce sus atributos (tipo, lugar).

> **En nuestro trabajo — ejemplo.** La usuaria 1 (prefiere «Beaches, Historical», viaja sola)
> tiene dos interacciones: Goa Beaches (ID 467, entrenamiento) y Taj Mahal (ID 941, prueba). Para
> simularla como **cliente nuevo**, se oculta su visita a Goa y cada estrategia recibe solo el
> perfil; luego se mira si el destino 941 aparece en el top-10. Repetido con los 606 usuarios de
> prueba, esto da el desempeño de cada estrategia para clientes nuevos (pregunta de negocio 1),
> **sin necesidad de otra partición**.
>
> **Ítems fríos en nuestros datos:** con la partición elegida (sección 14), el 33,7 % de los IDs
> de prueba no tiene ninguna interacción en entrenamiento. La popularidad y la factorización de
> matrices no pueden acertarlos (techo de HitRate ≈ 66 %); la red con contenido sí puede.

---

## 14. Partición de los datos sin fugas

**Regla del enunciado:** ninguna interacción de prueba se puede usar para entrenar, y todos los
modelos usan **las mismas particiones**.

Opciones:

| Estrategia | Cómo | Cuándo |
|---|---|---|
| **Temporal global** | todo lo anterior a la fecha $t_1$ = entrenamiento, $[t_1,t_2)$ = validación, posterior = prueba | si las fechas son reales y variadas; es lo más realista |
| **Leave-one-out / leave-k-out por usuario** | para cada usuario con suficientes interacciones, se reserva 1 (o $k$) en prueba y otra en validación | estándar en NCF; garantiza que el usuario existe en entrenamiento |
| **Aleatoria por interacción** | 80/10/10 al azar | simple, pero puede dejar usuarios sin entrenamiento |
| **Por usuario (usuarios fríos)** | se separan **usuarios** completos | para evaluar el arranque en frío |

**Problema típico:** usuarios con **una sola** interacción. Si va a prueba, el usuario no existe en
entrenamiento (es frío); si va a entrenamiento, no se puede evaluar. Hay que decidir de forma
explícita qué hacer con ellos.

**Otras fugas que hay que evitar:** calcular la popularidad, los escaladores o los codificadores
con datos de prueba. Todo se **ajusta solo con entrenamiento**.

**Para qué sirve cada conjunto (analogía del examen):**
- **Entrenamiento** = lo que el estudiante estudia.
- **Validación** = el examen de práctica: sirve para decidir *cómo* estudiar (los
  hiperparámetros) y se puede repetir.
- **Prueba** = el examen final: se presenta **una sola vez**, al final. Si se mira antes y se
  ajusta el modelo según su resultado, la nota deja de ser honesta.

**Flujo habitual:** (1) buscar hiperparámetros entrenando con *entrenamiento* y midiendo en
*validación*; (2) reentrenar la configuración ganadora con *entrenamiento + validación*; (3) medir
una vez en *prueba*.

> **En nuestro trabajo — dejar uno fuera por usuario.** A cada usuario con ≥ 2 interacciones se le
> esconde 1 (al azar, con semilla fija) para **prueba**; a los que tienen ≥ 3, otra para
> **validación**; el resto va a **entrenamiento**. Ejemplos reales:
>
> | Usuario | Interacciones | Entrenamiento | Validación | Prueba |
> |---|---|---|---|---|
> | 6 | 1 | Jaipur City (963) | — | — |
> | 1 | 2 | Goa Beaches (467) | — | Taj Mahal (941) |
> | 16 | 4 | Leh Ladakh (970), Taj Mahal (176) | Leh Ladakh (40) | Jaipur City (113) |
>
> El usuario 6 **sí entrena** (es la única interacción que tiene el destino 963 en
> entrenamiento), pero **no se evalúa**: no queda nada que esconder. Los usuarios con una sola
> interacción aportan 252 de las 1066 interacciones de entrenamiento.

---

## 15. Métricas de evaluación

Para cada usuario de prueba $u$, sea $L_u^K$ su lista top-K y $T_u$ el conjunto de ítems relevantes
en prueba.

### 15.1 Precision@K y Recall@K

```math
\text{Precision@}K(u) = \frac{\lvert L_u^K\cap T_u\rvert}{K}, \qquad
\text{Recall@}K(u) = \frac{\lvert L_u^K\cap T_u\rvert}{\lvert T_u\rvert}
```

Se promedian sobre los usuarios de prueba. **Ojo:** si cada usuario tiene uno o dos ítems en
prueba, la Precision@10 es como mucho 0,1 o 0,2. Los valores pequeños son normales; lo que
importa es la comparación entre modelos.

**HitRate@K** (relacionada): la fracción de usuarios con **al menos un** acierto en el top-K.

### 15.2 NDCG@K (*Normalized Discounted Cumulative Gain*)

Premia que los aciertos aparezcan **arriba** en la lista. Con relevancia $rel_k\in\{0,1\}$ (o
graduada, por ejemplo la calificación) en la posición $k$:

```math
\text{DCG@}K = \sum_{k=1}^{K}\frac{2^{rel_k}-1}{\log_2(k+1)}, \qquad
\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}
```

donde IDCG@K es el DCG de la lista ideal (todos los relevantes arriba), de modo que
$\text{NDCG}\in[0,1]$. Un acierto en la posición 1 vale 1; en la posición 3 vale $1/\log_2 4 = 0{,}5$.

### 15.3 Cobertura del catálogo

```math
\text{Cobertura@}K = \frac{\left\lvert \bigcup_{u} L_u^K \right\rvert}{n}
```

Es la fracción del catálogo que aparece en alguna lista. La popularidad tiene una cobertura
mínima ($K/n$).

### 15.4 Desempeño por grupo

Se reportan las métricas por separado para usuarios con **pocas** y **muchas** interacciones de
entrenamiento (el umbral se fija con la distribución, por ejemplo la mediana). Así se ve si el
modelo solo sirve a los usuarios «ricos en datos».

### 15.5 Variabilidad

Cada métrica sale de 30 semillas y se reportan la media, la desviación estándar, el mejor y el
peor valor. Si se quiere, también un intervalo bootstrap sobre los usuarios.

---

## 16. Concentración y sesgo de popularidad

La **pregunta de negocio 2** pide medir si las recomendaciones concentran a los clientes en pocos
destinos. Herramientas:

- **Coeficiente de Gini** de la distribución de apariciones de cada ítem en las listas
  ($0$ = reparto perfecto, $1$ = todos reciben el mismo ítem). Con las apariciones ordenadas de
  menor a mayor, $x_{(1)}\le\dots\le x_{(n)}$:

```math
G = \frac{\sum_{k=1}^{n}(2k-n-1)\,x_{(k)}}{n\sum_k x_{(k)}}
```

- **Proporción de recomendaciones que van al top-10 % de ítems más populares.**
- **Curva de Lorenz** o gráfico de «popularidad del ítem frente a frecuencia en las listas» (la
  figura mínima de cobertura y concentración).

**Conexión con el negocio:** si todos los clientes reciben el mismo destino, las rutas hacia ese
destino se saturan. Esto conecta con el módulo 1 (flota y capacidad).

---

## 17. Visualizar embeddings: t-SNE y UMAP

Los embeddings de destinos viven en $\mathbb R^d$. Para verlos se proyectan a 2D:

- **t-SNE:** conserva vecindarios locales (puntos cercanos siguen cercanos). Las distancias entre
  grupos lejanos **no** son interpretables. El hiperparámetro principal es la *perplexity*.
- **UMAP:** es parecido, más rápido y conserva algo mejor la estructura global.

**Interpretación:** se colorea cada punto por `Type` o `State`. Si los destinos del mismo tipo se
agrupan, el modelo capturó esa similitud. Si no se agrupan, hay que preguntarse si es por el
modelo o porque los **datos no contienen esa señal**.

---

## 18. Comparación justa entre modelos

Lo que el enunciado exige para todos los módulos, aplicado aquí:

- Mismo preprocesamiento y mismas particiones para la popularidad, la factorización de matrices y
  la red.
- Reportar **cuántas configuraciones** se probaron por familia y el **tiempo de cómputo**.
- Entrenar la configuración final con **30 semillas**.
- Reportar los **parámetros entrenables**: la popularidad tiene 0; la factorización de matrices,
  $(m+n)d$; la red, lo que diga `model.summary()` / `sum(p.numel())`.

---

## 19. Mapa: del enunciado a la teoría

| Lo que pide el enunciado | Secciones |
|---|---|
| Describir los datos (usuarios, destinos, densidad, escala) | 2, 3 |
| Tarea 1 — caso de uso, K | 1, 8 |
| Tarea 2 — análisis descriptivo, cola larga | 2, 16 |
| Tarea 3 — partición | 14 |
| Tarea 4 — popularidad + CF clásico | 5, 6, 7 |
| Tarea 5 — red neuronal con contenido; embeddings, negativos, pérdida | 9, 10, 11, 12 |
| Tarea 6 — Precision/Recall/NDCG, cobertura, grupos | 15 |
| Pregunta 1 — usuarios nuevos | 13 |
| Pregunta 2 — concentración | 16 |
| Pregunta 3 — campaña y conversiones | 10.3 |
| Fundamentación: factorización de matrices y arquitectura | 7, 12 |
| Fundamentación: derivación de la pérdida | 10 |
| Fundamentación: definición de las métricas | 15 |
| Figura de embeddings 2D | 17 |
| Regla de comparación justa | 18 |

---

## 20. Glosario

- **Ítem:** lo que se recomienda (aquí, un destino).
- **Interacción:** un evento usuario–ítem (visita, reseña).
- **Dispersión (*sparsity*):** $1 - \text{densidad}$.
- **Cola larga:** la mayoría de los ítems tiene muy pocas interacciones.
- **Factor latente:** una dimensión aprendida que no observamos directamente.
- **Embedding:** un vector denso aprendido que representa una entidad.
- **Negativo muestreado:** un par $(u,j)$ no observado que se trata como «no interacción».
- **Arranque en frío (*cold start*):** un usuario o ítem sin historial.
- **Fuga de datos (*leakage*):** usar información de prueba al entrenar.
- **top-K:** las K primeras posiciones de la lista ordenada.

---

## 21. Referencias

- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender
  Systems.* IEEE Computer, 42(8), 30–37.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback
  Datasets.* ICDM.
- Rendle, S., Freudenthaler, C., Gantner, Z., & Schmidt-Thieme, L. (2009). *BPR: Bayesian
  Personalized Ranking from Implicit Feedback.* UAI.
- He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017). *Neural Collaborative
  Filtering.* WWW.
- Covington, P., Adams, J., & Sargin, E. (2016). *Deep Neural Networks for YouTube
  Recommendations.* RecSys. (Origen popular del enfoque de dos torres o recuperación por
  candidatos.)
- Järvelin, K., & Kekäläinen, J. (2002). *Cumulated Gain-Based Evaluation of IR Techniques.* ACM
  TOIS. (Define DCG/NDCG.)
- Salakhutdinov, R., & Mnih, A. (2008). *Probabilistic Matrix Factorization.* NeurIPS.
  (Interpretación probabilística de la factorización de matrices.)
- Ricci, F., Rokach, L., & Shapira, B. (eds.) (2022). *Recommender Systems Handbook*, 3.ª ed.
  Springer. (Libro de referencia general.)
