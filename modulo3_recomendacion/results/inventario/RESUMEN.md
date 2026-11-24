# Inventario de los datos — Módulo 3

_Archivo generado automáticamente por `src/inventario.py`. No editar a mano: volver a ejecutar el script._

## 1. Tamaño y calidad de las tablas

Cuántas filas tiene cada archivo y si hay datos faltantes, IDs repetidos o interacciones que apunten a usuarios/destinos que no existen.

| tabla | filas | columnas | celdas_nulas | ids_repetidos | pares_usuario_destino_repetidos | usuarios_inexistentes | destinos_inexistentes |
|---|---|---|---|---|---|---|---|
| Destinos | 1000 | 6 | 0 | 0 | nan | nan | nan |
| Usuarios | 999 | 7 | 0 | 0 | nan | nan | nan |
| Reseñas | 999 | 5 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| Historial | 999 | 5 | 0 | 0 | 0.0 | 0.0 | 0.0 |

_CSV: `results/inventario/tabla_01.csv`_

## 2. Valores repetidos en ciclos exactos (evidencia de datos sintéticos)

Para cada columna se cuentan sus valores distintos y se verifica si su valor queda totalmente determinado por la posición del ID en un ciclo de k (por ejemplo, IDs 1, 6, 11, … siempre son el mismo destino). En datos reales esto no ocurre.

| tabla | columna | valores_distintos | id | k | determinada_por_(id-1)_mod_k |
|---|---|---|---|---|---|
| Destinos | Name | 5 | DestinationID | 5 | True |
| Destinos | Type | 5 | DestinationID | 5 | True |
| Destinos | State | 5 | DestinationID | 5 | True |
| Destinos | BestTimeToVisit | 5 | DestinationID | 5 | True |
| Destinos | Popularity | 1000 | DestinationID | 5 | False |
| Usuarios | Preferences | 3 | UserID | 3 | True |
| Usuarios | Name | 20 | UserID | 20 | False |
| Usuarios | Email | 20 | UserID | 20 | False |
| Reseñas | ReviewText | 3 | ReviewID | 3 | True |
| Historial | VisitDate | 3 | HistoryID | 3 | True |

_CSV: `results/inventario/tabla_02.csv`_

## 3. Los 1000 destinos son 5 lugares

Se agrupan los destinos por todos sus atributos descriptivos. Solo aparecen 5 combinaciones, cada una con 200 IDs; lo único que cambia entre copias es Popularity.

| Name | State | Type | BestTimeToVisit | ids | popularity_min | popularity_max |
|---|---|---|---|---|---|---|
| Goa Beaches | Goa | Beach | Nov-Mar | 200 | 7.5 | 9.5 |
| Jaipur City | Rajasthan | City | Oct-Mar | 200 | 7.51 | 9.5 |
| Kerala Backwaters | Kerala | Nature | Sep-Mar | 200 | 7.53 | 9.49 |
| Leh Ladakh | Jammu and Kashmir | Adventure | Apr-Jun | 200 | 7.53 | 9.48 |
| Taj Mahal | Uttar Pradesh | Historical | Nov-Feb | 200 | 7.51 | 9.48 |

_CSV: `results/inventario/tabla_03.csv`_

## 4. Distribución de las calificaciones

Si las 5 notas fueran igual de probables, cada una tendría ~20 %. La prueba chi-cuadrado de bondad de ajuste contrasta H0: «distribución uniforme». Un p > 0,05 significa que no hay evidencia para rechazar que las notas se sortearon al azar.

- Reseñas (Rating): media = 3.02; chi-cuadrado de uniformidad χ² = 2.44, p = 0.656
- Historial (ExperienceRating): media = 2.90; chi-cuadrado de uniformidad χ² = 7.36, p = 0.118

| nota | Historial (ExperienceRating) — conteo | Reseñas (Rating) — conteo | Historial (ExperienceRating) — porcentaje | Reseñas (Rating) — porcentaje |
|---|---|---|---|---|
| 1.0 | 206.0 | 198.0 | 20.6 | 19.8 |
| 2.0 | 228.0 | 199.0 | 22.8 | 19.9 |
| 3.0 | 201.0 | 185.0 | 20.1 | 18.5 |
| 4.0 | 183.0 | 216.0 | 18.3 | 21.6 |
| 5.0 | 181.0 | 201.0 | 18.1 | 20.1 |

_CSV: `results/inventario/tabla_04.csv`_

## 5. ¿La nota se relaciona con la popularidad del destino?

Se une cada interacción con su destino y se correlaciona la nota con Popularity. Pearson mide relación lineal; Spearman, si a más de una tiende a haber más de la otra. Valores cercanos a 0 con p alto = no hay relación.

| tabla | n | pearson | p_pearson | spearman | p_spearman |
|---|---|---|---|---|---|
| Reseñas | 999 | 0.044 | 0.165 | 0.044 | 0.167 |
| Historial | 999 | 0.026 | 0.412 | 0.028 | 0.385 |

_CSV: `results/inventario/tabla_05.csv`_

## 6. ¿Las preferencias declaradas explican lo que visitan?

Cada perfil declara 2 de los 5 tipos de destino, así que si los usuarios eligieran al azar, ~40 % de sus interacciones coincidiría con sus preferencias. Tres pruebas contrastan si lo observado supera al azar: binomial (contra 40 %), permutación (barajar los tipos 5000 veces) y chi-cuadrado de independencia perfil × tipo.

- Permutación (unión): azar promedio = 40.1 %, observado = 42.3 %, p = 0.019
- Chi-cuadrado de independencia (unión): χ² = 18.35, gl = 8, p = 0.019 → hay evidencia estadística de relación, pero el efecto es pequeño.

Coincidencia por perfil (unión):

| perfil | interacciones | coinciden_% |
|---|---|---|
| Beaches, Historical | 684 | 43.0 |
| City, Historical | 655 | 38.6 |
| Nature, Adventure | 659 | 45.4 |

Tabla cruzada perfil × tipo de destino (unión):

| Preferences | Adventure | Beach | City | Historical | Nature |
|---|---|---|---|---|---|
| Beaches, Historical | 150 | 134 | 131 | 160 | 109 |
| City, Historical | 125 | 126 | 125 | 128 | 151 |
| Nature, Adventure | 147 | 120 | 120 | 120 | 152 |

| fuente | interacciones | coinciden | observado_% | esperado_por_azar_% | p_binomial(observado>azar) |
|---|---|---|---|---|---|
| reseña | 999 | 429 | 42.9 | 40.0 | 0.031 |
| historial | 999 | 417 | 41.7 | 40.0 | 0.138 |
| unión | 1998 | 846 | 42.3 | 40.0 | 0.017 |

_CSV: `results/inventario/tabla_06.csv`_

## 7. ¿El texto de la reseña corresponde al destino?

Si los textos fueran reales, «Loved the beaches!» aparecería sobre todo en playas. Chi-cuadrado de independencia: χ² = 3.69, gl = 8, p = 0.884.

| Type | A historical wonder | Incredible monument! | Loved the beaches! |
|---|---|---|---|
| Adventure | 68 | 70 | 80 |
| Beach | 65 | 60 | 65 |
| City | 62 | 72 | 57 |
| Historical | 70 | 66 | 63 |
| Nature | 68 | 65 | 68 |

_CSV: `results/inventario/tabla_07.csv`_

## 8. Solapamiento entre reseñas e historial

Si los datos fueran reales, quien reseña un destino normalmente lo visitó. Aquí ningún par usuario–destino se repite entre tablas (por eso unirlas no crea duplicados).

| medida | valor |
|---|---|
| pares (usuario, destino) en ambas tablas | 0 |
| usuarios con interacciones en ambas tablas | 408 |
| usuarios solo en reseñas | 216 |
| usuarios solo en historial | 234 |

_CSV: `results/inventario/tabla_08.csv`_

## 9. Decisión 1 — ¿Qué es un destino? (1000 IDs frente a 5 lugares)

Compara las dos formas de definir «destino». Lugares distintos por usuario (opción B): 1 lugar(es): 322 usuarios, 2 lugar(es): 339 usuarios, 3 lugar(es): 148 usuarios, 4 lugar(es): 45 usuarios, 5 lugar(es): 4 usuarios.

| aspecto | A_1000_IDs | B_5_lugares |
|---|---|---|
| ítems en el catálogo | 1000 | 5 |
| ítems sin ninguna interacción | 134 | 0 |
| ítems con 1 sola interacción | 271 | 0 |
| interacciones del ítem más popular | 7 | 422 |
| interacciones del ítem menos popular (con ≥1) | 1 | 376 |
| usuarios que repetirían ítem al agrupar | 0 | 266 |
| K máximo con sentido | 10 o más | ≤ 4 |
| HitRate@10 de un recomendador al azar | ≈ 1 % | 100 % con K = 5 (trivial) |
| % de ítems de prueba vistos en entrenamiento | 73 % | 100 % |

_CSV: `results/inventario/tabla_09.csv`_

## 10. Decisiones 2 y 3 — ¿Qué cuenta como interacción?

Para evaluar a un usuario se necesitan al menos 2 interacciones (una para entrenar y otra escondida para prueba). La tabla muestra cuántos usuarios cumplen eso con cada definición posible.

| definición | interacciones | usuarios | destinos | usuarios_con_1 | usuarios_evaluables_(≥2) | media_por_usuario |
|---|---|---|---|---|---|---|
| Solo reseñas | 999 | 624 | 651 | 355 | 269 | 1.6 |
| Solo historial | 999 | 642 | 638 | 384 | 258 | 1.56 |
| Unión | 1998 | 858 | 866 | 252 | 606 | 2.33 |
| Unión, solo nota ≥ 3 | 1167 | 675 | 684 | 351 | 324 | 1.73 |
| Unión, solo nota ≥ 4 | 781 | 534 | 544 | 353 | 181 | 1.46 |

_CSV: `results/inventario/tabla_10.csv`_

## 11. Interacciones por usuario y por destino (unión)

Cuántos usuarios y destinos tienen 0, 1, 2, … interacciones. Densidad de la matriz usuario–destino: 1998 / (999 × 1000) = 0.20 %.

| n_interacciones | usuarios_con_n | destinos_con_n |
|---|---|---|
| 0 | 141 | 134 |
| 1 | 252 | 271 |
| 2 | 280 | 270 |
| 3 | 195 | 184 |
| 4 | 79 | 89 |
| 5 | 31 | 37 |
| 6 | 17 | 11 |
| 7 | 4 | 4 |

_CSV: `results/inventario/tabla_11.csv`_

## 12. Cifras clave del dataset

Las cifras que pide el enunciado. La densidad es la fracción de celdas llenas de la matriz usuario–destino: interacciones / (usuarios × destinos).

| medida | valor |
|---|---|
| usuarios en el catálogo | 999 |
| usuarios con ≥ 1 interacción | 858 |
| destinos (IDs) en el catálogo | 1000 |
| lugares reales distintos | 5 |
| destinos (IDs) con ≥ 1 interacción | 866 |
| interacciones: reseñas | 999 |
| interacciones: historial | 999 |
| interacciones: unión | 1998 |
| escala de las calificaciones | 1–5 (enteros) |
| densidad (solo reseñas) | 0.10 % |
| densidad (solo historial) | 0.10 % |
| densidad (unión) | 0.20 % |
| interacciones por usuario (media, unión) | 2.33 |

_CSV: `results/inventario/tabla_12.csv`_

## 13. Diccionario de variables

Todas las columnas de los 4 archivos, qué papel cumplen en un recomendador (identificador, contenido o dato de la interacción) y qué valores toman.

| tabla | variable | papel | valores |
|---|---|---|---|
| Destinos | DestinationID | ID del ítem | identificador, 1000 valores distintos |
| Destinos | Name | contenido (nombre) | Goa Beaches: 200; Jaipur City: 200; Kerala Backwaters: 200; Leh Ladakh: 200; Taj Mahal: 200 |
| Destinos | State | contenido del destino | Goa: 200; Jammu and Kashmir: 200; Kerala: 200; Rajasthan: 200; Uttar Pradesh: 200 |
| Destinos | Type | contenido del destino | Adventure: 200; Beach: 200; City: 200; Historical: 200; Nature: 200 |
| Destinos | Popularity | contenido del destino | continua, 7.50–9.50 (media 8.51) |
| Destinos | BestTimeToVisit | contenido del destino | Apr-Jun: 200; Nov-Feb: 200; Nov-Mar: 200; Oct-Mar: 200; Sep-Mar: 200 |
| Usuarios | UserID | ID del usuario | identificador, 999 valores distintos |
| Usuarios | Name | dato personal falso — no se usa | Aanya: 64; Aarav: 55; Aditya: 51; Anika: 54; Arjun: 48; Dhruv: 52; Diya: 38; Hitesh: 54; Isha: 47; Ishaan: 51; Karan: 46; Kavya: 47; Meera: 56; Pooja: 45; Ritvik: 48; Riya: 59; Rohan: 48; Saanvi: 43; Tanvi: 43; Vivaan: 50 |
| Usuarios | Email | dato personal falso — no se usa | aanya@example.com: 64; aarav@example.com: 55; aditya@example.com: 51; anika@example.com: 54; arjun@example.com: 48; dhruv@example.com: 52; diya@example.com: 38; hitesh@example.com: 54; isha@example.com: 47; ishaan@example.com: 51; karan@example.com: 46; kavya@example.com: 47; meera@example.com: 56; pooja@example.com: 45; ritvik@example.com: 48; riya@example.com: 59; rohan@example.com: 48; saanvi@example.com: 43; tanvi@example.com: 43; vivaan@example.com: 50 |
| Usuarios | Preferences | contenido del usuario | Beaches, Historical: 333; City, Historical: 333; Nature, Adventure: 333 |
| Usuarios | Gender | contenido del usuario | Female: 496; Male: 503 |
| Usuarios | NumberOfAdults | contenido del usuario | 1: 495; 2: 504 |
| Usuarios | NumberOfChildren | contenido del usuario | 0: 338; 1: 341; 2: 320 |
| Reseñas | ReviewID | ID de fila | identificador, 999 valores distintos |
| Reseñas | DestinationID | ID del ítem | identificador, 651 valores distintos |
| Reseñas | UserID | ID del usuario | identificador, 624 valores distintos |
| Reseñas | Rating | calificación de la interacción | 1: 198; 2: 199; 3: 185; 4: 216; 5: 201 |
| Reseñas | ReviewText | texto de la interacción | A historical wonder: 333; Incredible monument!: 333; Loved the beaches!: 333 |
| Historial | HistoryID | ID de fila | identificador, 999 valores distintos |
| Historial | UserID | ID del usuario | identificador, 642 valores distintos |
| Historial | DestinationID | ID del ítem | identificador, 638 valores distintos |
| Historial | VisitDate | fecha de la interacción | 2024-01-01: 333; 2024-02-15: 333; 2024-03-20: 333 |
| Historial | ExperienceRating | calificación de la interacción | 1: 206; 2: 228; 3: 201; 4: 183; 5: 181 |

_CSV: `results/inventario/tabla_13.csv`_

## 14. Tarea 1 — ¿Excluir los lugares ya visitados?

Cuántos lugares (de 5) le quedan a cada usuario si se excluyen los que ya visitó.

- Usuarios que recibirían menos de K lugares nuevos: K = 1: 4 usuarios (0.5 %); K = 2: 49 usuarios (5.7 %); K = 3: 197 usuarios (23.0 %); K = 4: 536 usuarios (62.5 %); K = 5: 858 usuarios (100.0 %).
- En una partición exploratoria (1 interacción escondida al azar por usuario con ≥ 2, 606 usuarios), el **33.8 %** de las interacciones escondidas es a un lugar que el usuario ya había visitado: excluirlos haría fallar por diseño esos casos.

| lugares_nuevos_disponibles | usuarios |
|---|---|
| 0 | 4 |
| 1 | 45 |
| 2 | 148 |
| 3 | 339 |
| 4 | 322 |

_CSV: `results/inventario/tabla_14.csv`_

## 15. Tarea 3 — Comparación de estrategias de partición

Cifras de cada estrategia posible (semilla 0). «Usuarios sin entrenamiento» son usuarios de validación/prueba que el modelo nunca vio: con ellos el filtrado colaborativo no puede hacer nada. «IDs no vistos» son destinos de validación/prueba sin ninguna interacción en entrenamiento: un modelo que solo usa IDs no puede recomendarlos; uno con contenido sí.

- **Partición temporal: no es viable.** El historial tiene solo 3 fechas distintas, asignadas en ciclo por el número de fila (tabla 2), y las reseñas no tienen fecha.

| estrategia | conjunto | interacciones_entrenamiento | interacciones_conjunto | usuarios_evaluados | %_usuarios_sin_entrenamiento | %_IDs_no_vistos_en_entrenamiento | %_lugares_no_vistos_en_entrenamiento |
|---|---|---|---|---|---|---|---|
| Dejar-uno-fuera por usuario | validación | 1066 | 326 | 326 | 0.0 | 34.4 | 0.0 |
| Dejar-uno-fuera por usuario | prueba | 1066 | 606 | 606 | 0.0 | 33.7 | 0.0 |
| Aleatoria 80/10/10 | validación | 1586 | 213 | 194 | 18.0 | 21.6 | 0.0 |
| Aleatoria 80/10/10 | prueba | 1586 | 199 | 181 | 18.8 | 21.1 | 0.0 |

_CSV: `results/inventario/tabla_15.csv`_
