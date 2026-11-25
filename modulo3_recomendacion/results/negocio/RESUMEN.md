# Preguntas de negocio

_Generado por `src/preguntas_negocio.py`. No editar a mano._

## P1. Usuarios nuevos (606 usuarios de prueba simulados como nuevos, 5 pliegues, 30 semillas)

| estrategia | Lugar@3 (%) | Lugar@1 (%) | HitRate@10 (%) | lugar@3: aciertos / esperados | lugar@3: p-valor corregido (m = 2) | lugar@1: aciertos / esperados | lugar@1: p-valor corregido (m = 2) |
|---|---|---|---|---|---|---|---|
| Azar | 60.1 | 19.8 | 1.0 | 364 / 363.6 | 1.0 | 120 / 121.2 | 1.0 |
| Popularidad | 58.9 | 17.7 | 0.64 | 357 / 363.6 | 1.0 | 107 / 121.2 | 1.0 |
| Regla: lo que más visita tu perfil | 60.7 | 24.6 | 1.36 | 368 / 363.6 | 0.748 | 149 / 121.2 | 0.007 |
| Red de dos torres | 61.5 | 21.9 | 0.95 | 373 / 363.6 | 0.461 | 133 / 121.2 | 0.252 |

| usuarios simulados como nuevos | red − popularidad en Lugar@3 (puntos) | p-valor pareado red vs popularidad |
|---|---|---|
| 606.0 | 2.6 | 0.122 |

## P2. Concentración (modelos finales, 606 usuarios, 30 semillas)

| modelo | Gini de las recomendaciones top-10 por ID | % de las recomendaciones que va al 10 % de destinos más populares | cobertura@10 (%) | máximo factor de capacidad por lugar |
|---|---|---|---|---|
| Azar | 0.043 | 9.9 | 100.0 | 1.02 |
| Popularidad | 0.985 | 100.0 | 1.6 | 4.97 |
| Factorización (BPR) | 0.577 | 1.2 | 91.3 | 1.11 |
| Red de dos torres | 0.923 | 14.1 | 21.4 | 1.7 |

| modelo | lugar | % de clientes que lo reciben 1.º | % que lo reciben entre sus 3 | factor de capacidad (1.º / 20 %) |
|---|---|---|---|---|
| Azar | Goa Beaches | 20.3 | 60.3 | 1.02 |
| Azar | Jaipur City | 20.1 | 60.3 | 1.01 |
| Azar | Kerala Backwaters | 19.8 | 59.9 | 0.99 |
| Azar | Leh Ladakh | 19.7 | 59.5 | 0.98 |
| Azar | Taj Mahal | 20.0 | 60.0 | 1.0 |
| Popularidad | Goa Beaches | 0.1 | 50.4 | 0.01 |
| Popularidad | Jaipur City | 99.5 | 99.7 | 4.97 |
| Popularidad | Kerala Backwaters | 0.2 | 49.0 | 0.01 |
| Popularidad | Leh Ladakh | 0.1 | 50.6 | 0.0 |
| Popularidad | Taj Mahal | 0.1 | 50.3 | 0.01 |
| Factorización (BPR) | Goa Beaches | 18.4 | 56.5 | 0.92 |
| Factorización (BPR) | Jaipur City | 22.1 | 62.7 | 1.11 |
| Factorización (BPR) | Kerala Backwaters | 18.5 | 56.5 | 0.93 |
| Factorización (BPR) | Leh Ladakh | 20.1 | 63.1 | 1.0 |
| Factorización (BPR) | Taj Mahal | 20.8 | 61.3 | 1.04 |
| Red de dos torres | Goa Beaches | 15.8 | 49.4 | 0.79 |
| Red de dos torres | Jaipur City | 5.6 | 59.9 | 0.28 |
| Red de dos torres | Kerala Backwaters | 34.1 | 66.1 | 1.7 |
| Red de dos torres | Leh Ladakh | 28.6 | 73.0 | 1.43 |
| Red de dos torres | Taj Mahal | 15.9 | 51.5 | 0.8 |

Figura: `figures/negocio_concentracion.png`

## P3. Campaña de un solo destino en noviembre (red guardada; 999 clientes)

| lugar | temporada | ¿en temporada en noviembre? | clientes del grupo (la red se lo recomienda 1.º) | perfiles del grupo | tasa en entrenamiento (%) | tasa en todos los clientes (%) | conversiones esperadas en el grupo | conversiones si se mostrara a un grupo al azar del mismo tamaño | comprobación en prueba: aciertos / usuarios | comprobación en prueba: tasa (%) e IC 95 % |
|---|---|---|---|---|---|---|---|---|---|---|
| Goa Beaches | Nov-Mar | sí | 333 | Beaches, Historical (333) | 20.8 | 19.5 | 69.1 | 64.8 | 35 / 207 | 16.9 (12.4–22.6) |
| Jaipur City | Oct-Mar | sí | 0 | nan | nan | nan | nan | nan | nan | nan |
| Kerala Backwaters | Sep-Mar | sí | 666 | Nature, Adventure (333); City, Historical (333) | 21.9 | 19.9 | 145.6 | 132.5 | 103 / 399 | 25.8 (21.8–30.3) |
| Leh Ladakh | Apr-Jun | no | 0 | nan | nan | nan | nan | nan | nan | nan |
| Taj Mahal | Nov-Feb | sí | 0 | nan | nan | nan | nan | nan | nan | nan |
