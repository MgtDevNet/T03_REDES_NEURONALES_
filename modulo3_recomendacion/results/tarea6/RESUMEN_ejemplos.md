# Tarea 6 — usuarios de ejemplo y embeddings

_Generado por `src/ejemplos_embeddings.py`. No editar a mano._

## 1. Usuarios de ejemplo (red guardada, semilla 0)

| UserID | por qué se eligió | perfil | historial (lo aprendido) | recomendación de la red (3 lugares) | lugar escondido | ¿acierta? (Lugar@3) |
|---|---|---|---|---|---|---|
| 1 | perfil «Beaches, Historical», sin niños | Beaches, Historical; Female; 1 adulto(s), 0 niño(s) | Goa Beaches (historial) | Goa Beaches (¿vuelves?), Taj Mahal, Jaipur City | Taj Mahal | sí |
| 3 | perfil «City, Historical», sin niños | City, Historical; Female; 2 adulto(s), 0 niño(s) | Taj Mahal (historial) | Kerala Backwaters, Leh Ladakh, Jaipur City | Kerala Backwaters | sí |
| 20 | perfil «Nature, Adventure», sin niños | Nature, Adventure; Female; 2 adulto(s), 0 niño(s) | Kerala Backwaters (reseña); Jaipur City (reseña); Taj Mahal (reseña); Taj Mahal (reseña); Taj Mahal (historial); Kerala Backwaters (reseña) | Kerala Backwaters (¿vuelves?), Leh Ladakh, Jaipur City (¿vuelves?) | Kerala Backwaters | sí |
| 2 | viaja con niños | Nature, Adventure; Male; 2 adulto(s), 2 niño(s) | Leh Ladakh (historial); Goa Beaches (historial) | Kerala Backwaters, Leh Ladakh (¿vuelves?), Jaipur City | Goa Beaches | no |
| 234 | el que más interacciones tiene para aprender (6) | City, Historical; Male; 1 adulto(s), 0 niño(s) | Leh Ladakh (reseña); Kerala Backwaters (historial); Goa Beaches (historial); Kerala Backwaters (historial); Leh Ladakh (historial); Taj Mahal (historial) | Kerala Backwaters (¿vuelves?), Leh Ladakh (¿vuelves?), Jaipur City | Goa Beaches | no |

## 2. Puntaje medio de cada lugar para cada perfil (producto punto u · v)

| index | Goa Beaches | Jaipur City | Kerala Backwaters | Leh Ladakh | Taj Mahal |
|---|---|---|---|---|---|
| Beaches + Historical | 0.03999999910593033 | 0.0020000000949949026 | -0.035999998450279236 | -0.020999999716877937 | 0.02199999988079071 |
| City + Historical | -0.02199999988079071 | -0.006000000052154064 | 0.017000000923871994 | 0.007000000216066837 | -0.019999999552965164 |
| Nature + Adventure | -0.2150000035762787 | -0.013000000268220901 | 0.19200000166893005 | 0.11400000005960464 | -0.11999999731779099 |

Figura: `figures/tarea6_embeddings.png`
