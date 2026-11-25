# Evidencia para la tarea 5 — red neuronal con contenido

_Generado por `src/evidencia_tarea5.py`. No editar a mano._

## 1. ¿Qué variables del destino aportan algo además del lugar?

| variable | valores distintos por lugar (mín–máx) | ¿aporta algo además del lugar? |
|---|---|---|
| Type | 1–1 | no: la fija el lugar |
| State | 1–1 | no: la fija el lugar |
| BestTimeToVisit | 1–1 | no: la fija el lugar |
| Popularity | 200–200 | sí: cambia entre IDs del mismo lugar |

## 2. ¿Cuántos perfiles de usuario distintos hay?

| variables | perfiles distintos |
|---|---|
| Preferences | 3 |
| Preferences + Gender | 6 |
| Preferences + Gender + NumberOfAdults + NumberOfChildren | 36 |

## 3. Máximo que puede lograr el contenido del destino (oráculo que siempre sabe el lugar, 30 semillas)

| escenario | modelo de referencia (oráculo) | HitRate@10 (%) |
|---|---|---|
| ajuste | sabe el lugar; IDs del lugar al azar | 5.13 |
| ajuste | sabe el lugar; IDs del lugar por Popularity | 6.13 |
| final | sabe el lugar; IDs del lugar al azar | 5.05 |
| final | sabe el lugar; IDs del lugar por Popularity | 4.79 |

## 4. ¿El perfil o el historial predicen el lugar? (prueba, 606 usuarios)

| predictor del lugar | aciertos | usuarios | acierto (%) | azar (%) | p-valor (H1: acierta más que el azar) |
|---|---|---|---|---|---|
| perfil: Preferences → lugar más frecuente del perfil | 145 | 606 | 23.9 | 20.0 | 0.01 |
| perfil: Preferences + Gender + NumberOfAdults + NumberOfChildren → lugar más frecuente del perfil | 121 | 606 | 20.0 | 20.0 | 0.524 |
| historial: el lugar escondido es uno que ya visitó | 195 | 606 | 32.2 | 32.4 | 0.56 |
