# Ideas para el informe — Módulo 3 (recomendación)

Archivo vivo: en cada paso se añaden ideas, cifras, frases y figuras para el reporte. Está
organizado según lo que pide el enunciado (`docs/instrucciones_trabajo_3.pdf`, pp. 5–7).
Marcas: ✅ ya tenemos la evidencia · ⏳ falta calcularlo o hacerlo · 💡 idea de redacción.

---

## 1. Datos

- ✅ Fuente: *Travel Recommendation Dataset* («India Travel Recommender System Dataset»), Aman
  Mehra, Kaggle. Versión 1, publicada el 2024-01-23 y nunca actualizada. Son 4 CSV (destinos,
  usuarios, reseñas, historial) que suman 177 KB.
- ✅ Tamaños: 1000 destinos (ID), 999 usuarios, 999 reseñas y 999 visitas (1998 interacciones en
  la unión). Densidad: 0,1 % por tabla y 0,2 % en la unión.
- ✅ Escala: `Rating` y `ExperienceRating` de 1 a 5, distribuidas casi uniformemente.
- ✅ Variables de usuario: `Preferences` (3 valores), `Gender`, `NumberOfAdults` (1–2),
  `NumberOfChildren` (0–2). `Name` y `Email` son identificadores falsos (20 valores) y no se usan.
- ✅ Variables de destino: `Name`, `State`, `Type`, `BestTimeToVisit` (5 combinaciones) y
  `Popularity` (7,5–9,5, continua).
- 💡 **¿Reales o sintéticos?** (el enunciado lo pide de forma explícita). Hay que argumentarlo con
  dos tipos de evidencia:
  1. *Documentación de la fuente:* el autor dice que el dataset «was generated for educational and
     research purposes» y que los perfiles se «enriquecieron» con diversidad de género. No describe
     ningún método de recolección ni ninguna fuente primaria. Las licencias son contradictorias
     (los metadatos dicen MIT; la descripción, CC0 y además «uso no comercial»), lo que indica poco
     cuidado editorial.
  2. *Evidencia interna (análisis propio):* los 1000 destinos son 5 lugares repetidos con
     `(ID−1) mod 5`; `Preferences`, `VisitDate` y `ReviewText` se repiten con `mod 3`; el texto de
     las reseñas no tiene relación con el destino; las calificaciones son uniformes; las visitas
     coinciden con las preferencias el 41,7 % de las veces frente al 40 % esperado por azar; las
     reseñas y el historial no comparten ningún par.
- 💡 **Implicaciones:** (a) no se puede afirmar nada sobre turistas reales ni sobre India; (b) si
  las interacciones son aleatorias, el techo de cualquier modelo es cercano al azar, y el valor del
  trabajo está en la **metodología** y en **detectar** que no hay señal; (c) cualquier
  «mejora» de la red sobre la popularidad debe contrastarse con las 30 semillas antes de creerla.

## 2. Caso de uso (tarea 1)
- ⏳ Canal y momento: por ejemplo, al comprar un tiquete o al iniciar sesión en la app de la
  empresa. Se decide K. Información disponible: el perfil (preferencias, tamaño del grupo) y el
  historial.

## 3. Análisis descriptivo (tarea 2)
- ✅ Usuarios por número de interacciones (unión): 1 → 252, 2 → 280, 3 → 195, 4 → 79, 5 → 31,
  6 → 17, 7 → 4.
- ✅ 134 destinos sin interacciones y 271 con una sola. El más popular tiene solo 7.
- ⏳ Figura de cola larga (por usuario y por destino).
- 💡 Hipótesis sobre las preferencias: el enunciado pide formularlas. Hay que plantearlas
  (familia → destinos de playa, etc.) **y contrastarlas con datos**. Por ahora el cruce
  preferencias × tipo visitado no muestra relación.

## 4. Partición (tarea 3)
- ⏳ Las 3 fechas de `VisitDate` son cíclicas y no sirven para una partición temporal realista.
  Hay que argumentarlo.

## 5–6. Modelos y evaluación
- ⏳ Añadir un recomendador **aleatorio** como piso de comparación, además de la popularidad y
  la factorización de matrices.
- 💡 Decisión sobre la granularidad del ítem (ver `DECISIONES.md`, paso 1): explicarla en el
  informe como una decisión de modelado.

## Preguntas de negocio
- ⏳ P1 (usuarios nuevos), P2 (concentración), P3 (campaña).

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
