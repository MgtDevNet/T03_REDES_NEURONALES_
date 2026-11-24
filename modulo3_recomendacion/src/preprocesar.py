"""Preprocesamiento del módulo 3: datos crudos -> data/processed/.

Aplica las decisiones de DECISIONES.md (pasos 1 a 4) y deja los archivos que
leen TODOS los modelos (popularidad, factorización de matrices, red neuronal)
y la app. Usar los mismos archivos es lo que garantiza la «comparación justa».

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/preprocesar.py

Salidas en modulo3_recomendacion/data/processed/:
    interacciones.csv   una fila por interacción, con su conjunto
                        (entrenamiento / validación / prueba)
    usuarios.csv        perfil de cada usuario (sin nombre ni correo) +
                        preferencias en formato multi-hot + conteos
    destinos.csv        catálogo de destinos con sus atributos
    preferencias.json   valores posibles de preferencias (para el
                        formulario de perfil nuevo de la app)
"""

import json
from pathlib import Path

import pandas as pd

from inventario import PREFERENCIA_A_TIPO, cargar, interacciones_unidas, particion_dejar_uno_fuera

MODULO = Path(__file__).resolve().parents[1]
PROCESADOS = MODULO / "data" / "processed"
SEMILLA_PARTICION = 0  # fija: la partición es la misma para todos los modelos
TIPOS = list(PREFERENCIA_A_TIPO.values())  # Beach, Historical, City, Nature, Adventure


def construir_interacciones(destinos, resenas, historial):
    """Decisiones 2 y 3 + partición (paso 4).

    - Unión de reseñas + historial; toda interacción es positiva.
    - Se conservan la calificación y la fuente por si se quieren usar después.
    - Se agrega el lugar real (decisión 1: el modelo usa IDs, el cliente ve lugares).
    """
    inter = interacciones_unidas(resenas, historial).merge(
        destinos[["DestinationID", "Name"]], on="DestinationID"
    )
    inter = particion_dejar_uno_fuera(inter, semilla=SEMILLA_PARTICION)
    return inter.rename(columns={"Name": "lugar"})[
        ["UserID", "DestinationID", "lugar", "fuente", "calificacion", "conjunto"]
    ]


def construir_usuarios(usuarios, inter):
    """Perfil de cada usuario, listo para usar como variables de contenido.

    - Se eliminan Name y Email: son ficticios (20 valores repetidos) y no aportan.
    - Preferences ("Beaches, Historical") se convierte a 5 columnas 0/1
      (pref_Beach, pref_Historical, ...). Así el modelo acepta CUALQUIER
      combinación de preferencias, también las que un usuario nuevo marque en
      la app aunque no existan en los datos.
    """
    u = usuarios.drop(columns=["Name", "Email"]).copy()
    elegidas = u.Preferences.str.split(", ").apply(lambda xs: {PREFERENCIA_A_TIPO[x] for x in xs})
    for tipo in TIPOS:
        u[f"pref_{tipo}"] = elegidas.apply(lambda s, t=tipo: int(t in s))
    cuenta = inter.groupby(["UserID", "conjunto"]).size().unstack(fill_value=0)
    u["n_interacciones"] = u.UserID.map(cuenta.sum(axis=1)).fillna(0).astype(int)
    u["n_entrenamiento"] = u.UserID.map(cuenta.get("entrenamiento")).fillna(0).astype(int)
    return u


def verificar(inter, usuarios, destinos):
    """Comprobaciones de que la partición cumple las reglas. Si algo falla, se detiene."""
    ent = inter[inter.conjunto == "entrenamiento"]
    for conj in ("validación", "prueba"):
        ev = inter[inter.conjunto == conj]
        # 1. Ningún par (usuario, destino) de validación/prueba está en entrenamiento.
        pares_ent = set(zip(ent.UserID, ent.DestinationID))
        assert not any(p in pares_ent for p in zip(ev.UserID, ev.DestinationID)), f"fuga en {conj}"
        # 2. Todo usuario evaluado tiene al menos 1 interacción de entrenamiento.
        assert ev.UserID.isin(ent.UserID).all(), f"usuario de {conj} sin entrenamiento"
        # 3. Exactamente 1 interacción por usuario evaluado.
        assert ev.UserID.is_unique, f"usuario con más de 1 interacción en {conj}"
    # 4. No se perdió ni se duplicó ninguna interacción.
    assert len(inter) == 1998 and not inter.duplicated(["UserID", "DestinationID"]).any()
    # 5. Los IDs son consecutivos desde 1 (se pueden usar directamente como índices - 1).
    assert (usuarios.UserID.sort_values().values == range(1, len(usuarios) + 1)).all()
    assert (destinos.DestinationID.sort_values().values == range(1, len(destinos) + 1)).all()


def main():
    PROCESADOS.mkdir(parents=True, exist_ok=True)
    destinos, usuarios, resenas, historial = cargar()

    inter = construir_interacciones(destinos, resenas, historial)
    usr = construir_usuarios(usuarios, inter)
    verificar(inter, usuarios, destinos)

    inter.to_csv(PROCESADOS / "interacciones.csv", index=False)
    usr.to_csv(PROCESADOS / "usuarios.csv", index=False)
    destinos.to_csv(PROCESADOS / "destinos.csv", index=False)
    opciones = {
        "tipos_de_destino": TIPOS,
        "etiquetas_en_datos": list(PREFERENCIA_A_TIPO.keys()),
        "perfiles_presentes_en_datos": sorted(usuarios.Preferences.unique().tolist()),
        "generos": sorted(usuarios.Gender.unique().tolist()),
        "adultos": sorted(int(x) for x in usuarios.NumberOfAdults.unique()),
        "ninos": sorted(int(x) for x in usuarios.NumberOfChildren.unique()),
    }
    (PROCESADOS / "preferencias.json").write_text(json.dumps(opciones, ensure_ascii=False, indent=2), encoding="utf-8")

    # Resumen en pantalla
    print("Interacciones por conjunto:")
    print(inter.conjunto.value_counts().to_string())
    print("\nUsuarios evaluados:", inter[inter.conjunto == "prueba"].UserID.nunique(), "(prueba),",
          inter[inter.conjunto == "validación"].UserID.nunique(), "(validación)")
    print("Usuarios por nº de interacciones de entrenamiento:",
          usr.n_entrenamiento.value_counts().sort_index().to_dict())
    print("\nEjemplos (usuarios 6, 1 y 16):")
    print(inter[inter.UserID.isin([6, 1, 16])].sort_values(["UserID", "conjunto"]).to_string(index=False))
    print(f"\nVerificaciones OK. Archivos en {PROCESADOS.relative_to(MODULO.parent)}")


if __name__ == "__main__":
    main()
