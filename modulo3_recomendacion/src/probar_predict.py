"""Pruebas de las funciones de inferencia (`predict.py`) que usará la app.

Se ejecutan como lo haría la app: importando desde la RAÍZ del repositorio.

Uso (desde la raíz del repo T03):
    .venv/bin/python modulo3_recomendacion/src/probar_predict.py
"""

import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

t0 = time.perf_counter()
from modulo3_recomendacion.src.predict import historial, recomendar_perfil_nuevo, recomendar_usuario  # noqa: E402


def debe_fallar(f, *args):
    try:
        f(*args)
    except ValueError as e:
        return str(e)
    raise AssertionError(f"{f.__name__}{args} debía fallar")


def main():
    # Usuario existente (usuaria 1 de la Tabla 11: Goa en entrenamiento, Taj Mahal escondido)
    h = historial(1)
    print("historial(1):\n", h.to_string(index=False), "\n")
    assert len(h) == 2 and set(h.lugar) == {"Goa Beaches", "Taj Mahal"}

    r = recomendar_usuario(1)
    print("recomendar_usuario(1):\n", r[["posicion", "lugar", "mejor_epoca", "destination_id", "mensaje"]].to_string(index=False), "\n")
    assert len(r) == 3 and r.lugar.is_unique
    assert not set(r.destination_id) & set(h.DestinationID), "no debe repetir un ID ya visto"
    assert r[r.lugar.isin(h.lugar)].ya_visitado.all(), "los lugares visitados deben marcarse «¿vuelves?»"

    # Perfil nuevo: mismas preferencias que la usuaria 1 → mismo orden de lugares (la red solo usa preferencias)
    n = recomendar_perfil_nuevo({"preferencias": ["Beach", "Historical"], "genero": "Female", "adultos": 1, "ninos": 0})
    print("recomendar_perfil_nuevo(Beach + Historical):\n", n[["posicion", "lugar", "mensaje"]].to_string(index=False), "\n")
    assert list(n.lugar) == list(r.lugar), "mismo perfil de preferencias → mismos lugares"
    assert not n.ya_visitado.any()
    # Etiquetas como en los datos y como texto
    assert list(recomendar_perfil_nuevo({"preferencias": "Beaches, Historical"}).lugar) == list(n.lugar)
    # Combinación que no existe en los datos (extrapolación) y k = 5
    e = recomendar_perfil_nuevo({"preferencias": ["Beach", "Nature"]}, k=5)
    print("recomendar_perfil_nuevo(Beach + Nature, k=5):", ", ".join(e.lugar), "\n")
    assert len(e) == 5 and e.lugar.is_unique

    # Errores claros
    print("Errores esperados:")
    for msg in (debe_fallar(recomendar_usuario, 0), debe_fallar(recomendar_usuario, 1, 6),
                debe_fallar(recomendar_perfil_nuevo, {"preferencias": []}),
                debe_fallar(recomendar_perfil_nuevo, {"preferencias": ["Montaña"]})):
        print("  -", msg)
    print(f"\nTodas las pruebas de predict.py pasaron ({time.perf_counter() - t0:.1f} s, incluida la carga del modelo).")


if __name__ == "__main__":
    main()
