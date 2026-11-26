"""Pruebas de la app sin abrir el navegador (Streamlit AppTest).

Uso, desde la raíz del repositorio:
    python app/probar_app.py
"""

import time
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent


def main():
    t0 = time.time()
    ini = AppTest.from_file(str(APP / "Inicio.py"), default_timeout=120).run()
    assert not ini.exception, ini.exception
    for pagina in ("1_Demanda_y_operacion.py", "2_Conduccion_distractiva.py"):
        a = AppTest.from_file(str(APP / "pages" / pagina), default_timeout=60).run()
        assert not a.exception, (pagina, a.exception)
    a = AppTest.from_file(str(APP / "pages" / "3_Recomendacion_de_destinos.py"), default_timeout=120).run()
    assert not a.exception, a.exception
    a.number_input[0].set_value(234).run()          # otro cliente
    assert not a.exception
    a.multiselect[0].set_value(["Playa", "Naturaleza"]).run()   # combinación nueva → aviso
    assert any("no aparece en nuestros datos" in w.value for w in a.warning)
    a.slider[0].set_value(5).run()                  # K = 5
    assert not a.exception
    a.multiselect[0].set_value([]).run()            # sin preferencias → aviso
    assert any("al menos un tipo" in w.value for w in a.warning)
    print(f"Todas las pruebas de la app pasaron ({time.time() - t0:.1f} s).")


if __name__ == "__main__":
    main()
