"""Generacion del calendario oficial de festivos y variables exogenas de Colombia.

Fuente primaria:
- Ley 51 de 1983 (Ley Emiliani), Congreso de la Republica de Colombia:
  establece los 18 dias festivos nacionales (fijos, trasladados al lunes siguiente
  y religiosos moviles dependientes del Domingo de Pascua).
- Decreto 1373 de 2007, Ministerio de Educacion Nacional:
  semana de receso estudiantil de octubre (semana inmediatamente anterior al lunes
  festivo del Dia de la Raza).
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

MODULE_DIR = Path(__file__).resolve().parents[1]
FESTIVOS_CSV = MODULE_DIR / "data" / "processed" / "festivos.csv"
CALENDARIO_DIARIO_CSV = MODULE_DIR / "data" / "processed" / "calendario_diario.csv"

DIAS_SEMANA_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
MESES_ES = [
    "", "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
FUENTE_LEY = "Ley 51 de 1983 (Congreso de la República de Colombia)"


def calcular_domingo_pascua(anio: int) -> date:
    """Calcula el Domingo de Pascua gregoriano (algoritmo de Meeus/Jones/Butcher)."""
    a = anio % 19
    b = anio // 100
    c = anio % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mes = (h + l - 7 * m + 114) // 31
    dia = ((h + l - 7 * m + 114) % 31) + 1
    return date(anio, mes, dia)


def trasladar_a_lunes_emiliani(fecha: date) -> date:
    """Aplica la Ley 51 de 1983: si no cae lunes, traslada al lunes siguiente."""
    if fecha.weekday() == 0:
        return fecha
    return fecha + timedelta(days=(7 - fecha.weekday()))


def festivos_colombia_anio(anio: int) -> list[dict]:
    """Devuelve los 18 festivos oficiales de Colombia para un anio dado."""
    pascua = calcular_domingo_pascua(anio)
    definiciones = [
        (date(anio, 1, 1), "Año Nuevo", "Fijo"),
        (trasladar_a_lunes_emiliani(date(anio, 1, 6)), "Reyes Magos", "Ley Emiliani"),
        (trasladar_a_lunes_emiliani(date(anio, 3, 19)), "San José", "Ley Emiliani"),
        (pascua - timedelta(days=3), "Jueves Santo", "Religioso móvil"),
        (pascua - timedelta(days=2), "Viernes Santo", "Religioso móvil"),
        (date(anio, 5, 1), "Día del Trabajo", "Fijo"),
        (trasladar_a_lunes_emiliani(pascua + timedelta(days=39)), "Ascensión del Señor", "Ley Emiliani"),
        (trasladar_a_lunes_emiliani(pascua + timedelta(days=60)), "Corpus Christi", "Ley Emiliani"),
        (trasladar_a_lunes_emiliani(pascua + timedelta(days=68)), "Sagrado Corazón de Jesús", "Ley Emiliani"),
        (trasladar_a_lunes_emiliani(date(anio, 6, 29)), "San Pedro y San Pablo", "Ley Emiliani"),
        (date(anio, 7, 20), "Día de la Independencia", "Fijo"),
        (date(anio, 8, 7), "Batalla de Boyacá", "Fijo"),
        (trasladar_a_lunes_emiliani(date(anio, 8, 15)), "Asunción de la Virgen", "Ley Emiliani"),
        (trasladar_a_lunes_emiliani(date(anio, 10, 12)), "Día de la Raza", "Ley Emiliani"),
        (trasladar_a_lunes_emiliani(date(anio, 11, 1)), "Todos los Santos", "Ley Emiliani"),
        (trasladar_a_lunes_emiliani(date(anio, 11, 11)), "Independencia de Cartagena", "Ley Emiliani"),
        (date(anio, 12, 8), "Inmaculada Concepción", "Fijo"),
        (date(anio, 12, 25), "Navidad", "Fijo"),
    ]

    rows = []
    for d, nombre, tipo in sorted(definiciones, key=lambda x: x[0]):
        rows.append(
            {
                "fecha": d.isoformat(),
                "anio": anio,
                "mes": d.month,
                "dia_semana_num": d.weekday(),
                "dia_semana": DIAS_SEMANA_ES[d.weekday()],
                "festivo_nombre": nombre,
                "tipo_festivo": tipo,
                "es_lunes_festivo": d.weekday() == 0,
                "es_semana_santa": nombre in ("Jueves Santo", "Viernes Santo"),
                "fuente": FUENTE_LEY,
            }
        )
    return rows


def exportar_festivos_csv(
    anios: Iterable[int] = range(2021, 2028),
    out_path: Path | str = FESTIVOS_CSV,
) -> pd.DataFrame:
    """Genera y guarda data/processed/festivos.csv para los anios indicados."""
    rows: list[dict] = []
    for y in anios:
        rows.extend(festivos_colombia_anio(y))
    df = pd.DataFrame(rows).sort_values("fecha").reset_index(drop=True)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def generar_calendario_exogenas(
    fecha_inicio: str | date = "2021-01-01",
    fecha_fin: str | date = "2026-12-31",
    festivos_csv: Path | str = FESTIVOS_CSV,
) -> pd.DataFrame:
    """Construye tabla diaria de variables exogenas de calendario a partir de festivos.csv.

    Usa la fecha calendario real (para cruzar 1:1 con fecha_despacho), garantizando que
    el dia de la semana (Lunes..Domingo) cambie correctamente cada anio y en anios bisiestos.
    Todas las variables son deterministicas y conocidas con >30 dias de anticipacion.
    """
    path = Path(festivos_csv)
    if not path.exists():
        exportar_festivos_csv(out_path=path)

    fest_df = pd.read_csv(path, parse_dates=["fecha"])
    fest_set = set(fest_df["fecha"])
    lunes_fest_set = set(fest_df.loc[fest_df["es_lunes_festivo"], "fecha"])
    nombre_map = dict(zip(fest_df["fecha"], fest_df["festivo_nombre"]))

    fechas = pd.date_range(fecha_inicio, fecha_fin, freq="D")
    cal = pd.DataFrame({"fecha": fechas})
    cal["anio"] = cal["fecha"].dt.year
    cal["mes"] = cal["fecha"].dt.month
    cal["nombre_mes"] = cal["mes"].map(lambda m: MESES_ES[m])
    cal["dia_mes"] = cal["fecha"].dt.day
    cal["dia_anio"] = cal["fecha"].dt.dayofyear
    cal["dia_semana_num"] = cal["fecha"].dt.dayofweek  # 0=Lunes ... 6=Domingo
    cal["dia_semana"] = cal["dia_semana_num"].map(lambda d: DIAS_SEMANA_ES[d])
    cal["es_fin_semana"] = cal["dia_semana_num"].isin([5, 6]).astype(int)

    cal["es_festivo"] = cal["fecha"].isin(fest_set).astype(int)
    cal["festivo_nombre"] = cal["fecha"].map(nombre_map).fillna("")
    cal["es_lunes_festivo"] = cal["fecha"].isin(lunes_fest_set).astype(int)
    cal["es_vispera_festivo"] = (cal["fecha"] + pd.Timedelta(days=1)).isin(fest_set).astype(int)

    # Puente festivo completo: viernes (salida), sabado, domingo y lunes festivo
    cal["es_puente"] = (
        cal["fecha"].isin(lunes_fest_set)
        | ((cal["dia_semana_num"] == 6) & (cal["fecha"] + pd.Timedelta(days=1)).isin(lunes_fest_set))
        | ((cal["dia_semana_num"] == 5) & (cal["fecha"] + pd.Timedelta(days=2)).isin(lunes_fest_set))
        | ((cal["dia_semana_num"] == 4) & (cal["fecha"] + pd.Timedelta(days=3)).isin(lunes_fest_set))
    ).astype(int)

    # Semana Santa completa (Domingo de Ramos a Domingo de Pascua)
    semana_santa_dias: set[pd.Timestamp] = set()
    for y in range(cal["anio"].min(), cal["anio"].max() + 1):
        p = pd.Timestamp(calcular_domingo_pascua(int(y)))
        for offset in range(-7, 1):
            semana_santa_dias.add(p + pd.Timedelta(days=offset))
    cal["es_semana_santa"] = cal["fecha"].isin(semana_santa_dias).astype(int)

    # Semana de receso escolar de octubre (Decreto 1373 de 2007: semana previa al Dia de la Raza)
    receso_oct_dias: set[pd.Timestamp] = set()
    raza_fechas = fest_df.loc[fest_df["festivo_nombre"] == "Día de la Raza", "fecha"]
    for f_raza in raza_fechas:
        for offset in range(-7, 1):
            receso_oct_dias.add(f_raza + pd.Timedelta(days=offset))
    cal["es_receso_octubre"] = cal["fecha"].isin(receso_oct_dias).astype(int)

    # Temporada fin de anio / Reyes (15 dic - 12 ene) y mitad de anio (15 jun - 15 jul)
    m_d = cal["fecha"].dt.month * 100 + cal["fecha"].dt.day
    cal["es_temporada_fin_anio"] = ((m_d >= 1215) | (m_d <= 112)).astype(int)
    cal["es_temporada_mitad_anio"] = ((m_d >= 615) & (m_d <= 715)).astype(int)

    return cal


def exportar_calendario_diario_csv(
    fecha_inicio: str | date = "2021-01-01",
    fecha_fin: str | date = "2027-12-31",
    out_path: Path | str = CALENDARIO_DIARIO_CSV,
) -> pd.DataFrame:
    """Exporta la tabla diaria de calendario y exogenas por separado para cruce futuro por fecha."""
    df = generar_calendario_exogenas(fecha_inicio=fecha_inicio, fecha_fin=fecha_fin)
    df_out = df.copy()
    df_out["fecha"] = df_out["fecha"].dt.strftime("%Y-%m-%d")
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False, encoding="utf-8")
    return df_out


if __name__ == "__main__":
    df_fest = exportar_festivos_csv()
    df_cal = exportar_calendario_diario_csv()
    print(f"Generado {FESTIVOS_CSV} con {len(df_fest)} festivos (2021-2027).")
    print(f"Generado {CALENDARIO_DIARIO_CSV} con {len(df_cal)} dias (2021-2027).")
