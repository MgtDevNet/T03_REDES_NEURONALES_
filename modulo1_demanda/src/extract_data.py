"""Extrae el dataset completo eh75-8ah6 de datos.gov.co por particiones y lo guarda en Parquet."""

from __future__ import annotations

import io
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
import requests

DATASET_ID = "eh75-8ah6"
BASE_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.csv"
COUNT_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.json"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PARTS_DIR = RAW_DIR / "parts"
OUTPUT_PARQUET = RAW_DIR / "operacion_pasajeros.parquet"

SCHEMA_OVERRIDES = {
    "terminal": pl.String,
    "clase_vehiculo": pl.String,
    "nivel_servicio": pl.String,
    "municipio_origen_ruta": pl.String,
    "municipio_destino_ruta": pl.String,
    "fecha_despacho": pl.String,
    "hora_despacho": pl.String,
    "tipo_despacho": pl.String,
    "despachos": pl.Int32,
    "pasajeros": pl.Int32,
}


def build_partitions() -> list[tuple[str, str]]:
    """Genera particiones (nombre, clausula $where) que cubren todo el dataset."""
    partitions: list[tuple[str, str]] = [
        ("2019", "fecha_despacho >= '2019-01-01' AND fecha_despacho < '2020-01-01'"),
        ("2020", "fecha_despacho >= '2020-01-01' AND fecha_despacho < '2021-01-01'"),
        ("2024", "fecha_despacho >= '2024-01-01' AND fecha_despacho < '2025-01-01'"),
        ("gt2026", "fecha_despacho >= '2027-01-01'"),
    ]
    for year in (2021, 2022, 2023, 2025, 2026):
        for month in range(1, 13):
            if year == 2026 and month > 6:
                continue
            start = f"{year}-{month:02d}-01"
            if month == 12:
                end = f"{year + 1}-01-01"
            else:
                end = f"{year}-{month + 1:02d}-01"
            partitions.append(
                (f"{year}_{month:02d}", f"fecha_despacho >= '{start}' AND fecha_despacho < '{end}'")
            )
    # Catch any remaining 2026 months (Jul-Dec) in case data extends later in 2026
    partitions.append(
        ("2026_h2", "fecha_despacho >= '2026-07-01' AND fecha_despacho < '2027-01-01'")
    )
    return partitions


def download_partition(name: str, where_clause: str, retries: int = 4) -> tuple[str, int]:
    """Descarga una particion en CSV y la guarda directamente como Parquet comprimido."""
    out_path = PARTS_DIR / f"{name}.parquet"
    if out_path.exists():
        rows = pl.scan_parquet(out_path).select(pl.len()).collect().item()
        return name, rows

    tmp_path = PARTS_DIR / f"{name}.parquet.tmp"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                BASE_URL,
                params={"$where": where_clause, "$limit": 5_000_000},
                timeout=300,
            )
            resp.raise_for_status()
            content = resp.content
            if not content or content.count(b"\n") <= 1:
                df = pl.DataFrame(schema={**SCHEMA_OVERRIDES, "fecha_despacho": pl.Date})
                df.write_parquet(tmp_path, compression="zstd")
                tmp_path.replace(out_path)
                return name, 0

            df = pl.read_csv(
                io.BytesIO(content),
                schema_overrides=SCHEMA_OVERRIDES,
            ).with_columns(
                pl.col("fecha_despacho").str.slice(0, 10).str.to_date("%Y-%m-%d").alias("fecha_despacho")
            )
            df.write_parquet(tmp_path, compression="zstd")
            tmp_path.replace(out_path)
            return name, df.height
        except Exception as exc:
            if attempt == retries:
                raise RuntimeError(f"Fallo particion {name}: {exc}") from exc
            wait = 2 ** attempt
            print(f"[{name}] Reintento {attempt}/{retries} tras error: {exc} (esperando {wait}s)")
            time.sleep(wait)
    raise AssertionError("unreachable")


def main() -> None:
    t0 = time.time()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

    # Limpiar CSV parcial previo si existe
    partial_csv = RAW_DIR / "operacion_pasajeros.csv"
    if partial_csv.exists():
        partial_csv.unlink()
        print("Eliminado CSV parcial previo.")

    expected_total = int(
        requests.get(COUNT_URL, params={"$select": "count(*)"}, timeout=30).json()[0]["count"]
    )
    print(f"Total esperado en datos.gov.co: {expected_total:,} filas")

    partitions = build_partitions()
    total_downloaded = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(download_partition, n, w): n for n, w in partitions}
        for i, fut in enumerate(as_completed(futures), 1):
            name, rows = fut.result()
            total_downloaded += rows
            print(f"[{i}/{len(partitions)}] {name}: {rows:,} filas (acumulado: {total_downloaded:,})")

    print("Combinando particiones en un unico Parquet con Polars streaming...")
    tmp_out = RAW_DIR / "operacion_pasajeros.parquet.tmp"
    (
        pl.scan_parquet(str(PARTS_DIR / "*.parquet"))
        .sink_parquet(tmp_out, compression="zstd")
    )
    tmp_out.replace(OUTPUT_PARQUET)

    final_rows = pl.scan_parquet(OUTPUT_PARQUET).select(pl.len()).collect().item()
    size_mb = OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
    print(
        f"Listo: {OUTPUT_PARQUET} | {final_rows:,} filas (esperadas: {expected_total:,}) | "
        f"{size_mb:.1f} MB | {time.time() - t0:.1f}s"
    )

    if final_rows == expected_total:
        shutil.rmtree(PARTS_DIR, ignore_errors=True)
        print("Carpeta temporal parts/ eliminada.")
    else:
        print("ADVERTENCIA: el conteo final difiere del esperado; se conservan parts/ para inspeccion.")


if __name__ == "__main__":
    main()
