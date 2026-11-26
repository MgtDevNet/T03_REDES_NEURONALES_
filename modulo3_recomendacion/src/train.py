"""Experimento principal del módulo 3 con UN SOLO COMANDO y parámetros en un archivo YAML.

    python -m modulo3_recomendacion.src.train --config modulo3_recomendacion/configs/demo.yaml
    python -m modulo3_recomendacion.src.train --config modulo3_recomendacion/configs/completo.yaml

Cualquier parámetro se puede cambiar sin editar código, en el YAML o desde la línea de comandos
con --set (puede repetirse; la clave usa puntos para entrar en el YAML):

    ... --config modulo3_recomendacion/configs/demo.yaml --set modelos.red.fijos.lr=0.1
    ... --set modelos.red.buscar.d=[4] --set semillas_finales=10 --set K_lugar=[1,3]

Qué hace, con la misma partición y el mismo evaluador que el informe:
  1. AJUSTE (entrenamiento → validación): para la factorización y la red, entrena cada combinación
     de la rejilla `buscar` con `semillas_ajuste` semillas y parada temprana; elige la de mejor
     `criterio`; si ni la mejor supera al azar en validación (binomial), usa `mas_simple`.
  2. FINAL (entrenamiento + validación → prueba): entrena la configuración elegida de cada modelo
     con `semillas_finales` semillas y reporta media, desviación estándar, mínimo y máximo de cada
     métrica, más las pruebas contra el azar y la comparación pareada de la red con los demás.
Los resultados van a la carpeta `salida` (no se sobrescriben los resultados oficiales del informe).
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dos_torres import DosTorres  # noqa: E402
from evaluacion import N_DESTINOS, cargar_escenario, evaluar  # noqa: E402
from factorizacion import FactorizacionBPR  # noqa: E402
from inventario import _a_markdown  # noqa: E402
from referencias import Azar, Popularidad  # noqa: E402

MODULO = SRC.parent
RAIZ = MODULO.parent
CLASES = {"factorizacion": FactorizacionBPR, "red": DosTorres}
NOMBRES = {"azar": "Azar", "popularidad": "Popularidad", "factorizacion": "Factorización (BPR)", "red": "Red de dos torres"}
torch.set_num_threads(1)


# ----------------------------------------------------------------------------- configuración
def cargar_config(ruta, cambios):
    cfg = yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))
    for cambio in cambios or []:
        clave, _, valor = cambio.partition("=")
        nodo = cfg
        partes = clave.strip().split(".")
        for p in partes[:-1]:
            nodo = nodo.setdefault(p, {})
        nodo[partes[-1]] = yaml.safe_load(valor)  # «0.1» → número, «[2,4]» → lista, «bce» → texto
    return cfg


def p_azar_id(aprende, evalua, k=10):
    n_u = evalua.UserID.map(aprende.groupby("UserID").size()).fillna(0).to_numpy()
    return float(np.mean(k / (N_DESTINOS - n_u)))


# ----------------------------------------------------------------------------- 1. ajuste
def ajustar_familia(clave, mcfg, cfg):
    """Búsqueda en validación para una familia (factorización o red)."""
    aprende, val = cargar_escenario("ajuste")
    criterio = mcfg["criterio"]
    claves = list(mcfg["buscar"])
    filas = []
    for valores in itertools.product(*(mcfg["buscar"][k] for k in claves)):
        hp = dict(zip(claves, valores))
        for s in range(cfg["semillas_ajuste"]):
            t0 = time.perf_counter()
            m = CLASES[clave](**hp, **mcfg.get("fijos", {}), epocas=mcfg["epocas_max"], semilla=s)
            if clave == "red":
                m.ajustar(aprende, validacion=val, paciencia=mcfg["paciencia"], metrica=criterio)
            else:
                m.ajustar(aprende, validacion=val, paciencia=mcfg["paciencia"])
            r = evaluar(m.puntuar, aprende, val, semilla=s, ks=(10,), ks_lugar=tuple(cfg["K_lugar"]))[0].iloc[0]
            filas.append({**hp, "semilla": s, "mejor_epoca": m.mejor_epoca, criterio: r[criterio.replace("@10", "")]
                          if criterio.startswith("NDCG") else r[criterio], "HitRate@10": r.HitRate,
                          "segundos": time.perf_counter() - t0, "parametros": m.parametros()})
        print(f"  {NOMBRES[clave]} {hp}: {criterio} en validación = {100 * np.mean([f[criterio] for f in filas[-cfg['semillas_ajuste']:]]):.2f} %")
    corridas = pd.DataFrame(filas)
    config = (corridas.groupby(claves).agg(**{"criterio_media": (criterio, "mean"), "HitRate@10_media": ("HitRate@10", "mean"),
                                              "mediana_mejor_epoca": ("mejor_epoca", "median"), "segundos": ("segundos", "sum"),
                                              "parametros": ("parametros", "first")})
              .reset_index().sort_values("criterio_media", ascending=False))
    mejor = config.iloc[0]
    n = len(val)
    if criterio.startswith("HitRate_lugar@"):  # prueba contra el azar en la misma métrica
        k_ok, p0 = int(round(mejor.criterio_media * n)), int(criterio.split("@")[1]) / 5
    else:                                      # por ID: HitRate@10 contra el azar
        k_ok, p0 = int(round(mejor["HitRate@10_media"] * n)), p_azar_id(aprende, val)
    p_val = stats.binomtest(k_ok, n, p0, alternative="greater").pvalue
    if p_val < 0.05:
        elegida, motivo = mejor, f"la mejor en validación supera al azar ({k_ok} de {n}; p-valor = {p_val:.3f})"
    else:
        simple = mcfg["mas_simple"]
        filtro = np.logical_and.reduce([config[k] == v for k, v in simple.items()])
        elegida = config[filtro].iloc[0] if filtro.any() else mejor
        motivo = (f"ninguna supera al azar en validación (la mejor acierta {k_ok} de {n} frente a {n * p0:.1f}; "
                  f"p-valor = {p_val:.3f}): se usa la más simple" if filtro.any() else
                  f"ninguna supera al azar (p-valor = {p_val:.3f}) y la más simple no está en la rejilla: se usa la mejor")
    # cada valor vuelve a su tipo original de la rejilla (el agrupado de pandas convierte 8 en 8.0)
    hp = {k: type(mcfg["buscar"][k][0])(elegida[k]) for k in claves}
    decision = {"hiperparametros": hp, "epocas": int(round(elegida.mediana_mejor_epoca)), "motivo": motivo,
                "configuraciones": len(config), "corridas": len(corridas), "segundos": round(float(corridas.segundos.sum()), 1)}
    return corridas, config, decision


# ----------------------------------------------------------------------------- 2. final
def crear(clave, s, mcfg, decision):
    if clave == "azar":
        return Azar(semilla=1000 + s)
    if clave == "popularidad":
        return Popularidad()
    return CLASES[clave](**decision["hiperparametros"], **mcfg.get("fijos", {}), epocas=decision["epocas"], semilla=s)


def fase_final(cfg, decisiones):
    aprende, pru = cargar_escenario("final")
    activos = [k for k, m in cfg["modelos"].items() if m.get("activo", True)]
    filas, usuarios = [], []
    for s in range(cfg["semillas_finales"]):
        for clave in activos:
            t0 = time.perf_counter()
            m = crear(clave, s, cfg["modelos"][clave], decisiones.get(clave)).ajustar(aprende)
            seg = time.perf_counter() - t0
            res, pu = evaluar(m.puntuar, aprende, pru, semilla=s, ks=tuple(cfg["K_id"]), ks_lugar=tuple(cfg["K_lugar"]))
            params = m.parametros() if hasattr(m, "parametros") else 0
            filas.append(res.assign(modelo=NOMBRES[clave], semilla=s, segundos=seg, parametros=params))
            usuarios.append(pu.assign(modelo=NOMBRES[clave], semilla=s))
    return pd.concat(filas, ignore_index=True), pd.concat(usuarios, ignore_index=True), aprende, pru


def resumir(por_semilla, k):
    """Media, desviación estándar, mínimo y máximo (en %) de cada métrica sobre las semillas."""
    d = por_semilla[por_semilla.K == k]
    metricas = ["HitRate", "Precision", "Recall", "NDCG", "Cobertura"] + [c for c in d.columns if c.startswith("HitRate_lugar@")]
    filas = []
    for modelo, g in d.groupby("modelo", sort=False):
        for m in metricas:
            v = 100 * g[m]
            nombre = m.replace("HitRate_lugar@", "Lugar@") if m.startswith("HitRate_lugar") else f"{m}@{k}"
            filas.append({"modelo": modelo, "métrica": nombre, "media (%)": round(v.mean(), 2),
                          "desv. estándar": round(v.std(ddof=1), 2) if len(v) > 1 else 0.0,
                          "mínimo": round(v.min(), 2), "máximo": round(v.max(), 2),
                          "parámetros": int(g.parametros.iloc[0])})
    return pd.DataFrame(filas)


def pruebas(por_usuario, aprende, pru, cfg, k=10):
    """Contra el azar (binomial unilateral, Bonferroni en las métricas por lugar) y pareadas red vs. resto."""
    n = pru.UserID.nunique()
    p0_id = p_azar_id(aprende, pru, k)
    ks_l = cfg["K_lugar"]
    filas, aciertos_l = [], {}
    for modelo, g in por_usuario.groupby("modelo", sort=False):
        fila = {"modelo": modelo}
        h = (g.rango < k).groupby(g.UserID).mean()
        k_ok = int(round(h.sum()))
        fila[f"HitRate@{k}: aciertos / esperados"] = f"{k_ok} / {n * p0_id:.1f}"
        fila[f"HitRate@{k}: p-valor"] = round(stats.binomtest(k_ok, n, p0_id, alternative="greater").pvalue, 3)
        for kl in ks_l:
            a = (g.rango_lugar < kl).groupby(g.UserID).mean()
            if kl == cfg["metrica_principal_lugar"]:
                aciertos_l[modelo] = a
            k_ok = int(round(a.sum()))
            p = stats.binomtest(k_ok, n, kl / 5, alternative="greater").pvalue
            fila[f"Lugar@{kl}: aciertos / esperados"] = f"{k_ok} / {n * kl / 5:.1f}"
            fila[f"Lugar@{kl}: p-valor corregido (m = {len(ks_l)})"] = round(min(1.0, len(ks_l) * p), 3)
        filas.append(fila)
    pareadas = []
    red = NOMBRES["red"]
    if red in aciertos_l:
        rng = np.random.default_rng(0)
        for otro, a in aciertos_l.items():
            if otro == red:
                continue
            dif = (aciertos_l[red] - a.reindex(aciertos_l[red].index)).to_numpy()
            nulas = (rng.choice([-1, 1], size=(10_000, len(dif))) * dif).mean(1)
            pareadas.append({"comparación": f"{red} − {otro}", "métrica": f"Lugar@{cfg['metrica_principal_lugar']}",
                             "diferencia (puntos)": round(100 * dif.mean(), 2),
                             "p-valor (pareada)": round(float(np.mean(np.abs(nulas) >= abs(dif.mean()))), 3)})
    return pd.DataFrame(filas), pd.DataFrame(pareadas)


def guardar_modelos(cfg, decisiones):
    """Reentrena con semilla 0 la configuración elegida y la guarda en models/ (la usa la app)."""
    aprende, _ = cargar_escenario("final")
    destino = MODULO / "models"
    if "factorizacion" in decisiones:
        m = crear("factorizacion", 0, cfg["modelos"]["factorizacion"], decisiones["factorizacion"]).ajustar(aprende)
        P, Q = m.factores()
        np.savez(destino / "factorizacion_bpr.npz", P=P, Q=Q)
    if "red" in decisiones:
        m = crear("red", 0, cfg["modelos"]["red"], decisiones["red"]).ajustar(aprende)
        torch.save(m.red.state_dict(), destino / "dos_torres.pt")
        meta = {**decisiones["red"]["hiperparametros"], "epocas": decisiones["red"]["epocas"], "semilla": 0,
                "oculta": m.oculta, "aprende_con": "entrenamiento + validación", "parametros": m.parametros()}
        (destino / "dos_torres.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(destino / "popularidad.npz", conteo=Popularidad().ajustar(aprende).conteo)


# ----------------------------------------------------------------------------- principal
def main(argv=None):
    ap = argparse.ArgumentParser(description="Experimento principal del módulo 3 (recomendación de destinos).")
    ap.add_argument("--config", default=str(MODULO / "configs" / "demo.yaml"), help="archivo YAML de configuración")
    ap.add_argument("--set", action="append", metavar="CLAVE=VALOR", help="cambia un parámetro del YAML (repetible)")
    args = ap.parse_args(argv)
    cfg = cargar_config(args.config, args.set)
    salida = RAIZ / cfg["salida"] if not Path(cfg["salida"]).is_absolute() else Path(cfg["salida"])
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "config_usada.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    t0 = time.perf_counter()
    print(f"Experimento «{cfg['nombre']}» → {salida.relative_to(RAIZ) if salida.is_relative_to(RAIZ) else salida}")

    decisiones, textos = {}, []
    for clave in ("factorizacion", "red"):
        mcfg = cfg["modelos"].get(clave, {})
        if not mcfg.get("activo", True):
            continue
        if "epocas" in mcfg:  # configuración fija: sin búsqueda (por ejemplo, en la demo)
            hp = {k: v[0] if isinstance(v, list) else v for k, v in mcfg["buscar"].items()}
            decisiones[clave] = {"hiperparametros": hp, "epocas": mcfg["epocas"], "motivo": "fijada en el YAML (sin búsqueda)",
                                 "configuraciones": 1, "corridas": 0, "segundos": 0.0}
            print(f"\n[1] {NOMBRES[clave]}: configuración fija {hp}, {mcfg['epocas']} épocas (sin búsqueda)")
            textos.append(f"### {NOMBRES[clave]}: configuración fija {hp}, {mcfg['epocas']} épocas (sin búsqueda)\n")
            continue
        print(f"\n[1] Ajuste de {NOMBRES[clave]} en validación")
        corridas, config, dec = ajustar_familia(clave, mcfg, cfg)
        decisiones[clave] = dec
        corridas.to_csv(salida / f"ajuste_{clave}_corridas.csv", index=False)
        config.to_csv(salida / f"ajuste_{clave}_configuraciones.csv", index=False)
        print(f"  → elegida {dec['hiperparametros']}, {dec['epocas']} épocas ({dec['motivo']})")
        textos += [f"### Ajuste de {NOMBRES[clave]} ({dec['configuraciones']} configuraciones, {dec['corridas']} corridas, "
                   f"{dec['segundos']} s)\n", _a_markdown(config.round(4)) + "\n",
                   f"**Elegida:** {dec['hiperparametros']}, {dec['epocas']} épocas — {dec['motivo']}.\n"]
    (salida / "decisiones.json").write_text(json.dumps(decisiones, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[2] Evaluación final en prueba con {cfg['semillas_finales']} semilla(s)")
    por_semilla, por_usuario, aprende, pru = fase_final(cfg, decisiones)
    por_semilla.to_csv(salida / "final_por_semilla.csv", index=False)
    resumen = resumir(por_semilla, 10 if 10 in cfg["K_id"] else max(cfg["K_id"]))
    contra_azar, pareadas = pruebas(por_usuario, aprende, pru, cfg)
    resumen.to_csv(salida / "final_resumen.csv", index=False)
    contra_azar.to_csv(salida / "final_contra_azar.csv", index=False)
    pareadas.to_csv(salida / "final_pareadas.csv", index=False)
    if cfg.get("guardar_modelos", False):
        guardar_modelos(cfg, decisiones)
        print("  modelos guardados en models/")

    total = time.perf_counter() - t0
    (salida / "RESUMEN.md").write_text("\n".join([
        f"# Experimento «{cfg['nombre']}» del módulo 3\n",
        f"_Generado por `src/train.py` con `{args.config}`" + (f" y cambios {args.set}" if args.set else "") + f". Tiempo total: {total:.0f} s._\n",
        "## 1. Ajuste en validación\n", *textos,
        f"## 2. Prueba: {cfg['semillas_finales']} semilla(s) — media, desviación, mínimo y máximo\n", _a_markdown(resumen) + "\n",
        "## 3. Contra el azar\n", _a_markdown(contra_azar) + "\n",
        "## 4. Red frente a los demás (pareada por usuario)\n", _a_markdown(pareadas) + "\n" if len(pareadas) else "—\n"]),
        encoding="utf-8")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    principal = resumen[resumen["métrica"].isin([f"Lugar@{cfg['metrica_principal_lugar']}", "HitRate@10"])]
    print("\n" + principal.to_string(index=False))
    print("\n" + contra_azar.to_string(index=False))
    if len(pareadas):
        print("\n" + pareadas.to_string(index=False))
    print(f"\nListo en {total:.0f} s. Resultados en {salida}/RESUMEN.md")


if __name__ == "__main__":
    main()
