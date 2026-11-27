"""Puntos 6 y 7, pregunta de negocio 1 y entregables para la app.

Desde la raíz del repositorio (después de `train_cnn final`):
    python -m modulo2_conduccion.src.evaluate_cnn --config modulo2_conduccion/configs/cnn.yaml

Evalúa models/scratch.pt y models/transfer.pt (semilla mediana de las 30), compara
con la regresión logística y elige el mejor modelo por F1 macro de VALIDACIÓN.
Para ese modelo escribe lo que necesita la app:
    models/metadata.json, configs/alertas.yaml y results/umbrales.csv
"""

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import torch
import yaml

from .modelos import (CLASSES, MEAN, STD, GradCAM, build_model, cam_layer_name, head_of, resize_side, to_image,
                      trainable)
from .prepare import ROOT, write_csv
from .train_cnn import load_config, loaders, outputs, predict, read_rows, scores


def load(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = build_model(checkpoint["params"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["params"]


def latency(model, size, runs=100):
    """Latencia por imagen en CPU, lote de 1: proxy del equipo embebido del punto 1."""
    model = model.cpu().eval()
    x = torch.randn(1, 3, size, size)
    with torch.inference_mode():
        for _ in range(10):
            model(x)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95)),
            "hilos": torch.get_num_threads()}


# ----------------------------------------------------------------------------
# Pregunta de negocio 1: política de alertas
# ----------------------------------------------------------------------------

def alert_curve(probs, labels, alert_ids, thresholds):
    """Puntaje de alerta = suma de las probabilidades de las clases con alerta.
    Para cada umbral t se alerta si el puntaje ≥ t. Se cuentan, por cada 1.000
    imágenes, las falsas alarmas (no había distracción y se alerta) y las
    distracciones no detectadas (había distracción y no se alerta)."""
    score = probs[:, alert_ids].sum(1)
    real = np.isin(labels, alert_ids)
    rows = []
    for t in thresholds:
        alert = score >= t
        fa, fn = int((alert & ~real).sum()), int((~alert & real).sum())
        rows.append({"umbral": round(float(t), 3), "falsas_alarmas": fa, "no_detectadas": fn,
                     "falsas_alarmas_x1000": round(1000 * fa / len(labels), 1),
                     "no_detectadas_x1000": round(1000 * fn / len(labels), 1),
                     "sensibilidad_alerta": round(float(alert[real].mean()), 4) if real.any() else 0.0,
                     "precision_alerta": round(float(real[alert].mean()), 4) if alert.any() else 0.0})
    return rows


def alert_policy(val, test, config, kind, out):
    """El umbral se elige en VALIDACIÓN minimizando el costo esperado y se reporta en
    PRUEBA, para no ajustar la decisión con los datos con los que se evalúa."""
    a = config["alert"]
    alert_ids = [CLASSES.index(c) for c in a["classes"]]
    thresholds = np.round(np.arange(a["threshold_step"], 1.0, a["threshold_step"]), 3)
    curves = {name: alert_curve(*data, alert_ids, thresholds) for name, data in (("val", val), ("test", test))}
    cost = lambda r: a["cost_missed"] * r["no_detectadas"] + a["cost_false_alarm"] * r["falsas_alarmas"]  # noqa: E731
    best = min(curves["val"], key=lambda r: (cost(r), abs(r["umbral"] - 0.5)))  # empate: el más cercano a 0,5
    chosen = next(r for r in curves["test"] if r["umbral"] == best["umbral"])
    write_csv(out["results"] / f"{kind}_umbrales_prueba.csv", curves["test"])

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    t = [r["umbral"] for r in curves["test"]]
    ax.plot(t, [r["falsas_alarmas_x1000"] for r in curves["test"]], label="Falsas alarmas", color="#c0392b")
    ax.plot(t, [r["no_detectadas_x1000"] for r in curves["test"]], label="Distracciones no detectadas",
            color="#2c3e50")
    ax.axvline(best["umbral"], ls="--", color="gray")
    ax.annotate(f"umbral elegido = {best['umbral']:.2f}\n(costo no detectada : falsa alarma = "
                f"{a['cost_missed']}:{a['cost_false_alarm']})", (best["umbral"], ax.get_ylim()[1] * 0.85),
                xytext=(8, 0), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Umbral sobre P(clase con alerta)", ylabel="Errores por cada 1.000 imágenes de prueba",
           title=f"Política de alertas · {kind}")
    ax.legend()
    fig.savefig(out["figures"] / f"{kind}_umbral_alertas.png", dpi=130)
    plt.close(fig)
    return {"umbral_elegido_en_validacion": best["umbral"], "prueba_umbral_elegido": chosen,
            "prueba_umbral_0_5": next(r for r in curves["test"] if abs(r["umbral"] - 0.5) < 1e-9),
            "curva_prueba": curves["test"]}


# ----------------------------------------------------------------------------
# Grad-CAM y ejemplos
# ----------------------------------------------------------------------------

def cams_for(model, dataset, indices, device):
    cam = GradCAM(model)
    images, maps = [], []
    for i in indices:
        x, _ = dataset[i]
        maps.append(cam(x[None].to(device))[0])
        images.append(to_image(x))
    cam.remove()
    return images, maps


def grid(images, titles, path, maps=None, cols=5):
    n_rows = max(1, int(np.ceil(len(images) / cols)))
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.2 * cols, 3.4 * n_rows), layout="constrained")
    for ax in np.ravel(axes):
        ax.axis("off")
    for k, (ax, image, title) in enumerate(zip(np.ravel(axes), images, titles)):
        ax.imshow(image)
        if maps is not None:
            ax.imshow(maps[k], cmap="jet", alpha=0.45)
        ax.set_title(title, fontsize=8)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def pick(probs, labels, correct, rng, per_class=2):
    """Dos ejemplos por clase real: aciertos al azar y, entre los errores, los de
    mayor confianza (los más informativos para explicar)."""
    pred, conf = probs.argmax(1), probs.max(1)
    chosen = []
    for c in range(len(CLASSES)):
        idx = np.flatnonzero((labels == c) & ((pred == labels) == correct))
        idx = rng.permutation(idx) if correct else idx[np.argsort(-conf[idx])]
        chosen.extend(idx[:per_class].tolist())
    return chosen


def energy_in_box(cam, box):
    """Fracción de la activación dentro de la caja del conductor (x0, y0, x1, y1 en
    [0, 1] sobre el recorte cuadrado central) y su cociente con el área de la caja.
    Cociente > 1: el modelo concentra la evidencia en el conductor más que el azar."""
    h, w = cam.shape
    x0, y0, x1, y1 = box
    inside = cam[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].sum() / max(cam.sum(), 1e-8)
    return float(inside), float(inside / ((x1 - x0) * (y1 - y0)))


def sanity_check(model, params, dataset, indices, device):
    """Prueba de aleatorización (Adebayo et al., 2018): si los mapas casi no cambian
    al reiniciar los pesos, el método refleja bordes de la imagen y no el modelo."""
    trained = cams_for(model, dataset, indices, device)[1]
    randomized = cams_for(build_model(params, pretrained=False).to(device), dataset, indices, device)[1]
    corr = [np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(trained, randomized)
            if a.std() > 0 and b.std() > 0]
    return float(np.mean(corr)) if corr else None


# ----------------------------------------------------------------------------
# Figuras comunes: arquitectura, curvas de aprendizaje, cajas de las 30 semillas
# ----------------------------------------------------------------------------

def architecture_figure(model, p, kind, out):
    """Diagrama de bloques de la red y resumen en texto (print del modelo)."""
    size = p["size"]
    if p["arch"] == "scratch":
        boxes, side, c = [f"Entrada\n{size}×{size}×3"], size, 3
        for i in range(p["depth"]):
            c, side = p["width"] * 2 ** i, side // 2
            boxes.append(f"Bloque {i + 1}\n2×[Conv 3×3, BN, ReLU]\n{c} filtros\nMaxPool → {side}×{side}")
        boxes += [f"Global\nAvgPool\n{c}", f"Dropout\np={p['dropout']:.2f}", f"Lineal\n{c}→{len(CLASSES)}",
                  "Softmax\n5 clases"]
    else:
        head = [type(m).__name__ for m in head_of(model).children()] if len(list(head_of(model).children())) \
            else [type(head_of(model)).__name__]
        boxes = [f"Entrada\n{size}×{size}×3", f"{p['arch']}\n(preentrenada en ImageNet,\najuste fino con\n"
                 f"lr × {p['backbone_mult']:.3f})", "Global\nAvgPool", "Cabeza nueva\n" + "\n".join(head),
                 "Softmax\n5 clases"]
    fig, ax = plt.subplots(figsize=(1.9 * len(boxes), 2.8), layout="constrained")
    for i, text in enumerate(boxes):
        ax.add_patch(plt.Rectangle((i * 1.3, 0), 1.0, 1.6, fc="#dbe9f6", ec="#2c3e50"))
        ax.text(i * 1.3 + 0.5, 0.8, text, ha="center", va="center", fontsize=7.5)
        if i:
            ax.annotate("", (i * 1.3, 0.8), (i * 1.3 - 0.3, 0.8), arrowprops={"arrowstyle": "->"})
    ax.set(xlim=(-0.1, len(boxes) * 1.3), ylim=(-0.1, 1.7))
    ax.axis("off")
    ax.set_title(f"{kind}: {trainable(model):,} parámetros entrenables".replace(",", "."))
    fig.savefig(out["figures"] / f"{kind}_arquitectura.png", dpi=130)
    plt.close(fig)
    (out["results"] / f"{kind}_arquitectura.txt").write_text(
        f"{p}\n\nParámetros entrenables: {trainable(model)}\n\n{model}\n", encoding="utf-8")


def learning_curves(kind, chosen_seed, out):
    path = out["results"] / f"{kind}_curvas.csv"
    if not path.exists():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for seed in sorted({int(r["seed"]) for r in rows}):
        r = [x for x in rows if int(x["seed"]) == seed]
        epoch = [int(x["epoch"]) for x in r]
        main = seed == chosen_seed
        kw = {"lw": 2 if main else 0.6, "alpha": 1 if main else 0.3}
        axes[0].plot(epoch, [float(x["train_loss"]) for x in r], color="#2c3e50", **kw,
                     label="entrenamiento" if main else None)
        axes[0].plot(epoch, [float(x["val_loss"]) for x in r], color="#c0392b", **kw,
                     label="validación" if main else None)
        axes[1].plot(epoch, [float(x["val_f1_macro"]) for x in r], color="#27ae60", **kw,
                     label=f"semilla guardada ({seed})" if main else None)
    axes[0].set(xlabel="Época", ylabel="Entropía cruzada", title="Pérdida (línea gruesa: modelo guardado)")
    axes[1].set(xlabel="Época", ylabel="F1 macro", title="F1 macro en validación (todas las semillas)")
    for ax in axes:
        ax.legend()
    fig.suptitle(f"Curvas de aprendizaje · {kind}")
    fig.savefig(out["figures"] / f"{kind}_curvas_aprendizaje.png", dpi=130)
    plt.close(fig)


def seeds_boxplot(out, baseline):
    data, names = [], []
    for kind in ("scratch", "transfer"):
        path = out["results"] / f"{kind}_semillas.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as stream:
                data.append([float(r["test_f1_macro"]) for r in csv.DictReader(stream)])
            names.append(f"{kind}\n(n={len(data[-1])})")
    if not data:
        return
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    ax.boxplot(data, tick_labels=names)
    for i, values in enumerate(data, 1):
        ax.scatter(np.full(len(values), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(values)), values,
                   s=12, alpha=0.6, color="#2c3e50")
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="#c0392b", label=f"regresión logística ({baseline:.3f})")
        ax.legend()
    ax.set(ylabel="F1 macro en prueba", title="Variación entre semillas")
    fig.savefig(out["figures"] / "semillas_boxplot.png", dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------

def evaluate_model(kind, config, rows, out, device, rng):
    model, p = load(out["models"] / f"{kind}.pt", device)
    data, split = loaders(rows, config, p["size"], "none", 64, config["seed"])
    val, test = predict(model, data["val"], device), predict(model, data["test"], device)
    probs, labels = test
    pred, conf = probs.argmax(1), probs.max(1)
    names = list(range(len(CLASSES)))
    summary = {"modelo": kind, "arquitectura": p["arch"], "params": p, **scores(*test),
               "validacion": scores(*val),
               "por_clase": classification_report(labels, pred, labels=names, target_names=CLASSES,
                                                  output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(labels, pred, labels=names).tolist(),
               "parametros_entrenables": trainable(model), "latencia_cpu": latency(model, p["size"]),
               "tamano_archivo_mb": round((out["models"] / f"{kind}.pt").stat().st_size / 2 ** 20, 2)}
    model.to(device)
    seeds_file = out["results"] / f"final_{kind}.json"
    if seeds_file.exists():
        summary["semillas"] = {k: v for k, v in json.loads(seeds_file.read_text()).items()
                               if k.startswith(("val_", "test_", "semilla", "mejor_", "peor_"))}
    policy = alert_policy(val, test, config, kind, out)
    summary["alertas"] = {k: v for k, v in policy.items() if k != "curva_prueba"}

    cm = np.asarray(summary["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(ax=axes[0], xticks_rotation=25, cmap="Blues",
                                                            colorbar=False)
    ConfusionMatrixDisplay(cm / np.maximum(cm.sum(1, keepdims=True), 1), display_labels=CLASSES).plot(
        ax=axes[1], xticks_rotation=25, cmap="Blues", colorbar=False, values_format=".2f")
    for ax in axes:
        ax.set(xlabel="Clase predicha", ylabel="Clase real")
    fig.suptitle(f"{kind} ({p['arch']}) · prueba · conteos y proporción por fila")
    fig.savefig(out["figures"] / f"{kind}_confusion.png", dpi=130)
    plt.close(fig)

    test_rows, dataset = split["test"], data["test"].dataset
    for name, correct in [("bien", True), ("mal", False)]:
        idx = pick(probs, labels, correct, rng)
        if not idx:
            continue
        images, maps = cams_for(model, dataset, idx, device)
        titles = [f"real: {CLASSES[labels[i]]}\npred: {CLASSES[pred[i]]} ({conf[i]:.2f})" for i in idx]
        grid(images, titles, out["figures"] / f"{kind}_{name}_clasificadas.png")
        grid(images, titles, out["figures"] / f"{kind}_{name}_gradcam.png", maps)
        write_csv(out["results"] / f"{kind}_{name}_clasificadas.csv", [   # columna para explicar cada caso
            {"path": test_rows[i]["path"], "real": CLASSES[labels[i]], "predicha": CLASSES[pred[i]],
             "confianza": round(float(conf[i]), 3), "explicacion": ""} for i in idx])

    # Mapa promedio por clase predicha: ¿hay una región fija (ventana, techo)?
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(16, 3.4), layout="constrained")
    for ax, c in zip(axes, range(len(CLASSES))):
        idx = rng.permutation(np.flatnonzero(pred == c))[:config["gradcam_mean_per_class"]]
        mean = np.mean(cams_for(model, dataset, idx, device)[1], axis=0) if len(idx) else np.zeros((2, 2))
        ax.imshow(mean, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"pred: {CLASSES[c]} (n={len(idx)})", fontsize=9)
        ax.axis("off")
    fig.savefig(out["figures"] / f"{kind}_gradcam_promedio.png", dpi=120)
    plt.close(fig)

    sample = rng.permutation(len(test_rows))[:config["sanity_images"]].tolist()
    summary["gradcam_correlacion_con_pesos_aleatorios"] = sanity_check(model, p, dataset, sample, device)

    boxes_path = ROOT / config.get("boxes", "data/processed/cajas_conductor.csv")
    if boxes_path.exists():
        index = {r["path"]: i for i, r in enumerate(test_rows)}
        with boxes_path.open(newline="", encoding="utf-8") as stream:
            boxes = [(index[b["path"]], tuple(float(b[k]) for k in ("x0", "y0", "x1", "y1")))
                     for b in csv.DictReader(stream) if b["path"] in index]
        if boxes:
            maps = cams_for(model, dataset, [i for i, _ in boxes], device)[1]
            values = np.array([energy_in_box(cam, box) for cam, (_, box) in zip(maps, boxes)])
            summary["gradcam_en_conductor"] = {
                "imagenes": len(boxes), "fraccion_media": values[:, 0].mean(),
                "cociente_vs_area_media": values[:, 1].mean(),
                "imagenes_con_cociente_mayor_1": int((values[:, 1] > 1).sum())}

    architecture_figure(model, p, kind, out)
    learning_curves(kind, summary.get("semillas", {}).get("semilla_guardada"), out)
    (out["results"] / f"{kind}_metricas.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[{kind}] prueba: exactitud {summary['accuracy']:.4f}, F1 macro {summary['f1_macro']:.4f}; "
          f"umbral de alerta {policy['umbral_elegido_en_validacion']}")
    return summary, policy, model


# ----------------------------------------------------------------------------
# Entregables para la app
# ----------------------------------------------------------------------------

def app_deliverables(kind, summary, policy, model, config, out):
    """metadata.json (cómo cargar y preprocesar), alertas.yaml (política) y
    umbrales.csv (tabla/curva que puede mostrar la app), para el mejor modelo."""
    p, a = summary["params"], config["alert"]
    metadata = {
        "clases": CLASSES,                                  # mismo orden que la salida del modelo
        "tamano_entrada": [p["size"], p["size"]],
        "redimension_lado_corto": resize_side(p["size"]),   # Resize(lado corto) + CenterCrop(tamano_entrada)
        "normalizacion": {"media": MEAN, "std": STD},
        "capa_gradcam": cam_layer_name(model),
        "framework": "pytorch",
        "archivo": f"{kind}.pt",
        "arquitectura": p["arch"],
        "params": p,                                        # necesarios para reconstruir la red
        "tamano_archivo_mb": summary["tamano_archivo_mb"],
        "parametros_entrenables": summary["parametros_entrenables"],
        "metricas_prueba": {"exactitud": summary["accuracy"], "f1_macro": summary["f1_macro"]},
        "referencia": {"archivo": "baseline.joblib", "metadata": "baseline_metadata.json"},
    }
    (out["models"] / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False,
                                                            default=float), encoding="utf-8")
    write_csv(out["results"] / "umbrales.csv", policy["curva_prueba"])
    alertas = {
        "modelo": kind,
        "clases_con_alerta": list(a["classes"]),
        "clases_sin_alerta": [c for c in CLASSES if c not in a["classes"]],
        "justificacion": a.get("justificacion", "").strip(),
        "puntaje": "suma de las probabilidades de las clases con alerta",
        "costo_no_detectada": a["cost_missed"],
        "costo_falsa_alarma": a["cost_false_alarm"],
        "umbral_recomendado": policy["umbral_elegido_en_validacion"],
        "criterio": "umbral que minimiza costo_no_detectada × no detectadas + costo_falsa_alarma × falsas "
                    "alarmas en VALIDACIÓN; resultados reportados en PRUEBA",
        "prueba_con_umbral_recomendado": {k: v for k, v in policy["prueba_umbral_elegido"].items()},
    }
    (out["configs"] / "alertas.yaml").write_text(
        "# Generado por evaluate_cnn.py. La justificación se edita en cnn.yaml (alert.justificacion).\n"
        + yaml.safe_dump(alertas, allow_unicode=True, sort_keys=False, width=100), encoding="utf-8")
    if summary["tamano_archivo_mb"] > 100:
        print(f"ADVERTENCIA: {kind}.pt pesa {summary['tamano_archivo_mb']} MB (> 100 MB para el servidor).")
    print(f"Entregables de la app: models/metadata.json, configs/alertas.yaml, results/umbrales.csv ({kind})")


def run(config, rows, out):
    device = torch.device(config["device"])
    rng = np.random.default_rng(config["seed"])
    table, evaluated = [], {}
    baseline_f1 = None
    baseline = out["results"] / "baseline_metricas.json"   # en la demo no existe: no se mezcla
    if baseline.exists():
        b = json.loads(baseline.read_text(encoding="utf-8"))
        baseline_f1 = b["test"]["f1_macro"]
        table.append({"modelo": "regresion_logistica", "arquitectura": "mobilenet_v3_small congelada",
                      "val_f1_macro": b["validation"]["f1_macro"], "test_accuracy": b["test"]["accuracy"],
                      "test_f1_macro": b["test"]["f1_macro"], "test_f1_media_semillas": "",
                      "test_f1_std_semillas": "", "parametros_entrenables": b["trainable_parameters"],
                      "latencia_p95_ms": "", "tamano_mb": ""})
    for kind in ("scratch", "transfer"):
        if not (out["models"] / f"{kind}.pt").exists():
            print(f"Sin models/{kind}.pt: corre primero `train_cnn final --model {kind}`.")
            continue
        s, policy, model = evaluate_model(kind, config, rows, out, device, rng)
        evaluated[kind] = (s, policy, model)
        seeds = s.get("semillas", {})
        table.append({"modelo": kind, "arquitectura": s["arquitectura"],
                      "val_f1_macro": seeds.get("val_f1_macro", {}).get("media", s["validacion"]["f1_macro"]),
                      "test_accuracy": s["accuracy"], "test_f1_macro": s["f1_macro"],
                      "test_f1_media_semillas": seeds.get("test_f1_macro", {}).get("media", ""),
                      "test_f1_std_semillas": seeds.get("test_f1_macro", {}).get("std", ""),
                      "parametros_entrenables": s["parametros_entrenables"],
                      "latencia_p95_ms": round(s["latencia_cpu"]["p95_ms"], 1), "tamano_mb": s["tamano_archivo_mb"]})
    seeds_boxplot(out, baseline_f1)
    if not table:
        return
    best = max(table, key=lambda r: r["val_f1_macro"])     # el «mejor modelo» se decide por validación
    for r in table:
        r["mejor_por_validacion"] = r is best
    write_csv(out["results"] / "comparacion_modelos.csv", table)
    print(f"Mejor modelo por F1 macro de validación: {best['modelo']}")
    if evaluated:   # la app usa la mejor RED (la referencia ya tiene su propio baseline.joblib)
        kind = max(evaluated, key=lambda k: next(r["val_f1_macro"] for r in table if r["modelo"] == k))
        app_deliverables(kind, *evaluated[kind], config, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cnn.yaml")
    parser.add_argument("--output", type=Path, help="carpeta de salida (obligatoria en modo demo)")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("limit_per_class_split") and args.output is None:
        parser.error("Indica --output para la demostración.")
    torch.set_num_threads(config["threads"])
    run(config, read_rows(config), outputs(args.output or ROOT))


if __name__ == "__main__":
    main()
    