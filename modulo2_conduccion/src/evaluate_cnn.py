"""Puntos 6 y 7 y pregunta de negocio 1: evaluación en prueba, política de alertas,
ejemplos bien y mal clasificados, y Grad-CAM.

Desde la raíz del repositorio (después de `train_cnn final`):
    python -m modulo2_conduccion.src.evaluate_cnn --config modulo2_conduccion/configs/cnn.yaml

Analiza models/scratch.pt y models/transfer.pt (semilla mediana) y resume las 30
semillas desde results/<modelo>_semillas.csv.
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
import torch.nn.functional as F

from .train_cnn import (MEAN, STD, build_model, cam_layer, load_config, loaders, outputs, predict, read_rows,
                   scores, trainable)
from .prepare import CLASSES, ROOT, write_csv


def load(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
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
    """Puntaje de alerta = P(escribir) + P(hablar) + P(otras) según el softmax.
    Para cada umbral t se alerta si el puntaje ≥ t. Se cuentan, por cada 1.000
    imágenes, las falsas alarmas (sin alerta real, pero se alerta) y las
    distracciones no detectadas (alerta real, pero no se alerta)."""
    score = probs[:, alert_ids].sum(1)
    real = np.isin(labels, alert_ids)
    rows = []
    for t in thresholds:
        alert = score >= t
        fa, fn = int((alert & ~real).sum()), int((~alert & real).sum())
        rows.append({"umbral": round(float(t), 3), "falsas_alarmas": fa, "no_detectadas": fn,
                     "falsas_alarmas_x1000": 1000 * fa / len(labels), "no_detectadas_x1000": 1000 * fn / len(labels),
                     "sensibilidad_alerta": float(alert[real].mean()) if real.any() else 0.0,
                     "precision_alerta": float(real[alert].mean()) if alert.any() else 0.0})
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
    default = next(r for r in curves["test"] if abs(r["umbral"] - 0.5) < 1e-9)
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
    return {"clases_con_alerta": a["classes"], "costo_no_detectada": a["cost_missed"],
            "costo_falsa_alarma": a["cost_false_alarm"], "umbral_elegido_en_validacion": best["umbral"],
            "prueba_umbral_elegido": chosen, "prueba_umbral_0_5": default}


# ----------------------------------------------------------------------------
# Grad-CAM (Selvaraju et al., 2017)
# ----------------------------------------------------------------------------

class GradCAM:
    """Pondera cada mapa A_k de la capa objetivo por el promedio espacial del
    gradiente ∂y_c/∂A_k y aplica ReLU: L_c = ReLU(Σ_k α_k A_k)."""

    def __init__(self, model, layer):
        self.model, self.store = model, {}
        self.handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.store["act"] = output
        output.register_hook(lambda grad: self.store.__setitem__("grad", grad))

    def __call__(self, x):
        self.model.eval()
        logits = self.model(x.clone().requires_grad_(True))
        self.model.zero_grad(set_to_none=True)
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
        act, grad = self.store["act"], self.store["grad"]
        cam = F.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin((1, 2), keepdim=True)
        return (cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-8)).detach().cpu().numpy()


def cams_for(model, dataset, indices, device):
    cam = GradCAM(model, cam_layer(model))
    images, maps = [], []
    for i in indices:
        x, _ = dataset[i]
        maps.append(cam(x[None].to(device))[0])
        images.append((x.numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)).clip(0, 1))
    cam.handle.remove()
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
    return float(np.nanmean([np.corrcoef(a.ravel(), b.ravel())[0, 1] for a, b in zip(trained, randomized)]))


# ----------------------------------------------------------------------------

def evaluate_model(kind, config, rows, out, device, rng):
    model, p = load(out["models"] / f"{kind}.pt", device)
    data, split = loaders(rows, config, p["size"], "none", 64, config["seed"])
    val, test = predict(model, data["val"], device), predict(model, data["test"], device)
    probs, labels = test
    pred, conf = probs.argmax(1), probs.max(1)
    names = list(range(len(CLASSES)))
    summary = {"modelo": kind, "arquitectura": p["arch"], **scores(*test),
               "por_clase": classification_report(labels, pred, labels=names, target_names=CLASSES,
                                                  output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(labels, pred, labels=names).tolist(),
               "parametros_entrenables": trainable(model), "latencia_cpu": latency(model, p["size"])}
    model.to(device)
    seeds_file = out["results"] / f"final_{kind}.json"
    if seeds_file.exists():
        summary["semillas"] = {k: v for k, v in json.loads(seeds_file.read_text()).items()
                               if k.startswith(("val_", "test_", "semilla"))}
    summary["alertas"] = alert_policy(val, test, config, kind, out)

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
        if not correct:   # columna vacía para escribir la explicación de cada error
            write_csv(out["results"] / f"{kind}_errores.csv", [
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

    (out["results"] / f"{kind}_metricas.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"[{kind}] prueba: exactitud {summary['accuracy']:.4f}, F1 macro {summary['f1_macro']:.4f}; "
          f"umbral de alerta {summary['alertas']['umbral_elegido_en_validacion']}")
    return summary


def run(config, rows, out):
    device = torch.device(config["device"])
    rng = np.random.default_rng(config["seed"])
    table = []
    baseline = out["results"] / "baseline_metricas.json"   # en la demo no existe: no se mezcla
    if baseline.exists():
        b = json.loads(baseline.read_text(encoding="utf-8"))
        table.append({"modelo": "regresion_logistica", "arquitectura": "mobilenet_v3_small congelada",
                      "val_f1_macro_media": b["validation"]["f1_macro"], "test_accuracy": b["test"]["accuracy"],
                      "test_f1_macro": b["test"]["f1_macro"], "test_f1_macro_std_semillas": 0.0,
                      "parametros_entrenables": b["trainable_parameters"], "latencia_p95_ms": ""})
    for kind in ("scratch", "transfer"):
        if not (out["models"] / f"{kind}.pt").exists():
            print(f"Sin models/{kind}.pt: corre primero `train_cnn final --model {kind}`.")
            continue
        s = evaluate_model(kind, config, rows, out, device, rng)
        seeds = s.get("semillas", {})
        table.append({"modelo": kind, "arquitectura": s["arquitectura"],
                      "val_f1_macro_media": seeds.get("val_f1_macro", {}).get("media", ""),
                      "test_accuracy": s["accuracy"], "test_f1_macro": s["f1_macro"],
                      "test_f1_macro_std_semillas": seeds.get("test_f1_macro", {}).get("std", ""),
                      "parametros_entrenables": s["parametros_entrenables"],
                      "latencia_p95_ms": round(s["latencia_cpu"]["p95_ms"], 1)})
    if table:
        # El «mejor modelo» se decide por validación, no por prueba.
        best = max((r for r in table if r["val_f1_macro_media"] != ""), key=lambda r: r["val_f1_macro_media"])
        for r in table:
            r["mejor_por_validacion"] = r is best
        write_csv(out["results"] / "comparacion_modelos.csv", table)
        print(f"Mejor modelo por F1 macro de validación: {best['modelo']}")


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