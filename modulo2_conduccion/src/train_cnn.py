"""Punto 5: CNN desde cero y ajuste fino, con búsqueda de hiperparámetros (Optuna/TPE).

Desde la raíz del repositorio:
    # Experimento completo (caché → búsquedas → 30 semillas → evaluación)
    python -m modulo2_conduccion.src.train_cnn todo --config modulo2_conduccion/configs/cnn.yaml
    # Modo demo (< 2 min en CPU, usa las imágenes de muestra versionadas)
    python -m modulo2_conduccion.src.train_cnn todo --config modulo2_conduccion/configs/cnn_demo.yaml --output demo_redes

Pasos sueltos: cache | search --model scratch|transfer | final --model scratch|transfer.
La partición viene de data/processed/particiones.csv (no se modifica). La
búsqueda y la parada temprana usan solo validación.
"""

import argparse
import csv
import json
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models as tvm
from torchvision import transforms as T
import yaml

from .prepare import CLASSES, ROOT, write_csv
from .train import select_rows

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------------
# Configuración y datos
# ----------------------------------------------------------------------------

def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("device", "auto") == "auto":
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return config


def read_rows(config, manifest=ROOT / "data/processed/particiones.csv"):
    with Path(manifest).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    # Misma selección estratificada que la demo de la línea base (train.select_rows).
    return select_rows(rows, config.get("limit_per_class_split", 0), config["seed"])


def cache_images(config, zip_path):
    """Decodifica el ZIP una vez y guarda JPEG con lado corto de 256 px (nombre =
    SHA-256 del original). Leer fotos de 3072 × 4096 en cada época domina el
    tiempo; con la caché cada trial es varias veces más rápido. Además copia una
    muestra pequeña a `demo_cache_dir`, que sí se versiona, para el modo demo."""
    from io import BytesIO
    from zipfile import ZipFile
    cache = ROOT / config["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    rows = read_rows(config)
    with ZipFile(zip_path) as archive:
        for i, row in enumerate(rows):
            target = cache / f"{row['sha256']}.jpg"
            if target.exists():
                continue
            with Image.open(BytesIO(archive.read(row["path"]))) as image:
                image.draft("RGB", (256, 256))
                image = ImageOps.exif_transpose(image).convert("RGB")
                scale = 256 / min(image.size)
                size = (round(image.width * scale), round(image.height * scale))
                image.resize(size, Image.Resampling.LANCZOS).save(target, quality=95)
            if i % 500 == 0:
                print(f"Caché: {i}/{len(rows)}", flush=True)
    demo = config.get("export_demo_per_class_split")
    if demo:
        target = ROOT / config["demo_cache_dir"]
        target.mkdir(parents=True, exist_ok=True)
        for row in select_rows(rows, demo, config["seed"]):
            shutil.copy(cache / f"{row['sha256']}.jpg", target / f"{row['sha256']}.jpg")
        print(f"Muestra demo copiada en {target}")


class CachedImages(Dataset):
    def __init__(self, rows, folder, transform):
        self.rows, self.folder, self.transform = rows, Path(folder), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.folder / f"{row['sha256']}.jpg") as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row["class"])


def make_transforms(size, aug):
    """Aumento en línea. El espejo horizontal es coherente con el conjunto: los
    autores ya publicaron versiones espejadas de las fotos (ver punto 2)."""
    test = T.Compose([T.Resize(round(size * 256 / 224)), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
    if aug == "none":
        return test, test
    strong = aug == "strong"
    train = T.Compose([
        T.RandomResizedCrop(size, scale=(0.6 if strong else 0.8, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(15 if strong else 7),
        T.ColorJitter(*(0.4, 0.4, 0.3, 0.05) if strong else (0.2, 0.2, 0.1)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        *([T.RandomErasing(p=0.25, scale=(0.02, 0.12))] if strong else []),
    ])
    return train, test


def loaders(rows, config, size, aug, batch_size, seed):
    train_tf, test_tf = make_transforms(size, aug)
    folder = ROOT / config["cache_dir"]
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    workers = config["workers"]
    kw = dict(batch_size=batch_size, num_workers=workers, pin_memory=config["device"] == "cuda",
              persistent_workers=workers > 0)
    return {
        "train": DataLoader(CachedImages(split["train"], folder, train_tf), shuffle=True,
                            drop_last=len(split["train"]) > batch_size,
                            generator=torch.Generator().manual_seed(seed), **kw),
        "val": DataLoader(CachedImages(split["val"], folder, test_tf), **kw),
        "test": DataLoader(CachedImages(split["test"], folder, test_tf), **kw),
    }, split


# ----------------------------------------------------------------------------
# Modelos
# ----------------------------------------------------------------------------

class ScratchCNN(nn.Module):
    """VGG compacta: `depth` bloques [conv-BN-ReLU] × 2 + max-pool, luego GAP."""

    def __init__(self, width=32, depth=4, dropout=0.3, n_classes=len(CLASSES)):
        super().__init__()
        layers, c_in = [], 3
        for i in range(depth):
            c_out = width * 2 ** i
            layers += [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
                       nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(),
                       nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(c_in, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


BACKBONES = {
    "mobilenet_v3_small": (tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v3_large": (tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(p, pretrained=True):
    n = len(CLASSES)
    if p["arch"] == "scratch":
        return ScratchCNN(p["width"], p["depth"], p["dropout"])
    fn, weights = BACKBONES[p["arch"]]
    model = fn(weights=weights if pretrained else None)
    if p["arch"].startswith("mobilenet"):        # classifier = [Linear, Hardswish, Dropout, Linear]
        model.classifier[2].p = p["dropout"]
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, n)
    elif p["arch"].startswith("efficientnet"):   # classifier = [Dropout, Linear]
        model.classifier[0].p = p["dropout"]
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    else:                                         # ResNet
        model.fc = nn.Sequential(nn.Dropout(p["dropout"]), nn.Linear(model.fc.in_features, n))
    return model


def head_of(model):
    if isinstance(model, ScratchCNN):
        return model.head
    return model.fc if isinstance(model, tvm.ResNet) else model.classifier


def cam_layer(model):
    """Última capa convolucional (con resolución espacial) para Grad-CAM."""
    if isinstance(model, ScratchCNN):
        return model.features[-2]                 # última ReLU antes del pooling final
    if isinstance(model, tvm.ResNet):
        return model.layer4[-1]
    return model.features[-1]                     # MobileNetV3 / EfficientNet


def trainable(model):
    return sum(t.numel() for t in model.parameters() if t.requires_grad)


# ----------------------------------------------------------------------------
# Entrenamiento
# ----------------------------------------------------------------------------

def make_optimizer(model, p):
    """Ajuste fino con tasas discriminativas: el backbone aprende `backbone_mult`
    veces más lento que la cabeza nueva, para no destruir los rasgos de ImageNet."""
    head_ids = {id(t) for t in head_of(model).parameters()}
    body = [t for t in model.parameters() if id(t) not in head_ids]
    head = [t for t in model.parameters() if id(t) in head_ids]
    groups = [{"params": body, "lr": p["lr"] * p.get("backbone_mult", 1.0)}, {"params": head, "lr": p["lr"]}]
    if p["optimizer"] == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True, weight_decay=p["weight_decay"])
    return torch.optim.AdamW(groups, weight_decay=p["weight_decay"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device, non_blocking=True)), 1).float().cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def scores(probs, labels):
    pred = probs.argmax(1)
    return {"accuracy": float(accuracy_score(labels, pred)),
            "f1_macro": float(f1_score(labels, pred, average="macro", labels=range(len(CLASSES)),
                                       zero_division=0))}


def class_weights(rows):
    counts = np.bincount([CLASSES.index(r["class"]) for r in rows], minlength=len(CLASSES))
    return torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)


def fit(p, data, split, device, trial=None, pretrained=True):
    """Calentamiento lineal (1 época) + coseno; parada temprana por F1 macro en validación."""
    torch.manual_seed(p["seed"])
    np.random.seed(p["seed"])
    model = build_model(p, pretrained=pretrained).to(device)
    optimizer = make_optimizer(model, p)
    warm = max(1, len(data["train"]))
    steps = max(warm + 1, p["epochs"] * len(data["train"]))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(
        (s + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(1.0, max(0, s - warm) / (steps - warm))))))
    weight = class_weights(split["train"]).to(device) if p["class_weights"] else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=p["label_smoothing"])
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=amp)

    best, best_state, history, stale = -1.0, None, [], 0
    for epoch in range(p["epochs"]):
        model.train()
        started, total, seen = time.perf_counter(), 0.0, 0
        for x, y in data["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total, seen = total + loss.item() * len(y), seen + len(y)
        val = scores(*predict(model, data["val"], device))
        row = {"epoch": epoch + 1, "train_loss": total / max(seen, 1), "val_accuracy": val["accuracy"],
               "val_f1_macro": val["f1_macro"], "seconds": time.perf_counter() - started}
        history.append(row)
        print(f"  época {row['epoch']:2d}  pérdida {row['train_loss']:.3f}  "
              f"F1 val {row['val_f1_macro']:.3f}  ({row['seconds']:.0f} s)", flush=True)
        if row["val_f1_macro"] > best:
            best, stale = row["val_f1_macro"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if trial is not None:
            trial.report(row["val_f1_macro"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if stale >= p["patience"]:
            break
    model.load_state_dict(best_state)
    return model, best, history


# ----------------------------------------------------------------------------
# Búsqueda de hiperparámetros
# ----------------------------------------------------------------------------

def suggest(trial, kind, config):
    """TPE (Bergstra et al., 2011) modela qué regiones dan mejor F1 y concentra allí
    los trials; la poda por mediana corta los trials que van peor que la mediana
    en la misma época (Akiba et al., 2019). Rangos en el YAML (search_space)."""
    s, m = config["search_space"], config[kind]
    p = {
        "lr": trial.suggest_float("lr", *m["lr"], log=True),
        "weight_decay": trial.suggest_float("weight_decay", *s["weight_decay"], log=True),
        "optimizer": trial.suggest_categorical("optimizer", s["optimizers"]),
        "dropout": trial.suggest_float("dropout", *s["dropout"]),
        "aug": trial.suggest_categorical("aug", s["aug"]),
        "label_smoothing": trial.suggest_float("label_smoothing", *s["label_smoothing"]),
        "class_weights": trial.suggest_categorical("class_weights", s["class_weights"]),
        "batch_size": config["batch_size"], "patience": config["patience"], "seed": config["seed"],
        "size": m["size"], "epochs": m["epochs"],
    }
    if kind == "scratch":
        p.update(arch="scratch", width=trial.suggest_categorical("width", m["widths"]),
                 depth=trial.suggest_int("depth", *m["depth"]))
    else:
        p.update(arch=trial.suggest_categorical("arch", m["archs"]),
                 backbone_mult=trial.suggest_float("backbone_mult", *m["backbone_mult"], log=True))
    return p


def search(kind, config, rows, out):
    device = torch.device(config["device"])
    study = optuna.create_study(
        study_name=f"conduccion_{kind}", direction="maximize", load_if_exists=True,
        storage=f"sqlite:///{(out['results'] / f'optuna_{kind}.db').as_posix()}",  # se puede reanudar
        sampler=optuna.samplers.TPESampler(seed=config["seed"], n_startup_trials=config["startup_trials"]),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    started = time.perf_counter()

    def objective(trial):
        p = suggest(trial, kind, config)
        print(f"Trial {trial.number}: {json.dumps(p)}", flush=True)
        data, split = loaders(rows, config, p["size"], p["aug"], p["batch_size"], p["seed"])
        _, best, history = fit(p, data, split, device, trial)
        trial.set_user_attr("params_full", p)
        trial.set_user_attr("epochs_run", len(history))
        return best

    pending = config[kind]["trials"] - len(study.trials)
    if pending > 0:
        study.optimize(objective, n_trials=pending, gc_after_trial=True)
    study.trials_dataframe().to_csv(out["results"] / f"busqueda_{kind}.csv", index=False)
    best = study.best_trial
    states = [t.state.name for t in study.trials]
    summary = {"trial": best.number, "val_f1_macro": best.value, "params": best.user_attrs["params_full"],
               "configuraciones_probadas": len(states), "completas": states.count("COMPLETE"),
               "podadas": states.count("PRUNED"),
               "segundos_esta_corrida": round(time.perf_counter() - started, 1),
               "segundos_todos_los_trials": round(sum(
                   (t.datetime_complete - t.datetime_start).total_seconds()
                   for t in study.trials if t.datetime_complete), 1)}
    (out["results"] / f"mejor_{kind}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Mejor trial {best.number}: F1 val = {best.value:.4f}")
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
        for name, plot in [("historia", plot_optimization_history), ("importancia", plot_param_importances)]:
            with plt.rc_context():          # Optuna cambia el estilo global; se restaura al salir
                ax = plot(study)
                ax.figure.set_size_inches(9, 5)
                ax.figure.savefig(out["figures"] / f"optuna_{name}_{kind}.png", dpi=130, bbox_inches="tight")
                plt.close(ax.figure)
    except Exception as error:   # noqa: BLE001 — con muy pocos trials la importancia falla
        print(f"(figura de Optuna omitida: {error})")


def final(kind, config, rows, out):
    """Reentrena la mejor configuración con N semillas (las reglas del equipo piden 30).
    Aquí la variación sí es informativa (inicialización, orden de lotes, aumento).
    Se guarda en models/<kind>.pt el modelo con F1 de validación mediano, que es
    el que analiza evaluate_cnn.py y carga la app; el resto queda en models/checkpoints/."""
    device = torch.device(config["device"])
    p0 = json.loads((out["results"] / f"mejor_{kind}.json").read_text())["params"]
    checkpoints = out["models"] / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in range(config["final_seeds"]):
        p = {**p0, "seed": seed}
        data, split = loaders(rows, config, p["size"], p["aug"], p["batch_size"], seed)
        print(f"[{kind}] semilla {seed + 1}/{config['final_seeds']}", flush=True)
        started = time.perf_counter()
        model, _, history = fit(p, data, split, device)
        val = scores(*predict(model, data["val"], device))
        test = scores(*predict(model, data["test"], device))
        torch.save({"params": p, "classes": CLASSES, "state_dict": model.state_dict()},
                   checkpoints / f"{kind}_seed{seed}.pt")
        runs.append({"seed": seed, "val_accuracy": val["accuracy"], "val_f1_macro": val["f1_macro"],
                     "test_accuracy": test["accuracy"], "test_f1_macro": test["f1_macro"],
                     "epocas": len(history), "segundos": round(time.perf_counter() - started, 1)})
    write_csv(out["results"] / f"{kind}_semillas.csv", runs)
    f1 = np.array([r["val_f1_macro"] for r in runs])
    chosen = runs[int(np.argsort(f1)[len(f1) // 2])]["seed"]          # semilla mediana en validación
    shutil.copy(checkpoints / f"{kind}_seed{chosen}.pt", out["models"] / f"{kind}.pt")
    summary = {"modelo": kind, "params": p0, "semillas": len(runs), "semilla_guardada": chosen,
               "parametros_entrenables": trainable(build_model(p0, pretrained=False)),
               "segundos_total": round(sum(r["segundos"] for r in runs), 1)}
    for key in ["val_accuracy", "val_f1_macro", "test_accuracy", "test_f1_macro"]:
        values = np.array([r[key] for r in runs])
        summary[key] = {"media": values.mean(), "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                        "min": values.min(), "max": values.max()}
    (out["results"] / f"final_{kind}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False,
                                                                  default=float))
    print(f"[{kind}] F1 val {summary['val_f1_macro']['media']:.4f} ± {summary['val_f1_macro']['std']:.4f}")


def outputs(base):
    out = {name: base / name for name in ("models", "results", "figures")}
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["cache", "search", "final", "todo"])
    parser.add_argument("--model", choices=["scratch", "transfer"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cnn.yaml")
    parser.add_argument("--zip", type=Path, default=ROOT / "data/raw/dataset.zip")
    parser.add_argument("--output", type=Path, help="carpeta de salida (obligatoria en modo demo)")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.get("limit_per_class_split") and args.output is None:
        parser.error("Indica --output para guardar la demostración por separado.")
    if args.command in ("search", "final") and args.model is None:
        parser.error("Indica --model scratch o --model transfer.")
    torch.set_num_threads(config["threads"])
    torch.backends.cudnn.benchmark = True
    if args.command == "cache":
        return cache_images(config, args.zip)
    if not (ROOT / config["cache_dir"]).exists():
        if config.get("limit_per_class_split") or not args.zip.exists():
            parser.error(f"No existe {config['cache_dir']} ni el ZIP en {args.zip}.")
        cache_images(config, args.zip)
    out = outputs(args.output or ROOT)
    rows = read_rows(config)
    kinds = [args.model] if args.model else ["scratch", "transfer"]
    started = time.perf_counter()
    for kind in kinds:
        if args.command in ("search", "todo"):
            search(kind, config, rows, out)
        if args.command in ("final", "todo"):
            final(kind, config, rows, out)
    if args.command == "todo":
        from .evaluate_cnn import run as evaluate
        evaluate(config, rows, out)
    print(f"Tiempo total: {time.perf_counter() - started:.1f} s")


if __name__ == "__main__":
    main()