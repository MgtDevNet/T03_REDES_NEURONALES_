"""Definiciones compartidas por el entrenamiento, la evaluación y la app.

Solo depende de torch, torchvision, numpy y Pillow, para que la app pueda
importarlo sin cargar scikit-learn ni Optuna.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models as tvm
from torchvision import transforms as T

CLASSES = ["safe_driving", "turning", "texting_phone", "talking_phone", "other_activities"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def resize_side(size):
    """Lado corto antes del recorte central (256 para una entrada de 224)."""
    return round(size * 256 / 224)


def test_transform(size):
    return T.Compose([T.Resize(resize_side(size)), T.CenterCrop(size), T.ToTensor(), T.Normalize(MEAN, STD)])


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
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(p, pretrained=True):
    """Construye la red a partir del diccionario de hiperparámetros `p`."""
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


def cam_layer_name(model):
    target = cam_layer(model)
    return next(name for name, module in model.named_modules() if module is target)


def trainable(model):
    return sum(t.numel() for t in model.parameters() if t.requires_grad)


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017): pondera cada mapa A_k de la capa objetivo
    por el promedio espacial del gradiente ∂y_c/∂A_k y aplica ReLU:
    L_c = ReLU(Σ_k α_k A_k), con α_k = mean_ij ∂y_c/∂A_k[i, j]."""

    def __init__(self, model, layer=None):
        self.model, self.store = model, {}
        self.handle = (layer or cam_layer(model)).register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.store["act"] = output
        output.register_hook(lambda grad: self.store.__setitem__("grad", grad))

    def __call__(self, x, target=None):
        """x: lote normalizado (N, 3, H, W). target: clases (N,) o None = clase predicha.
        Devuelve mapas (N, H, W) en [0, 1]."""
        self.model.eval()
        with torch.enable_grad():
            logits = self.model(x.clone().requires_grad_(True))
            if target is None:
                target = logits.argmax(1)
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, target.view(-1, 1)).sum().backward()
        act, grad = self.store["act"], self.store["grad"]
        cam = F.relu((grad.mean((2, 3), keepdim=True) * act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin((1, 2), keepdim=True)
        return (cam / cam.amax((1, 2), keepdim=True).clamp_min(1e-8)).detach().cpu().numpy()

    def remove(self):
        self.handle.remove()


def to_image(tensor):
    """Deshace la normalización de un tensor (3, H, W) para mostrarlo."""
    return (tensor.cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)).clip(0, 1)