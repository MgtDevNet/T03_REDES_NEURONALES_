"""Factorización de matrices con pérdida BPR (tarea 4, decisión 7.3), en PyTorch.

Modelo:   puntaje(u, i) = p_u · q_i        (p_u, q_i ∈ R^d se aprenden)
Pérdida:  para cada trío (u, i, j), con i visitado por u y j no visitado (muestreado al azar):
              −ln σ(puntaje(u,i) − puntaje(u,j)) + λ (‖p_u‖² + ‖q_i‖² + ‖q_j‖²)
          Minimizarla = maximizar la probabilidad de que u prefiera lo que visitó sobre lo que no
          (Rendle et al., 2009 — referencia pendiente de verificar en la fuente).

Parámetros entrenables: (n_usuarios + n_destinos) · d.

Este archivo solo define el modelo y su entrenamiento. Las pruebas están en
`probar_factorizacion.py` y el ajuste/evaluación en `ajustar_factorizacion.py`.
"""

import numpy as np
import torch
from torch import nn

from evaluacion import N_DESTINOS, evaluar

N_USUARIOS = 999


class ModeloBPR(nn.Module):
    def __init__(self, n_usuarios, n_destinos, d):
        super().__init__()
        self.P = nn.Embedding(n_usuarios, d)  # fila u-1 = vector del usuario u
        self.Q = nn.Embedding(n_destinos, d)  # fila i-1 = vector del destino i
        nn.init.normal_(self.P.weight, std=0.1)
        nn.init.normal_(self.Q.weight, std=0.1)

    def forward(self, u, i):
        return (self.P(u) * self.Q(i)).sum(dim=1)


def muestrear_negativos(usuarios, positivos, n_destinos, rng):
    """Para cada usuario, un destino que NO esté entre sus positivos (muestreo uniforme)."""
    j = rng.integers(0, n_destinos, size=len(usuarios))
    while True:
        malos = np.isin(usuarios * n_destinos + j, positivos)
        if not malos.any():
            return j
        j[malos] = rng.integers(0, n_destinos, size=malos.sum())


class FactorizacionBPR:
    """Envoltorio con la misma interfaz que los modelos de `referencias.py`: ajustar / puntuar."""

    nombre = "Factorización (BPR)"

    def __init__(self, d=16, lam=1e-3, epocas=100, semilla=0, lr=0.01, lote=256,
                 n_usuarios=N_USUARIOS, n_destinos=N_DESTINOS):
        self.d, self.lam, self.epocas, self.semilla = d, lam, epocas, semilla
        self.lr, self.lote = lr, lote
        self.n_usuarios, self.n_destinos = n_usuarios, n_destinos
        self.historia = []  # una fila por época: pérdida y, si hay validación, NDCG@10
        self.mejor_epoca = None

    def parametros(self):
        return sum(p.numel() for p in self.modelo.parameters())

    def ajustar(self, aprende, validacion=None, paciencia=10):
        """Entrena. Si se da `validacion`, para cuando NDCG@10 no mejora en `paciencia` épocas
        y se queda con los pesos de la mejor época (se guarda en `mejor_epoca`)."""
        torch.manual_seed(self.semilla)
        rng = np.random.default_rng(self.semilla)
        self.modelo = ModeloBPR(self.n_usuarios, self.n_destinos, self.d)
        opt = torch.optim.Adam(self.modelo.parameters(), lr=self.lr)

        u_all = aprende.UserID.to_numpy() - 1
        i_all = aprende.DestinationID.to_numpy() - 1
        positivos = np.unique(u_all * self.n_destinos + i_all)

        if validacion is not None:  # pérdida de validación: negativos fijos, con su propio generador
            uv = validacion.UserID.to_numpy() - 1
            iv = validacion.DestinationID.to_numpy() - 1
            todos_pos = np.union1d(positivos, uv * self.n_destinos + iv)
            jv = muestrear_negativos(uv, todos_pos, self.n_destinos, np.random.default_rng(10_000 + self.semilla))
            self._val = tuple(torch.as_tensor(x) for x in (uv, iv, jv))

        mejor, sin_mejora, mejores_pesos = -1.0, 0, None
        for epoca in range(1, self.epocas + 1):
            self.modelo.train()
            orden = rng.permutation(len(u_all))
            j_all = muestrear_negativos(u_all[orden], positivos, self.n_destinos, rng)
            total = 0.0
            for ini in range(0, len(orden), self.lote):
                idx = orden[ini:ini + self.lote]
                u = torch.as_tensor(u_all[idx])
                i = torch.as_tensor(i_all[idx])
                j = torch.as_tensor(j_all[ini:ini + self.lote])
                pu, qi, qj = self.modelo.P(u), self.modelo.Q(i), self.modelo.Q(j)
                x = (pu * qi).sum(1) - (pu * qj).sum(1)
                reg = self.lam * (pu.pow(2).sum(1) + qi.pow(2).sum(1) + qj.pow(2).sum(1))
                perdida = (-nn.functional.logsigmoid(x) + reg).mean()
                opt.zero_grad()
                perdida.backward()
                opt.step()
                total += perdida.item() * len(idx)
            fila = {"epoca": epoca, "perdida": total / len(orden)}

            if validacion is not None:
                fila["perdida_val"] = self._perdida_val()
                ndcg = evaluar(self.puntuar, aprende, validacion, semilla=self.semilla, ks=(10,))[0].NDCG.iloc[0]
                fila["NDCG@10_val"] = ndcg
                if ndcg > mejor:
                    mejor, sin_mejora, self.mejor_epoca = ndcg, 0, epoca
                    mejores_pesos = {k: v.clone() for k, v in self.modelo.state_dict().items()}
                else:
                    sin_mejora += 1
            self.historia.append(fila)
            if validacion is not None and sin_mejora >= paciencia:
                break

        if mejores_pesos is not None:
            self.modelo.load_state_dict(mejores_pesos)
        else:
            self.mejor_epoca = self.epocas
        return self

    @torch.no_grad()
    def _perdida_val(self):
        """Misma pérdida que en el entrenamiento (BPR + regularización), sobre los pares de validación."""
        u, i, j = self._val
        pu, qi, qj = self.modelo.P(u), self.modelo.Q(i), self.modelo.Q(j)
        x = (pu * qi).sum(1) - (pu * qj).sum(1)
        reg = self.lam * (pu.pow(2).sum(1) + qi.pow(2).sum(1) + qj.pow(2).sum(1))
        return float((-nn.functional.logsigmoid(x) + reg).mean())

    @torch.no_grad()
    def puntuar(self, usuarios):
        self.modelo.eval()
        P = self.modelo.P.weight[torch.as_tensor(np.asarray(usuarios) - 1)]
        return (P @ self.modelo.Q.weight.T).numpy()

    def factores(self):
        """Los vectores aprendidos como arreglos numpy (para guardarlos en .npz para la app)."""
        return self.modelo.P.weight.detach().numpy().copy(), self.modelo.Q.weight.detach().numpy().copy()
