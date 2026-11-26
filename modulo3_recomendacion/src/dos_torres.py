"""Red neuronal de dos torres con variables de contenido (tarea 5, DECISIONES.md paso 8).

    Torre del usuario:  variables del perfil  ──► capa oculta (16, ReLU) ──► vector u ∈ R^d
    Torre del destino:  [embedding del lugar ∈ R^d, Popularity estandarizada]
                                              ──► capa oculta (16, ReLU) ──► vector v ∈ R^d
    Puntaje:            u · v

Ninguna torre usa el ID: la torre del usuario funciona igual para un perfil nuevo de la app (sin
ID), y la del destino puntúa también los IDs que nadie tocó al aprender («IDs fríos»).

Variables del usuario:
    - "preferencias": las 5 casillas pref_Beach … pref_Adventure (0/1).
    - "completo":     + género (2 casillas), adultos (1 o 2) y niños (0, 1 o 2), en casillas 0/1.

Pérdidas (ambas se comparan en el ajuste):
    - BCE con negativos: cada par (usuario, destino) es una moneda «visitó / no visitó»;
      −[y ln σ(s) + (1 − y) ln(1 − σ(s))] con y = 1 para el visitado y 0 para los negativos.
    - BPR: −ln σ(s_visitado − s_negativo), como en la factorización.
Negativos: uniformes entre los 1000 IDs, sin los positivos del usuario; k por positivo.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn

from evaluacion import N_DESTINOS, PROCESADOS, evaluar, lugar_de_cada_id
from factorizacion import muestrear_negativos

TIPOS = ["Beach", "Historical", "City", "Nature", "Adventure"]
GENEROS, ADULTOS, NINOS = ["Female", "Male"], [1, 2], [0, 1, 2]


def variables_usuario(perfiles, tipo):
    """Matriz (n × f) de variables 0/1 a partir de un DataFrame con las columnas de usuarios.csv
    (o de un perfil nuevo armado con las mismas columnas)."""
    cols = [perfiles[f"pref_{t}"].to_numpy(float) for t in TIPOS]
    if tipo == "completo":
        cols += [(perfiles.Gender == g).to_numpy(float) for g in GENEROS]
        cols += [(perfiles.NumberOfAdults == a).to_numpy(float) for a in ADULTOS]
        cols += [(perfiles.NumberOfChildren == c).to_numpy(float) for c in NINOS]
    elif tipo != "preferencias":
        raise ValueError(tipo)
    return np.stack(cols, axis=1).astype(np.float32)


def variables_destino():
    """(lugar 0–4 de cada ID, Popularity estandarizada), en el orden de los DestinationID 1..1000."""
    lugar_id, _ = lugar_de_cada_id()
    pop = pd.read_csv(PROCESADOS / "destinos.csv").set_index("DestinationID").Popularity.reindex(range(1, N_DESTINOS + 1)).to_numpy()
    return lugar_id.astype(np.int64), ((pop - pop.mean()) / pop.std()).astype(np.float32)


class RedDosTorres(nn.Module):
    def __init__(self, n_var_usuario, d, oculta=16, n_lugares=5):
        super().__init__()
        self.torre_usuario = nn.Sequential(nn.Linear(n_var_usuario, oculta), nn.ReLU(), nn.Linear(oculta, d))
        self.emb_lugar = nn.Embedding(n_lugares, d)
        self.torre_destino = nn.Sequential(nn.Linear(d + 1, oculta), nn.ReLU(), nn.Linear(oculta, d))

    def vec_usuario(self, x):
        return self.torre_usuario(x)

    def vec_destino(self, lugar, pop):
        return self.torre_destino(torch.cat([self.emb_lugar(lugar), pop[:, None]], dim=1))


class DosTorres:
    """Misma interfaz que los demás modelos: ajustar(aprende, validacion) / puntuar(usuarios)."""

    nombre = "Red de dos torres"

    def __init__(self, d=4, perdida="bpr", negativos=1, usuario="preferencias", semilla=0,
                 epocas=200, lr=0.01, decaimiento=1e-4, lote=256, oculta=16, usuarios=None):
        self.d, self.perdida, self.negativos, self.tipo_usuario = d, perdida, negativos, usuario
        self.semilla, self.epocas, self.lr, self.decaimiento = semilla, epocas, lr, decaimiento
        self.lote, self.oculta = lote, oculta
        u = usuarios if usuarios is not None else pd.read_csv(PROCESADOS / "usuarios.csv")
        u = u.set_index("UserID").sort_index()
        self.X_usuario = torch.as_tensor(variables_usuario(u, usuario))  # fila UserID − 1
        lugar, pop = variables_destino()
        self.lugar, self.pop = torch.as_tensor(lugar), torch.as_tensor(pop)
        self.historia, self.mejor_epoca = [], None

    def parametros(self):
        return sum(p.numel() for p in self.red.parameters())

    def _puntajes_pares(self, u, i):
        return (self.red.vec_usuario(self.X_usuario[u]) * self.red.vec_destino(self.lugar[i], self.pop[i])).sum(1)

    def ajustar(self, aprende, validacion=None, paciencia=10, metrica="HitRate_lugar@3"):
        torch.manual_seed(self.semilla)
        rng = np.random.default_rng(self.semilla)
        self.red = RedDosTorres(self.X_usuario.shape[1], self.d, self.oculta)
        opt = torch.optim.Adam(self.red.parameters(), lr=self.lr, weight_decay=self.decaimiento)
        u_all = aprende.UserID.to_numpy() - 1
        i_all = aprende.DestinationID.to_numpy() - 1
        positivos = np.unique(u_all * N_DESTINOS + i_all)
        k = self.negativos

        if validacion is not None:  # pérdida de validación: negativos fijos, con su propio generador
            uv = validacion.UserID.to_numpy() - 1
            iv = validacion.DestinationID.to_numpy() - 1
            todos_pos = np.union1d(positivos, uv * N_DESTINOS + iv)
            jv = muestrear_negativos(np.repeat(uv, k), todos_pos, N_DESTINOS, np.random.default_rng(10_000 + self.semilla))
            self._val = tuple(torch.as_tensor(x) for x in (uv, iv, jv))

        mejor, sin_mejora, mejores_pesos = -1.0, 0, None
        for epoca in range(1, self.epocas + 1):
            self.red.train()
            orden = rng.permutation(len(u_all))
            total = 0.0
            for ini in range(0, len(orden), self.lote):
                idx = orden[ini:ini + self.lote]
                u_np, i_np = u_all[idx], i_all[idx]
                j_np = muestrear_negativos(np.repeat(u_np, k), positivos, N_DESTINOS, rng)  # k por positivo
                u, i, j = (torch.as_tensor(x) for x in (u_np, i_np, j_np))
                s_pos = self._puntajes_pares(u, i)
                s_neg = self._puntajes_pares(u.repeat_interleave(k), j)
                if self.perdida == "bce":
                    logits = torch.cat([s_pos, s_neg])
                    y = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
                    perdida = nn.functional.binary_cross_entropy_with_logits(logits, y)
                elif self.perdida == "bpr":
                    perdida = -nn.functional.logsigmoid(s_pos.repeat_interleave(k) - s_neg).mean()
                else:
                    raise ValueError(self.perdida)
                opt.zero_grad()
                perdida.backward()
                opt.step()
                total += perdida.item() * len(idx)
            fila = {"epoca": epoca, "perdida": total / len(orden)}
            if validacion is not None:
                fila["perdida_val"] = self._perdida_val()
                r = evaluar(self.puntuar, aprende, validacion, semilla=self.semilla, ks=(10,))[0].iloc[0]
                fila[metrica + "_val"] = r[metrica]
                if r[metrica] > mejor:
                    mejor, sin_mejora, self.mejor_epoca = r[metrica], 0, epoca
                    mejores_pesos = {n: p.clone() for n, p in self.red.state_dict().items()}
                else:
                    sin_mejora += 1
            self.historia.append(fila)
            if validacion is not None and sin_mejora >= paciencia:
                break
        if mejores_pesos is not None:
            self.red.load_state_dict(mejores_pesos)
        else:
            self.mejor_epoca = self.epocas
        return self

    @torch.no_grad()
    def _perdida_val(self):
        """Misma pérdida que en el entrenamiento (BCE o BPR), sobre los pares de validación."""
        self.red.eval()
        u, i, j = self._val
        k = self.negativos
        s_pos = self._puntajes_pares(u, i)
        s_neg = self._puntajes_pares(u.repeat_interleave(k), j)
        if self.perdida == "bce":
            logits = torch.cat([s_pos, s_neg])
            y = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
            return float(nn.functional.binary_cross_entropy_with_logits(logits, y))
        return float(-nn.functional.logsigmoid(s_pos.repeat_interleave(k) - s_neg).mean())

    @torch.no_grad()
    def puntuar_variables(self, x_usuario):
        """Puntajes (n × 1000) a partir de variables de usuario ya armadas: sirve para perfiles nuevos."""
        self.red.eval()
        U = self.red.vec_usuario(torch.as_tensor(x_usuario))
        V = self.red.vec_destino(self.lugar, self.pop)
        return (U @ V.T).numpy()

    def puntuar(self, usuarios):
        return self.puntuar_variables(self.X_usuario[torch.as_tensor(np.asarray(usuarios) - 1)])
