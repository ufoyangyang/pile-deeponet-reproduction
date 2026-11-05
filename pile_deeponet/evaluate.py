"""Prediction on the (time x radius) window grid and the metrics of the paper
(Eq. 31-34)."""

from __future__ import annotations

import numpy as np
import torch

from .batch import WindowDataset
from .config import Config


@torch.no_grad()
def predict_window(model, wds: WindowDataset, i, cfg: Config, n_t=None, device="cpu"):
    """Predicted fields of condition `i` on the (n_t, nr) window grid."""
    m, p = cfg.model, cfg.physics
    n_t = n_t or int(round(m.lead_time / wds.dt)) + 1
    t, T_true, c_true, gas_true, p_true = wds.window_grid(i, n_t)
    n_t = T_true.shape[0]
    nr = wds.nr

    d = torch.linspace(0.0, 1.0, n_t, device=device)
    x = torch.linspace(0.0, 1.0, nr, device=device)
    D, X = torch.meshgrid(d, x, indexing="ij")
    X = X.reshape(-1, 1)
    D = D.reshape(-1, 1)
    N = X.shape[0]

    u1 = wds.u1[i].expand(N, 1)
    u2 = wds._norm_temp(wds.u2[i]).expand(N, -1)
    u3 = wds._norm_temp(wds.u3[i]).expand(N, -1)

    Ts, cs, gs = [], [], []
    bs = 8192
    for s in range(0, N, bs):
        sl = slice(s, min(s + bs, N))
        Tb, cb, gb = model(u1[sl], u2[sl], u3[sl], X[sl], D[sl])
        Ts.append(Tb)
        cs.append(cb)
        gs.append(gb)
    T = torch.cat(Ts).reshape(n_t, nr)
    c = torch.cat(cs).reshape(n_t, nr)
    gas = torch.cat(gs).reshape(n_t, nr)

    T_mean = T.mean(dim=1)
    p_pred = gas.mean(dim=1) * p.Rg * (T_mean + 273.15) / p.free_volume
    return dict(t=t.cpu().numpy(), T=T.cpu().numpy(), c=c.cpu().numpy(),
                gas=gas.cpu().numpy(), p=p_pred.cpu().numpy(),
                T_true=T_true.cpu().numpy(), c_true=c_true.cpu().numpy(),
                gas_true=gas_true.cpu().numpy(), p_true=p_true.cpu().numpy())


def wape(y, yhat) -> float:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    den = np.abs(y).sum()
    return float(100.0 * np.abs(y - yhat).sum() / den) if den > 0 else float("nan")


def arep(y, yhat) -> float:
    return float(100.0 * abs(y - yhat) / abs(y))


def metrics_window(res) -> dict:
    """Per-condition metrics; the total WAPE is the sum of the four variables."""
    T, Tt = res["T"], res["T_true"]
    c, ct = res["c"].mean(axis=1), res["c_true"]
    g, gt = res["gas"].mean(axis=1), res["gas_true"]
    pp, pt = res["p"], res["p_true"]

    m = dict(
        wape_T=wape(Tt, T),
        wape_P=wape(pt, pp),
        wape_c=wape(ct, c),
        wape_gas=wape(gt, g),
    )
    m["wape_total"] = m["wape_T"] + m["wape_P"] + m["wape_c"] + m["wape_gas"]

    Tmean, Ttmean = T.mean(axis=1), Tt.mean(axis=1)
    ip, it = int(np.argmax(Tmean)), int(np.argmax(Ttmean))
    m["peak_T"] = float(Tmean[ip])
    m["peak_T_true"] = float(Ttmean[it])
    m["arep_T"] = arep(Ttmean[it], Tmean[ip])
    m["peak_t"] = float(res["t"][ip])
    m["peak_t_true"] = float(res["t"][it])
    m["arep_t"] = arep(res["t"][it], res["t"][ip])

    m["rmse_surface"] = float(np.sqrt(np.mean((T[:, -1] - Tt[:, -1]) ** 2)))
    m["rmse_center"] = float(np.sqrt(np.mean((T[:, 0] - Tt[:, 0]) ** 2)))
    m["mae_surface"] = float(np.mean(np.abs(T[:, -1] - Tt[:, -1])))
    m["mae_center"] = float(np.mean(np.abs(T[:, 0] - Tt[:, 0])))
    m["rmse_T"] = float(np.sqrt(np.mean((T - Tt) ** 2)))
    return m


def evaluate_split(model, wds: WindowDataset, mask, cfg: Config, device="cpu", verbose=False):
    idx = np.arange(len(wds)) if mask is None else np.where(mask)[0]
    rows = []
    for i in idx:
        res = predict_window(model, wds, int(i), cfg, device=device)
        rows.append(metrics_window(res))
    keys = [k for k in rows[0]]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = float(np.nanmean(vals))
        out[k + "_std"] = float(np.nanstd(vals))
    out["n"] = len(rows)
    if verbose:
        print(f"  n={out['n']}  WAPE total={out['wape_total']:.2f}%  "
              f"(T {out['wape_T']:.2f} P {out['wape_P']:.2f} "
              f"c {out['wape_c']:.2f} g {out['wape_gas']:.2f})  "
              f"AREP_T={out['arep_T']:.2f}%")
    return out
