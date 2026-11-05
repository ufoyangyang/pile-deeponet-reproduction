"""Figures reproducing the main plots of the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 150})


def _save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_case(res, path, title="test-set 2 (experiment condition)"):
    """Temperature / reaction / gas / pressure history of one condition."""
    t = res["t"] - res["t"][0]
    fig, ax = plt.subplots(2, 3, figsize=(9, 4.6))

    a = ax[0, 0]
    a.plot(t, res["T_true"].mean(1), "k-", lw=1.4, label="reference")
    a.plot(t, res["T"].mean(1), "r--", lw=1.2, label="PILE-DeepONet")
    a.set_xlabel("time from prediction start [s]")
    a.set_ylabel("mean temperature [°C]")
    a.legend(frameon=False)
    a.set_title("(a) cell temperature")

    a = ax[0, 1]
    a.plot(t, res["T_true"][:, -1], "k-", lw=1.0, label="surface")
    a.plot(t, res["T"][:, -1], "r--", lw=1.0)
    a.plot(t, res["T_true"][:, 0], "b-", lw=1.0, label="centre")
    a.plot(t, res["T"][:, 0], "c--", lw=1.0)
    a.set_xlabel("time [s]")
    a.set_ylabel("temperature [°C]")
    a.legend(frameon=False)
    a.set_title("(b) surface / centre")

    a = ax[0, 2]
    a.plot(t, res["c_true"], "k-", lw=1.2)
    a.plot(t, res["c"].mean(1), "r--", lw=1.2)
    a.set_xlabel("time [s]")
    a.set_ylabel("normalised reactant c")
    a.set_title("(c) equivalent reactant")

    a = ax[1, 0]
    a.plot(t, res["gas_true"], "k-", lw=1.2)
    a.plot(t, res["gas"].mean(1), "r--", lw=1.2)
    a.set_xlabel("time [s]")
    a.set_ylabel("gas mass [g]")
    a.set_title("(d) gas generation")

    a = ax[1, 1]
    a.plot(t, res["p_true"] / 1e3, "k-", lw=1.2)
    a.plot(t, res["p"] / 1e3, "r--", lw=1.2)
    a.set_xlabel("time [s]")
    a.set_ylabel("pressure [kPa]")
    a.set_title("(e) internal pressure")

    a = ax[1, 2]
    tt = t / max(t[-1], 1)
    X, Y = np.meshgrid(res["t"] - res["t"][0], np.linspace(0, 1, res["T"].shape[1]))
    im = a.contourf(tt, Y[:, 0], res["T"].T, levels=20, cmap="inferno")
    a.set_xlabel("normalised time")
    a.set_ylabel("normalised radius")
    fig.colorbar(im, ax=a, label="°C")
    a.set_title("(f) predicted field")
    fig.suptitle(title, y=1.02, fontsize=9)
    return _save(fig, path)


def plot_wape_bars(summary: dict, path):
    """Grouped WAPE per variable for val / test1 / test2 / test3."""
    keys = ["wape_T", "wape_P", "wape_c", "wape_gas"]
    labels = ["T", "P", "c", "Gas"]
    splits = [k for k in summary]
    x = np.arange(len(keys))
    w = 0.8 / max(len(splits), 1)
    fig, a = plt.subplots(figsize=(5.2, 2.6))
    for i, s in enumerate(splits):
        vals = [summary[s][k] for k in keys]
        a.bar(x + i * w - 0.4, vals, w, label=s)
    a.set_xticks(x)
    a.set_xticklabels(labels)
    a.set_ylabel("WAPE [%]")
    a.legend(frameon=False, ncol=len(splits))
    return _save(fig, path)


def plot_profiles(res, path, times=(0, 300, 600, 900, 1200)):
    """Radial temperature profiles at several times (paper Fig. 8 style)."""
    fig, a = plt.subplots(figsize=(3.6, 2.8))
    r = np.linspace(0, 1, res["T"].shape[1])
    for tq in times:
        i = int(np.argmin(np.abs(res["t"] - res["t"][0] - tq)))
        a.plot(r, res["T_true"][i], "k-", lw=1.0)
        a.plot(r, res["T"][i], "r--", lw=1.0)
    a.set_xlabel("normalised radius")
    a.set_ylabel("temperature [°C]")
    a.set_title("radial profiles (black: reference)")
    return _save(fig, path)


def plot_saliency(model, wds, cfg, path, i=0, n_t=25):
    """Sensitivity of the surface temperature to the latent features (Eq. 35)."""
    m = cfg.model
    dev = wds.u1.device
    d = torch.linspace(0.0, 1.0, n_t, device=dev)
    x = torch.ones(n_t, 1, device=dev)
    u1 = wds.u1[i].expand(n_t, 1)
    u2 = wds.u2[i].expand(n_t, -1)
    u3 = wds.u3[i].expand(n_t, -1)
    # detach so that u4 is a leaf: the saliency is dT/dT_F, not dT/d(network weights)
    with torch.no_grad():
        u4 = model.u4(u1, u2, u3, x)
    u4 = u4.detach().requires_grad_(True)
    d = d.unsqueeze(-1)
    tau = torch.remainder(d, m.local_window / m.lead_time) / (m.local_window / m.lead_time)
    tT, _, _ = model.pred([u1, u2, u3, u4], model._trunk_inputs(x, d, tau))
    red = tT.sum(dim=1, keepdim=True) * model.gain if m.reduce == "sum" else tT.mean(dim=1, keepdim=True)
    T = m.T_min + (m.T_max - m.T_min) * torch.sigmoid(red + model.b_T)
    T = T.squeeze(-1)
    grads = []
    for j in range(u4.shape[1]):
        g = torch.autograd.grad(T.sum(), u4, retain_graph=True)[0]
        grads.append(g[:, j].abs())
    G = torch.stack(grads, 1).detach().cpu().numpy()
    fig, a = plt.subplots(figsize=(3.6, 2.4))
    for j in range(G.shape[1]):
        a.plot(d.squeeze(-1).detach().cpu().numpy() * m.lead_time, G[:, j], label=f"$d_{j}$")
    a.set_xlabel("time [s]")
    a.set_ylabel(r"$|\partial \hat T / \partial \hat T_F|$")
    a.legend(frameon=False, ncol=4, fontsize=6)
    return _save(fig, path)
