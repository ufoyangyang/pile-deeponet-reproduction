"""Synthetic dataset construction (paper Section 3.1 / Table 2).

The 124 thermal-abuse conditions of the paper (84 train / 10 validation /
11 in-distribution test / 1 experimental / 18 out-of-distribution) are
generated with the in-house 1-D radial solver of `physics.py` and stored as a
single compressed `.npz` file.

For every condition the dataset also provides the operator inputs of the paper:

    u1  cooling performance h_eq                       (1 value)
    u2  historical surface-temperature trend           (v = 31 samples, 30 s)
        sampled before the surface reaches 120 degC
    u3  short-term surface-temperature trend           (m = 11 samples, 30 s)
        sampled before the prediction start t_d

and the prediction window [t_d, t_d + lead_time] with lead_time = 1200 s.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config, SplitConfig
from .physics import simulate


# --------------------------------------------------------------------------- #
# conditions
# --------------------------------------------------------------------------- #
def build_conditions(sc: SplitConfig):
    """Return a list of (split, t_ramp, h_cool)."""
    pool = [(float(r), float(h)) for r in sc.ramps_train for h in sc.h_cool]
    rng = np.random.default_rng(sc.seed)
    perm = rng.permutation(len(pool))
    train = [pool[i] for i in perm[: sc.n_train]]
    val = [pool[i] for i in perm[sc.n_train: sc.n_train + sc.n_val]]
    test1 = [pool[i] for i in perm[sc.n_train + sc.n_val:]]
    test2 = [sc.exp_condition]
    test3 = [(float(r), float(h)) for r in sc.ramps_ood for h in sc.h_ood]

    out = []
    for name, conds in (("train", train), ("val", val),
                        ("test1", test1), ("test2", test2), ("test3", test3)):
        for (ramp, h) in conds:
            out.append((name, ramp, h))
    return out


def _interp_surface(surf: np.ndarray, t_grid: np.ndarray, t_query: np.ndarray):
    return np.interp(t_query, t_grid, surf)


def _window_features(t, T, h_cool, mcfg, save_dt):
    """u2 / u3 histories and the prediction start of one condition."""
    surf = T[:, -1]
    t_thr = float(t[np.argmax(surf >= mcfg.t_threshold)]) if (surf >= mcfg.t_threshold).any() else float(t[-1])

    n2 = mcfg.history_len
    t2 = t_thr - mcfg.sample_dt * np.arange(n2 - 1, -1, -1)          # ascending
    u2 = _interp_surface(surf, t, t2)
    u2 = np.where(t2 < 0, T[0, -1], u2)

    peak_i = int(np.argmax(T.mean(axis=1)))
    t_peak = float(t[peak_i])
    t_d = t_peak - mcfg.lead_time
    t_d = max(t_d, float(save_dt) * (mcfg.short_len + 2))
    t_d = float(np.floor(t_d / save_dt) * save_dt)

    n3 = mcfg.short_len
    t3 = t_d - mcfg.sample_dt * np.arange(n3 - 1, -1, -1)
    u3 = _interp_surface(surf, t, t3)

    return dict(t_thr=t_thr, t_d=t_d, u2=u2, u3=u3, h_cool=float(h_cool),
                t_peak=t_peak, peak_T=float(T.mean(axis=1)[peak_i]))


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def generate_dataset(cfg: Config, out_path, verbose: bool = True):
    sc, mcfg, pcfg = cfg.split, cfg.model, cfg.physics
    conds = build_conditions(sc)
    ramps = np.array([c[1] for c in conds])
    hcool = np.array([c[2] for c in conds])

    if verbose:
        print(f"[data] simulating {len(conds)} conditions ...")
    sim = simulate(pcfg, hcool, ramps, progress=verbose)

    T, c, gas, p = sim["T"], sim["c"], sim["gas"], sim["p"]
    t = sim["t"].astype(np.float64)

    u1 = (hcool / 20.0).astype(np.float32)
    u2 = np.zeros((len(conds), mcfg.history_len), dtype=np.float32)
    u3 = np.zeros((len(conds), mcfg.short_len), dtype=np.float32)
    t_d = np.zeros(len(conds), dtype=np.float32)
    t_thr = np.zeros(len(conds), dtype=np.float32)
    t_peak = np.zeros(len(conds), dtype=np.float32)
    peak_T = np.zeros(len(conds), dtype=np.float32)

    for i in range(len(conds)):
        f = _window_features(t, T[i].astype(np.float64), hcool[i], mcfg, pcfg.save_dt)
        u2[i] = f["u2"]
        u3[i] = f["u3"]
        t_d[i] = f["t_d"]
        t_thr[i] = f["t_thr"]
        t_peak[i] = f["t_peak"]
        peak_T[i] = f["peak_T"]

    split = np.array([c[0] for c in conds])
    np.savez_compressed(
        out_path,
        t=t.astype(np.float32), T=T, c=c, gas=gas, p=p,
        u1=u1, u2=u2, u3=u3, t_d=t_d, t_thr=t_thr, t_peak=t_peak, peak_T=peak_T,
        ramp=ramps.astype(np.float32), h_cool=hcool.astype(np.float32), split=split,
        x=(np.linspace(0, 1, pcfg.n_r) * pcfg.radius).astype(np.float32),
    )
    if verbose:
        print(f"[data] wrote {out_path}  T={T.shape}  mean peak={peak_T.mean():.1f} degC")
    return out_path


# --------------------------------------------------------------------------- #
# torch-side view
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    t: "np.ndarray"
    T: "np.ndarray"
    c: "np.ndarray"
    gas: "np.ndarray"
    p: "np.ndarray"
    u1: "np.ndarray"
    u2: "np.ndarray"
    u3: "np.ndarray"
    t_d: "np.ndarray"
    t_thr: "np.ndarray"
    t_peak: "np.ndarray"
    peak_T: "np.ndarray"
    ramp: "np.ndarray"
    h_cool: "np.ndarray"
    split: "np.ndarray"
    x: "np.ndarray"

    def subset(self, mask) -> "Dataset":
        idx = np.where(mask)[0]
        n = self.T.shape[0]
        fields = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            fields[k] = v[idx] if v.shape[0] == n else v
        return Dataset(**fields)

    def __len__(self):
        return self.T.shape[0]


def load_dataset(path) -> Dataset:
    z = np.load(str(path), allow_pickle=False)
    return Dataset(**{k: z[k] for k in Dataset.__dataclass_fields__})
