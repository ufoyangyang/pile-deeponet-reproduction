"""Predict one thermal-abuse case from a checkpoint and print a compact report.

Example::

    python scripts/demo_case.py --ckpt runs/pile/best.pt --ramp 2600 --h-cool 7
    python scripts/demo_case.py --ckpt runs/pile/best.pt --split test3 --index 2

The script prints the paper's AREP/WAPE numbers for that single trajectory and
(optionally) writes the six-panel figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pile_deeponet.batch import WindowDataset
from pile_deeponet.config import config_from_dict
from pile_deeponet.data import load_dataset
from pile_deeponet.evaluate import metrics_window, predict_window
from pile_deeponet.model import PILEDeepONet


def load_checkpoint(path, device="cpu"):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    cfg = config_from_dict(ck["cfg"])
    model = PILEDeepONet(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("iter", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/pile/best.pt")
    ap.add_argument("--data", default="data/lfp_4680.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="test2", choices=["train", "val", "test1", "test2", "test3"])
    ap.add_argument("--index", type=int, default=0, help="index inside the chosen split")
    ap.add_argument("--ramp", type=float, default=None, help="select by ramp-up time instead")
    ap.add_argument("--h-cool", type=float, default=None)
    ap.add_argument("--out", default="results", help="folder for the figure ('' to skip)")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    model, cfg, it = load_checkpoint(args.ckpt, args.device)
    ds = load_dataset(args.data)

    if args.ramp is not None:
        mask = np.isclose(ds.ramp, args.ramp)
        if args.h_cool is not None:
            mask &= np.isclose(ds.h_cool, args.h_cool)
        if not mask.any():
            raise SystemExit(f"no condition with ramp={args.ramp} h_cool={args.h_cool}")
        idx = int(np.where(mask)[0][0])
    else:
        mask = ds.split == args.split
        if not mask.any():
            raise SystemExit(f"split {args.split} is empty")
        idx = int(np.where(mask)[0][args.index])

    wds = WindowDataset(ds, cfg, device=args.device)
    res = predict_window(model, wds, idx, cfg, device=args.device)
    m = metrics_window(res)

    print(f"checkpoint  {args.ckpt}  (iter {it})")
    print(f"condition   #{idx}  split={ds.split[idx]}  ramp={float(ds.ramp[idx]):.0f}s  "
          f"h_cool={float(ds.h_cool[idx]):.0f}  peak_true={float(ds.peak_T[idx]):.1f}C")
    print("-" * 62)
    print(f"  WAPE   T {m['wape_T']:6.2f}%   P {m['wape_P']:6.2f}%   "
          f"c {m['wape_c']:6.2f}%   gas {m['wape_gas']:6.2f}%   total {m['wape_total']:6.2f}%")
    print(f"  peak   {m['peak_T']:.2f}C @ {m['peak_t']:.0f}s  vs true "
          f"{m['peak_T_true']:.2f}C @ {m['peak_t_true']:.0f}s   "
          f"(AREP_T {m['arep_T']:.2f}%, AREP_t {m['arep_t']:.2f}%)")
    print(f"  RMSE   surface {m['rmse_surface']:.2f}K  centre {m['rmse_center']:.2f}K")
    print(f"  MAE    surface {m['mae_surface']:.2f}K  centre {m['mae_center']:.2f}K")

    t = res["t"]
    picks = np.linspace(0, len(t) - 1, 8).astype(int)
    print("\n  t[s]   x=0 (centre)        x=1 (surface)        P[kPa]")
    print("         pred  true           pred  true           pred  true")
    for k in picks:
        print(f"  {t[k]:5.0f}  {res['T'][k, 0]:6.1f} {res['T_true'][k, 0]:6.1f}      "
              f"{res['T'][k, -1]:6.1f} {res['T_true'][k, -1]:6.1f}      "
              f"{res['p'][k]:6.1f} {res['p_true'][k]:6.1f}")

    if args.out:
        from pile_deeponet.plots import plot_case
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        tag = args.tag or f"{ds.split[idx]}{idx}"
        path = out / f"fig_case_{tag}.png"
        plot_case(res, path)
        print(f"\nfigure -> {path}")


if __name__ == "__main__":
    main()
