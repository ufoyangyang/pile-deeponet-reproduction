"""Single-condition overfit sanity check.

Trains the full PILE-DeepONet on ONE thermal-abuse condition and reports the
error on a dense (time x radius) grid.  This is the fastest way to verify that
the architecture, the sampler and the loss are wired correctly: if the model
cannot memorise a single trajectory, something is broken.  A healthy run
reaches a mean absolute error of a few kelvin within ~600 iterations.

Usage::

    python scripts/overfit.py --cond 0 --iters 600 --device cuda
    python scripts/overfit.py --ramp 2600 --h-cool 7 --iters 600
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from pile_deeponet.batch import WindowDataset
from pile_deeponet.config import Config
from pile_deeponet.data import load_dataset
from pile_deeponet.evaluate import predict_window
from pile_deeponet.train import Trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lfp_4680.npz")
    ap.add_argument("--cond", type=int, default=0, help="index inside the train split")
    ap.add_argument("--ramp", type=float, default=None, help="pick the condition by ramp (s)")
    ap.add_argument("--h-cool", type=float, default=None)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="runs/overfit")
    args = ap.parse_args()

    cfg = Config()
    cfg.train.iters = args.iters
    cfg.train.use_physics = False          # pure supervised memorisation test
    cfg.train.use_ntk = False
    cfg.train.log_every = 50
    cfg.train.eval_every = 0
    cfg.train.n_phy = 0

    ds = load_dataset(args.data)
    mask = ds.split == "train"
    if args.ramp is not None:
        mask = mask & np.isclose(ds.ramp, args.ramp) & np.isclose(ds.h_cool, args.h_cool)
        if not mask.any():
            raise SystemExit(f"no train condition matches ramp={args.ramp} h_cool={args.h_cool}")
    idx = np.where(mask)[0][args.cond]
    one = ds.subset(np.arange(len(ds)) == idx)
    wds = WindowDataset(one, cfg, device=args.device)

    print(f"[overfit] condition {idx}: ramp={float(one.ramp[0])}s h_cool={float(one.h_cool[0])} "
          f"peak={float(one.peak_T[0]):.1f}C t_d={float(one.t_d[0]):.0f}s")

    t0 = time.time()
    trainer = Trainer(cfg, wds, None, device=args.device, out_dir=args.out)
    trainer.fit()
    print(f"[overfit] trained in {time.time() - t0:.1f}s")

    res = predict_window(trainer.model, wds, 0, cfg, device=args.device)
    T, That = res["T_true"], res["T"]
    mae = float(np.abs(T - That).mean())
    rmse = float(np.sqrt(((T - That) ** 2).mean()))
    print(f"[overfit] dense-grid MAE = {mae:.2f} K, RMSE = {rmse:.2f} K "
          f"(pred std {That.std():.2f} K vs target std {T.std():.2f} K)")
    if That.std() < 1.0:
        print("[overfit] WARNING: prediction is (nearly) constant -> check sampler shapes "
              "(batch['T'] must be (N, 1), not (N, N, 1))")
    return mae


if __name__ == "__main__":
    main()
