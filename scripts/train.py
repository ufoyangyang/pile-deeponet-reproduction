"""Train PILE-DeepONet (or one of its ablations)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pile_deeponet.config import Config, load_config
from pile_deeponet.runner import train_variant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lfp_4680.npz")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--iters", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # ablations ----------------------------------------------------------
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--no-latent", action="store_true")
    ap.add_argument("--no-stan", action="store_true")
    ap.add_argument("--no-separated-trunk", action="store_true")
    ap.add_argument("--no-ntk", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else Config()
    if args.iters:
        cfg.train.iters = args.iters
    if args.no_physics:
        cfg.train.use_physics = False
    if args.no_latent:
        cfg.model.use_latent_operator = False
    if args.no_stan:
        cfg.model.stan = False
    if args.no_separated_trunk:
        cfg.model.separated_trunk = False
    if args.no_ntk:
        cfg.train.use_ntk = False
    if args.seed is not None:
        cfg.train.seed = args.seed

    print(f"[cfg] device={args.device} iters={cfg.train.iters} "
          f"physics={cfg.train.use_physics} latent={cfg.model.use_latent_operator} "
          f"stan={cfg.model.stan} ntk={cfg.train.use_ntk}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.to_json(out / "config.json")

    trainer, _ = train_variant(cfg, args.data, out, args.device)
    with open(out / "train_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"iters": cfg.train.iters, "best_val_wape": trainer.best,
                   "history": trainer.history}, fh, indent=2)


if __name__ == "__main__":
    main()
