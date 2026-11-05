"""Ablation / baseline study (paper Table 9).

Variants
--------
pile            full PILE-DeepONet (physics + latent ensemble + stan +
                separated trunk + NTK weights)
no_physics      LE-DeepONet : latent ensemble without physics
no_latent       PI-DeepONet : physics-informed without the latent ensemble
no_stan         tanh instead of the trainable-slope stan activation
no_separated    a single trunk for (x, d) instead of separated trunks
no_ntk          fixed unit loss weights
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pile_deeponet.config import Config
from pile_deeponet.runner import evaluate_model, save_summary, train_variant

VARIANTS = {
    "pile": {},
    "no_physics": dict(use_physics=False),
    "no_latent": dict(use_latent_operator=False),
    "no_stan": dict(stan=False),
    "no_separated": dict(separated_trunk=False),
    "no_ntk": dict(use_ntk=False),
}
TRAIN_KEYS = {"use_physics", "use_ntk"}
MODEL_KEYS = {"use_latent_operator", "stan", "separated_trunk"}


def make_config(over: dict, iters: int, seed: int) -> Config:
    cfg = Config()
    cfg.train.iters = iters
    cfg.train.seed = seed
    for k, v in over.items():
        if k in TRAIN_KEYS:
            setattr(cfg.train, k, v)
        elif k in MODEL_KEYS:
            setattr(cfg.model, k, v)
        else:
            raise KeyError(k)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/lfp_4680.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", default="runs/ablation")
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    path = out / "ablation.json"
    if path.exists():
        results = json.loads(path.read_text(encoding="utf-8"))

    for name in args.variants:
        print(f"\n=== {name} ===", flush=True)
        cfg = make_config(VARIANTS[name], args.iters, args.seed)
        trainer, wds_all = train_variant(cfg, args.data, Path(args.runs) / name, args.device)
        summary = evaluate_model(trainer.model, cfg, wds_all, device=args.device)
        results[name] = {
            "settings": VARIANTS[name],
            "iters": args.iters,
            "best_val_wape": trainer.best,
            "splits": summary,
            "wape_total_mean": float(np.mean([s["wape_total"] for s in summary.values()])),
        }
        save_summary(results, path)
        print(f"[{name}] mean WAPE total = {results[name]['wape_total_mean']:.2f}%")

    if results:
        _plot_ablation(results, out / "fig_ablation.png")
    print(f"[ablation] wrote {path}")


def _plot_ablation(results: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(results)
    vals = [results[n]["wape_total_mean"] for n in names]
    fig, a = plt.subplots(figsize=(4.2, 2.4))
    a.barh(names, vals, color="#c0392b")
    a.set_xlabel("mean total WAPE [%]")
    for i, v in enumerate(vals):
        a.text(v, i, f" {v:.2f}%", va="center", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
