"""Evaluate a trained PILE-DeepONet on the validation / test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pile_deeponet.batch import WindowDataset
from pile_deeponet.config import config_from_dict
from pile_deeponet.data import load_dataset
from pile_deeponet.evaluate import evaluate_split, predict_window
from pile_deeponet.model import PILEDeepONet
from pile_deeponet.plots import plot_case, plot_profiles, plot_saliency, plot_wape_bars

SPLITS = ["val", "test1", "test2", "test3"]


def load_checkpoint(path, device="cpu"):
    ck = torch.load(str(path), map_location=device, weights_only=False)
    cfg = config_from_dict(ck["cfg"])
    model = PILEDeepONet(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg, ck.get("iter", -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--data", default="data/lfp_4680.npz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="pile")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    model, cfg, it = load_checkpoint(args.ckpt, args.device)
    ds = load_dataset(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for sp in SPLITS:
        if not (ds.split == sp).any():
            continue
        wds = WindowDataset(ds.subset(ds.split == sp), cfg, device=args.device)
        print(f"[eval] {sp} (n={len(wds)})")
        summary[sp] = evaluate_split(model, wds, np.ones(len(wds), dtype=bool), cfg,
                                     args.device, verbose=True)

    summary["_meta"] = dict(ckpt=str(args.ckpt), iter=it,
                            device=args.device, tag=args.tag)
    path = out / f"metrics_{args.tag}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[eval] wrote {path}")

    if not args.no_plots:
        wds2 = WindowDataset(ds.subset(ds.split == "test2"), cfg, device=args.device)
        res = predict_window(model, wds2, 0, cfg, device=args.device)
        plot_case(res, out / f"fig_case_{args.tag}.png")
        plot_profiles(res, out / f"fig_profiles_{args.tag}.png")
        plot_wape_bars({k: v for k, v in summary.items() if not k.startswith("_")},
                       out / f"fig_wape_{args.tag}.png")
        try:
            plot_saliency(model, wds2, cfg, out / f"fig_saliency_{args.tag}.png", i=0)
        except Exception as exc:      # noqa: BLE001
            print(f"[warn] saliency plot skipped: {exc}")
        print(f"[eval] figures written to {out}")


if __name__ == "__main__":
    main()
