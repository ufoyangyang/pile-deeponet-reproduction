"""Generate the synthetic thermal-runaway dataset (paper Table 2)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from pile_deeponet.config import Config
from pile_deeponet.data import build_conditions, generate_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/lfp_4680.npz")
    ap.add_argument("--config", default=None)
    ap.add_argument("--save-dt", type=float, default=None)
    ap.add_argument("--stats", action="store_true", help="print dataset statistics and exit")
    args = ap.parse_args()

    cfg = Config()
    if args.config:
        from pile_deeponet.config import load_config
        cfg = load_config(args.config)
    if args.save_dt:
        cfg.physics.save_dt = args.save_dt

    if args.stats:
        _stats(Path(args.out))
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    generate_dataset(cfg, out)
    cfg.to_json(out.with_suffix(".json"))
    print(f"[make_dataset] {time.time() - t0:.1f}s, conditions: {len(build_conditions(cfg.split))}")


def _stats(path: Path):
    """Summarise an existing dataset without regenerating it."""
    import numpy as np
    from pile_deeponet.data import load_dataset

    ds = load_dataset(path)
    print(f"file        {path}  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"conditions  {len(ds)}   t: {len(ds.t)} steps x {ds.T.shape[2]} radial nodes")
    print(f"t range     {ds.t[0]:.0f} .. {ds.t[-1]:.0f} s (dt={ds.t[1] - ds.t[0]:.0f}s)")
    for sp in ("train", "val", "test1", "test2", "test3"):
        m = ds.split == sp
        if not m.any():
            continue
        print(f"  {sp:6s} n={int(m.sum()):3d}  ramp {ds.ramp[m].min():.0f}-{ds.ramp[m].max():.0f}s  "
              f"h_cool {[int(v) for v in sorted(set(ds.h_cool[m]))]}  "
              f"peak_T {ds.peak_T[m].mean():.1f}+-{ds.peak_T[m].std():.1f} C")
    print(f"peak_T      mean {ds.peak_T.mean():.1f} C   Pmax mean "
          f"{ds.p.max(axis=1).mean() / 1e3:.1f} kPa")
    print(f"t_d         mean {ds.t_d.mean():.0f} s  (prediction horizon "
          f"{ds.t_d.min():.0f}-{ds.t_d.max():.0f} s)")
    print(f"u2 {ds.u2.shape}   u3 {ds.u3.shape}   u1 range "
          f"{ds.u1.min():.3f}-{ds.u1.max():.3f}")


if __name__ == "__main__":
    main()
