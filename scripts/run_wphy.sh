#!/usr/bin/env bash
# Sensitivity of the fit to the physics weight w_phy (short, parallel runs).
# Usage: bash scripts/run_wphy.sh [iters]
set -u
ITERS="${1:-800}"
export PYTHONPATH="F:\\deepseek_workspace\\ai_job\\pile-deeponet;F:\\deepseek_workspace\\ai_job\\_libs_cuda"
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs logs results/wphy

pids=()
for w in 0.0 0.01 0.1 1.0; do
  echo "[launch] w_phy=$w"
  python -X utf8 -c "
import json, sys
sys.argv = ['x']
from pile_deeponet.config import Config
from pile_deeponet.runner import train_variant
cfg = Config()
cfg.train.iters = $ITERS
cfg.train.w_phy = $w
tr, wds = train_variant(cfg, 'data/lfp_4680.npz', 'runs/wphy_$w', 'cuda')
json.dump({'w_phy': $w, 'iters': $ITERS, 'best_val_wape': tr.best,
           'history': tr.history}, open('results/wphy/w_$w.json', 'w'), indent=2)
print('DONE w_phy=$w best', tr.best)
" > "logs/wphy_$w.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "[fail] $p"; done
echo "[done] w_phy sweep"
grep -h "DONE" logs/wphy_*.log
