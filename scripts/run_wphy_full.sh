#!/usr/bin/env bash
# Physics-weight sweep at the full training budget (the physics term is ~2 orders
# of magnitude larger than the data term, so w_phy=0.1 (paper's implicit scale)
# over-weights it for our 1-D solver).  Usage: bash scripts/run_wphy_full.sh [iters]
set -u
ITERS="${1:-2500}"
export PYTHONPATH="F:\\deepseek_workspace\\ai_job\\pile-deeponet;F:\\deepseek_workspace\\ai_job\\_libs_cuda"
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs logs results

pids=()
for w in 0.003 0.01 0.03; do
  echo "[launch] w_phy=$w"
  python -X utf8 -c "
import json
from pile_deeponet.config import Config
from pile_deeponet.runner import train_variant
cfg = Config(); cfg.train.iters = $ITERS; cfg.train.w_phy = $w
tr, wds = train_variant(cfg, 'data/lfp_4680.npz', 'runs/pile_w$w', 'cuda')
json.dump({'w_phy': $w, 'iters': $ITERS, 'best_val_wape': tr.best, 'history': tr.history},
          open('results/wphy_w$w.json', 'w'), indent=2)
print('DONE w_phy=$w best', tr.best)
" > "logs/wphy_full_$w.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "[fail] $p"; done
echo "[done] sweep"
grep -h "DONE" logs/wphy_full_*.log
