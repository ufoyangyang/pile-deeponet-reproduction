#!/usr/bin/env bash
# Final model: late-window sampling (d_power) + the best physics weight from the sweep.
set -u
export PYTHONPATH="F:\deepseek_workspace\ai_job\pile-deeponet;F:\deepseek_workspace\ai_job\_libs_cuda"
cd "$(dirname "$0")/.." || exit 1
python -X utf8 -c "
import json
from pile_deeponet.config import Config
from pile_deeponet.runner import train_variant
cfg = Config(); cfg.train.iters = 2500; cfg.train.w_phy = 0.01; cfg.train.d_power = 3.0
tr, wds = train_variant(cfg, 'data/lfp_4680.npz', 'runs/pile_final', 'cuda')
print('DONE final best', tr.best)
" > logs/final.log 2>&1
tail -3 logs/final.log
