#!/usr/bin/env bash
# Late-window sampling with the physics term OFF: the clean test of whether
# concentrating collocation points on the runaway transient helps the data fit.
set -u
export PYTHONPATH="F:\deepseek_workspace\ai_job\pile-deeponet;F:\deepseek_workspace\ai_job\_libs_cuda"
cd "$(dirname "$0")/.." || exit 1
python -X utf8 -c "
from pile_deeponet.config import Config
from pile_deeponet.runner import train_variant
cfg = Config(); cfg.train.iters = 2500; cfg.train.use_physics = False; cfg.train.d_power = 3.0
tr, wds = train_variant(cfg, 'data/lfp_4680.npz', 'runs/pile_late', 'cuda')
print('DONE late best', tr.best)
" > logs/late.log 2>&1
tail -2 logs/late.log
