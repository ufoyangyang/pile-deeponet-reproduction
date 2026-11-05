#!/usr/bin/env bash
# Launch the full model and all ablations at an identical budget, in parallel.
# Usage: bash scripts/run_all.sh [iters]
set -u
ITERS="${1:-2500}"
export PYTHONPATH="F:\\deepseek_workspace\\ai_job\\pile-deeponet;F:\\deepseek_workspace\\ai_job\\_libs_cuda"
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs logs results

pids=()
launch() {  # name, command...
  local name="$1"; shift
  echo "[launch] $name"
  python -X utf8 "$@" > "logs/${name}.log" 2>&1 &
  pids+=($!)
}

launch pile         scripts/train.py --iters "$ITERS" --out runs/pile
for v in no_physics no_latent no_stan no_separated no_ntk; do
  launch "$v" scripts/ablation.py --iters "$ITERS" --variants "$v" \
      --out "results/abl_$v" --runs "runs/abl_$v"
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "[fail] job ${pids[$i]} exited non-zero"
    fail=1
  fi
done
echo "[done] all jobs finished (fail=$fail)"
for v in pile no_physics no_latent no_stan no_separated no_ntk; do
  echo "--- $v ---"
  tail -2 "logs/${v}.log" 2>/dev/null
done
