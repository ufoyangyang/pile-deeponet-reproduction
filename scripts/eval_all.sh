#!/usr/bin/env bash
# Evaluate every trained checkpoint through the same code path (scripts/evaluate.py)
# so that the numbers in docs/04 are mutually comparable.
set -u
export PYTHONPATH="F:\deepseek_workspace\ai_job\pile-deeponet;F:\deepseek_workspace\ai_job\_libs_cuda"
cd "$(dirname "$0")/.." || exit 1

eval_one() {  # tag, ckpt
  local tag="$1" ckpt="$2"
  if [ ! -f "$ckpt" ]; then echo "[skip] $tag ($ckpt missing)"; return; fi
  echo "[eval] $tag <- $ckpt"
  python -X utf8 scripts/evaluate.py --ckpt "$ckpt" --out results --tag "$tag" --no-plots > "logs/eval_$tag.log" 2>&1 \
    && tail -1 "logs/eval_$tag.log" || echo "[fail] $tag"
}

eval_one no_physics   runs/abl_no_physics/no_physics/best.pt
eval_one no_latent    runs/abl_no_latent/no_latent/best.pt
eval_one no_stan      runs/abl_no_stan/no_stan/best.pt
eval_one no_separated runs/abl_no_separated/no_separated/best.pt
eval_one no_ntk       runs/abl_no_ntk/no_ntk/best.pt
eval_one pile         runs/pile/best.pt
eval_one pile_final   runs/pile_final/pile_final/best.pt
for w in 0.003 0.01 0.03; do
  eval_one "wphy_$w" "runs/pile_w$w/pile_w$w/best.pt"
done
echo "[done]"
