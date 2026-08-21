#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
train_dir="experiments/original_sudormrf_lrs2_train"
eval_dir="experiments/original_baselines_lrs2"

if [[ ! -f "${train_dir}/last.pt" ]]; then
  echo "Missing resume checkpoint: ${train_dir}/last.pt" >&2
  exit 1
fi

"${python_bin}" -u -m reproduction.train_original \
  --model sudormrf \
  --data-root data/LRS2-2Mix \
  --epochs 500 \
  --resume "${train_dir}/last.pt" \
  --output-dir "${train_dir}" \
  --num-workers 4

"${python_bin}" -u -m reproduction.evaluate_original \
  --model sudormrf \
  --checkpoint "${train_dir}/best.pt" \
  --data-root data/LRS2-2Mix \
  --output-dir "${eval_dir}"
