#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
mask_path="${repo_root}/experiments/tdanet_lrs2_mask/mask.pt"
output_dir="${repo_root}/experiments/tdanet_lrs2_eval_full"

cd "${repo_root}"
mkdir -p logs "${output_dir}"

"${python_bin}" -u -m reproduction.evaluate_tdanet \
  --data-root data/LRS2-2Mix \
  --split tt \
  --device cuda \
  --output-dir "${output_dir}" \
  2>&1 | tee logs/tdanet_original_full_eval.log

"${python_bin}" -u -m reproduction.evaluate_tdanet \
  --data-root data/LRS2-2Mix \
  --split tt \
  --device cuda \
  --mask "${mask_path}" \
  --output-dir "${output_dir}" \
  2>&1 | tee logs/tdanet_pruned_full_eval.log

