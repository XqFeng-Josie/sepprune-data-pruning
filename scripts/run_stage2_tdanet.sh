#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
mask_path="${repo_root}/experiments/tdanet_lrs2_mask/mask.pt"
inherited_dir="${repo_root}/experiments/tdanet_lrs2_inherited_1epoch"
random_dir="${repo_root}/experiments/tdanet_lrs2_random_1epoch"
eval_dir="${repo_root}/experiments/tdanet_lrs2_stage2_eval"

cd "${repo_root}"
mkdir -p logs "${inherited_dir}" "${random_dir}" "${eval_dir}"

if [[ ! -f "${inherited_dir}/final.pt" ]]; then
  inherited_resume=()
  if [[ -f "${inherited_dir}/last.pt" ]]; then
    inherited_resume=(--resume "${inherited_dir}/last.pt")
  fi
  "${python_bin}" -u -m reproduction.finetune_tdanet \
    --data-root data/LRS2-2Mix \
    --mask "${mask_path}" \
    --initialization inherited \
    --device cuda \
    --epochs 1 \
    --checkpoint-every 1000 \
    --log-every 100 \
    --output-dir "${inherited_dir}" \
    "${inherited_resume[@]}" \
    2>&1 | tee logs/tdanet_inherited_1epoch.log
fi

if [[ ! -f "${eval_dir}/pruned_inherited_1epoch_tt_summary.json" ]]; then
  "${python_bin}" -u -m reproduction.evaluate_tdanet \
    --data-root data/LRS2-2Mix \
    --split tt \
    --device cuda \
    --mask "${mask_path}" \
    --trained-state "${inherited_dir}/final.pt" \
    --output-dir "${eval_dir}" \
    2>&1 | tee logs/tdanet_inherited_1epoch_eval.log
fi

if [[ ! -f "${random_dir}/final.pt" ]]; then
  random_resume=()
  if [[ -f "${random_dir}/last.pt" ]]; then
    random_resume=(--resume "${random_dir}/last.pt")
  fi
  "${python_bin}" -u -m reproduction.finetune_tdanet \
    --data-root data/LRS2-2Mix \
    --mask "${mask_path}" \
    --initialization random \
    --device cuda \
    --epochs 1 \
    --checkpoint-every 1000 \
    --log-every 100 \
    --output-dir "${random_dir}" \
    "${random_resume[@]}" \
    2>&1 | tee logs/tdanet_random_1epoch.log
fi

if [[ ! -f "${eval_dir}/pruned_random_1epoch_tt_summary.json" ]]; then
  "${python_bin}" -u -m reproduction.evaluate_tdanet \
    --data-root data/LRS2-2Mix \
    --split tt \
    --device cuda \
    --mask "${mask_path}" \
    --trained-state "${random_dir}/final.pt" \
    --output-dir "${eval_dir}" \
    2>&1 | tee logs/tdanet_random_1epoch_eval.log
fi

