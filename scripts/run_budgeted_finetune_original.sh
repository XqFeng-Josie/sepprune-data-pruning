#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {afrcnn12|sudormrf}" >&2
  exit 2
fi

model="$1"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"

case "${model}" in
  afrcnn12)
    baseline="experiments/original_afrcnn12_lrs2_train/best.pt"
    masks="experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026/masks.pt"
    train_dir="experiments/seprune_budgeted_afrcnn12_lrs2_finetune"
    eval_dir="experiments/seprune_budgeted_afrcnn12_lrs2_eval"
    ;;
  sudormrf)
    baseline="experiments/original_sudormrf_lrs2_train/best.pt"
    masks="experiments/seprune_budgeted_sudormrf_lrs2_e07_seed2026/masks.pt"
    train_dir="experiments/seprune_budgeted_sudormrf_lrs2_finetune"
    eval_dir="experiments/seprune_budgeted_sudormrf_lrs2_eval"
    ;;
  *)
    echo "unsupported model: ${model}" >&2
    exit 2
    ;;
esac

mkdir -p "${train_dir}" "${eval_dir}"

if [[ ! -f "${train_dir}/epoch1.pt" ]]; then
  "${python_bin}" -u -m reproduction.train_pruned_original \
    --model "${model}" \
    --baseline-checkpoint "${baseline}" \
    --masks "${masks}" \
    --data-root data/LRS2-2Mix \
    --device cuda \
    --epochs 1 \
    --num-workers 4 \
    --output-dir "${train_dir}"
  cp "${train_dir}/best.pt" "${train_dir}/epoch1.pt"
  cp "${train_dir}/result.json" "${train_dir}/epoch1_result.json"
  cp "${train_dir}/config.json" "${train_dir}/epoch1_config.json"
fi

if [[ ! -f "${eval_dir}/${model}_budgeted_1epoch_tt_summary.json" ]]; then
  "${python_bin}" -u -m reproduction.evaluate_pruned_original \
    --model "${model}" \
    --baseline-checkpoint "${baseline}" \
    --masks "${masks}" \
    --checkpoint "${train_dir}/epoch1.pt" \
    --data-root data/LRS2-2Mix \
    --split tt \
    --device cuda \
    --label budgeted_1epoch \
    --output-dir "${eval_dir}"
fi

if [[ ! -f "${train_dir}/converged_result.json" ]]; then
  "${python_bin}" -u -m reproduction.train_pruned_original \
    --model "${model}" \
    --baseline-checkpoint "${baseline}" \
    --masks "${masks}" \
    --data-root data/LRS2-2Mix \
    --device cuda \
    --epochs 500 \
    --resume "${train_dir}/last.pt" \
    --num-workers 4 \
    --output-dir "${train_dir}"
  cp "${train_dir}/result.json" "${train_dir}/converged_result.json"
  cp "${train_dir}/config.json" "${train_dir}/converged_config.json"
fi

if [[ ! -f "${eval_dir}/${model}_budgeted_final_tt_summary.json" ]]; then
  "${python_bin}" -u -m reproduction.evaluate_pruned_original \
    --model "${model}" \
    --baseline-checkpoint "${baseline}" \
    --masks "${masks}" \
    --checkpoint "${train_dir}/best.pt" \
    --data-root data/LRS2-2Mix \
    --split tt \
    --device cuda \
    --label budgeted_final \
    --output-dir "${eval_dir}"
fi

