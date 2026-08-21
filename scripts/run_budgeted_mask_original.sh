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
    checkpoint="experiments/original_afrcnn12_lrs2_train/best.pt"
    output_dir="experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026"
    ;;
  sudormrf)
    checkpoint="experiments/original_sudormrf_lrs2_train/best.pt"
    output_dir="experiments/seprune_budgeted_sudormrf_lrs2_e07_seed2026"
    ;;
  *)
    echo "unsupported model: ${model}" >&2
    exit 2
    ;;
esac

"${python_bin}" -u -m reproduction.train_budgeted_mask_original \
  --model "${model}" \
  --checkpoint "${checkpoint}" \
  --data-root data/LRS2-2Mix \
  --device cuda \
  --iterations 500 \
  --temperature 1.0 \
  --learning-rate 0.1 \
  --num-workers 4 \
  --output-dir "${output_dir}"

