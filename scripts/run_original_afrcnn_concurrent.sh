#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
output_dir="experiments/original_afrcnn12_lrs2_train"
mkdir -p "${output_dir}" logs

# Keep the shell PID alive for the sequential launcher to wait on. The marker
# intentionally remains after exit; a dead PID is treated as a resumable run.
printf '%s\n' "$$" > "${output_dir}/concurrent.pid"
resume_args=()
if [[ -f "${output_dir}/last.pt" ]]; then
  resume_args=(--resume "${output_dir}/last.pt")
fi

"${python_bin}" -u -m reproduction.train_original \
  --model afrcnn12 \
  --data-root data/LRS2-2Mix \
  --output-dir "${output_dir}" \
  --num-workers 4 \
  "${resume_args[@]}"
