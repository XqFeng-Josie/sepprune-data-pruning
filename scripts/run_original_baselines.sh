#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-.venv/bin/python}"
mkdir -p logs experiments/original_baselines_lrs2

if [[ ! -f experiments/original_baselines_lrs2/tdanet_tt_summary.json ]]; then
  "${python_bin}" -u -m reproduction.evaluate_original \
    --model tdanet \
    --data-root data/LRS2-2Mix \
    --output-dir experiments/original_baselines_lrs2 \
    2>&1 | tee logs/original_tdanet_eval.log
fi

sudo_dir="experiments/original_sudormrf_lrs2_train"
if [[ ! -f "${sudo_dir}/result.json" ]] || ! grep -q '"status": "completed"' "${sudo_dir}/result.json"; then
  resume_args=()
  if [[ -f "${sudo_dir}/last.pt" ]]; then
    resume_args=(--resume "${sudo_dir}/last.pt")
  fi
  "${python_bin}" -u -m reproduction.train_original \
    --model sudormrf \
    --data-root data/LRS2-2Mix \
    --output-dir "${sudo_dir}" \
    --num-workers 4 \
    "${resume_args[@]}" \
    2>&1 | tee -a logs/original_sudormrf_train.log
fi
"${python_bin}" -u -m reproduction.evaluate_original \
  --model sudormrf \
  --checkpoint "${sudo_dir}/best.pt" \
  --data-root data/LRS2-2Mix \
  --output-dir experiments/original_baselines_lrs2 \
  2>&1 | tee logs/original_sudormrf_eval.log

afrcnn_dir="experiments/original_afrcnn12_lrs2_train"
# A-FRCNN may be launched concurrently while SuDoRM-RF is still training.
# If so, wait for that owner process instead of starting a duplicate writer in
# the same experiment directory. A stale PID is harmless: kill -0 fails and
# the normal resume path below takes over from last.pt.
afrcnn_owner_file="${afrcnn_dir}/concurrent.pid"
if [[ -f "${afrcnn_owner_file}" ]]; then
  afrcnn_owner_pid="$(<"${afrcnn_owner_file}")"
  while [[ ! -f "${afrcnn_dir}/result.json" ]] && kill -0 "${afrcnn_owner_pid}" 2>/dev/null; do
    sleep 60
  done
fi
if [[ ! -f "${afrcnn_dir}/result.json" ]] || ! grep -q '"status": "completed"' "${afrcnn_dir}/result.json"; then
  resume_args=()
  if [[ -f "${afrcnn_dir}/last.pt" ]]; then
    resume_args=(--resume "${afrcnn_dir}/last.pt")
  fi
  "${python_bin}" -u -m reproduction.train_original \
    --model afrcnn12 \
    --data-root data/LRS2-2Mix \
    --output-dir "${afrcnn_dir}" \
    --num-workers 4 \
    "${resume_args[@]}" \
    2>&1 | tee -a logs/original_afrcnn12_train.log
fi
"${python_bin}" -u -m reproduction.evaluate_original \
  --model afrcnn12 \
  --checkpoint "${afrcnn_dir}/best.pt" \
  --data-root data/LRS2-2Mix \
  --output-dir experiments/original_baselines_lrs2 \
  2>&1 | tee logs/original_afrcnn12_eval.log
