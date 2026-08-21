#!/usr/bin/env bash
# Stage A of the data-pruning study: score once, build every subset, then train
# each arm under an identical fixed-update budget.
#
# Resumable. Every stage is skipped when its artefact already exists, and each
# training arm resumes from its own last.pt, so the launcher can be re-run after
# an interruption or after the GPU frees up.
#
#   PYTHON_BIN=.venv/bin/python bash scripts/run_data_pruning_stage_a.sh
#
# Environment overrides:
#   MODEL          afrcnn12 | sudormrf | tdanet    (default: afrcnn12)
#   TOTAL_UPDATES / SNAPSHOTS / NUM_WORKERS also honoured; TDANet needs a
#   lower PARALLEL than the others because each run peaks at ~3.0 GB of GPU
#   memory against A-FRCNN's ~1.4 GB.
#   KEEPS          space-separated keep budgets    (default: "10000")
#   SEEDS          space-separated arm seeds       (default: "2026")
#   ARMS           space-separated arm names       (default: all of stage A1)
#   TOTAL_UPDATES  optimizer updates per arm       (default: 40000)
#   PARALLEL       arms trained concurrently       (default: 3)
#   ROOT           output directory                (default: experiments/data_pruning)
#
# The A0 power gate is `SEEDS="2026 7 13" ARMS="S1" ...`: three seeds of the
# random-subset arm alone, whose spread must come out below the pre-registered
# 0.15 dB minimum detectable effect before the other arms are worth running.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MODEL="${MODEL:-afrcnn12}"
KEEPS="${KEEPS:-10000}"
SEEDS="${SEEDS:-2026}"
ARMS="${ARMS:-S0 S1 S2 S3 S3p}"
TOTAL_UPDATES="${TOTAL_UPDATES:-40000}"
# K0 is the distillation anchor: full data, frozen dense teacher, no selection.
# Add it with ARMS="... K0". It is deliberately NOT compute-matched with S0 --
# the teacher forward makes every update dearer -- so read its wall-clock apart
# from the data arms and compare it to S0 only at equal update counts.
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
# How many arms to train at once. Measured on this A100: 3 concurrent arms
# finish in 2.8 h versus 5.9 h sequentially, and see identical conditions.
PARALLEL="${PARALLEL:-3}"
# Snapshot points. Low keep budgets overtrain within the 40,000-update budget
# (see the plan's §13.4), so those sweeps add a 10,000-update point.
SNAPSHOTS="${SNAPSHOTS:-20000,30000,40000}"
# Dataloader workers per run. This box has 12 cores, so a wide batch must cut
# this down or the worker processes contend with each other; batch-size-1
# training on 2 s clips needs very little decode throughput anyway.
NUM_WORKERS="${NUM_WORKERS:-4}"

# Frozen inputs per backbone. The hashes are asserted by both the scorer and
# every training run, so a swapped checkpoint fails immediately instead of
# silently producing a subset that no longer means what its name says.
case "${MODEL}" in
  afrcnn12)
    DENSE="experiments/original_afrcnn12_lrs2_train/best.pt"
    MASKS="experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026/masks.pt"
    DENSE_SHA="6f9dc2c700b03ed38bf6070e0b0929269fa2f43d1b8b0239229724145c322da6"
    MASKS_SHA="13a774ee64c587ac7b7f9e82e2e37070c1ee6258a5aeb9206ffe4e2bce540433"
    DEFAULT_ROOT="experiments/data_pruning" ;;
  sudormrf)
    DENSE="experiments/original_sudormrf_lrs2_train/best.pt"
    MASKS="experiments/seprune_budgeted_sudormrf_lrs2_e07_seed2026/masks.pt"
    DENSE_SHA="2ef26d2e6707e3094839f8e37f7752d7c6038399d79f724847fef58b13fad5e1"
    MASKS_SHA="69ed6fe4a66bfec0e7a602f00b232b0c4ce38c414e8b38a842e2c7fe42b175c2"
    DEFAULT_ROOT="experiments/data_pruning_sudormrf" ;;
  tdanet)
    # The dense TDANet is the published checkpoint, not one trained here, so it
    # is resolved from the Hugging Face cache. Its attention mixes across the
    # batch (see the plan's §13.2), which is why scoring is pinned to batch 1.
    DENSE="${TDANET_CHECKPOINT:-$(python - <<'PYX'
from huggingface_hub import hf_hub_download
print(hf_hub_download("JusperLee/TDANetBest-4ms-LRS2", "pytorch_model.bin",
                      revision="d10e423ef25bc6f09f907455feb3f1030e9e3add"))
PYX
)}"
    MASKS="experiments/tdanet_lrs2_mask/mask.pt"
    DENSE_SHA="0048bdb31c71e8ec9c694e828ef268e7483051358bb1d309c8959bafb9a4b958"
    MASKS_SHA="500d1437dfb73c30d75334784d04c20c1864d270cb77b6ffc48beab4baa772da"
    DEFAULT_ROOT="experiments/data_pruning_tdanet" ;;
  *)
    echo "unknown MODEL: ${MODEL} (expected afrcnn12, sudormrf or tdanet)" >&2; exit 1 ;;
esac
# Each backbone keeps its own tree: the scores, the subsets derived from them and
# the runs are all specific to one dense/mask pair.
ROOT="${ROOT:-${DEFAULT_ROOT}}"

SCORES="${ROOT}/scores/${MODEL}_tr_init"
LOGS="logs"
mkdir -p "${ROOT}" "${LOGS}"

# Subsets and runs live under their keep budget, so a keep-ratio sweep never
# clobbers an earlier one. S0 and K0 train on the whole split and are therefore
# independent of the budget: they only ever need to be run once, under any keep.
subset_dir() { echo "${ROOT}/subsets/keep${1}"; }
run_dir()    { echo "${ROOT}/runs/keep${1}/${2}_seed${3}"; }

# Method and subset file behind each arm label, for a given keep budget.
arm_subset() {  # arm seed keep
  local dir; dir="$(subset_dir "$3")"
  case "$1" in
    S0)  echo "" ;;
    S1)  echo "${dir}/random_seed${2}.json" ;;
    S2)  echo "${dir}/hard_seed${2}.json" ;;
    S3)  echo "${dir}/gap_db_seed${2}.json" ;;
    S3p) echo "${dir}/gap_rank_seed${2}.json" ;;
    S4)  echo "${dir}/gap_rank_2d_seed${2}.json" ;;
    K0)  echo "" ;;
    *)   echo "unknown arm: $1" >&2; exit 1 ;;
  esac
}

echo "=== model=${MODEL}  root=${ROOT} ==="
echo "=== acceptance checks ==="
"${PYTHON_BIN}" -m reproduction.data_pruning.verify --resume-test 2>&1 | tee "${LOGS}/dp_${MODEL}_verify.log"

if [ -f "${SCORES}.jsonl" ]; then
  echo "=== scoring: already done, skipping ==="
else
  echo "=== scoring the training split ==="
  "${PYTHON_BIN}" -m reproduction.data_pruning.score_lrs2 \
    --model "${MODEL}" \
    --baseline-checkpoint "${DENSE}" --masks "${MASKS}" \
    --expect-baseline-sha256 "${DENSE_SHA}" --expect-masks-sha256 "${MASKS_SHA}" \
    --split tr --batch-size 32 --output-dir "${ROOT}/scores" \
    2>&1 | tee "${LOGS}/dp_${MODEL}_score.log"
fi

for keep in ${KEEPS}; do
  dir="$(subset_dir "${keep}")"
  mkdir -p "${dir}"
  echo "=== building subsets: keep=${keep}, seeds: ${SEEDS} ==="
  "${PYTHON_BIN}" -m reproduction.data_pruning.samplers \
    --scores "${SCORES}" \
    --methods random,hard,gap_db,gap_rank,gap_rank_2d \
    --seeds "$(echo "${SEEDS}" | tr ' ' ',')" \
    --keep "${keep}" --score-fraction 0.75 --snr-strata 4 \
    --output-dir "${dir}" 2>&1 | tee "${LOGS}/dp_${MODEL}_subsets_keep${keep}.log"
  echo "=== auditing subsets: keep=${keep} ==="
  "${PYTHON_BIN}" -m reproduction.data_pruning.audit_selection \
    --scores "${SCORES}" --subsets "${dir}" 2>&1 | tee "${LOGS}/dp_${MODEL}_audit_keep${keep}.log"
done

launch() {  # arm seed keep
  local arm="$1" seed="$2" keep="$3"
  local out; out="$(run_dir "${keep}" "${arm}" "${seed}")"
  local subset distill resume
  mkdir -p "${out}"
  subset="$(arm_subset "${arm}" "${seed}" "${keep}")"
  distill=""
  [ "${arm}" = "K0" ] && distill="--distill-alpha ${DISTILL_ALPHA}"
  resume=""
  [ -f "${out}/last.pt" ] && resume="--resume"
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" -m reproduction.data_pruning.train_data_pruned \
    --model "${MODEL}" \
    --arm "${arm}" --seed "${seed}" \
    --baseline-checkpoint "${DENSE}" --masks "${MASKS}" \
    --expect-baseline-sha256 "${DENSE_SHA}" --expect-masks-sha256 "${MASKS_SHA}" \
    ${subset:+--subset "${subset}"} \
    ${distill} \
    --total-updates "${TOTAL_UPDATES}" \
    --validate-every 5000 --monitor-size 1000 \
    --snapshot-at "${SNAPSHOTS}" \
    --num-workers "${NUM_WORKERS}" \
    --output-dir "${out}" ${resume} \
    > "${LOGS}/dp_${MODEL}_keep${keep}_${arm}_seed${seed}.log" 2>&1
}

# Batch-size-1 separation training is launch-latency bound, so several arms share
# the GPU cheaply: measured on this A100, a second job costs the incumbent 3.7%
# and five concurrent jobs still nearly triple aggregate throughput. Arms in the
# same batch also see identical load (measured spread < 0.5% ms/step), which is
# what makes their wall-clock comparable for the training-efficiency endpoint.
# Arms in different batches may not be, so keep arms you intend to compare
# together, or report only the fixed-update endpoint across batches.
pending=()
for keep in ${KEEPS}; do
  for seed in ${SEEDS}; do
    for arm in ${ARMS}; do
      out="$(run_dir "${keep}" "${arm}" "${seed}")"
      if [ -f "${out}/result.json" ] && grep -q '"status": "completed"' "${out}/result.json"; then
        echo "=== keep=${keep} ${arm} seed=${seed}: already completed, skipping ==="
        continue
      fi
      pending+=("${arm}:${seed}:${keep}")
    done
  done
done

batch=()
run_batch() {
  [ ${#batch[@]} -eq 0 ] && return 0
  echo "=== batch: ${batch[*]} (${TOTAL_UPDATES} updates each) ==="
  local pids=() job arm seed keep
  for job in "${batch[@]}"; do
    IFS=: read -r arm seed keep <<< "${job}"
    launch "${arm}" "${seed}" "${keep}" &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
  [ ${status} -ne 0 ] && { echo "a run in this batch failed; see ${LOGS}/" >&2; exit 1; }
  batch=()
}

for job in "${pending[@]}"; do
  batch+=("${job}")
  [ ${#batch[@]} -ge "${PARALLEL}" ] && run_batch
done
run_batch

echo "=== stage A training complete ==="
echo "Evaluate the 20,000- and 40,000-update snapshots on the full test split with"
echo "reproduction.evaluate_pruned_original --checkpoint <run>/step_020000.pt (batch 1),"
echo "which is the same evaluator that produced the existing 10.522 dB anchor."
