# sepprune-data-pruning

An independent reproduction of **SepPrune** (structured channel pruning for deep
speech separation, AAAI 2026) on LRS2-2Mix, and a study built on top of it:
**does choosing *which* training data the pruned model recovers on actually
matter?**

This is not a fork of the SepPrune release. The authors' repository ships
neither their modified `look2hear` package nor the `ChannelMask1D` and physical
pruning modules its scripts import, so everything here is a reconstruction
written against the public model implementations, with the mask search and
physical pruning re-derived from the channel dependencies the released
fine-tuning scripts expose.

## Headline results

**Reproduction.** Pruning is lossless on one backbone and lossy on the other,
matching the paper's direction on SuDoRM-RF but not its magnitude on A-FRCNN:

| model | parameters | dense SI-SDRi | pruned SI-SDRi | change | paper's change |
|---|---|---:|---:|---:|---:|
| A-FRCNN-12 | 5.13M → 3.06M (−40.3%) | 11.487 | 11.497 | **+0.010** | +1.750 |
| SuDoRM-RF | 2.72M → 1.54M (−43.4%) | 10.164 | 9.336 | **−0.828** | −1.120 |

The paper reports that pruning *improves* A-FRCNN-12 by 1.75 dB. We measure
+0.01 dB. The most likely explanation is that the paper's dense baseline is
undertrained — it sits 0.99 dB below ours, and the pruning stage is itself a
long round of extra training — but that explanation is **consistent with the
evidence rather than verified**, and it does not account for the paper's pruned
model being 0.75 dB above ours in absolute terms.

**Data pruning.** Selecting training data helps only once data is genuinely
scarce, and the direction of the selection rule matters more than its strength:

| keep ratio | random subset's loss vs full data | `Gap_rank` − random | pre-registered criteria |
|---:|---:|---:|:--:|
| 50% | 0.019 dB (no loss at all) | +0.023 | ✗ |
| 10% | 0.270 dB | +0.100 (p=0.017) | ✗ below 0.15 dB threshold |
| **5%** | **0.603 dB** | **+0.162 (p=0.005)** | **✓** |

At 5% keep, `Gap_rank` also beats hard-example mining by +0.232 dB (p=0.0001),
and **hard-example mining is worse than uniform random** (−0.070 dB, p=0.021):
it spends a tiny budget on mixtures that neither the dense nor the pruned model
can separate. Conditioning on what the dense teacher can actually do is what
makes selection work.

Two findings matter more in practice than the 0.162 dB:

- **A 50% uniform random subset matches the full training split**, with no
  selection method involved at all.
- Under a fixed optimizer-update budget none of this saves wall-clock. Every
  claim here is about *data* efficiency; see `docs/` for the accounting.

All results are single-dataset (LRS2-2Mix) and, for the data-pruning study,
currently single-backbone. A cross-model attempt on SuDoRM-RF was
**inconclusive** — at an affordable budget that model never reaches the regime
where data is the binding constraint. See `docs/实验结果总报告.md`.

## Setup

```bash
python -m pip install -r requirements.txt   # plus a CUDA build of torch/torchaudio
bash scripts/bootstrap.sh                   # pins TDANet, AFRCNN, sudo_rm_rf
python scripts/download_lrs2mix.py          # 16.5 GB LRS2-2Mix
python -m reproduction.data_pruning.verify --resume-test
```

`verify` is the gate, not a formality: it asserts the invariants the study
depends on, and it has caught real defects — TF32 perturbing percentile ranks,
and an upstream TDANet attention bug that makes its output depend on batch size.
Run it after any change.

## Reproducing the experiments

```bash
# Dense baselines, mask search, and pruned fine-tuning (days of GPU time)
PYTHON_BIN=.venv/bin/python bash scripts/run_original_baselines.sh
PYTHON_BIN=.venv/bin/python bash scripts/run_budgeted_mask_original.sh
PYTHON_BIN=.venv/bin/python bash scripts/run_budgeted_finetune_original.sh

# Data-pruning study: score once, build every subset, train each arm
MODEL=afrcnn12 KEEPS="1000" SEEDS="2026 7 13 42 99" ARMS="S1 S2 S3p" \
  PARALLEL=15 PYTHON_BIN=.venv/bin/python bash scripts/run_data_pruning_stage_a.sh
```

The launcher is resumable and skips completed runs. Arms: `S0` full data,
`S1` uniform random, `S2` hard-example mining, `S3` raw-dB gap, `S3p` the
percentile-rank gap (the main method), `K0` dense-teacher distillation.

Batch-size-1 separation training is launch-latency bound, so several arms share
one GPU almost for free — a second job costs the incumbent 3.7%, and aggregate
throughput saturates around 20 steps/s. `PARALLEL` exploits that; arms in one
batch also see identical load, which is what makes their wall-clock comparable.

## Layout

```
reproduction/               models, training, evaluation, physical pruning
  data_pruning/             the study: scoring, samplers, training, audit, verify
docs/                       plan of record, pre-registration, daily log, results
  数据剪枝与模型剪枝协同方案调研.md   design + pre-registered protocol + per-workpoint results
  实验结果总报告.md                    consolidated results across both lines
  TDANet复现流程详解.md                step-by-step TDANet pipeline
  复现实验日志.md                      dated log, including every decision and mishap
scripts/                    bootstrap, dataset download, experiment launchers
tools/export_results.py     extracts the committable slice of an experiment tree
results/                    per-sample score sets and summary tables (see below)
```

The package is still called `reproduction/` because that is what it is; renaming
it would only churn imports.

## What is committed, and what is not

`results/` holds the two things that are expensive to recompute but small:

- `results/<model>/scores/*.jsonl` — per-sample dense and pruned SI-SDR for all
  20,000 training mixtures. Every subset in the study is a deterministic
  function of these plus a method name and a seed, so `samplers.py` rebuilds any
  subset bit for bit.
- `results/runs.csv`, `results/tt_eval.csv` — one row per run and per evaluated
  checkpoint, with provenance hashes.

Not committed: checkpoints (15 GB), the subset files (50 MB, regenerable), and
the per-sample arrays behind `tt_eval.csv`. Regenerate the extract with
`python tools/export_results.py --experiments <dir>`.

## Method, in one paragraph

The pruned model inherits its surviving weights from the dense model, so both
are available for free at the start of recovery. Score every training mixture
with each: `Q_dense` and `Q_pruned`, PIT-best SI-SDR. Convert each to a
percentile rank over the training split and take the difference,
`Gap_rank = pct(Q_dense) − pct(Q_pruned)`. Ranking before subtracting is what
keeps the score from collapsing into plain hard-example mining: the raw dB
difference correlates 0.81 with pruned loss because the pruned term has 2.2×
the variance, while the rank difference correlates 0.50. Select the top 75% of
the budget by that score and fill the rest uniformly at random, under identical
mixing-SNR stratum quotas across every arm.

## Provenance and licensing

Our code is MIT (see `LICENSE`). The three upstream model implementations are
cloned at pinned commits by `scripts/bootstrap.sh` rather than vendored, and
keep their own licences: MIT for
[AFRCNN](https://github.com/JusperLee/AFRCNN-For-Speech-Separation) and
[sudo_rm_rf](https://github.com/etzinis/sudo_rm_rf), Apache-2.0 for
[TDANet](https://github.com/JusperLee/TDANet). The TDANet dense weights are the
public `JusperLee/TDANetBest-4ms-LRS2` checkpoint; the A-FRCNN-12 and SuDoRM-RF
dense models are trained here from scratch because the SepPrune release does not
publish them.

Paper: A. Li et al., *SepPrune: Structured Pruning for Efficient Deep Speech
Separation*, AAAI 2026.
