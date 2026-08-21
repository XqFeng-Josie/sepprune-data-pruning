# TDANet 复现流程详解

The released repository is missing the authors' modified `look2hear` package,
dependency lock file, pretrained checkpoints and portable dataset paths. The
code under `reproduction/` is therefore an auditable reconstruction of the
TDANet path revealed by `mask_learning_tdanet.py` and `finetune_tdanet.py`, not
a claim that the paper's tables have already been reproduced.

## Bootstrap

Use Python with a CUDA-compatible PyTorch and torchaudio installation. Then:

```bash
python -m pip install -r requirements-repro.txt
bash scripts/bootstrap.sh
```

The bootstrap script pins the official TDANet repository to commit
`565af18692e18bf695e5bb0ca54ba466c4a86a2a`.

## Synthetic smoke test

```bash
PYTHON_BIN=/path/to/python bash scripts/run_smoke_tdanet.sh
```

This downloads the public `JusperLee/TDANetBest-4ms-LRS2` checkpoint, loads it
with PyTorch's `weights_only=True` safety mode, freezes all pretrained weights,
updates only one 1024-channel mask, physically slices the FFN dependency chain,
and verifies that the smaller network produces finite `[1, 2, T]` output.

Artifacts are written to `experiments/smoke_tdanet/`. The synthetic tones are
only an execution/gradient test and do not produce publishable SDRi/SI-SDRi.
For the paper methodology and the remaining dataset/checkpoint requirements,
see `论文解读与复现指南.md`.

## LRS2-2Mix

The original Google Drive link in TDANet's README currently returns 404. A
16.5 GB copy is available from the same maintainer on Hugging Face:

```bash
python scripts/download_lrs2mix.py
```

The archive is intentionally ignored by Git. Inspect its member paths before
extracting it, then generate portable JSON indexes with
`DataPreProcess/process_lrs2.py`.

After extraction, run the real-data mask search and evaluation with:

```bash
.venv/bin/python -m reproduction.train_mask_tdanet --iterations 500
.venv/bin/python -m reproduction.evaluate_tdanet --limit 10
.venv/bin/python -m reproduction.evaluate_tdanet \
  --mask experiments/tdanet_lrs2_mask/mask.pt --limit 10
```

Remove `--limit` for the full 3000-example test split. The evaluation script
duplicates the SDR argument order used by the released `MetricsTracker`, even
though it is unusual, so results can be compared to the authors' pipeline.
The helper `scripts/run_full_eval_tdanet.sh` runs the original and pruned
3000-example evaluations sequentially.

## Three original LRS2 backbones

TDANet uses the public `JusperLee/TDANetBest-4ms-LRS2` checkpoint. The SepPrune
release does not include its LRS2 A-FRCNN-12 or SuDoRM-RF checkpoints, so those
two models must be trained from scratch. Their official architecture sources
are pinned under `third_party/AFRCNN` and `third_party/sudo_rm_rf`; the SepPrune
configurations reproduce the paper parameter counts (5,127,688 and 2,720,417).

Run the resumable sequential baseline pipeline with:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_original_baselines.sh
```

It evaluates TDANet, trains/evaluates SuDoRM-RF for the paper-reported 86
epochs, then trains/evaluates A-FRCNN-12 for 136 epochs. Training uses the
released Look2Hear PIT negative-SNR objective, Adam at 1e-3, batch size 1,
2-second segments, validation SI-SDR, ReduceLROnPlateau, and gradient clipping
at 5. Checkpoints are overwritten every 1,000 steps and at every epoch, so the
launcher resumes automatically from `last.pt` after interruption. Final test
summaries are written to `experiments/original_baselines_lrs2/`.

## One-epoch recovery and random baseline

The stage-two launcher fine-tunes the inherited pruned weights for one epoch,
evaluates all 3000 test mixtures, then trains and evaluates the identical
pruned structure from random initialization:

```bash
PYTHON_BIN=.venv/bin/python bash scripts/run_stage2_tdanet.sh
```

Both runs use Adam at `1e-3`, batch size 1, negative SI-SDR, gradient clipping
at 5, seed 2026, and the same 711/1024-channel mask. A recoverable `last.pt` is
overwritten every 1000 steps. Re-running the launcher skips completed stages
and resumes an interrupted training stage from `last.pt`.

Complexity and CUDA timing can be measured with:

```bash
.venv/bin/python -m reproduction.analyze_tdanet
.venv/bin/python -m reproduction.analyze_tdanet --mask path/to/mask.pt
.venv/bin/python -m reproduction.benchmark_tdanet --repeats 1000
.venv/bin/python -m reproduction.benchmark_tdanet --mask path/to/mask.pt --repeats 1000
```
