# Migrating this work to another machine

Almost nothing here needs to move. Of the 19 GB experiment tree, **179 MB is
irreplaceable**; the rest is either already in this repository, re-downloadable,
or already reduced to the numbers that were extracted from it.

## What must be carried: 179 MB

Five checkpoints and the training curves behind them. Together they represent
roughly 400 GPU-hours, and nothing in this repository can regenerate them.

| bundle path | GPU-hours to retrain | why it is needed |
|---|---:|---|
| `checkpoints/original_afrcnn12_lrs2_train_best.pt` | 136 | dense A-FRCNN: the scoring teacher **and** the source of every pruned model's inherited weights |
| `checkpoints/original_sudormrf_lrs2_train_best.pt` | ~90 | same role for SuDoRM-RF |
| `checkpoints/seprune_budgeted_afrcnn12_lrs2_finetune_best.pt` | 72 | the converged pruned A-FRCNN behind the 11.497 dB reproduction result |
| `checkpoints/seprune_budgeted_sudormrf_lrs2_finetune_best.pt` | 105 | same for SuDoRM-RF, 9.336 dB |
| `checkpoints/seprune_budgeted_afrcnn12_lrs2_finetune_epoch1.pt` | 1 | the 20,000-update anchor whose 10.522 dB defines the pre-registered E2 recovery threshold |
| `curves/*` | — | per-epoch validation SI-SDR and the final `result.json`; the convergence and learning-rate analyses read these |

Build and verify the bundle:

```bash
# on the old machine
cd <bundle>
sha256sum checkpoints/* curves/* > MANIFEST.sha256

# on the new machine
sha256sum -c MANIFEST.sha256
```

The two dense checkpoints must hash to the values this repository already
asserts — `6f9dc2c7…` for A-FRCNN and `2ef26d2e…` for SuDoRM-RF. Every scorer
and training run checks them, so a corrupted transfer fails immediately rather
than producing subtly wrong subsets.

## What is already here

- **`results/<model>/scores/*.jsonl`** — the per-sample score sets. These cost
  955 s, 249 s and **26,656 s** of GPU respectively; the TDANet one is slow
  because its upstream attention bug forces batch-size-1 scoring. Every subset
  in the study is a deterministic function of a score set plus a method and a
  seed, so `samplers.py` rebuilds any of them bit for bit.
- **`results/reproduction/masks/`** — the frozen channel masks. Without these
  the pruned structures cannot be reconstructed at all.
- **`results/runs.csv`, `results/tt_eval.csv`** — every run's provenance and
  every evaluated checkpoint's test-set SI-SDRi.

## What to leave behind

| | size | why |
|---|---:|---|
| `experiments/*/runs/*/step_*.pt` | 12 GB | intermediate checkpoints of the data-pruning arms. Their only use was the test-set evaluation, which is already in `tt_eval.csv` |
| `data/` | 32 GB | LRS2-2Mix, re-downloadable with `python scripts/download_lrs2mix.py` |
| `experiments/*/training.csv` | 1.1 GB | per-step loss rows. Every analysis reads `validation.csv`, which is carried |
| `third_party/` | 39 MB | re-fetched at pinned commits by `scripts/bootstrap.sh` |

## Restoring on the new machine

```bash
git clone <this repo> && cd sepprune-data-pruning
bash scripts/bootstrap.sh
python -m pip install -r requirements.txt        # plus a CUDA torch build
python scripts/download_lrs2mix.py               # 16.5 GB, ~32 GB extracted

# put the bundle back where the code expects it
mkdir -p experiments/original_afrcnn12_lrs2_train \
         experiments/original_sudormrf_lrs2_train \
         experiments/seprune_budgeted_afrcnn12_lrs2_finetune \
         experiments/seprune_budgeted_sudormrf_lrs2_finetune \
         experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026 \
         experiments/seprune_budgeted_sudormrf_lrs2_e07_seed2026 \
         experiments/tdanet_lrs2_mask
# checkpoints: <dir>__<file>.pt in the bundle maps back to <dir>/<file>.pt
# masks come from results/reproduction/masks/ under the same naming rule

python -m reproduction.data_pruning.verify --resume-test
```

`verify` re-asserts the checkpoint and mask hashes, so a successful run is
positive evidence that the migration is complete and intact — not just that the
files copied.

## One caveat about what a rebuilt tree can and cannot reproduce

Training is bit-reproducible on an idle GPU (72/72 tensors identical across two
runs) but not on a shared one: cuDNN picks convolution algorithms with a
memory-aware heuristic, so a run launched under different load can drift by
~3e-3 in the weights over a few Adam steps. Re-running an arm on the new
machine will therefore land near, not exactly on, the numbers in
`results/tt_eval.csv`. The seed-to-seed standard deviations reported in the
docs (0.03–0.09 dB) already include this.
