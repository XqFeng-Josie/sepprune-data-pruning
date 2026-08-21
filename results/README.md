# results/

The committed slice of an 18 GB experiment tree. Regenerate with
`python tools/export_results.py --experiments <dir>`.

```
<model>/scores/*.jsonl        per-sample dense and pruned SI-SDR for all 20,000
                              training mixtures, plus a _meta.json recording the
                              checkpoint hashes, TF32 state and scoring time
reproduction/test_summaries/  full 3,000-mixture test results behind every table
                              in docs/ (named <experiment>__<summary>.json)
reproduction/masks/           the frozen channel masks defining each pruned
                              structure; their sha256 are cited in docs/
reproduction/curves/          per-epoch validation SI-SDR and the final
                              result.json for the four long training runs
reproduction/preflight/       the read-only probe behind §5 of the plan
runs.csv                      one row per data-pruning training run
tt_eval.csv                   one row per evaluated checkpoint
```

## Why these and not others

The score sets are the expensive irreplaceable part: 955 s of GPU for A-FRCNN,
249 s for SuDoRM-RF and **26,656 s for TDANet**, whose upstream attention bug
forces batch-size-1 scoring. Every subset in the study is a deterministic
function of a score set plus a method name and a seed, so `samplers.py` rebuilds
any of them bit for bit — which is why the 50 MB of subset files are not here.

Checkpoints (15 GB) and per-step `training.csv` (200-365 MB each) are excluded;
nothing in the analysis reads them once a run has finished.

## Two things that look like duplicates but are not

`*_budgeted_final_*` and `*_budgeted_converged_*` evaluate the same checkpoint.
The first was produced automatically by the fine-tuning launcher when the run
early-stopped, the second by hand afterwards. They agree to four decimal places
(A-FRCNN 11.4969, SuDoRM-RF 9.3359), which is a free check that the evaluator is
deterministic across processes.

`masks/seprune_{afrcnn12,sudormrf}_lrs2_mask_*` are the earlier non-budgeted
mask searches, kept as the audit trail for why the budget-constrained variant
was needed. The masks actually used by every experiment are the
`seprune_budgeted_*` pair and `tdanet_lrs2_mask__mask.pt`.
