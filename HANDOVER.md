# Handover

State as of 2026-08-21 18:00 UTC. Read this before running anything; the
protocol has pre-registered rules that a well-meaning change can quietly break.

## Where things stand

| line | state |
|---|---|
| Reproduction, A-FRCNN-12 | **done.** Converged pruned model at 11.497 dB SI-SDRi, +0.010 over its dense baseline |
| Reproduction, SuDoRM-RF | **done.** 9.336 dB, −0.828 vs dense — the paper's direction, smaller magnitude |
| Reproduction, TDANet | **not done.** Only a non-budget-aligned mask (2.03M vs the paper's 1.92M) and a 1-epoch result exist |
| Data pruning, A-FRCNN-12 | **done.** Four keep ratios (50 / 10 / 5 / 2.5%), 73 runs |
| Data pruning, SuDoRM-RF | **done, inconclusive.** No keep ratio produces a measurable deficit on this model at an affordable budget, so there is nothing for selection to recover |
| Data pruning, TDANet | **training done, evaluation in flight.** 14 runs complete; test-set evaluation was at 10/28 snapshots when this was written |

## Finishing the one job still running

Training is complete. Only the evaluation remains:

```bash
cd <experiment tree>
python -m reproduction.data_pruning.evaluate_snapshots --model tdanet \
  --baseline-checkpoint "$(python -c 'from huggingface_hub import hf_hub_download;print(hf_hub_download("JusperLee/TDANetBest-4ms-LRS2","pytorch_model.bin",revision="d10e423ef25bc6f09f907455feb3f1030e9e3add"))')" \
  --masks experiments/tdanet_lrs2_mask/mask.pt --limit 1000 \
  --output experiments/data_pruning_tdanet/tt_eval_1000.json \
  experiments/data_pruning_tdanet/runs/keep500/*/step_00{7,10}000.pt
```

It is an accumulator: already-evaluated checkpoints are skipped, so re-running
it simply continues. Then `python tools/export_results.py --experiments <tree>`
and commit.

**The pre-registered endpoint is `E1t = mean(step 7,000, step 10,000)`**, and the
hypotheses in priority order are `S3′ − S2 > 0` and `S2 − S1 < 0`. `S3′ − S1` is
secondary — it was not significant on A-FRCNN at this keep ratio either. Use S0
to decide whether keep=500 even puts TDANet in a regime where data binds; if S0
and S1 are indistinguishable, the honest verdict is *inconclusive*, exactly as
it was for SuDoRM-RF. Do not reach for a different endpoint to rescue a null.
The full pre-registration is §13.9 of the plan document.

## The result this study actually supports

Not "prune-aware selection beats random" — that cleared the pre-registered bar
at exactly one of four working points. What holds across two working points,
with a mechanism that is consistent across three:

> Under a scarce training budget, mining hard examples by the pruned model's own
> loss **actively harms** recovery, and the harm grows as data gets scarcer.
> Conditioning on what the dense teacher can do avoids that harm.

`S2 − S1`: −0.016 → +0.026 → −0.070 → **−0.243** across 50/10/5/2.5% keep, the
last past the 0.15 dB materiality threshold (p=0.005). `S3′ − S2`: **+0.232** at
5% and **+0.328** at 2.5%, both past it. The subset audit says why, identically
at every budget: hard mining's mean `Q_dense` sits *below* the population's, so
it spends a shrinking budget on mixtures neither model can separate.

Also worth keeping: a 50% uniform random subset matches the full split
(−0.019 dB), with no method involved.

## Rules that are easy to break by accident

- **Endpoints and thresholds are pre-registered.** MDE is 0.15 dB; anything
  smaller is reported as indistinguishable *even when p < 0.05*. This has
  already mattered twice (§14.4, §14.5.3). Every endpoint change so far is
  recorded with its date and the evidence that motivated it, and each was
  written before the comparison it affects. Keep that discipline or the audit
  trail loses its value.
- **`verify --resume-test` is the gate.** 30 checks. It has caught two real
  defects; run it after any change.
- **TDANet must be evaluated and scored at batch 1.** Its released attention is
  built with `batch_first=False` and fed `[batch, time, channels]`, so torch
  treats the batch axis as the sequence axis and every output depends on the
  other items in the batch. The scorer refuses batch > 1; keep it that way.
- **TF32 stays off for scoring.** On an A100 it moves a per-sample SI-SDR by up
  to 1.6e-3 dB, and neighbouring percentile ranks are ~1e-3 dB apart, so it can
  permute the ranks a subset is built from.
- **Concurrency is bounded by per-model memory, not by a habit.** A-FRCNN peaks
  at ~1.4 GB per run, TDANet at ~3.0 GB. Assuming the former for the latter cost
  9 runs to OOM. Check `peak_cuda_bytes` in a finished `result.json` first.
- **Never edit a bash script while it is running.** Bash reads scripts
  incrementally by byte offset; changing the length mid-run makes it resume at a
  garbage boundary. This silently truncated one experiment batch. Copy the
  script and run the copy.
- **Launch long jobs with `setsid`.** A plain `nohup ... &` from a tool call dies
  when the caller's process group is signalled; that killed one run at step
  1,600 of 40,000.

## Suggested next steps, in the order I would do them

1. **Finish the TDANet evaluation** (above). It is the difference between a
   single-model finding and a replicated one.
2. **Test the undertrained-baseline hypothesis.** The paper reports pruning
   *improving* A-FRCNN-12 by 1.75 dB; we measure +0.010. Its dense baseline sits
   0.99 dB below ours, and the pruning stage is itself a long round of extra
   training. Train a dense model to ~epoch 40 (validation SI-SDR ≈ 10.15, the
   paper's level), prune it, fine-tune, and see whether the +1.75 dB appears.
   ~40 h for the dense stage plus ~130 h for the recovery. This is the single
   experiment that would settle the largest open question in the reproduction,
   though note it would not explain why the paper's *pruned* model is also 0.75
   dB above ours in absolute terms.
3. **Fill in the inverted U.** `Gap_rank − S1` runs +0.023 / +0.100 / +0.162 /
   +0.085 across 50/10/5/2.5%. The peak at 5% has no confirmed mechanism;
   speaker-coverage collapse was checked and ruled out. A working point between
   5% and 2.5% would say whether the peak is real or an artefact of two noisy
   points.
4. **TDANet budget-aligned pruning**, if the reproduction table needs its third
   row. This needs new code: `parameter_channel_costs` and
   `physically_prune_original` only handle A-FRCNN and SuDoRM-RF. Note that the
   *data-pruning* study does not need it — any frozen mask works.

## What is not worth redoing

- SuDoRM-RF data pruning at a larger budget. Reaching the data-limited regime
  would need ~1M updates per run across 15 runs; its long fine-tune took 4.1M
  updates and eight days to converge.
- Chasing significance on `Gap_rank − S1` with more seeds. Resolving 0.05 dB
  would take ~10 seeds per arm, and 0.05 dB does not change any engineering
  decision — for scale, pruning removed 40.3% of the parameters for 0.010 dB.

## Reading order

`README.md` → `docs/实验结果总报告.md` (consolidated results) →
`docs/数据剪枝与模型剪枝协同方案调研.md` §9 (design and pre-registration), §14
(results in chronological order) → `docs/复现实验日志.md` (dated log, including
every mishap) → `MIGRATION.md` if the machine changes.
