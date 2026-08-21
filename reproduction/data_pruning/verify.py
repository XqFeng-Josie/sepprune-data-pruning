"""Executable acceptance checks for the data-pruning pipeline.

This is the gate described in the plan's "强制验收项": nothing here trains a
model for real, but every invariant the stage-A protocol relies on is asserted
against live code, not against a description of it.

    .venv/bin/python -m reproduction.data_pruning.verify              # everything
    .venv/bin/python -m reproduction.data_pruning.verify --quick      # no GPU, no subprocess

Groups:

    metrics   the batched PIT SI-SDR agrees with the scalar helper already in
              use, is permutation-symmetric, and self-scores to a zero gap
    schema    score files round-trip, recompute their own derived columns, and
              reject tampering
    sampler   subsets hit their budgets exactly, keep their parts disjoint, and
              depend on the seed only through the random-exploration slice
    training  the step -> example map is deterministic and resume-exact
    model     batching a real checkpoint does not move the score, and
              threshold_byloss really is inert at batch size 1
    resume    a run stopped at step k and resumed reaches the same weights as
              an uninterrupted run
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

import torch

from ..evaluate_original import best_si_sdr
from ..lrs2 import LRS2MixDataset
from ..original_models import build_original_model
from ..train_original import pit_negative_sdr
from ..train_pruned_original import build_inherited_pruned_model
from .common import set_tf32, sha256_text
from .distill_baseline import align_to_targets, distillation_loss
from .metrics import pit_best_si_sdr
from .samplers import build_subset, load_subset, split_quota
from .score_schema import derive_columns, load_score_set, percentile_ranks, write_score_set
from .train_data_pruned import (
    build_monitor_indices,
    per_sample_pit_loss,
    resolve_subset_indices,
    training_order,
)

DEFAULT_BASELINE = "experiments/original_afrcnn12_lrs2_train/best.pt"
DEFAULT_MASKS = "experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026/masks.pt"

BATCH_TOLERANCE = 1e-4
"""Batched and single-example scores may differ only by kernel-selection noise.

`score_lrs2` disables TF32, which puts the observed deviation at ~6e-6 dB. The
tolerance is set an order of magnitude above that and still two orders below
the ~1e-3 dB spacing between neighbouring percentile ranks. With TF32 left on
the deviation is ~1.6e-3 dB, i.e. large enough to permute adjacent ranks, which
is exactly why the scorer turns it off.
"""

RESUME_LOSS_TOLERANCE = 1e-2
"""A resumed run must replay the same examples in the same order.

Bitwise equality is deliberately not required. cuDNN picks convolution
algorithms with a memory-aware heuristic, so a run launched while the GPU is
busy with other jobs can select a different algorithm and drift by ~1e-3 over a
few Adam steps. What a real defect looks like is different: feeding the wrong
example at step k moves that step's loss by several dB, which this tolerance
catches immediately while ignoring kernel noise.
"""

RESUME_WEIGHT_TOLERANCE = 5e-3

_REGISTRY: list[tuple[str, str, Callable[[argparse.Namespace], str]]] = []


def check(group: str, name: str):
    def decorate(function: Callable[[argparse.Namespace], str]):
        _REGISTRY.append((group, name, function))
        return function

    return decorate


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #


def _random_pair(batch: int, samples: int = 4096) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260813)
    estimate = torch.randn(batch, 2, samples, generator=generator)
    target = torch.randn(batch, 2, samples, generator=generator)
    return estimate, target


@check("metrics", "batched PIT matches the scalar helper element by element")
def _metrics_matches_scalar(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(16)
    batched = pit_best_si_sdr(estimate, target)
    scalar = torch.stack([best_si_sdr(estimate[i], target[i]) for i in range(estimate.shape[0])])
    deviation = float((batched - scalar).abs().max())
    if deviation != 0.0:
        raise AssertionError(f"max |batched - scalar| = {deviation:g}, expected exactly 0")
    return "exact agreement over 16 examples"


@check("metrics", "PIT score is invariant to swapping the two reference sources")
def _metrics_swap_invariant(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(16)
    straight = pit_best_si_sdr(estimate, target)
    swapped = pit_best_si_sdr(estimate, target.flip(1))
    deviation = float((straight - swapped).abs().max())
    if deviation != 0.0:
        raise AssertionError(f"max deviation {deviation:g} after swapping s1/s2")
    return "exact invariance over 16 examples"


@check("metrics", "identical models score a zero gap")
def _metrics_self_gap(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(16)
    gap = pit_best_si_sdr(estimate, target) - pit_best_si_sdr(estimate, target)
    if float(gap.abs().max()) != 0.0:
        raise AssertionError("scoring one model against itself did not cancel")
    return "gap == 0 for all 16 examples"


@check("metrics", "threshold_byloss is inert at batch size 1")
def _metrics_threshold_noop(args: argparse.Namespace) -> str:
    generator = torch.Generator().manual_seed(7)
    worst = 0.0
    for _ in range(64):
        estimate = torch.randn(1, 2, 2048, generator=generator)
        target = torch.randn(1, 2, 2048, generator=generator)
        scale = float(torch.rand((), generator=generator)) * 1e3
        estimate = estimate * scale
        filtered = pit_negative_sdr(estimate, target, kind="snr", threshold_byloss=True)
        plain = pit_negative_sdr(estimate, target, kind="snr", threshold_byloss=False)
        worst = max(worst, abs(float(filtered - plain)))
    if worst != 0.0:
        raise AssertionError(f"threshold_byloss changed the loss by {worst:g} at batch size 1")
    return "identical over 64 randomised single-example batches"


@check("metrics", "per-sample validation loss averages to the batched loss")
def _metrics_per_sample_mean(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(12)
    mean = float(per_sample_pit_loss(estimate, target, "sisdr").mean())
    reference = float(pit_negative_sdr(estimate, target, kind="sisdr", threshold_byloss=False))
    deviation = abs(mean - reference)
    if deviation > 1e-6:
        raise AssertionError(f"per-sample mean differs by {deviation:g}")
    return f"|difference| = {deviation:.2e}"


@check("metrics", "validation loss is independent of the validation batch size")
def _metrics_batch_size_invariance(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(24)
    whole = float(per_sample_pit_loss(estimate, target, "sisdr").mean())
    chunks = torch.cat(
        [per_sample_pit_loss(estimate[i : i + 5], target[i : i + 5], "sisdr") for i in range(0, 24, 5)]
    )
    deviation = abs(whole - float(chunks.mean()))
    if deviation > 1e-6:
        raise AssertionError(f"chunked mean differs by {deviation:g}")
    return f"|difference| = {deviation:.2e} between one batch of 24 and chunks of 5"


@check("distill", "alpha=0 reproduces the plain task loss exactly")
def _distill_alpha_zero(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(16)
    teacher, _ = _random_pair(16)
    terms = distillation_loss(estimate, target, teacher, alpha=0.0, kind="snr")
    if float((terms["total"] - terms["task"]).abs().max()) != 0.0:
        raise AssertionError("alpha=0 still mixed in a distillation term")
    reference = float(pit_negative_sdr(estimate, target, kind="snr", threshold_byloss=False))
    deviation = abs(float(terms["task"].mean()) - reference)
    if deviation > 1e-6:
        raise AssertionError(f"task loss differs from the shared PIT loss by {deviation:g}")
    return f"total == task, and task == pit_negative_sdr (|Δ|={deviation:.2e})"


@check("distill", "a perfect teacher makes the distillation term equal the task term")
def _distill_perfect_teacher(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(16)
    reports = []
    for label, student in (("aligned", estimate), ("swapped", estimate.flip(1))):
        terms = distillation_loss(student, target, target.clone(), alpha=0.5, kind="snr")
        deviation = float((terms["distillation"] - terms["task"]).abs().max())
        if deviation > 1e-5:
            raise AssertionError(
                f"{label} student: with teacher == target the distillation term is "
                f"{deviation:g} away from the task term, so the two permutations disagreed"
            )
        reports.append(f"{label} max|Δ|={deviation:.1e}")
    return "both permutation regimes: " + ", ".join(reports)


@check("distill", "the loss is invariant to the teacher's own source ordering")
def _distill_teacher_permutation(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(16)
    teacher, _ = _random_pair(16)
    straight = distillation_loss(estimate, target, teacher, alpha=0.5, kind="snr")["total"]
    flipped = distillation_loss(estimate, target, teacher.flip(1), alpha=0.5, kind="snr")["total"]
    deviation = float((straight - flipped).abs().max())
    if deviation > 1e-5:
        raise AssertionError(
            f"flipping the teacher's two outputs moved the loss by {deviation:g}; "
            "the teacher is not being aligned to the targets"
        )
    aligned = align_to_targets(teacher, target, "snr")
    if float((aligned - align_to_targets(aligned, target, "snr")).abs().max()) != 0.0:
        raise AssertionError("align_to_targets is not idempotent")
    return f"max|Δ|={deviation:.1e}, and alignment is idempotent"


@check("distill", "alpha interpolates between the two terms")
def _distill_interpolation(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(16)
    teacher, _ = _random_pair(16)
    terms = distillation_loss(estimate, target, teacher, alpha=0.25, kind="snr")
    expected = 0.75 * terms["task"] + 0.25 * terms["distillation"]
    deviation = float((terms["total"] - expected).abs().max())
    if deviation > 1e-6:
        raise AssertionError(f"total is not the declared blend, off by {deviation:g}")
    whole = distillation_loss(estimate, target, teacher, alpha=1.0, kind="snr")
    if float((whole["total"] - whole["distillation"]).abs().max()) != 0.0:
        raise AssertionError("alpha=1 did not reduce to the distillation term")
    return f"alpha=0.25 blend exact (|Δ|={deviation:.1e}), alpha=1 reduces to distillation"


@check("distill", "the teacher never joins the student's autograd graph")
def _distill_teacher_detached(args: argparse.Namespace) -> str:
    estimate, target = _random_pair(4)
    teacher, _ = _random_pair(4)
    estimate = estimate.clone().requires_grad_(True)
    teacher = teacher.clone().requires_grad_(True)
    distillation_loss(estimate, target, teacher, alpha=0.5, kind="snr")["total"].mean().backward()
    if teacher.grad is not None:
        raise AssertionError("gradient flowed into the teacher output")
    if estimate.grad is None or not torch.isfinite(estimate.grad).all():
        raise AssertionError("the student received no finite gradient")
    return "teacher.grad is None, student gradient finite"


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #


def _synthetic_rows(count: int = 400) -> list[dict]:
    generator = torch.Generator().manual_seed(4242)
    dense = torch.randn(count, generator=generator) * 1.6 + 14.0
    pruned = torch.randn(count, generator=generator) * 3.6 + 6.7
    gains = torch.rand(count, generator=generator) * 5.0
    rows = []
    for index in range(count):
        gain = round(float(gains[index]), 4)
        speaker1 = f"{5_000_000_000_000_000_000 + index % 97:d}"
        speaker2 = f"{6_000_000_000_000_000_000 + index % 89:d}"
        name = f"{speaker1}_{index:05d}_{gain}_{speaker2}_{index % 71:05d}_{-gain}.wav"
        rows.append(
            {
                "name": name,
                "q_dense_sisdr": float(dense[index]),
                "q_pruned_sisdr": float(pruned[index]),
                "mixture_sisdr_baseline": float(torch.randn((), generator=generator)) * 0.18,
                "snr_abs": abs(gain),
                "spk1": speaker1,
                "spk2": speaker2,
            }
        )
    return rows


def _synthetic_meta() -> dict:
    return {
        "model": "afrcnn12",
        "split": "tr",
        "pruned_label": "init",
        "dense_checkpoint": {"path": "synthetic", "sha256": "0" * 64},
        "masks": {"path": "synthetic", "sha256": "1" * 64},
        "pruned_checkpoint": None,
    }


def _write_synthetic(directory: Path):
    rows = derive_columns(_synthetic_rows())
    write_score_set(directory / "synthetic", _synthetic_meta(), rows)
    return load_score_set(directory / "synthetic")


@check("schema", "percentile ranks are a bijection onto an even grid")
def _schema_percentiles(args: argparse.Namespace) -> str:
    values = [3.0, 1.0, 2.0, 1.0]
    keys = ["d", "a", "c", "b"]
    ranks = percentile_ranks(values, keys)
    if sorted(ranks) != [0.0, 1 / 3, 2 / 3, 1.0]:
        raise AssertionError(f"ranks are not evenly spaced: {ranks}")
    if not ranks[1] < ranks[3]:
        raise AssertionError("ties were not broken by the key in ascending order")
    return "even spacing and deterministic tie-breaking"


@check("schema", "score sets round-trip and recompute their derived columns")
def _schema_round_trip(args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        loaded = _write_synthetic(Path(directory))
        for row in loaded.rows:
            if abs(row["gap_db"] - (row["q_dense_sisdr"] - row["q_pruned_sisdr"])) > 1e-12:
                raise AssertionError(f"gap_db is inconsistent for {row['name']}")
            if abs(row["gap_rank"] - (row["pct_dense"] - row["pct_pruned"])) > 1e-12:
                raise AssertionError(f"gap_rank is inconsistent for {row['name']}")
        return f"{len(loaded.rows)} rows verified row by row"


@check("schema", "a tampered score file fails to load")
def _schema_tamper(args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        _write_synthetic(Path(directory))
        rows_path = Path(directory) / "synthetic.jsonl"
        lines = rows_path.read_text(encoding="utf-8").splitlines()
        payload = json.loads(lines[5])
        payload["q_pruned_sisdr"] += 1.0
        lines[5] = json.dumps(payload, ensure_ascii=False)
        rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            load_score_set(Path(directory) / "synthetic")
        except ValueError as error:
            return f"rejected with: {str(error)[:70]}"
        raise AssertionError("a modified score file was accepted")


@check("schema", "inconsistent speaker or snr fields are rejected")
def _schema_name_consistency(args: argparse.Namespace) -> str:
    rows = derive_columns(_synthetic_rows(40))
    rows[3]["snr_abs"] = rows[3]["snr_abs"] + 0.5
    with tempfile.TemporaryDirectory() as directory:
        try:
            write_score_set(Path(directory) / "bad", _synthetic_meta(), rows)
        except ValueError as error:
            return f"rejected with: {str(error)[:70]}"
    raise AssertionError("snr_abs inconsistent with the file name was accepted")


# --------------------------------------------------------------------------- #
# sampler
# --------------------------------------------------------------------------- #


def _subsets(scores, methods=("random", "hard", "gap_db", "gap_rank", "gap_rank_2d"), seed=2026):
    return {
        method: build_subset(
            scores,
            method=method,
            seed=seed,
            keep=200,
            score_fraction=0.0 if method == "random" else 0.75,
            snr_strata=4,
            dense_strata=4,
        )
        for method in methods
    }


@check("sampler", "quota arithmetic sums exactly and is order-independent")
def _sampler_quota(args: argparse.Namespace) -> str:
    for total, parts in ((7500, 4), (10000, 4), (150, 4), (1875, 4), (0, 3), (7, 3)):
        quota = split_quota(total, parts)
        if sum(quota) != total or len(quota) != parts:
            raise AssertionError(f"split_quota({total}, {parts}) = {quota}")
        if max(quota) - min(quota) > 1:
            raise AssertionError(f"split_quota({total}, {parts}) is not balanced: {quota}")
    return "checked 6 configurations including the production 7500/4 and 10000/4"


@check("sampler", "every arm hits its budget with disjoint parts")
def _sampler_budgets(args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        scores = _write_synthetic(Path(directory))
        for method, payload in _subsets(scores).items():
            names = payload["names"]
            if len(names) != 200 or len(set(names)) != 200:
                raise AssertionError(f"{method}: kept {len(set(names))} unique names, expected 200")
            selected, explored = payload["score_selected"], payload["random_explored"]
            if set(selected) & set(explored):
                raise AssertionError(f"{method}: the two parts overlap")
            expected = 0 if method == "random" else 150
            if len(selected) != expected:
                raise AssertionError(f"{method}: {len(selected)} score-selected, expected {expected}")
            population = set(scores.names)
            if not set(names) <= population:
                raise AssertionError(f"{method}: selected names outside the population")
        return "5 methods checked for budget, disjointness and containment"


@check("sampler", "all arms share identical stratum boundaries and keep quotas")
def _sampler_strata_identical(args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        scores = _write_synthetic(Path(directory))
        subsets = _subsets(scores)
        reference = [
            (item["size"], item["snr_abs_min"], item["snr_abs_max"], item["keep_quota"])
            for item in subsets["random"]["strata"]
        ]
        for method, payload in subsets.items():
            current = [
                (item["size"], item["snr_abs_min"], item["snr_abs_max"], item["keep_quota"])
                for item in payload["strata"]
            ]
            if current != reference:
                raise AssertionError(f"{method} uses different strata: {current} vs {reference}")
        return f"{len(reference)} strata identical across 5 methods"


@check("sampler", "the seed moves only the random-exploration slice")
def _sampler_seed_scope(args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        scores = _write_synthetic(Path(directory))
        first = _subsets(scores, methods=("gap_rank",), seed=2026)["gap_rank"]
        again = _subsets(scores, methods=("gap_rank",), seed=2026)["gap_rank"]
        other = _subsets(scores, methods=("gap_rank",), seed=7)["gap_rank"]
        if first["names"] != again["names"]:
            raise AssertionError("the same seed produced a different subset")
        if first["score_selected"] != other["score_selected"]:
            raise AssertionError("the score-selected part changed with the seed")
        if first["random_explored"] == other["random_explored"]:
            raise AssertionError("the random-exploration part did not change with the seed")
        moved = len(set(first["random_explored"]) ^ set(other["random_explored"])) // 2
        return f"score part identical, {moved}/{len(first['random_explored'])} explored names moved"


@check("sampler", "a tampered subset file fails to load")
def _sampler_tamper(args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        scores = _write_synthetic(Path(directory))
        payload = _subsets(scores, methods=("gap_rank",))["gap_rank"]
        path = Path(directory) / "subset.json"
        payload["names"][0] = payload["names"][0].replace(".wav", "x.wav")
        path.write_text(json.dumps(payload), encoding="utf-8")
        try:
            load_subset(path, scores)
        except ValueError as error:
            return f"rejected with: {str(error)[:70]}"
        raise AssertionError("a modified subset file was accepted")


@check("sampler", "gap_rank and hard mining select measurably different sets")
def _sampler_decorrelation(args: argparse.Namespace) -> str:
    with tempfile.TemporaryDirectory() as directory:
        scores = _write_synthetic(Path(directory))
        subsets = _subsets(scores)
        hard = set(subsets["hard"]["score_selected"])
        overlaps = {
            method: len(hard & set(subsets[method]["score_selected"])) / len(hard)
            for method in ("gap_db", "gap_rank", "gap_rank_2d")
        }
        if overlaps["gap_rank"] >= overlaps["gap_db"]:
            raise AssertionError(
                f"gap_rank should decorrelate from hard mining more than gap_db: {overlaps}"
            )
        return " ".join(f"{key}∩hard={value:.3f}" for key, value in overlaps.items())


# --------------------------------------------------------------------------- #
# training plumbing
# --------------------------------------------------------------------------- #


@check("training", "the step -> example map is deterministic and covers each pass")
def _training_order(args: argparse.Namespace) -> str:
    indices = list(range(1000))
    order = training_order(indices, 2500, seed=2026)
    if order != training_order(indices, 2500, seed=2026):
        raise AssertionError("training_order is not deterministic")
    if len(order) != 2500:
        raise AssertionError(f"expected 2500 steps, got {len(order)}")
    if sorted(order[:1000]) != indices or sorted(order[1000:2000]) != indices:
        raise AssertionError("a full pass is not a permutation of the subset")
    if order[:1000] == order[1000:2000]:
        raise AssertionError("consecutive passes used the same permutation")
    if training_order(indices, 2500, seed=7)[:1000] == order[:1000]:
        raise AssertionError("different seeds produced the same first pass")
    return "2 full passes verified as permutations, seeds separate"


@check("training", "resuming replays the exact remaining step sequence")
def _training_resume_order(args: argparse.Namespace) -> str:
    indices = list(range(137))
    order = training_order(indices, 400, seed=11)
    for cut in (0, 1, 137, 200, 399):
        if order[cut:] != training_order(indices, 400, seed=11)[cut:]:
            raise AssertionError(f"tail from step {cut} is not reproducible")
    return "verified at cuts 0, 1, 137, 200 and 399"


@check("training", "the CV monitor subset does not depend on the arm seed")
def _training_monitor(args: argparse.Namespace) -> str:
    if args.quick:
        return "skipped (--quick)"
    dataset = LRS2MixDataset(args.data_root, "cv", segment_samples=32000)
    first = build_monitor_indices(dataset, 1000)
    again = build_monitor_indices(dataset, 1000)
    if first != again or len(set(first)) != 1000:
        raise AssertionError("the monitor subset is not a stable 1000-example draw")
    digest = sha256_text("\n".join(dataset.names[i] for i in first) + "\n")
    return f"1000 CV examples, sha256={digest[:16]}…"


@check("training", "subset names resolve to the right dataset positions")
def _training_resolve(args: argparse.Namespace) -> str:
    if args.quick:
        return "skipped (--quick)"
    dataset = LRS2MixDataset(args.data_root, "tr", segment_samples=32000)
    sample = [dataset.names[i] for i in (0, 5, 19999)]
    positions = resolve_subset_indices(dataset, sample)
    if positions != [0, 5, 19999]:
        raise AssertionError(f"resolved to {positions}")
    try:
        resolve_subset_indices(dataset, ["not-a-real-file.wav"])
    except ValueError:
        return "positions correct and unknown names rejected"
    raise AssertionError("an unknown file name was accepted")


# --------------------------------------------------------------------------- #
# real checkpoints
# --------------------------------------------------------------------------- #


@check("model", "rebuilding the pruned model twice gives identical weights")
def _model_deterministic_build(args: argparse.Namespace) -> str:
    if args.quick:
        return "skipped (--quick)"
    first, _ = build_inherited_pruned_model("afrcnn12", args.baseline_checkpoint, args.masks)
    second, _ = build_inherited_pruned_model("afrcnn12", args.baseline_checkpoint, args.masks)
    left, right = first.state_dict(), second.state_dict()
    if left.keys() != right.keys():
        raise AssertionError("the two rebuilds have different parameter names")
    worst = max(float((left[key] - right[key]).abs().max()) for key in left)
    if worst != 0.0:
        raise AssertionError(f"rebuilds differ by {worst:g}")
    return f"{len(left)} tensors identical bit for bit"


@check("model", "batched scoring matches batch-size-1 scoring on real audio")
def _model_batch_invariance(args: argparse.Namespace) -> str:
    if args.quick:
        return "skipped (--quick)"
    device = torch.device(args.device)
    dataset = LRS2MixDataset(args.data_root, "tr", segment_samples=None)
    count = args.batch_check_samples
    mixtures = torch.stack([dataset[i][0] for i in range(count)]).to(device)
    sources = torch.stack([dataset[i][1] for i in range(count)]).to(device)

    dense = build_original_model("afrcnn12")
    dense.load_state_dict(
        torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=True)["state_dict"],
        strict=True,
    )
    pruned, _ = build_inherited_pruned_model("afrcnn12", args.baseline_checkpoint, args.masks)

    def deviation_of(model, tf32: bool) -> float:
        set_tf32(tf32)
        with torch.inference_mode():
            batched = pit_best_si_sdr(model(mixtures), sources)
            single = torch.stack(
                [
                    pit_best_si_sdr(model(mixtures[i : i + 1]).squeeze(0), sources[i])
                    for i in range(count)
                ]
            )
        return float((batched - single).abs().max())

    report = []
    try:
        for label, model in (("dense", dense), ("pruned", pruned)):
            model = model.to(device).eval()
            strict = deviation_of(model, False)
            loose = deviation_of(model, True)
            if strict > BATCH_TOLERANCE:
                raise AssertionError(
                    f"{label}: with TF32 off, batching still moved the score by {strict:g} dB "
                    f"(tolerance {BATCH_TOLERANCE:g})"
                )
            report.append(f"{label} tf32-off={strict:.1e} (tf32-on={loose:.1e})")
            model.to("cpu")
    finally:
        set_tf32(False)
    return f"{count} real mixtures, max|Δ| dB: " + ", ".join(report)


@check("model", "TDANet is refused at batch > 1 (upstream attention bug)")
def _model_tdanet_batch_guard(args: argparse.Namespace) -> str:
    """The released TDANet mixes information across the batch.

    `MultiHeadAttention` builds `nn.MultiheadAttention` with the default
    batch_first=False and then hands it `[batch, time, channels]`, so torch
    reads the batch axis as the sequence axis. At batch 1 the softmax spans a
    single element and the attention degenerates; at batch 4 each utterance
    attends to the other three. Every TDANet number in this repository is
    produced at batch 1, and the scorer must refuse anything else rather than
    silently returning batch-size-dependent scores.
    """

    from .score_lrs2 import SUPPORTED_MODELS

    if "tdanet" not in SUPPORTED_MODELS:
        raise AssertionError("tdanet is no longer a supported scoring model")
    source = (Path(__file__).parent / "score_lrs2.py").read_text(encoding="utf-8")
    if 'args.model == "tdanet" and args.batch_size != 1' not in source:
        raise AssertionError("the TDANet batch guard is missing from score_lrs2")
    return "guard present; TDANet scoring is pinned to batch 1"


@check("model", "the frozen inputs still match their recorded sha256")
def _model_hashes(args: argparse.Namespace) -> str:
    if args.quick:
        return "skipped (--quick)"
    from .common import sha256_file

    expected = {
        args.baseline_checkpoint: args.expect_baseline_sha256,
        args.masks: args.expect_masks_sha256,
    }
    reported = []
    for path, want in expected.items():
        digest = sha256_file(path)
        if want and digest != want:
            raise AssertionError(f"{path} is {digest}, expected {want}")
        reported.append(f"{Path(path).name}={digest[:16]}…")
    return " ".join(reported)


# --------------------------------------------------------------------------- #
# end-to-end resume
# --------------------------------------------------------------------------- #


@check("resume", "an interrupted run replays the same steps as an uninterrupted one")
def _resume_end_to_end(args: argparse.Namespace) -> str:
    if args.quick or not args.resume_test:
        return "skipped (pass --resume-test to run)"
    steps, cut = args.resume_steps, args.resume_steps // 2
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        common = [
            sys.executable,
            "-m",
            "reproduction.data_pruning.train_data_pruned",
            "--data-root",
            args.data_root,
            "--baseline-checkpoint",
            args.baseline_checkpoint,
            "--masks",
            args.masks,
            "--device",
            args.device,
            "--total-updates",
            str(steps),
            "--validate-every",
            "0",
            "--snapshot-at",
            "",
            "--checkpoint-every",
            str(steps),
            "--log-every",
            str(steps),
            "--num-workers",
            "2",
            "--skip-length-check",
        ]
        subprocess.run(
            common + ["--arm", "resume-ref", "--output-dir", str(root / "reference")],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            common
            + ["--arm", "resume-cut", "--output-dir", str(root / "split"), "--stop-after", str(cut)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            common + ["--arm", "resume-cut", "--output-dir", str(root / "split"), "--resume"],
            check=True,
            capture_output=True,
        )

        def loss_trace(run: Path) -> list[float]:
            with (run / "training.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            if [int(row["global_step"]) for row in rows] != list(range(1, steps + 1)):
                raise AssertionError(f"{run.name}: training.csv does not cover steps 1..{steps}")
            return [float(row["loss"]) for row in rows]

        reference_losses, split_losses = loss_trace(root / "reference"), loss_trace(root / "split")
        worst_loss = max(abs(a - b) for a, b in zip(reference_losses, split_losses))
        if worst_loss > RESUME_LOSS_TOLERANCE:
            raise AssertionError(
                f"step losses diverge by {worst_loss:g} dB after resuming, which means the "
                "resumed run is not replaying the same examples"
            )

        reference = torch.load(root / "reference" / "last.pt", map_location="cpu", weights_only=True)
        split = torch.load(root / "split" / "last.pt", map_location="cpu", weights_only=True)
        if reference["global_step"] != steps or split["global_step"] != steps:
            raise AssertionError(
                f"steps {reference['global_step']} and {split['global_step']}, expected {steps}"
            )
        worst_weight = max(
            float((reference["state_dict"][key] - split["state_dict"][key]).abs().max())
            for key in reference["state_dict"]
        )
        if worst_weight > RESUME_WEIGHT_TOLERANCE:
            raise AssertionError(
                f"resumed weights differ by {worst_weight:g} (tolerance {RESUME_WEIGHT_TOLERANCE:g})"
            )
        return (
            f"{steps} steps, cut at {cut}, max|Δloss|={worst_loss:.2e} dB, "
            f"max|Δweight|={worst_weight:.2e}"
        )


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE)
    parser.add_argument("--masks", default=DEFAULT_MASKS)
    parser.add_argument(
        "--expect-baseline-sha256",
        default="6f9dc2c700b03ed38bf6070e0b0929269fa2f43d1b8b0239229724145c322da6",
    )
    parser.add_argument(
        "--expect-masks-sha256",
        default="13a774ee64c587ac7b7f9e82e2e37070c1ee6258a5aeb9206ffe4e2bce540433",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-check-samples", type=int, default=16)
    parser.add_argument("--resume-test", action="store_true")
    parser.add_argument("--resume-steps", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="schema/sampler/metrics only")
    parser.add_argument("--only", default=None, help="comma-separated group filter")
    args = parser.parse_args()

    groups = {item.strip() for item in args.only.split(",")} if args.only else None
    failures = 0
    current_group = None
    for group, name, function in _REGISTRY:
        if groups and group not in groups:
            continue
        if group != current_group:
            print(f"\n[{group}]")
            current_group = group
        started = time.monotonic()
        try:
            detail = function(args)
            status = "SKIP" if detail.startswith("skipped") else "PASS"
        except Exception as error:  # noqa: BLE001 - the report is the product
            detail = f"{type(error).__name__}: {error}"
            status = "FAIL"
            failures += 1
        print(f"  {status}  {name}\n        {detail}  ({time.monotonic() - started:.2f}s)")

    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
