"""Pre-flight validation of the Dense-Pruned data-selection hypotheses.

Scores a random subset of the LRS2-2Mix training split with up to four
A-FRCNN-12 checkpoints that all share one frozen starting point:

    dense       the unpruned baseline, used only as a scorer / teacher
    pruned@0    physically pruned from `dense` with the frozen masks, no fine-tuning
    pruned@1ep  the same structure after 20,000 optimizer updates
    pruned@Nep  the same structure after long fine-tuning (optional)

It answers the three questions that decide whether the plan in
`docs/数据剪枝与模型剪枝协同方案调研.md` is executable at all:

1. Is the Gap signal collinear with plain hard mining?  (§4.2)
2. How badly does the dense teacher memorize the training split?  (§5.2)
3. How fast does a statically computed score go stale?  (§5.3)

The script is read-only: it never writes a checkpoint and never trains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import subprocess
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from ..evaluate_original import best_si_sdr
from ..lrs2 import LRS2MixDataset
from ..original_models import build_original_model, parameter_count
from ..train_pruned_original import build_inherited_pruned_model

SELECTED_FRACTION = 0.375
"""7,500 score-selected out of 20,000, matching the S2/S3/S3' construction."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def percentile_rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for position, index in enumerate(order):
        ranks[index] = position / (len(values) - 1)
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    x, y = percentile_rank(left), percentile_rank(right)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    covariance = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    spread = (
        sum((a - mean_x) ** 2 for a in x) * sum((b - mean_y) ** 2 for b in y)
    ) ** 0.5
    return covariance / spread


def top_indices(scores: list[float], fraction: float) -> set[int]:
    count = int(len(scores) * fraction)
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:count])


def load_pruned(
    baseline: str, masks: str, checkpoint: str | None, device: torch.device
) -> nn.Module:
    model, _ = build_inherited_pruned_model("afrcnn12", baseline, masks)
    if checkpoint is not None:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = parameter_count(model)
        if int(payload.get("parameters", -1)) != expected:
            raise ValueError(
                f"{checkpoint} has a different pruned structure than the frozen masks"
            )
        model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def score_split(
    model: nn.Module,
    dataset: LRS2MixDataset,
    indices: list[int],
    device: torch.device,
) -> list[float]:
    scores: list[float] = []
    with torch.inference_mode():
        for index in indices:
            mixture, sources, _ = dataset[index]
            mixture = mixture.to(device)
            sources = sources.to(device)
            estimate = model(mixture.unsqueeze(0)).squeeze(0)
            scores.append(float(best_si_sdr(estimate, sources).cpu()))
    return scores


def mixture_baseline(
    dataset: LRS2MixDataset, indices: list[int], device: torch.device
) -> list[float]:
    """PIT-averaged SI-SDR of the mixture itself, i.e. the SI-SDRi reference."""

    values: list[float] = []
    with torch.inference_mode():
        for index in indices:
            mixture, sources, _ = dataset[index]
            mixture = mixture.to(device)
            sources = sources.to(device)
            values.append(float(best_si_sdr(mixture.expand_as(sources), sources).cpu()))
    return values


def describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "sd": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint",
        default="experiments/original_afrcnn12_lrs2_train/best.pt",
    )
    parser.add_argument(
        "--masks",
        default="experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026/masks.pt",
    )
    parser.add_argument(
        "--pruned-1ep",
        default="experiments/seprune_budgeted_afrcnn12_lrs2_finetune/epoch1.pt",
        help="checkpoint at 20,000 updates; used for the score-staleness gate",
    )
    parser.add_argument(
        "--pruned-long",
        default="experiments/seprune_budgeted_afrcnn12_lrs2_finetune/best.pt",
        help="optional long fine-tuning checkpoint; pass an empty string to skip",
    )
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="experiments/data_pruning_preflight")
    args = parser.parse_args()

    device = torch.device(args.device)
    started = time.monotonic()

    train = LRS2MixDataset(args.data_root, "tr", segment_samples=None)
    test = LRS2MixDataset(args.data_root, "tt", segment_samples=None)
    rng = random.Random(args.seed)
    train_indices = rng.sample(range(len(train)), args.samples)
    test_indices = rng.sample(range(len(test)), args.samples)

    dense = build_original_model("afrcnn12")
    dense.load_state_dict(
        torch.load(args.baseline_checkpoint, map_location="cpu", weights_only=True)[
            "state_dict"
        ],
        strict=True,
    )
    dense = dense.to(device).eval()
    pruned0 = load_pruned(args.baseline_checkpoint, args.masks, None, device)
    pruned1 = load_pruned(args.baseline_checkpoint, args.masks, args.pruned_1ep, device)
    pruned_long = (
        load_pruned(args.baseline_checkpoint, args.masks, args.pruned_long, device)
        if args.pruned_long
        else None
    )

    q_dense = score_split(dense, train, train_indices, device)
    q_dense_tt = score_split(dense, test, test_indices, device)
    q_p0 = score_split(pruned0, train, train_indices, device)
    q_p1 = score_split(pruned1, train, train_indices, device)
    q_plong = (
        score_split(pruned_long, train, train_indices, device) if pruned_long else None
    )
    baseline_si_sdr = mixture_baseline(train, train_indices, device)

    pct_dense = percentile_rank(q_dense)
    pct_pruned = percentile_rank(q_p0)
    hard = [-value for value in q_p0]
    gap_db = [a - b for a, b in zip(q_dense, q_p0)]
    gap_rank = [a - b for a, b in zip(pct_dense, pct_pruned)]
    gap_1ep = [a - b for a, b in zip(q_dense, q_p1)]
    improvement = [b - a for a, b in zip(q_p0, q_p1)]
    snr = [
        abs(float(train.names[index][:-4].split("_")[2])) for index in train_indices
    ]

    selections = {
        "S2_hard": top_indices(hard, SELECTED_FRACTION),
        "S3_gap_db": top_indices(gap_db, SELECTED_FRACTION),
        "S3p_gap_rank": top_indices(gap_rank, SELECTED_FRACTION),
    }
    names = list(selections)

    def mean_over(subset: set[int], values: list[float]) -> float:
        return sum(values[i] for i in subset) / len(subset)

    report: dict[str, object] = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit(),
        "device": str(device),
        "samples": args.samples,
        "seed": args.seed,
        "selected_fraction": SELECTED_FRACTION,
        "checkpoints": {
            "dense": {
                "path": args.baseline_checkpoint,
                "sha256": sha256(Path(args.baseline_checkpoint)),
            },
            "masks": {"path": args.masks, "sha256": sha256(Path(args.masks))},
            "pruned_1ep": {
                "path": args.pruned_1ep,
                "sha256": sha256(Path(args.pruned_1ep)),
            },
            "pruned_long": (
                {"path": args.pruned_long, "sha256": sha256(Path(args.pruned_long))}
                if args.pruned_long
                else None
            ),
        },
        "pruned_parameters": parameter_count(pruned0),
        "distributions": {
            "q_dense_tr": describe(q_dense),
            "q_dense_tt": describe(q_dense_tt),
            "q_pruned_at_init": describe(q_p0),
            "q_pruned_at_1ep": describe(q_p1),
            "gap_db_at_init": describe(gap_db),
            "gap_db_at_1ep": describe(gap_1ep),
            "mixture_si_sdr_baseline": describe(baseline_si_sdr),
            "improvement_over_first_epoch": describe(improvement),
        },
        "teacher_memorization_db": (sum(q_dense) / len(q_dense))
        - (sum(q_dense_tt) / len(q_dense_tt)),
        "collinearity": {
            key: {
                "rho_with_hard_mining": spearman(score, hard),
                "rho_with_q_dense": spearman(score, q_dense),
                "rho_with_gap_at_1ep": spearman(score, gap_1ep),
                "rho_with_first_epoch_improvement": spearman(score, improvement),
            }
            for key, score in (
                ("hard", hard),
                ("gap_db", gap_db),
                ("gap_rank", gap_rank),
            )
        },
        "subset_overlap": {
            f"{a}|{b}": len(selections[a] & selections[b]) / len(selections[a])
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        },
        "random_overlap_reference": SELECTED_FRACTION,
        "selected_subset_means": {
            key: {
                "q_dense": mean_over(subset, q_dense),
                "q_pruned_at_init": mean_over(subset, q_p0),
                "snr_abs": mean_over(subset, snr),
            }
            for key, subset in selections.items()
        }
        | {
            "full_set": {
                "q_dense": sum(q_dense) / len(q_dense),
                "q_pruned_at_init": sum(q_p0) / len(q_p0),
                "snr_abs": sum(snr) / len(snr),
            }
        },
        "score_staleness": {
            "rho_gap_db_init_vs_1ep": spearman(gap_db, gap_1ep),
            "overlap_gap_db_init_vs_1ep": len(
                top_indices(gap_db, SELECTED_FRACTION)
                & top_indices(gap_1ep, SELECTED_FRACTION)
            )
            / len(top_indices(gap_db, SELECTED_FRACTION)),
            "rho_gap_rank_init_vs_1ep": spearman(gap_rank, gap_1ep),
        },
        "elapsed_seconds": time.monotonic() - started,
    }

    if q_plong is not None:
        gap_long = [a - b for a, b in zip(q_dense, q_plong)]
        report["distributions"]["q_pruned_long"] = describe(q_plong)
        report["distributions"]["gap_db_long"] = describe(gap_long)
        report["score_staleness"]["rho_gap_db_init_vs_long"] = spearman(gap_db, gap_long)
        report["score_staleness"]["overlap_gap_db_init_vs_long"] = len(
            top_indices(gap_db, SELECTED_FRACTION)
            & top_indices(gap_long, SELECTED_FRACTION)
        ) / len(top_indices(gap_db, SELECTED_FRACTION))
        report["score_staleness"]["rho_gap_rank_init_vs_long"] = spearman(
            gap_rank, gap_long
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    per_sample = output_dir / "preflight_scores.json"
    per_sample.write_text(
        json.dumps(
            [
                {
                    "name": train.names[index],
                    "q_dense": q_dense[position],
                    "q_pruned_at_init": q_p0[position],
                    "q_pruned_at_1ep": q_p1[position],
                    "mixture_si_sdr": baseline_si_sdr[position],
                    "gap_db": gap_db[position],
                    "gap_rank": gap_rank[position],
                    "snr_abs": snr[position],
                }
                for position, index in enumerate(train_indices)
            ],
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
