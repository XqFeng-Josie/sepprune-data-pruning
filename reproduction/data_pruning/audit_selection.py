"""Audit the subsets before spending GPU time on them.

Reports the things that decide whether an arm is a fair comparison rather than
an accident: how much each score-based arm overlaps plain hard mining, whether
the |SNR| quotas really are identical, how far speaker coverage collapses, and
how the four regions of the (pct_dense, gap_rank) plane are represented.

    .venv/bin/python -m reproduction.data_pruning.audit_selection \\
        --scores experiments/data_pruning_scores/afrcnn12_tr_init \\
        --subsets experiments/data_pruning_subsets

Nothing here can fail an experiment on its own; it produces the numbers that
the plan requires to be reported alongside the results.
"""

from __future__ import annotations

import argparse
import statistics
from itertools import combinations
from pathlib import Path

from .common import write_json
from .samplers import load_subset
from .score_schema import ScoreSet, load_score_set, percentile_ranks

QUADRANTS = ("A_learnable_sensitive", "B_easy_robust", "C_unlearnable", "D_hard_sensitive")

STALENESS_FLOOR = 0.30
"""Below this the frozen score no longer orders the samples it is meant to order.

Calibrated in the plan's §5.3: over the 20,000-update window the measured
rank correlation is 0.43-0.48, and over 800,000 updates it falls to ~0.25. A
score set that cannot clear 0.30 against the 20,000-update checkpoint cannot
support a static arm at all, and the study should go straight to the dynamic
refresh variant.
"""


def spearman(left: list[float], right: list[float]) -> float:
    ranks = [percentile_ranks(values, [str(i) for i in range(len(values))]) for values in (left, right)]
    means = [statistics.fmean(column) for column in ranks]
    covariance = sum(
        (a - means[0]) * (b - means[1]) for a, b in zip(ranks[0], ranks[1])
    )
    spread = (
        sum((a - means[0]) ** 2 for a in ranks[0]) * sum((b - means[1]) ** 2 for b in ranks[1])
    ) ** 0.5
    return covariance / spread


def teacher_bias(scores: ScoreSet, holdout: ScoreSet) -> dict:
    """Compare the dense teacher on the split it trained on and on a held-out one.

    The dense model has seen every training mixture, so `Q_dense` there is part
    memorisation. Quantifying the gap is what stops `gap_rank` from being read
    as an unbiased reducible-loss estimate.
    """

    train = [row["q_dense_sisdr"] for row in scores.rows]
    held = [row["q_dense_sisdr"] for row in holdout.rows]
    return {
        "holdout_split": holdout.meta["split"],
        "train_mean": statistics.fmean(train),
        "holdout_mean": statistics.fmean(held),
        "memorization_gap_db": statistics.fmean(train) - statistics.fmean(held),
        "train_sd": statistics.pstdev(train),
        "holdout_sd": statistics.pstdev(held),
        "sd_ratio": statistics.pstdev(train) / statistics.pstdev(held),
    }


def staleness(scores: ScoreSet, later: ScoreSet) -> dict:
    """Rank correlation between the frozen score and the same score re-measured later."""

    if [row["name"] for row in scores.rows] != [row["name"] for row in later.rows]:
        raise ValueError("the two score sets cover different file names")
    correlation = spearman(
        [row["gap_rank"] for row in scores.rows], [row["gap_rank"] for row in later.rows]
    )
    return {
        "later_label": later.meta["pruned_label"],
        "rho_gap_rank": correlation,
        "rho_gap_db": spearman(
            [row["gap_db"] for row in scores.rows], [row["gap_db"] for row in later.rows]
        ),
        "floor": STALENESS_FLOOR,
        "passes": correlation >= STALENESS_FLOOR,
    }


def quadrant_of(row: dict, dense_split: float, gap_split: float) -> str:
    high_dense = row["pct_dense"] >= dense_split
    high_gap = row["gap_rank"] >= gap_split
    if high_dense and high_gap:
        return "A_learnable_sensitive"
    if high_dense:
        return "B_easy_robust"
    if high_gap:
        return "D_hard_sensitive"
    return "C_unlearnable"


def describe_subset(payload: dict, scores: ScoreSet, splits: tuple[float, float]) -> dict:
    by_name = scores.by_name()
    rows = [by_name[name] for name in payload["names"]]
    speakers = {row["spk1"] for row in rows} | {row["spk2"] for row in rows}
    population_speakers = {row["spk1"] for row in scores.rows} | {
        row["spk2"] for row in scores.rows
    }
    counts = {key: 0 for key in QUADRANTS}
    for row in rows:
        counts[quadrant_of(row, *splits)] += 1
    return {
        "method": payload["method"],
        "seed": payload["seed"],
        "kept": len(rows),
        "score_selected": payload["counts"]["score_selected"],
        "random_explored": payload["counts"]["random_explored"],
        "keep_quota_per_stratum": [item["keep_quota"] for item in payload["strata"]],
        "stratum_sizes": [item["size"] for item in payload["strata"]],
        "snr_boundaries": [
            [item["snr_abs_min"], item["snr_abs_max"]] for item in payload["strata"]
        ],
        "mean_q_dense": statistics.fmean(row["q_dense_sisdr"] for row in rows),
        "mean_q_pruned": statistics.fmean(row["q_pruned_sisdr"] for row in rows),
        "mean_gap_db": statistics.fmean(row["gap_db"] for row in rows),
        "mean_snr_abs": statistics.fmean(row["snr_abs"] for row in rows),
        "distinct_speakers": len(speakers),
        "speaker_coverage": len(speakers) / len(population_speakers),
        "quadrant_share": {key: value / len(rows) for key, value in counts.items()},
    }


def population_summary(scores: ScoreSet, splits: tuple[float, float]) -> dict:
    rows = scores.rows
    counts = {key: 0 for key in QUADRANTS}
    for row in rows:
        counts[quadrant_of(row, *splits)] += 1
    speakers = {row["spk1"] for row in rows} | {row["spk2"] for row in rows}
    return {
        "count": len(rows),
        "population_complete": scores.meta.get("population_complete", True),
        "mean_q_dense": statistics.fmean(row["q_dense_sisdr"] for row in rows),
        "mean_q_pruned": statistics.fmean(row["q_pruned_sisdr"] for row in rows),
        "mean_gap_db": statistics.fmean(row["gap_db"] for row in rows),
        "sd_q_dense": statistics.pstdev(row["q_dense_sisdr"] for row in rows),
        "sd_q_pruned": statistics.pstdev(row["q_pruned_sisdr"] for row in rows),
        "mean_snr_abs": statistics.fmean(row["snr_abs"] for row in rows),
        "distinct_speakers": len(speakers),
        "quadrant_share": {key: value / len(rows) for key, value in counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--subsets", required=True, help="directory of subset JSON files")
    parser.add_argument(
        "--holdout-scores",
        default=None,
        help="a dense score set on cv, for the teacher-memorisation check",
    )
    parser.add_argument(
        "--staleness-scores",
        default=None,
        help="the same split rescored with a later pruned checkpoint, for the static gate",
    )
    parser.add_argument("--output", default=None, help="defaults to <subsets>/audit.json")
    args = parser.parse_args()

    scores = load_score_set(args.scores)
    dense_split = statistics.median(row["pct_dense"] for row in scores.rows)
    gap_split = statistics.median(row["gap_rank"] for row in scores.rows)
    splits = (dense_split, gap_split)

    paths = sorted(Path(args.subsets).glob("*.json"))
    paths = [path for path in paths if path.name != "audit.json"]
    if not paths:
        raise SystemExit(f"No subset files found in {args.subsets}")
    payloads = {path.stem: load_subset(path, scores) for path in paths}

    population = population_summary(scores, splits)
    subsets = {
        name: describe_subset(payload, scores, splits) for name, payload in payloads.items()
    }

    reference = None
    quota_mismatch = []
    for name, summary in subsets.items():
        if summary["method"] == "full":
            continue
        signature = (summary["stratum_sizes"], summary["snr_boundaries"], summary["keep_quota_per_stratum"])
        if reference is None:
            reference = (name, signature)
        elif signature != reference[1]:
            quota_mismatch.append(f"{name} differs from {reference[0]}")

    overlaps = {}
    for left, right in combinations(sorted(payloads), 2):
        first, second = payloads[left], payloads[right]
        if not first["score_selected"] or not second["score_selected"]:
            continue
        if first["seed"] != second["seed"]:
            continue
        shared = set(first["score_selected"]) & set(second["score_selected"])
        overlaps[f"{left} ∩ {right}"] = len(shared) / len(first["score_selected"])

    report = {
        "score_set": {
            "stem": str(Path(args.scores).resolve()),
            "rows_sha256": scores.meta["rows_sha256"],
            "model": scores.meta["model"],
            "pruned_label": scores.meta["pruned_label"],
        },
        "plane_splits": {"pct_dense_median": dense_split, "gap_rank_median": gap_split},
        "population": population,
        "subsets": subsets,
        "score_selected_overlap": overlaps,
        "random_overlap_reference": {
            name: summary["score_selected"] / population["count"]
            for name, summary in subsets.items()
            if summary["score_selected"]
        },
        "teacher_bias": (
            teacher_bias(scores, load_score_set(args.holdout_scores))
            if args.holdout_scores
            else None
        ),
        "staleness": (
            staleness(scores, load_score_set(args.staleness_scores))
            if args.staleness_scores
            else None
        ),
        "warnings": (
            ([] if population["population_complete"] else ["score set was built with --limit"])
            + [f"stratum quotas are not identical: {item}" for item in quota_mismatch]
        ),
    }
    if report["staleness"] and not report["staleness"]["passes"]:
        report["warnings"].append(
            f"static assumption gate FAILED: rho(gap_rank init, "
            f"{report['staleness']['later_label']}) = {report['staleness']['rho_gap_rank']:.3f} "
            f"< {STALENESS_FLOOR}; a static subset cannot carry this study"
        )

    output = Path(args.output) if args.output else Path(args.subsets) / "audit.json"
    write_json(output, report)

    print(f"population: {population['count']} mixtures, "
          f"Q_dense {population['mean_q_dense']:.3f}±{population['sd_q_dense']:.3f}, "
          f"Q_pruned {population['mean_q_pruned']:.3f}±{population['sd_q_pruned']:.3f}, "
          f"{population['distinct_speakers']} speakers")
    header = f"{'subset':<26}{'kept':>7}{'Qdense':>9}{'Qpruned':>9}{'|SNR|':>7}{'spk%':>7}{'A':>7}{'C':>7}"
    print("\n" + header)
    print("-" * len(header))
    for name in sorted(subsets):
        summary = subsets[name]
        print(
            f"{name:<26}{summary['kept']:>7}{summary['mean_q_dense']:>9.3f}"
            f"{summary['mean_q_pruned']:>9.3f}{summary['mean_snr_abs']:>7.2f}"
            f"{summary['speaker_coverage'] * 100:>6.1f}%"
            f"{summary['quadrant_share']['A_learnable_sensitive'] * 100:>6.1f}%"
            f"{summary['quadrant_share']['C_unlearnable'] * 100:>6.1f}%"
        )
    if overlaps:
        print("\nscore-selected overlap (same seed):")
        for key, value in sorted(overlaps.items(), key=lambda item: -item[1]):
            print(f"  {value:.3f}  {key}")
    if report["teacher_bias"]:
        bias = report["teacher_bias"]
        print(
            f"\nteacher memorisation: Q_dense(tr)={bias['train_mean']:.3f} vs "
            f"Q_dense({bias['holdout_split']})={bias['holdout_mean']:.3f} "
            f"-> gap {bias['memorization_gap_db']:.3f} dB, sd ratio {bias['sd_ratio']:.2f}"
        )
    if report["staleness"]:
        stale = report["staleness"]
        print(
            f"\nstatic assumption gate: rho(gap_rank@init, gap_rank@{stale['later_label']}) = "
            f"{stale['rho_gap_rank']:.3f} (floor {stale['floor']}) -> "
            f"{'PASS' if stale['passes'] else 'FAIL'}"
        )
    for warning in report["warnings"]:
        print(f"\nWARNING: {warning}")
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
