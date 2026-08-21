"""Deterministic training-subset construction for the static (stage A) arms.

Every arm keeps exactly the same number of mixtures, drawn under exactly the
same |SNR| stratum quotas, so the only thing that differs between S1/S2/S3/S3'
is the ranking function. Each subset is `score_fraction` of the budget taken by
score and the remainder drawn uniformly from that stratum's *unselected* pool,
which keeps a fixed slice of random exploration in every score-based arm.

    S0  full        every training mixture
    S1  random      no ranking at all, the data-volume baseline
    S2  hard        rank by pruned loss, the hard-mining baseline
    S3  gap_db      rank by the raw dB difference (ablation: collinear with S2)
    S3' gap_rank    rank by the percentile-rank difference (main method)
    S4  gap_rank_2d gap_rank ranked inside pct_dense sub-strata

Subsets are reproducible from `(score set, method, seed)` alone and are stored
with the sha256 of both the name list and the score file they came from.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

from .common import git_commit, sha256_text, utc_now, seeded_rng, write_json, read_json
from .score_schema import ScoreSet, load_score_set

__all__ = [
    "SUBSET_SCHEMA_VERSION",
    "SCORE_FUNCTIONS",
    "split_quota",
    "build_subset",
    "load_subset",
]

SUBSET_SCHEMA_VERSION = "data-pruning-subset/1"

ScoreFunction = Callable[[dict], float]

SCORE_FUNCTIONS: dict[str, ScoreFunction | None] = {
    "full": None,
    "random": None,
    "hard": lambda row: -row["q_pruned_sisdr"],
    "gap_db": lambda row: row["gap_db"],
    "gap_rank": lambda row: row["gap_rank"],
    "gap_rank_2d": lambda row: row["gap_rank"],
}

TWO_DIMENSIONAL = frozenset({"gap_rank_2d"})


def split_quota(total: int, parts: int) -> list[int]:
    """Split `total` into `parts` integers that sum exactly to `total`.

    The remainder goes to the leading parts, so the result depends only on the
    two integers and never on data order.
    """

    if parts < 1:
        raise ValueError("parts must be positive")
    if total < 0:
        raise ValueError("total must be non-negative")
    base, remainder = divmod(total, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def _contiguous_groups(ordered: Sequence[dict], parts: int) -> list[list[dict]]:
    sizes = split_quota(len(ordered), parts)
    groups: list[list[dict]] = []
    start = 0
    for size in sizes:
        groups.append(list(ordered[start : start + size]))
        start += size
    if start != len(ordered):
        raise AssertionError("stratification lost rows")
    return groups


def _rank_by_score(rows: Sequence[dict], score: ScoreFunction, quota: int) -> list[dict]:
    if quota > len(rows):
        raise ValueError(f"Cannot take {quota} rows from a pool of {len(rows)}")
    ordered = sorted(rows, key=lambda row: (-score(row), row["name"]))
    return ordered[:quota]


def build_subset(
    scores: ScoreSet,
    *,
    method: str,
    seed: int,
    keep: int,
    score_fraction: float,
    snr_strata: int,
    dense_strata: int,
) -> dict:
    """Construct one arm's subset. Pure function of its arguments."""

    if method not in SCORE_FUNCTIONS:
        raise ValueError(f"Unknown method {method!r}; choose from {sorted(SCORE_FUNCTIONS)}")
    rows = sorted(scores.rows, key=lambda row: row["name"])
    total = len(rows)

    if method == "full":
        keep, score_fraction = total, 0.0
    if not 0 < keep <= total:
        raise ValueError(f"keep must be in (0, {total}], got {keep}")
    if not 0.0 <= score_fraction <= 1.0:
        raise ValueError(f"score_fraction must be in [0, 1], got {score_fraction}")

    score_function = SCORE_FUNCTIONS[method]
    if score_function is None and score_fraction != 0.0:
        raise ValueError(f"Method {method!r} has no ranking; score_fraction must be 0")

    score_total = round(keep * score_fraction)
    keep_quota = split_quota(keep, snr_strata)
    score_quota = split_quota(score_total, snr_strata)

    by_snr = _contiguous_groups(
        sorted(rows, key=lambda row: (row["snr_abs"], row["name"])), snr_strata
    )

    selected: list[dict] = []
    explored: list[dict] = []
    stratum_report: list[dict] = []
    for index, stratum in enumerate(by_snr):
        take_by_score = score_quota[index]
        take_at_random = keep_quota[index] - take_by_score
        if take_at_random < 0:
            raise ValueError("score quota exceeds the keep quota in a stratum")
        if keep_quota[index] > len(stratum):
            raise ValueError(
                f"Stratum {index} holds {len(stratum)} rows but the quota is {keep_quota[index]}"
            )

        if score_function is None or take_by_score == 0:
            chosen: list[dict] = []
        elif method in TWO_DIMENSIONAL:
            by_dense = _contiguous_groups(
                sorted(stratum, key=lambda row: (row["pct_dense"], row["name"])), dense_strata
            )
            sub_quota = split_quota(take_by_score, dense_strata)
            chosen = [
                row
                for group, quota in zip(by_dense, sub_quota)
                for row in _rank_by_score(group, score_function, quota)
            ]
        else:
            chosen = _rank_by_score(stratum, score_function, take_by_score)

        chosen_names = {row["name"] for row in chosen}
        if len(chosen_names) != take_by_score:
            raise AssertionError("score selection produced duplicates")
        remainder = sorted(
            (row for row in stratum if row["name"] not in chosen_names),
            key=lambda row: row["name"],
        )
        rng = seeded_rng("subset", SUBSET_SCHEMA_VERSION, method, seed, index)
        drawn = rng.sample(remainder, take_at_random)

        selected.extend(chosen)
        explored.extend(drawn)
        stratum_report.append(
            {
                "index": index,
                "size": len(stratum),
                "snr_abs_min": stratum[0]["snr_abs"],
                "snr_abs_max": stratum[-1]["snr_abs"],
                "keep_quota": keep_quota[index],
                "score_quota": take_by_score,
                "random_quota": take_at_random,
            }
        )

    selected_names = sorted(row["name"] for row in selected)
    explored_names = sorted(row["name"] for row in explored)
    overlap = set(selected_names) & set(explored_names)
    if overlap:
        raise AssertionError(f"score and random parts overlap on {len(overlap)} names")
    names = sorted(selected_names + explored_names)
    if len(set(names)) != keep:
        raise AssertionError(f"expected {keep} unique names, produced {len(set(names))}")

    payload = {
        "schema_version": SUBSET_SCHEMA_VERSION,
        "method": method,
        "seed": seed,
        "keep": keep,
        "keep_fraction": keep / total,
        "score_fraction": score_fraction,
        "snr_strata": snr_strata,
        "dense_strata": dense_strata if method in TWO_DIMENSIONAL else None,
        "counts": {
            "score_selected": len(selected_names),
            "random_explored": len(explored_names),
            "total": len(names),
            "population": total,
        },
        "strata": stratum_report,
        "source_score_set": {
            "rows_sha256": scores.meta["rows_sha256"],
            "model": scores.meta["model"],
            "split": scores.meta["split"],
            "pruned_label": scores.meta["pruned_label"],
            "dense_checkpoint_sha256": scores.meta["dense_checkpoint"]["sha256"],
            "masks_sha256": scores.meta["masks"]["sha256"],
            "pruned_checkpoint_sha256": (
                scores.meta["pruned_checkpoint"]["sha256"]
                if scores.meta.get("pruned_checkpoint")
                else None
            ),
        },
        "git_commit": git_commit(),
        "built_at_utc": utc_now(),
        "score_selected": selected_names,
        "random_explored": explored_names,
        "names": names,
    }
    payload["names_sha256"] = sha256_text("\n".join(names) + "\n")
    return payload


def load_subset(path: str | Path, scores: ScoreSet | None = None) -> dict:
    """Read a subset file and re-verify every invariant it claims."""

    payload = read_json(path)
    if payload.get("schema_version") != SUBSET_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema {payload.get('schema_version')!r}, "
            f"expected {SUBSET_SCHEMA_VERSION!r}"
        )
    names = payload["names"]
    if names != sorted(names):
        raise ValueError(f"{path}: names must be sorted")
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: names contain duplicates")
    if len(names) != payload["counts"]["total"] or len(names) != payload["keep"]:
        raise ValueError(f"{path}: name count disagrees with the declared budget")
    if payload["names_sha256"] != sha256_text("\n".join(names) + "\n"):
        raise ValueError(f"{path}: names_sha256 does not match the stored list")

    selected, explored = payload["score_selected"], payload["random_explored"]
    if set(selected) & set(explored):
        raise ValueError(f"{path}: score and random parts overlap")
    if sorted(selected + explored) != names:
        raise ValueError(f"{path}: the two parts do not reconstruct the name list")
    if len(selected) != payload["counts"]["score_selected"]:
        raise ValueError(f"{path}: score_selected count disagrees with the list")

    if scores is not None:
        if payload["source_score_set"]["rows_sha256"] != scores.meta["rows_sha256"]:
            raise ValueError(f"{path} was built from a different score set")
        population = set(scores.names)
        missing = [name for name in names if name not in population]
        if missing:
            raise ValueError(f"{path}: {len(missing)} names are absent from the score set")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, help="stem or .jsonl of the score set")
    parser.add_argument(
        "--methods",
        default="full,random,hard,gap_db,gap_rank",
        help="comma-separated subset of " + ",".join(SCORE_FUNCTIONS),
    )
    parser.add_argument("--seeds", default="2026", help="comma-separated integer seeds")
    parser.add_argument("--keep", type=int, default=10000)
    parser.add_argument("--score-fraction", type=float, default=0.75)
    parser.add_argument("--snr-strata", type=int, default=4)
    parser.add_argument("--dense-strata", type=int, default=4)
    parser.add_argument("--output-dir", default="experiments/data_pruning_subsets")
    args = parser.parse_args()

    scores = load_score_set(args.scores)
    output_dir = Path(args.output_dir)
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = [method for method in methods if method not in SCORE_FUNCTIONS]
    if unknown:
        raise SystemExit(f"Unknown methods {unknown}; choose from {sorted(SCORE_FUNCTIONS)}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]

    for method in methods:
        for seed in seeds:
            fraction = 0.0 if SCORE_FUNCTIONS[method] is None else args.score_fraction
            payload = build_subset(
                scores,
                method=method,
                seed=seed,
                keep=args.keep,
                score_fraction=fraction,
                snr_strata=args.snr_strata,
                dense_strata=args.dense_strata,
            )
            path = output_dir / (
                "full.json" if method == "full" else f"{method}_seed{seed}.json"
            )
            write_json(path, payload)
            load_subset(path, scores)
            print(
                f"{method:<12} seed={seed} kept={payload['counts']['total']} "
                f"(score {payload['counts']['score_selected']} / "
                f"random {payload['counts']['random_explored']}) "
                f"sha256={payload['names_sha256'][:16]}… -> {path}",
                flush=True,
            )
            if method == "full":
                break  # the full split does not depend on the seed


if __name__ == "__main__":
    main()
