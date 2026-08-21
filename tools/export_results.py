"""Extract the committable slice of an experiment tree into `results/`.

A full run tree is ~18 GB, almost all of it checkpoints. What is worth keeping
under version control is the part that is expensive to recompute but small:

    scores/       per-sample dense/pruned scores for the whole training split
                  (20 min to 7 h of GPU each) - every subset in the study is a
                  deterministic function of these plus a method and a seed
    masks/        the frozen channel masks that define each pruned structure;
                  without them nothing downstream can be rebuilt
    reproduction/ the full-test-set summaries and validation curves behind the
                  reproduction tables, plus the pre-flight probe
    runs.csv      one row per training run: arm, seed, keep budget, timings,
                  provenance hashes
    tt_eval.csv   one row per evaluated checkpoint: the test-set SI-SDRi

Deliberately excluded: checkpoints (15 GB), the per-sample arrays inside
tt_eval.json (~18 MB of floats that only the aggregate is ever read from), and
the subset files (50 MB, and `samplers.py` rebuilds them bit for bit from the
score set).

    python tools/export_results.py --experiments /path/to/experiments
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

RUN_PATTERN = re.compile(r"runs/keep(\d+)/(\w+)_seed(\d+)$")
CKPT_PATTERN = re.compile(r"runs/keep(\d+)/(\w+)_seed(\d+)/step_(\d+)\.pt$")


def model_of(tree: Path) -> str:
    """`data_pruning` is the A-FRCNN tree; later ones carry the model in the name."""

    suffix = tree.name.removeprefix("data_pruning")
    return suffix.lstrip("_") or "afrcnn12"


def export_scores(tree: Path, out: Path) -> list[str]:
    written = []
    source = tree / "scores"
    if not source.is_dir():
        return written
    target = out / "scores"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.suffix in {".jsonl", ".json"}:
            shutil.copy2(path, target / path.name)
            written.append(path.name)
    return written


def export_runs(tree: Path, model: str, rows: list[dict]) -> None:
    for config_path in sorted(tree.glob("runs/keep*/*/config.json")):
        run = config_path.parent
        match = RUN_PATTERN.search(str(run))
        if not match:
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result_path = run / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
        subset = config.get("subset") or {}
        rows.append(
            {
                "model": model,
                "keep": int(match.group(1)),
                "arm": match.group(2),
                "seed": int(match.group(3)),
                "status": result.get("status", "incomplete"),
                "global_step": result.get("global_step", ""),
                "total_updates": config.get("total_updates", ""),
                "training_examples": config.get("training_examples", ""),
                "parameters": config.get("parameters", ""),
                "learning_rate": config.get("learning_rate", ""),
                "distill_alpha": (config.get("distillation") or {}).get("alpha", ""),
                "ms_per_update": round(result["ms_per_update"], 2) if "ms_per_update" in result else "",
                "gpu_processes": result.get("gpu_processes_at_end", ""),
                "subset_method": subset.get("method", ""),
                "subset_sha256": (subset.get("names_sha256") or "")[:16],
                "baseline_sha256": config.get("baseline_sha256", "")[:16],
                "masks_sha256": config.get("masks_sha256", "")[:16],
            }
        )


def export_tt_eval(tree: Path, model: str, rows: list[dict]) -> None:
    """Collect every tt_eval*.json in the tree.

    `n_test` is not decoration. TDANet must be evaluated one mixture at a time
    (its attention is batch-dependent), which is slow enough that its runs use a
    fixed 1,000-mixture subset while the other two use all 3,000. Rows from the
    two regimes are directly comparable within a model but not across, so the
    count travels with every row.
    """

    for path in sorted(tree.glob("tt_eval*.json")):
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            match = CKPT_PATTERN.search(key)
            rows.append(
                {
                    "model": model,
                    "keep": int(match.group(1)) if match else "",
                    "arm": value.get("arm", ""),
                    "seed": value.get("seed", ""),
                    "global_step": value.get("global_step", ""),
                    "n_test": value.get("n_test", len(value.get("per_sample", [])) or 3000),
                    "si_sdri": round(value["si_sdri"], 6),
                }
            )


SMOKE = ("smoke", "_smoke")


def export_reproduction(root: Path, out: Path) -> dict[str, int]:
    """Copy the reproduction line's small artefacts.

    Summaries and masks are tiny and are what every table in `docs/` cites.
    `validation.csv` is kept because the convergence and learning-rate analyses
    read it; `training.csv` is not - it is 200-365 MB of per-step rows whose
    only aggregate use is the timing already recorded in the summaries.
    """

    counts = {"summaries": 0, "masks": 0, "curves": 0, "preflight": 0}

    target = out / "reproduction" / "test_summaries"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.glob("*/*_tt_summary.json")):
        if any(token in path.parent.name for token in SMOKE):
            continue
        shutil.copy2(path, target / f"{path.parent.name}__{path.name}")
        counts["summaries"] += 1

    target = out / "reproduction" / "masks"
    target.mkdir(parents=True, exist_ok=True)
    for pattern in ("*/masks.pt", "*/mask.pt"):
        for path in sorted(root.glob(pattern)):
            if any(token in path.parent.name for token in SMOKE):
                continue
            shutil.copy2(path, target / f"{path.parent.name}__{path.name}")
            counts["masks"] += 1

    target = out / "reproduction" / "curves"
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(root.glob("*/validation.csv")):
        if any(token in path.parent.name for token in SMOKE) or "data_pruning" in path.parent.name:
            continue
        shutil.copy2(path, target / f"{path.parent.name}__validation.csv")
        counts["curves"] += 1
    for path in sorted(root.glob("*/result.json")):
        if any(token in path.parent.name for token in SMOKE) or "data_pruning" in path.parent.name:
            continue
        shutil.copy2(path, target / f"{path.parent.name}__result.json")

    probe = root / "data_pruning_preflight"
    if probe.is_dir():
        target = out / "reproduction" / "preflight"
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(probe.glob("*.json")):
            shutil.copy2(path, target / path.name)
            counts["preflight"] += 1
    return counts


def write_csv(path: Path, rows: list[dict], sort_by: tuple[str, ...]) -> int:
    if not rows:
        return 0
    rows.sort(key=lambda r: tuple(str(r.get(k, "")) for k in sort_by))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", required=True, help="directory holding data_pruning* trees")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    root = Path(args.experiments)
    out = Path(args.output)
    # A tree only counts if it holds scores or runs; sibling directories such as
    # the preflight probe share the prefix but hold neither.
    trees = sorted(
        p for p in root.glob("data_pruning*")
        if p.is_dir() and ((p / "scores").is_dir() or (p / "runs").is_dir())
    )
    if not trees:
        raise SystemExit(f"no data_pruning* trees under {root}")

    run_rows: list[dict] = []
    eval_rows: list[dict] = []
    for tree in trees:
        model = model_of(tree)
        scores = export_scores(tree, out / model)
        export_runs(tree, model, run_rows)
        export_tt_eval(tree, model, eval_rows)
        print(f"{tree.name:<28} model={model:<10} score files: {len(scores)}")

    counts = export_reproduction(root, out)
    print(
        f"\nreproduction/  {counts['summaries']} test summaries, {counts['masks']} masks, "
        f"{counts['curves']} validation curves, {counts['preflight']} pre-flight files"
    )

    runs = write_csv(out / "runs.csv", run_rows, ("model", "keep", "arm", "seed"))
    evals = write_csv(out / "tt_eval.csv", eval_rows, ("model", "keep", "arm", "seed", "global_step", "n_test"))
    print(f"\nresults/runs.csv     {runs} rows")
    print(f"results/tt_eval.csv  {evals} rows")


if __name__ == "__main__":
    main()
