"""Extract the committable slice of an experiment tree into `results/`.

A full run tree is ~18 GB, almost all of it checkpoints. What is worth keeping
under version control is the part that is expensive to recompute but small:

    scores/    the per-sample dense/pruned scores for the whole training split
               (~20 min of GPU each, 8 MB) - every subset in the study is a
               deterministic function of these plus a method and a seed
    runs.csv   one row per training run: its arm, seed, keep budget, timings
               and provenance hashes
    tt_eval.csv  one row per evaluated checkpoint: the test-set SI-SDRi

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
    path = tree / "tt_eval.json"
    if not path.is_file():
        return
    for key, value in json.loads(path.read_text(encoding="utf-8")).items():
        match = CKPT_PATTERN.search(key)
        rows.append(
            {
                "model": model,
                "keep": int(match.group(1)) if match else "",
                "arm": value.get("arm", ""),
                "seed": value.get("seed", ""),
                "global_step": value.get("global_step", ""),
                "si_sdri": round(value["si_sdri"], 6),
            }
        )


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

    runs = write_csv(out / "runs.csv", run_rows, ("model", "keep", "arm", "seed"))
    evals = write_csv(out / "tt_eval.csv", eval_rows, ("model", "keep", "arm", "seed", "global_step"))
    print(f"\nresults/runs.csv     {runs} rows")
    print(f"results/tt_eval.csv  {evals} rows")


if __name__ == "__main__":
    main()
