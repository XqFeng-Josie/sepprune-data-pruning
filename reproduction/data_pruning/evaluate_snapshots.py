"""Score training snapshots on the LRS2-2Mix test split.

Every number in the results tables comes from here. The one subtlety is the
batch policy: TDANet's released `MultiHeadAttention` is built with the default
`batch_first=False` and then handed `[batch, time, channels]`, so torch reads
the batch axis as the sequence axis and each output depends on the other items
in its batch. TDANet is therefore pinned to batch 1, which costs ~150 ms per
mixture and is why its evaluations use a fixed test subset rather than all
3,000 mixtures.

That subset is statistically sound for the comparisons it serves: every arm is
evaluated on exactly the same mixtures, so test-set sampling error cancels in
the paired differences and only seed noise remains. It does mean TDANet's
absolute SI-SDRi is not directly comparable to a 3,000-mixture number, which is
why `n_test` is recorded alongside every row.

    python -m reproduction.data_pruning.evaluate_snapshots --model tdanet \
        --baseline-checkpoint <dense> --masks <masks> --limit 1000 \
        --output experiments/data_pruning_tdanet/tt_eval_1000.json \
        experiments/data_pruning_tdanet/runs/keep500/*/step_0*.pt
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..lrs2 import LRS2MixDataset
from ..original_models import parameter_count
from ..train_pruned_original import build_inherited_pruned_model
from .common import seeded_rng, set_tf32
from .metrics import pit_best_si_sdr

BATCH_INVARIANT = {"afrcnn12": True, "sudormrf": True, "tdanet": False}


def test_indices(dataset: LRS2MixDataset, limit: int | None) -> list[int]:
    """A deterministic subset, drawn at random rather than taken from the front.

    The test split is sorted by file name, which groups mixtures by speaker, so
    the first N would be a biased sample.
    """

    if not limit or limit >= len(dataset):
        return list(range(len(dataset)))
    return sorted(seeded_rng("tt-subset", len(dataset), limit).sample(range(len(dataset)), limit))


@torch.inference_mode()
def evaluate(model, dataset, indices, device, batch_size) -> list[float]:
    if batch_size == 1:
        values = []
        for index in indices:
            mixture, sources, _ = dataset[index]
            mixture, sources = mixture.to(device), sources.to(device)
            values.append(
                float(
                    pit_best_si_sdr(model(mixture.unsqueeze(0)).squeeze(0), sources)
                    - pit_best_si_sdr(mixture.expand_as(sources), sources)
                )
            )
        return values
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    values: list[float] = []
    for mixture, sources, _ in loader:
        mixture, sources = mixture.to(device), sources.to(device)
        values += (
            pit_best_si_sdr(model(mixture), sources)
            - pit_best_si_sdr(mixture.expand_as(sources), sources)
        ).tolist()
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(BATCH_INVARIANT), required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--limit", type=int, default=None, help="evaluate a fixed random subset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", required=True, help="JSON accumulator; existing rows are kept")
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    batch_size = args.batch_size
    if not BATCH_INVARIANT[args.model] and batch_size != 1:
        print(f"{args.model} is not batch-invariant; forcing batch size 1", flush=True)
        batch_size = 1

    set_tf32(False)
    device = torch.device(args.device)
    dataset = LRS2MixDataset(args.data_root, "tt", segment_samples=None)
    indices = test_indices(dataset, args.limit)
    print(f"evaluating {len(indices)}/{len(dataset)} mixtures at batch size {batch_size}", flush=True)

    output = Path(args.output)
    done = json.loads(output.read_text(encoding="utf-8")) if output.is_file() else {}
    todo = [c for c in args.checkpoints if c not in done and Path(c).is_file()]
    print(f"{len(done)} already evaluated, {len(todo)} to go", flush=True)

    for checkpoint in todo:
        model, _ = build_inherited_pruned_model(args.model, args.baseline_checkpoint, args.masks)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if int(payload["parameters"]) != parameter_count(model):
            raise ValueError(f"{checkpoint} has a different pruned structure than the masks")
        model.load_state_dict(payload["state_dict"], strict=True)
        model = model.to(device).eval()
        values = evaluate(model, dataset, indices, device, batch_size)
        done[checkpoint] = {
            "si_sdri": statistics.fmean(values),
            "n_test": len(values),
            "global_step": int(payload["global_step"]),
            "arm": payload["config"]["arm"],
            "seed": payload["config"]["seed"],
            "per_sample": values,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(done), encoding="utf-8")
        row = done[checkpoint]
        print(
            f"  {row['arm']:<4} seed{row['seed']:<5} step {row['global_step']:>6}: "
            f"{row['si_sdri']:.4f}  (n={row['n_test']})",
            flush=True,
        )
        model.to("cpu")
    print("all done", flush=True)


if __name__ == "__main__":
    main()
