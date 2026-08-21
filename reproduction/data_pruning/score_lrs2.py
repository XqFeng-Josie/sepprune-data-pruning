"""Score every LRS2-2Mix training mixture with the dense and pruned models.

Produces the score set consumed by `samplers.py`. Both models are evaluated
from one frozen starting point: the dense baseline checkpoint, and the
physically pruned network rebuilt from that same checkpoint plus the frozen
channel masks. Nothing is trained and nothing is written back to a checkpoint.

Batching is a correctness-sensitive optimisation: at batch size 1 the full
training split takes tens of minutes per model. Every model in this repository
normalises per sample (`GlobLN` reduces over all non-batch axes) and pads only
by the shared time length, so batching is semantically inert; `verify.py`
checks the residual kernel-selection noise numerically.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ..lrs2 import LRS2MixDataset
from ..original_models import build_original_model, parameter_count
from ..train_pruned_original import build_inherited_pruned_model
from .common import git_commit, set_tf32, sha256_file, utc_now
from .metrics import pit_best_si_sdr
from .score_schema import derive_columns, load_score_set, parse_name, write_score_set

SUPPORTED_MODELS = ("afrcnn12", "sudormrf", "tdanet")


def build_dense(model_name: str, checkpoint: str | Path, device: torch.device) -> nn.Module:
    if model_name == "tdanet":
        return build_original_model("tdanet", tdanet_checkpoint=str(checkpoint)).to(device).eval()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("model_name") != model_name:
        raise ValueError(f"Baseline is for {payload.get('model_name')}, not {model_name}")
    model = build_original_model(model_name)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def build_pruned(
    model_name: str,
    baseline: str | Path,
    masks: str | Path,
    checkpoint: str | Path | None,
    device: torch.device,
) -> nn.Module:
    model, _ = build_inherited_pruned_model(model_name, baseline, masks)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("model_name") not in (None, model_name):
            raise ValueError(f"Checkpoint is for {payload.get('model_name')}, not {model_name}")
        if int(payload.get("parameters", -1)) != parameter_count(model):
            raise ValueError(
                f"{checkpoint} has a different pruned structure than the frozen masks"
            )
        model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()


def assert_uniform_lengths(dataset: LRS2MixDataset) -> int:
    """Fail before scoring if the split is not fixed length.

    Variable-length items would make the default collate raise mid-run, and a
    silently truncated batch would corrupt every downstream percentile.
    """

    lengths = {sf.info(str(dataset.paths["mix"][name])).frames for name in dataset.names}
    if len(lengths) != 1:
        raise RuntimeError(
            f"Split {dataset.split!r} is not fixed length ({sorted(lengths)[:5]}...); "
            "batched scoring requires a uniform sample count"
        )
    return lengths.pop()


@torch.inference_mode()
def score_models(
    dense: nn.Module | None,
    pruned: nn.Module,
    dataset: LRS2MixDataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    log_every: int,
    limit: int | None = None,
) -> list[dict]:
    population = dataset if limit is None else Subset(dataset, list(range(limit)))
    loader = DataLoader(
        population,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    rows: list[dict] = []
    seen = 0
    for mixture, sources, names in loader:
        mixture = mixture.to(device, non_blocking=True)
        sources = sources.to(device, non_blocking=True)
        pruned_score = pit_best_si_sdr(pruned(mixture), sources)
        dense_score = pit_best_si_sdr(dense(mixture), sources) if dense else None
        baseline = pit_best_si_sdr(mixture.expand_as(sources), sources)
        for index, name in enumerate(names):
            speaker1, speaker2, snr = parse_name(name)
            rows.append(
                {
                    "name": name,
                    "q_dense_sisdr": (
                        float(dense_score[index]) if dense_score is not None else None
                    ),
                    "q_pruned_sisdr": float(pruned_score[index]),
                    "mixture_sisdr_baseline": float(baseline[index]),
                    "snr_abs": snr,
                    "spk1": speaker1,
                    "spk2": speaker2,
                }
            )
        seen += len(names)
        if log_every and (seen % log_every < batch_size or seen == len(population)):
            print(f"scored {seen}/{len(population)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="afrcnn12")
    parser.add_argument(
        "--baseline-checkpoint",
        default="experiments/original_afrcnn12_lrs2_train/best.pt",
        help="dense checkpoint; supplies both the teacher and the pruned initialisation",
    )
    parser.add_argument(
        "--masks",
        default="experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026/masks.pt",
    )
    parser.add_argument(
        "--pruned-checkpoint",
        default=None,
        help="optional fine-tuned pruned checkpoint; omit to score the frozen start",
    )
    parser.add_argument(
        "--label",
        default="init",
        help="tag for the pruned model state, e.g. init or step020000",
    )
    parser.add_argument(
        "--reuse-dense",
        default=None,
        help="stem of an existing score set whose dense column should be reused",
    )
    parser.add_argument("--expect-baseline-sha256", default=None)
    parser.add_argument("--expect-masks-sha256", default=None)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--split", choices=["tr", "cv", "tt"], default="tr")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=2000)
    parser.add_argument("--skip-length-check", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N files; for integration tests only, since the "
        "percentiles would then be taken over a truncated population",
    )
    parser.add_argument("--output-dir", default="experiments/data_pruning_scores")
    parser.add_argument("--stem", default=None, help="defaults to <model>_<split>_<label>")
    args = parser.parse_args()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.model == "tdanet" and args.batch_size != 1:
        # The released TDANet builds nn.MultiheadAttention with the default
        # batch_first=False and then feeds it [batch, time, channels], so torch
        # reads the batch axis as the sequence axis: the attention mixes across
        # utterances and every output depends on the batch size. Scoring must
        # therefore run one mixture at a time, which is also how every published
        # TDANet number in this repository was produced.
        raise SystemExit(
            "TDANet is not batch-invariant (upstream attention bug); use --batch-size 1"
        )
    for path, expected in (
        (args.baseline_checkpoint, args.expect_baseline_sha256),
        (args.masks, args.expect_masks_sha256),
    ):
        # A mismatch invalidates every downstream subset, so it is fatal, not a warning.
        if expected and sha256_file(path) != expected:
            raise SystemExit(f"{path} is not the expected checkpoint (sha256 {expected})")
    device = torch.device(args.device)
    # Scores feed a percentile ranking whose neighbouring entries are ~1e-3 dB
    # apart, which is the same size as TF32's error. See `common.set_tf32`.
    tf32 = set_tf32(False)
    started = time.monotonic()

    dataset = LRS2MixDataset(args.data_root, args.split, segment_samples=None)
    samples = None if args.skip_length_check else assert_uniform_lengths(dataset)

    reused = load_score_set(args.reuse_dense) if args.reuse_dense else None
    if reused is not None:
        if reused.meta["dense_checkpoint"]["sha256"] != sha256_file(args.baseline_checkpoint):
            raise ValueError(
                "--reuse-dense was produced by a different dense checkpoint; refusing to mix"
            )
        if reused.meta["split"] != args.split:
            raise ValueError("--reuse-dense was produced on a different split")

    dense = None if reused is not None else build_dense(args.model, args.baseline_checkpoint, device)
    pruned = build_pruned(
        args.model, args.baseline_checkpoint, args.masks, args.pruned_checkpoint, device
    )

    rows = score_models(
        dense,
        pruned,
        dataset,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_every=args.log_every,
        limit=args.limit,
    )

    if reused is not None:
        dense_by_name = {row["name"]: row["q_dense_sisdr"] for row in reused.rows}
        if set(dense_by_name) != {row["name"] for row in rows}:
            raise ValueError("--reuse-dense covers a different set of file names")
        for row in rows:
            row["q_dense_sisdr"] = dense_by_name[row["name"]]

    stem = Path(args.output_dir) / (args.stem or f"{args.model}_{args.split}_{args.label}")
    meta = {
        "model": args.model,
        "split": args.split,
        "pruned_label": args.label,
        "data_root": str(Path(args.data_root).resolve()),
        "audio_root": str(dataset.root),
        "samples_per_mixture": samples,
        "dense_checkpoint": {
            "path": str(Path(args.baseline_checkpoint).resolve()),
            "sha256": sha256_file(args.baseline_checkpoint),
        },
        "masks": {
            "path": str(Path(args.masks).resolve()),
            "sha256": sha256_file(args.masks),
        },
        "pruned_checkpoint": (
            {
                "path": str(Path(args.pruned_checkpoint).resolve()),
                "sha256": sha256_file(args.pruned_checkpoint),
            }
            if args.pruned_checkpoint
            else None
        ),
        "dense_reused_from": str(Path(args.reuse_dense).resolve()) if args.reuse_dense else None,
        "pruned_parameters": parameter_count(pruned),
        "batch_size": args.batch_size,
        "limit": args.limit,
        "population_complete": args.limit is None,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "tf32": tf32,
        "git_commit": git_commit(),
        "scored_at_utc": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
    }
    written = write_score_set(stem, meta, derive_columns(rows))
    print(
        f"wrote {written['count']} rows to {stem.with_suffix('.jsonl')} "
        f"(rows_sha256={written['rows_sha256'][:16]}…) in {written['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
