"""Fine-tune the physically pruned model on a fixed data subset.

Differences from `reproduction.train_pruned_original`, all required by the
stage-A protocol:

* the budget is a fixed number of optimizer updates, not a number of epochs,
  so a 50% subset and the full split are compared at equal compute;
* the learning rate is constant and there is no ReduceLROnPlateau and no early
  stopping, which removes schedule-trigger frequency as a confounder;
* validation runs on one fixed CV subset shared by every arm, on a global-step
  cadence rather than at epoch boundaries;
* the step -> example mapping is precomputed from `(seed, epoch)` with Python's
  stable Mersenne Twister, so a resume replays the exact remaining sequence
  instead of fast-forwarding a DataLoader.

Everything else - the model construction, the PIT negative-SNR objective, the
gradient clip and the optimizer - is imported unchanged from the existing
pipeline so that the two are directly comparable.
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from ..lrs2 import LRS2MixDataset
from ..original_models import parameter_count
from ..train_original import pairwise_negative_sdr, pit_negative_sdr
from ..train_pruned_original import build_inherited_pruned_model
from .distill_baseline import build_teacher, describe_teacher, distillation_loss
from .common import (
    git_commit,
    gpu_process_count,
    seeded_rng,
    set_tf32,
    sha256_file,
    sha256_text,
    tf32_state,
    utc_now,
    write_json,
)
from .samplers import load_subset
from .score_lrs2 import assert_uniform_lengths

TRAIN_LOSS_KIND = "snr"
"""Matches the released Look2Hear objective used by every other run here."""

VALIDATION_LOSS_KIND = "sisdr"


@dataclass
class Timings:
    train_seconds: float = 0.0
    validation_seconds: float = 0.0
    checkpoint_seconds: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "train_seconds": self.train_seconds,
            "validation_seconds": self.validation_seconds,
            "checkpoint_seconds": self.checkpoint_seconds,
            "accounted_seconds": self.train_seconds
            + self.validation_seconds
            + self.checkpoint_seconds,
        }


@dataclass
class RunState:
    global_step: int = 0
    timings: Timings = field(default_factory=Timings)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def per_sample_pit_loss(estimate: Tensor, target: Tensor, kind: str) -> Tensor:
    """Per-example PIT loss, i.e. `pit_negative_sdr` without the batch mean.

    Averaging these afterwards makes validation independent of the validation
    batch size, which a mean-of-batch-means would not be.
    """

    pair = pairwise_negative_sdr(estimate, target, kind)
    direct = 0.5 * (pair[:, 0, 0] + pair[:, 1, 1])
    swapped = 0.5 * (pair[:, 0, 1] + pair[:, 1, 0])
    return torch.minimum(direct, swapped)


def build_monitor_indices(dataset: LRS2MixDataset, size: int) -> list[int]:
    """Fixed CV subset used by every arm.

    It depends only on the split contents and `size`, never on the arm seed, so
    all arms are monitored on identical audio.
    """

    if size > len(dataset):
        raise ValueError(f"Monitor size {size} exceeds the {len(dataset)}-example split")
    if size == len(dataset):
        return list(range(len(dataset)))
    rng = seeded_rng("cv-monitor", dataset.split, len(dataset), size)
    return sorted(rng.sample(range(len(dataset)), size))


def training_order(indices: list[int], total_updates: int, seed: int) -> list[int]:
    """Expand the subset into an exact `total_updates`-long stream of examples.

    Each pass is an independent permutation seeded by `(seed, pass index)`, so
    step `k` maps to the same example no matter where training resumed.
    """

    if not indices:
        raise ValueError("cannot build a training order from an empty subset")
    order: list[int] = []
    epoch = 0
    while len(order) < total_updates:
        shuffled = list(indices)
        seeded_rng("order", seed, epoch).shuffle(shuffled)
        order.extend(shuffled)
        epoch += 1
    return order[:total_updates]


def resolve_subset_indices(dataset: LRS2MixDataset, names: list[str]) -> list[int]:
    position = {name: index for index, name in enumerate(dataset.names)}
    missing = [name for name in names if name not in position]
    if missing:
        raise ValueError(f"{len(missing)} subset names are absent from the split, e.g. {missing[:3]}")
    return [position[name] for name in names]


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataset: LRS2MixDataset,
    indices: list[int],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    was_training = model.training
    model.eval()
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    total = 0.0
    count = 0
    for mixture, sources, _ in loader:
        estimate = model(mixture.to(device, non_blocking=True))
        losses = per_sample_pit_loss(
            estimate, sources.to(device, non_blocking=True), VALIDATION_LOSS_KIND
        )
        total += float(losses.sum())
        count += losses.numel()
    if was_training:
        model.train()
    if count != len(indices):
        raise AssertionError(f"validated {count} examples, expected {len(indices)}")
    return total / count


def save_state(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: RunState,
    config: dict,
) -> None:
    payload = {
        "model_name": config["model"],
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "global_step": state.global_step,
        "parameters": parameter_count(model),
        "timings": state.timings.as_dict(),
        "config": config,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["afrcnn12", "sudormrf", "tdanet"], default="afrcnn12")
    parser.add_argument(
        "--baseline-checkpoint",
        default="experiments/original_afrcnn12_lrs2_train/best.pt",
    )
    parser.add_argument(
        "--masks",
        default="experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026/masks.pt",
    )
    parser.add_argument("--expect-baseline-sha256", default=None)
    parser.add_argument("--expect-masks-sha256", default=None)
    parser.add_argument(
        "--subset",
        default=None,
        help="subset JSON from samplers.py; omit to train on the whole split",
    )
    parser.add_argument("--arm", required=True, help="label recorded in every artefact, e.g. S3p")
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=0.0,
        help="K0: weight of the dense-teacher distillation term; 0 disables it entirely",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="defaults to --baseline-checkpoint, which is the model the student inherited from",
    )
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--total-updates", type=int, default=40000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--allow-batched-threshold", action="store_true")
    parser.add_argument("--monitor-size", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=5000)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--snapshot-at", default="20000,40000")
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="off by default so runs stay comparable with the existing finetune pipeline",
    )
    parser.add_argument("--skip-length-check", action="store_true")
    parser.add_argument("--resume", action="store_true", help="continue from last.pt if present")
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="stop once this global step is reached, leaving a resumable last.pt (smoke tests)",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.batch_size != 1 and not args.allow_batched_threshold:
        raise SystemExit(
            "Training uses threshold_byloss=True, which only behaves as a no-op at batch size 1. "
            "Pass --allow-batched-threshold to accept the sample-dropping behaviour explicitly."
        )
    for path, expected in (
        (args.baseline_checkpoint, args.expect_baseline_sha256),
        (args.masks, args.expect_masks_sha256),
    ):
        if expected and sha256_file(path) != expected:
            raise SystemExit(f"{path} does not match the expected sha256 {expected}")

    seed_everything(args.seed)
    # Leave the CUDA defaults alone unless explicitly asked. They are asymmetric
    # on A100 (cuDNN conv TF32 on, cuBLAS matmul TF32 off) and every existing
    # checkpoint in this repository was produced under them.
    tf32 = set_tf32(False) if args.disable_tf32 else tf32_state()
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, masks = build_inherited_pruned_model(args.model, args.baseline_checkpoint, args.masks)
    initial_state_sha = sha256_text(
        "\n".join(
            f"{key}:{torch.as_tensor(value).flatten()[:8].tolist()}"
            for key, value in sorted(model.state_dict().items())
        )
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)

    if not 0.0 <= args.distill_alpha <= 1.0:
        raise SystemExit(f"--distill-alpha must be in [0, 1], got {args.distill_alpha}")
    teacher = None
    teacher_meta = None
    if args.distill_alpha > 0.0:
        teacher_path = args.teacher_checkpoint or args.baseline_checkpoint
        teacher = build_teacher(args.model, teacher_path, device)
        teacher_meta = describe_teacher(teacher, teacher_path) | {"alpha": args.distill_alpha}

    train_dataset = LRS2MixDataset(args.data_root, "tr", segment_samples=32000)
    val_dataset = LRS2MixDataset(args.data_root, "cv", segment_samples=32000)
    if not args.skip_length_check:
        # Every LRS2-2Mix clip is exactly 2.0 s, so the dataset's random crop is
        # a no-op. If that ever stopped holding, validation would silently become
        # stochastic and the arms would no longer be monitored on identical audio.
        for dataset in (train_dataset, val_dataset):
            frames = assert_uniform_lengths(dataset)
            if frames != 32000:
                raise SystemExit(
                    f"Split {dataset.split!r} holds {frames}-sample clips; "
                    "the 32,000-sample crop would no longer be deterministic"
                )
    monitor_indices = build_monitor_indices(val_dataset, args.monitor_size)
    monitor_sha = sha256_text("\n".join(val_dataset.names[i] for i in monitor_indices) + "\n")

    if args.subset:
        subset = load_subset(args.subset)
        subset_names = subset["names"]
        subset_meta = {
            "path": str(Path(args.subset).resolve()),
            "method": subset["method"],
            "seed": subset["seed"],
            "names_sha256": subset["names_sha256"],
            "source_score_set": subset["source_score_set"],
        }
    else:
        subset_names = list(train_dataset.names)
        subset_meta = {"path": None, "method": "full", "seed": None, "names_sha256": None}
    subset_indices = resolve_subset_indices(train_dataset, subset_names)
    order = training_order(subset_indices, args.total_updates, args.seed)
    snapshot_at = sorted(
        {int(item) for item in args.snapshot_at.split(",") if item.strip()}
    )
    if any(step > args.total_updates for step in snapshot_at):
        raise SystemExit("--snapshot-at contains a step beyond --total-updates")

    config = {
        "arm": args.arm,
        "model": args.model,
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "baseline_sha256": sha256_file(args.baseline_checkpoint),
        "masks": str(Path(args.masks).resolve()),
        "masks_sha256": sha256_file(args.masks),
        "channels_kept": [int(mask.sum()) for mask in masks],
        "parameters": parameter_count(model),
        "initial_state_fingerprint": initial_state_sha,
        "subset": subset_meta,
        "training_examples": len(subset_indices),
        "total_updates": args.total_updates,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lr_schedule": "constant",
        "early_stopping": False,
        "gradient_clip": args.gradient_clip,
        "loss": "Look2Hear PIT pairwise negative SNR (threshold_byloss=True)",
        "distillation": teacher_meta,
        "validation_loss": "PIT negative SI-SDR, per-sample mean",
        "monitor_split": "cv",
        "monitor_size": len(monitor_indices),
        "monitor_sha256": monitor_sha,
        "validate_every": args.validate_every,
        "snapshot_at": snapshot_at,
        "seed": args.seed,
        "segment_samples": 32000,
        "deterministic": args.deterministic,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "tf32": tf32,
        "git_commit": git_commit(),
        "started_at_utc": utc_now(),
        "gpu_processes_at_start": gpu_process_count(),
    }

    state = RunState()
    last_path = output_dir / "last.pt"
    if args.resume and last_path.is_file():
        payload = torch.load(last_path, map_location="cpu", weights_only=True)
        stored = payload["config"]
        if (stored.get("distillation") or {}).get("alpha") != (teacher_meta or {}).get("alpha"):
            raise SystemExit("Resume refused: the distillation weight changed")
        for key in ("arm", "baseline_sha256", "masks_sha256", "total_updates", "seed"):
            if stored.get(key) != config[key]:
                raise SystemExit(f"Resume refused: {key} changed from {stored.get(key)!r}")
        if stored["subset"].get("names_sha256") != subset_meta.get("names_sha256"):
            raise SystemExit("Resume refused: the subset changed")
        if int(payload["parameters"]) != parameter_count(model):
            raise SystemExit("Resume refused: pruned structure changed")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state.global_step = int(payload["global_step"])
        stored_timings = payload["timings"]
        state.timings = Timings(
            train_seconds=stored_timings["train_seconds"],
            validation_seconds=stored_timings["validation_seconds"],
            checkpoint_seconds=stored_timings["checkpoint_seconds"],
        )
        print(f"resumed {args.arm} at global_step={state.global_step}", flush=True)

    write_json(output_dir / "config.json", config)
    append = state.global_step > 0
    training_path = output_dir / "training.csv"
    validation_path = output_dir / "validation.csv"

    def snapshot(tag: str) -> None:
        started = time.monotonic()
        save_state(output_dir / f"{tag}.pt", model=model, optimizer=optimizer, state=state, config=config)
        state.timings.checkpoint_seconds += time.monotonic() - started

    with training_path.open("a" if append else "w", newline="", encoding="utf-8") as train_stream, (
        validation_path.open("a" if append else "w", newline="", encoding="utf-8")
    ) as val_stream:
        train_writer = csv.DictWriter(
            train_stream,
            fieldnames=[
                "global_step",
                "loss",
                "task_loss",
                "distillation_loss",
                "gradient_norm",
                "learning_rate",
                "train_seconds",
            ],
        )
        val_writer = csv.DictWriter(
            val_stream,
            fieldnames=["global_step", "val_loss", "val_si_sdr", "train_seconds", "wall_seconds"],
        )
        if not append:
            train_writer.writeheader()
            val_writer.writeheader()

        def run_validation() -> None:
            started = time.monotonic()
            loss = validate(
                model,
                val_dataset,
                monitor_indices,
                device=device,
                batch_size=args.val_batch_size,
                num_workers=args.num_workers,
            )
            state.timings.validation_seconds += time.monotonic() - started
            val_writer.writerow(
                {
                    "global_step": state.global_step,
                    "val_loss": loss,
                    "val_si_sdr": -loss,
                    "train_seconds": state.timings.train_seconds,
                    "wall_seconds": state.timings.as_dict()["accounted_seconds"],
                }
            )
            val_stream.flush()
            print(
                f"arm={args.arm} step={state.global_step}/{args.total_updates} "
                f"val_si_sdr={-loss:.4f}",
                flush=True,
            )

        if state.global_step == 0 and args.validate_every:
            run_validation()

        remaining = order[state.global_step :]
        if remaining:
            loader = DataLoader(
                Subset(train_dataset, remaining),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            model.train()
            for mixture, sources, _ in loader:
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                mixture = mixture.to(device, non_blocking=True)
                sources = sources.to(device, non_blocking=True)
                estimate = model(mixture)
                if teacher is None:
                    # Untouched path: every data arm uses the exact call the rest
                    # of the repository's pruned fine-tuning already uses.
                    loss = pit_negative_sdr(
                        estimate, sources, kind=TRAIN_LOSS_KIND, threshold_byloss=True
                    )
                    task_value = float(loss.detach())
                    distill_value = 0.0
                else:
                    with torch.no_grad():
                        teacher_estimate = teacher(mixture)
                    terms = distillation_loss(
                        estimate,
                        sources,
                        teacher_estimate,
                        alpha=args.distill_alpha,
                        kind=TRAIN_LOSS_KIND,
                    )
                    loss = terms["total"].mean()
                    task_value = float(terms["task"].mean().detach())
                    distill_value = float(terms["distillation"].mean().detach())
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at step {state.global_step + 1}: {loss}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.gradient_clip
                )
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError(f"Non-finite gradient norm at step {state.global_step + 1}")
                optimizer.step()
                state.global_step += 1
                state.timings.train_seconds += time.monotonic() - started

                train_writer.writerow(
                    {
                        "global_step": state.global_step,
                        "loss": float(loss.detach()),
                        "task_loss": task_value,
                        "distillation_loss": distill_value,
                        "gradient_norm": float(gradient_norm.detach()),
                        "learning_rate": args.learning_rate,
                        "train_seconds": state.timings.train_seconds,
                    }
                )
                if state.global_step % args.log_every == 0:
                    train_stream.flush()
                    print(
                        f"arm={args.arm} step={state.global_step}/{args.total_updates} "
                        f"loss={float(loss.detach()):.4f} grad={float(gradient_norm):.4f}",
                        flush=True,
                    )
                if state.global_step % args.checkpoint_every == 0:
                    snapshot("last")
                if state.global_step in snapshot_at:
                    snapshot(f"step_{state.global_step:06d}")
                if args.validate_every and state.global_step % args.validate_every == 0:
                    run_validation()
                if args.stop_after and state.global_step >= args.stop_after:
                    break
            train_stream.flush()

    snapshot("last")
    stopped_early = bool(args.stop_after) and state.global_step < args.total_updates
    if not stopped_early and state.global_step != args.total_updates:
        raise AssertionError(
            f"finished at step {state.global_step}, expected {args.total_updates}"
        )

    result = {
        "arm": args.arm,
        "status": "stopped_early" if stopped_early else "completed",
        "global_step": state.global_step,
        "training_examples": len(subset_indices),
        "unique_examples_seen": len(set(order)),
        "passes_over_subset": args.total_updates / len(subset_indices),
        "audio_seconds_processed": args.total_updates * 32000 / 16000,
        "timings": state.timings.as_dict(),
        "distillation": teacher_meta,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "finished_at_utc": utc_now(),
        "gpu_processes_at_end": gpu_process_count(),
        "ms_per_update": state.timings.train_seconds / max(1, state.global_step) * 1000,
        "output_dir": str(output_dir.resolve()),
    }
    write_json(output_dir / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
