"""Learn SepPrune channel masks for A-FRCNN-12 or SuDoRM-RF on LRS2-2Mix."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .lrs2 import LRS2MixDataset
from .original_models import build_original_model, parameter_count
from .seprune_originals import (
    attach_original_masks,
    deterministic_masks,
    mask_parameters,
    physically_prune_original,
)
from .train_original import pit_negative_sdr, seed_everything


PAPER_PRUNED_PARAMETERS = {"afrcnn12": 3_060_000, "sudormrf": 1_540_000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["afrcnn12", "sudormrf"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--epsilon", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--initial-probability-low", type=float, default=0.55)
    parser.add_argument("--initial-probability-high", type=float, default=0.85)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the released mask YAML: 1 for A-FRCNN, 8 for SuDoRM-RF.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    batch_size = args.batch_size or (8 if args.model == "sudormrf" else 1)
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("model_name") != args.model:
        raise ValueError(f"Checkpoint is for {payload.get('model_name')}, not {args.model}")

    model = build_original_model(args.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    original_parameters = parameter_count(model)
    if original_parameters != int(payload["parameters"]):
        raise RuntimeError(
            f"Checkpoint parameter count mismatch: model={original_parameters}, "
            f"checkpoint={payload['parameters']}"
        )
    model = model.to(device).eval()
    wrappers = attach_original_masks(
        model,
        args.model,
        epsilon=args.epsilon,
        temperature=args.temperature,
        seed=args.seed,
        initial_probability_low=args.initial_probability_low,
        initial_probability_high=args.initial_probability_high,
    )
    trainable = mask_parameters(wrappers)
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate, weight_decay=0.0)

    dataset = LRS2MixDataset(args.data_root, "tr", segment_samples=32000)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "implementation": "independent reconstruction from released mask/finetune scripts",
        "known_missing_source": "authors' private look2hear *_pruned.py and ChannelMask1D",
        "model": args.model,
        "baseline_checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset.root),
        "training_examples": len(dataset),
        "iterations": args.iterations,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "temperature": args.temperature,
        "initial_probability_low": args.initial_probability_low,
        "initial_probability_high": args.initial_probability_high,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "gradient_clip": args.gradient_clip,
        "loss": "released YAML: PIT pairwise negative SNR",
        "batch_size": batch_size,
        "segment_samples": 32000,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False), flush=True)

    started = time.monotonic()
    history_path = output_dir / "mask_learning.csv"
    iterator = iter(loader)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        fields = ["iteration", "loss", "gradient_norm", "channels_kept", "elapsed_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for iteration in range(1, args.iterations + 1):
            try:
                mixture, sources, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                mixture, sources, _ = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            estimate = model(mixture.to(device, non_blocking=True))
            loss = pit_negative_sdr(
                estimate,
                sources.to(device, non_blocking=True),
                kind="snr",
                threshold_byloss=True,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite mask loss at iteration {iteration}: {loss}")
            loss.backward()
            missing_grad = [index for index, parameter in enumerate(trainable) if parameter.grad is None]
            if missing_grad:
                raise RuntimeError(f"Mask parameters without gradients: {missing_grad}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite mask gradient at iteration {iteration}")
            optimizer.step()
            kept = [int(mask.sum()) for mask in deterministic_masks(wrappers)]
            writer.writerow(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "channels_kept": json.dumps(kept),
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            if iteration == 1 or iteration % args.log_every == 0:
                stream.flush()
                print(
                    f"model={args.model} iteration={iteration}/{args.iterations} "
                    f"loss={float(loss.detach()):.4f} grad={float(gradient_norm):.4f} "
                    f"kept={kept}",
                    flush=True,
                )

    masks = deterministic_masks(wrappers)
    for index, mask in enumerate(masks):
        if int(mask.sum()) in (0, mask.numel()):
            raise RuntimeError(
                f"Degenerate final mask {index}: kept {int(mask.sum())}/{mask.numel()}"
            )
    pruned = physically_prune_original(model, args.model, masks)
    pruned_parameters = parameter_count(pruned)
    result = {
        "status": "completed",
        **config,
        "mask_count": len(masks),
        "channels_total": [int(mask.numel()) for mask in masks],
        "channels_kept": [int(mask.sum()) for mask in masks],
        "original_parameters": original_parameters,
        "pruned_parameters": pruned_parameters,
        "paper_pruned_parameters": PAPER_PRUNED_PARAMETERS[args.model],
        "parameter_reduction": 1.0 - pruned_parameters / original_parameters,
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(masks, output_dir / "masks.pt")
    torch.save([wrapper.mask.alpha.detach().cpu() for wrapper in wrappers], output_dir / "alphas.pt")
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
