"""Learn Gumbel-STE masks under the SepPrune paper's parameter budget."""

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
    attach_budgeted_original_masks,
    prepare_budgeted_masks,
    physically_prune_original,
)
from .train_original import pit_negative_sdr, seed_everything


PAPER_TARGETS = {"afrcnn12": 3_060_000, "sudormrf": 1_540_000}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(PAPER_TARGETS), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--target-parameters", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    batch_size = args.batch_size or (8 if args.model == "sudormrf" else 1)
    target_parameters = args.target_parameters or PAPER_TARGETS[args.model]
    seed_everything(args.seed)
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("model_name") != args.model:
        raise ValueError(f"Checkpoint is for {payload.get('model_name')}, not {args.model}")
    model = build_original_model(args.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    original_parameters = parameter_count(model)
    model = model.to(device).eval()
    wrappers, controller = attach_budgeted_original_masks(
        model,
        args.model,
        target_parameters=target_parameters,
        temperature=args.temperature,
        seed=args.seed,
    )
    optimizer = torch.optim.Adam(controller.parameters(), lr=args.learning_rate, weight_decay=0.0)

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
        "implementation": "paper-guided independent reconstruction with explicit parameter-budget projection",
        "known_missing_source": "authors' private ChannelMask1D and *_pruned.py",
        "paper_components": "Gumbel masks, hard binary forward, clipped STE, frozen backbone, 500 updates, lr=0.1",
        "explicit_completion": "hard masks projected to paper Table 2 parameter budget",
        "model": args.model,
        "baseline_checkpoint": str(checkpoint_path),
        "dataset_root": str(dataset.root),
        "training_examples": len(dataset),
        "iterations": args.iterations,
        "seed": args.seed,
        "paper_reported_epsilon": 0.7,
        "temperature": args.temperature,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "gradient_clip": args.gradient_clip,
        "loss": "released YAML: PIT pairwise negative SNR",
        "batch_size": batch_size,
        "segment_samples": 32000,
        "original_parameters": original_parameters,
        "target_parameters": target_parameters,
        "fixed_parameters": controller.fixed_parameters,
        "variable_budget": controller.variable_budget,
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False), flush=True)

    parameters = list(controller.parameters())
    iterator = iter(loader)
    started = time.monotonic()
    history_path = output_dir / "mask_learning.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["iteration", "loss", "gradient_norm", "channels_kept", "hard_parameters", "elapsed_seconds"],
        )
        writer.writeheader()
        for iteration in range(1, args.iterations + 1):
            try:
                mixture, sources, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                mixture, sources, _ = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            masks = prepare_budgeted_masks(wrappers, controller, stochastic=True)
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
            missing = [index for index, parameter in enumerate(parameters) if parameter.grad is None]
            if missing:
                raise RuntimeError(f"Controller parameters without gradients: {missing}")
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(f"Non-finite mask gradient at iteration {iteration}")
            optimizer.step()
            kept = [int(mask.detach().sum()) for mask in masks]
            writer.writerow(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "channels_kept": json.dumps(kept),
                    "hard_parameters": controller.last_hard_parameters,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            if iteration == 1 or iteration % args.log_every == 0:
                stream.flush()
                print(
                    f"budgeted={args.model} iteration={iteration}/{args.iterations} "
                    f"loss={float(loss.detach()):.4f} grad={float(gradient_norm):.4f} "
                    f"kept={kept} parameters={controller.last_hard_parameters}",
                    flush=True,
                )

    final_masks = controller.deterministic_masks()
    prepare_budgeted_masks(wrappers, controller, stochastic=False)
    pruned = physically_prune_original(model, args.model, final_masks)
    pruned_parameters = parameter_count(pruned)
    if pruned_parameters != controller.last_hard_parameters:
        raise RuntimeError(
            f"Budget accounting mismatch: physical={pruned_parameters}, controller={controller.last_hard_parameters}"
        )
    max_cost = max(int(cost.max()) for cost in controller.costs)
    if not 0 <= target_parameters - pruned_parameters < max_cost:
        raise RuntimeError(
            f"Physical model misses target by more than one channel cost: "
            f"target={target_parameters}, physical={pruned_parameters}, max_cost={max_cost}"
        )
    result = {
        "status": "completed",
        **config,
        "mask_count": len(final_masks),
        "channels_total": [int(mask.numel()) for mask in final_masks],
        "channels_kept": [int(mask.sum()) for mask in final_masks],
        "pruned_parameters": pruned_parameters,
        "target_gap": target_parameters - pruned_parameters,
        "parameter_reduction": 1.0 - pruned_parameters / original_parameters,
        "elapsed_seconds": time.monotonic() - started,
    }
    torch.save(final_masks, output_dir / "masks.pt")
    torch.save([logits.detach().cpu() for logits in controller.logits], output_dir / "logits.pt")
    torch.save(controller.state_dict(), output_dir / "controller.pt")
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

