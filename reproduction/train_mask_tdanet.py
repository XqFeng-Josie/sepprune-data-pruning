"""Learn the reconstructed SepPrune TDANet mask on real LRS2-2Mix audio."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .lrs2 import LRS2MixDataset
from .tdanet_seprune import (
    attach_mask,
    load_official_tdanet,
    parameter_count,
    physically_prune_ffn,
    pit_negative_si_sdr,
    seed_everything,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--output-dir", default="experiments/tdanet_lrs2_mask")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    dataset = LRS2MixDataset(
        args.data_root,
        "tr",
        segment_samples=round(args.segment_seconds * 16000),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )

    model = load_official_tdanet(args.checkpoint).to(device).eval()
    original_parameters = parameter_count(model)
    masked = attach_mask(model, args.epsilon, args.temperature, args.seed)
    optimizer = torch.optim.Adam([masked.mask.alpha], lr=args.learning_rate)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "mask_learning.csv"
    iterator = iter(loader)
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["iteration", "loss", "gradient_norm", "deterministic_kept"],
        )
        writer.writeheader()
        for iteration in range(1, args.iterations + 1):
            try:
                mixture, sources, _ = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                mixture, sources, _ = next(iterator)
            mixture = mixture.to(device, non_blocking=True)
            sources = sources.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            estimate = model(mixture)
            loss = pit_negative_si_sdr(estimate, sources)
            loss.backward()
            gradient_norm = float(masked.mask.alpha.grad.norm().detach().cpu())
            torch.nn.utils.clip_grad_norm_([masked.mask.alpha], max_norm=1.0)
            optimizer.step()
            kept = int(masked.mask.deterministic_mask().sum().item())
            writer.writerow(
                {
                    "iteration": iteration,
                    "loss": float(loss.detach().cpu()),
                    "gradient_norm": gradient_norm,
                    "deterministic_kept": kept,
                }
            )
            stream.flush()
            if iteration == 1 or iteration % 10 == 0:
                print(
                    f"iteration={iteration}/{args.iterations} "
                    f"loss={float(loss.detach()):.4f} grad={gradient_norm:.4f} "
                    f"kept={kept}/1024",
                    flush=True,
                )

    keep_mask = masked.mask.deterministic_mask().cpu()
    kept = int(keep_mask.sum().item())
    if kept == 0 or kept == keep_mask.numel():
        raise RuntimeError(f"Degenerate final mask: kept {kept}/{keep_mask.numel()}")
    pruned = physically_prune_ffn(model, keep_mask)
    result = {
        "status": "completed",
        "implementation": "independent reconstruction from released scripts",
        "dataset_root": str(dataset.root),
        "training_examples": len(dataset),
        "iterations": args.iterations,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "temperature": args.temperature,
        "channels_total": int(keep_mask.numel()),
        "channels_kept": kept,
        "original_parameters": original_parameters,
        "pruned_parameters": parameter_count(pruned),
    }
    torch.save(keep_mask, output_dir / "mask.pt")
    torch.save(masked.mask.alpha.detach().cpu(), output_dir / "alpha.pt")
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

