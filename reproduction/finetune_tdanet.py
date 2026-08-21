"""Train a physically pruned TDANet from inherited or random weights."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .lrs2 import LRS2MixDataset
from .tdanet_seprune import (
    load_official_tdanet,
    parameter_count,
    physically_prune_ffn,
    pit_negative_si_sdr,
    seed_everything,
)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    label: str,
    initialization: str,
    completed_steps: int,
    epoch: int,
    mask_path: str,
    seed: int,
) -> None:
    payload = {
        "label": label,
        "initialization": initialization,
        "state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_steps": completed_steps,
        "epoch": epoch,
        "mask_path": mask_path,
        "seed": seed,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mask", default="experiments/tdanet_lrs2_mask/mask.pt")
    parser.add_argument(
        "--initialization", choices=["inherited", "random"], default="inherited"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    output_dir = Path(
        args.output_dir
        or f"experiments/tdanet_lrs2_{args.initialization}_1epoch"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    label = f"pruned_{args.initialization}_1epoch"

    mask = torch.load(args.mask, map_location="cpu", weights_only=True)
    base = load_official_tdanet(
        args.checkpoint, load_weights=args.initialization == "inherited"
    )
    model = physically_prune_ffn(base, mask).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    completed_steps = 0
    start_epoch = 0
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(resume_payload["state_dict"], strict=True)
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        completed_steps = int(resume_payload["completed_steps"])
        start_epoch = int(resume_payload["epoch"])

    dataset = LRS2MixDataset(args.data_root, "tr", segment_samples=32000)
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
    total_target_steps = args.epochs * len(loader)
    if args.max_steps is not None:
        total_target_steps = min(total_target_steps, args.max_steps)

    history_path = output_dir / "training.csv"
    history_mode = "a" if args.resume and history_path.exists() else "w"
    started = time.monotonic()
    with history_path.open(history_mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["step", "epoch", "loss", "gradient_norm", "elapsed_seconds"],
        )
        if history_mode == "w":
            writer.writeheader()
        stop = False
        for epoch in range(start_epoch, args.epochs):
            for mixture, sources, _ in loader:
                if completed_steps >= total_target_steps:
                    stop = True
                    break
                mixture = mixture.to(device, non_blocking=True)
                sources = sources.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                estimate = model(mixture)
                loss = pit_negative_si_sdr(estimate, sources)
                loss.backward()
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.gradient_clip
                    ).detach().cpu()
                )
                optimizer.step()
                completed_steps += 1
                writer.writerow(
                    {
                        "step": completed_steps,
                        "epoch": epoch + 1,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm": gradient_norm,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                stream.flush()
                if completed_steps == 1 or completed_steps % args.log_every == 0:
                    print(
                        f"initialization={args.initialization} "
                        f"step={completed_steps}/{total_target_steps} "
                        f"loss={float(loss.detach()):.4f} grad={gradient_norm:.4f}",
                        flush=True,
                    )
                if completed_steps % args.checkpoint_every == 0:
                    save_checkpoint(
                        output_dir / "last.pt",
                        model,
                        optimizer,
                        label=label,
                        initialization=args.initialization,
                        completed_steps=completed_steps,
                        epoch=epoch,
                        mask_path=args.mask,
                        seed=args.seed,
                    )
            if stop:
                break

    final_path = output_dir / "final.pt"
    save_checkpoint(
        final_path,
        model,
        optimizer,
        label=label,
        initialization=args.initialization,
        completed_steps=completed_steps,
        epoch=args.epochs,
        mask_path=args.mask,
        seed=args.seed,
    )
    result = {
        "status": "completed",
        "label": label,
        "initialization": args.initialization,
        "epochs_requested": args.epochs,
        "completed_steps": completed_steps,
        "training_examples": len(dataset),
        "parameters": parameter_count(model),
        "mask_kept": int(mask.sum().item()),
        "learning_rate": args.learning_rate,
        "final_checkpoint": str(final_path.resolve()),
        "elapsed_seconds": time.monotonic() - started,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

