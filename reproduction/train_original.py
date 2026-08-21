"""Train an unpruned A-FRCNN-12 or SuDoRM-RF baseline on LRS2-2Mix."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from .lrs2 import LRS2MixDataset
from .original_models import MODEL_SPECS, build_original_model, parameter_count


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pairwise_negative_sdr(estimate: Tensor, target: Tensor, kind: str) -> Tensor:
    """Released Look2Hear pairwise negative SNR/SI-SDR implementation."""

    if estimate.shape != target.shape or estimate.ndim != 3:
        raise ValueError(f"Expected matching [batch, source, time], got {estimate.shape} and {target.shape}")
    target = target - target.mean(dim=-1, keepdim=True)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target_pair = target.unsqueeze(1)
    estimate_pair = estimate.unsqueeze(2)
    if kind == "sisdr":
        dot = (estimate_pair * target_pair).sum(dim=-1, keepdim=True)
        energy = target_pair.square().sum(dim=-1, keepdim=True) + 1e-8
        reference = dot * target_pair / energy
    elif kind == "snr":
        reference = target_pair.expand(-1, target.shape[1], -1, -1)
    else:
        raise ValueError(kind)
    noise = estimate_pair - (target_pair if kind == "snr" else reference)
    ratio = reference.square().sum(dim=-1) / (noise.square().sum(dim=-1) + 1e-8)
    return -10.0 * torch.log10(ratio + 1e-8)


def pit_negative_sdr(
    estimate: Tensor,
    target: Tensor,
    *,
    kind: str,
    threshold_byloss: bool,
) -> Tensor:
    pair = pairwise_negative_sdr(estimate, target, kind)
    direct = 0.5 * (pair[:, 0, 0] + pair[:, 1, 1])
    swapped = 0.5 * (pair[:, 0, 1] + pair[:, 1, 0])
    minimum = torch.minimum(direct, swapped)
    if threshold_byloss and torch.any(minimum > -30.0):
        minimum = minimum[minimum > -30.0]
    return minimum.mean()


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(
    path: Path,
    *,
    model_name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch_index: int,
    step_in_epoch: int,
    global_step: int,
    best_val_loss: float,
    bad_epochs: int,
    seed: int,
) -> None:
    payload: dict[str, Any] = {
        "model_name": model_name,
        "state_dict": cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch_index": epoch_index,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "bad_epochs": bad_epochs,
        "seed": seed,
        "parameters": parameter_count(model),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def make_loader(
    dataset: LRS2MixDataset,
    *,
    shuffle: bool,
    seed: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        generator=generator,
    )


def validate(
    model: nn.Module,
    dataset: LRS2MixDataset,
    *,
    device: torch.device,
    num_workers: int,
    limit: int | None,
) -> float:
    loader = make_loader(
        dataset,
        shuffle=False,
        seed=0,
        num_workers=num_workers,
        device=device,
    )
    model.eval()
    total = 0.0
    count = min(len(dataset), limit) if limit else len(dataset)
    with torch.inference_mode():
        for index, (mixture, sources, _) in enumerate(loader):
            if index >= count:
                break
            estimate = model(mixture.to(device, non_blocking=True))
            loss = pit_negative_sdr(
                estimate,
                sources.to(device, non_blocking=True),
                kind="sisdr",
                threshold_byloss=False,
            )
            total += float(loss.cpu())
    model.train()
    return total / count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["afrcnn12", "sudormrf"], required=True)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--scheduler-patience", type=int, default=15)
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    spec = MODEL_SPECS[args.model]
    epochs = args.epochs or spec.reported_training_epochs
    assert epochs is not None
    output_dir = Path(args.output_dir or f"experiments/original_{args.model}_lrs2_train")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_original_model(args.model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=args.scheduler_patience,
        factor=0.5,
    )
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    bad_epochs = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload["model_name"] != args.model:
            raise ValueError(f"Checkpoint is for {payload['model_name']}, not {args.model}")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch_index"])
        start_step_in_epoch = int(payload["step_in_epoch"])
        global_step = int(payload["global_step"])
        best_val_loss = float(payload["best_val_loss"])
        bad_epochs = int(payload["bad_epochs"])

    train_dataset = LRS2MixDataset(args.data_root, "tr", segment_samples=32000)
    val_dataset = LRS2MixDataset(args.data_root, "cv", segment_samples=32000)
    parameters = parameter_count(model)
    metadata = {
        "model": args.model,
        "parameters": parameters,
        "paper_parameters": spec.paper_parameters,
        "epochs": epochs,
        "training_examples": len(train_dataset),
        "validation_examples": len(val_dataset),
        "loss": "released Look2Hear PIT pairwise negative SNR; validation negative SI-SDR",
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "batch_size": 1,
        "segment_samples": 32000,
        "seed": args.seed,
    }
    (output_dir / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False), flush=True)

    training_path = output_dir / "training.csv"
    validation_path = output_dir / "validation.csv"
    training_exists = training_path.exists() and args.resume
    validation_exists = validation_path.exists() and args.resume
    started = time.monotonic()
    stop_reason = "epochs_completed"
    with training_path.open("a" if training_exists else "w", newline="", encoding="utf-8") as train_stream, validation_path.open(
        "a" if validation_exists else "w", newline="", encoding="utf-8"
    ) as val_stream:
        train_writer = csv.DictWriter(
            train_stream,
            fieldnames=["global_step", "epoch", "step_in_epoch", "loss", "gradient_norm", "learning_rate", "elapsed_seconds"],
        )
        val_writer = csv.DictWriter(
            val_stream,
            fieldnames=["epoch", "global_step", "val_loss", "val_si_sdri_proxy", "learning_rate", "elapsed_seconds"],
        )
        if not training_exists:
            train_writer.writeheader()
        if not validation_exists:
            val_writer.writeheader()

        should_stop = False
        model.train()
        for epoch in range(start_epoch, epochs):
            loader = make_loader(
                train_dataset,
                shuffle=True,
                seed=args.seed + epoch,
                num_workers=args.num_workers,
                device=device,
            )
            resume_at = start_step_in_epoch if epoch == start_epoch else 0
            for batch_index, (mixture, sources, _) in enumerate(loader):
                if batch_index < resume_at:
                    continue
                optimizer.zero_grad(set_to_none=True)
                estimate = model(mixture.to(device, non_blocking=True))
                loss = pit_negative_sdr(
                    estimate,
                    sources.to(device, non_blocking=True),
                    kind="snr",
                    threshold_byloss=True,
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch + 1}, step={batch_index + 1}: {loss}")
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
                global_step += 1
                step_in_epoch = batch_index + 1
                train_writer.writerow(
                    {
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "step_in_epoch": step_in_epoch,
                        "loss": float(loss.detach().cpu()),
                        "gradient_norm": float(gradient_norm.detach().cpu()),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
                if global_step == 1 or global_step % args.log_every == 0:
                    train_stream.flush()
                    print(
                        f"model={args.model} epoch={epoch + 1}/{epochs} "
                        f"step={step_in_epoch}/{len(train_dataset)} global_step={global_step} "
                        f"loss={float(loss.detach()):.4f} grad={float(gradient_norm):.4f}",
                        flush=True,
                    )
                if global_step % args.checkpoint_every == 0:
                    save_checkpoint(
                        output_dir / "last.pt",
                        model_name=args.model,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch_index=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        best_val_loss=best_val_loss,
                        bad_epochs=bad_epochs,
                        seed=args.seed,
                    )
                if args.max_steps is not None and global_step >= args.max_steps:
                    stop_reason = "max_steps_reached"
                    should_stop = True
                    break
            start_step_in_epoch = 0
            if should_stop:
                save_checkpoint(
                    output_dir / "final.pt",
                    model_name=args.model,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch_index=epoch,
                    step_in_epoch=step_in_epoch,
                    global_step=global_step,
                    best_val_loss=best_val_loss,
                    bad_epochs=bad_epochs,
                    seed=args.seed,
                )
                break

            val_loss = validate(
                model,
                val_dataset,
                device=device,
                num_workers=args.num_workers,
                limit=args.limit_val,
            )
            scheduler.step(val_loss)
            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
                bad_epochs = 0
            else:
                bad_epochs += 1
            val_writer.writerow(
                {
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "val_loss": val_loss,
                    "val_si_sdri_proxy": -val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            val_stream.flush()
            checkpoint_args = dict(
                model_name=args.model,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch_index=epoch + 1,
                step_in_epoch=0,
                global_step=global_step,
                best_val_loss=best_val_loss,
                bad_epochs=bad_epochs,
                seed=args.seed,
            )
            save_checkpoint(output_dir / "last.pt", **checkpoint_args)
            if improved:
                save_checkpoint(output_dir / "best.pt", **checkpoint_args)
            print(
                f"model={args.model} epoch={epoch + 1}/{epochs} val_si_sdr={-val_loss:.4f} "
                f"best={-best_val_loss:.4f} lr={optimizer.param_groups[0]['lr']:.6g} bad_epochs={bad_epochs}",
                flush=True,
            )
            if bad_epochs >= args.early_stop_patience:
                stop_reason = "early_stopped"
                should_stop = True
                break

    result = {
        "status": "completed" if stop_reason != "max_steps_reached" else "smoke_completed",
        "stop_reason": stop_reason,
        "model": args.model,
        "parameters": parameters,
        "global_step": global_step,
        "best_val_si_sdr": None if best_val_loss == float("inf") else -best_val_loss,
        "elapsed_seconds_this_run": time.monotonic() - started,
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

