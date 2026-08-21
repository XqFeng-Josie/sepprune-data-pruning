"""Fine-tune a physically pruned A-FRCNN-12 or SuDoRM-RF model."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from .lrs2 import LRS2MixDataset
from .original_models import build_original_model, parameter_count
from .seprune_originals import physically_prune_original
from .train_original import (
    make_loader,
    pit_negative_sdr,
    save_checkpoint,
    seed_everything,
    validate,
)


def build_inherited_pruned_model(
    model_name: str, baseline_checkpoint: str | Path, masks_path: str | Path
) -> tuple[torch.nn.Module, list[torch.Tensor]]:
    masks = torch.load(masks_path, map_location="cpu", weights_only=True)
    # TDANet's mask search predates this module and stores one bare tensor for
    # the single FFN hidden dimension it prunes; the other two backbones store a
    # list, one tensor per masked layer. Normalise so callers see a list.
    if isinstance(masks, torch.Tensor):
        masks = [masks]
    if not isinstance(masks, list) or not all(isinstance(mask, torch.Tensor) for mask in masks):
        raise TypeError("masks.pt must contain a tensor or a list of tensors")

    if model_name == "tdanet":
        # The dense TDANet is the published checkpoint rather than one this
        # repository trained, so it carries no `model_name` field and is loaded
        # through the official-architecture helper.
        model = build_original_model("tdanet", tdanet_checkpoint=str(baseline_checkpoint))
    else:
        baseline = torch.load(baseline_checkpoint, map_location="cpu", weights_only=True)
        if baseline.get("model_name") != model_name:
            raise ValueError(f"Baseline is for {baseline.get('model_name')}, not {model_name}")
        model = build_original_model(model_name)
        model.load_state_dict(baseline["state_dict"], strict=True)
    return physically_prune_original(model, model_name, masks), masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["afrcnn12", "sudormrf"], required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--scheduler-patience", type=int, default=15)
    parser.add_argument(
        "--plateau-threshold",
        type=float,
        default=0.0,
        help="dB of validation SI-SDR an epoch must gain to count as an improvement. "
        "The default 0 keeps the original behaviour, where any gain however small "
        "resets both the scheduler patience and the early-stopping counter; with a "
        "flat validation curve that can stop the learning rate from ever decaying.",
    )
    parser.add_argument(
        "--override-learning-rate",
        type=float,
        default=None,
        help="force this learning rate after --resume. Needed because loading the "
        "optimizer state restores the stored rate and silently overrides "
        "--learning-rate.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=30)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    model, masks = build_inherited_pruned_model(
        args.model, args.baseline_checkpoint, args.masks
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    # Only pass the threshold when it is actually requested, so the default path
    # keeps PyTorch's own defaults and stays identical to every previous run.
    plateau_kwargs = (
        {"threshold": args.plateau_threshold, "threshold_mode": "abs"}
        if args.plateau_threshold
        else {}
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.scheduler_patience, factor=0.5, **plateau_kwargs
    )
    start_epoch = 0
    start_step_in_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    bad_epochs = 0
    if args.resume:
        payload = torch.load(args.resume, map_location="cpu", weights_only=True)
        if payload.get("model_name") != args.model:
            raise ValueError(f"Resume checkpoint is for {payload.get('model_name')}, not {args.model}")
        if int(payload.get("parameters", -1)) != parameter_count(model):
            raise ValueError("Resume checkpoint has a different pruned structure")
        model.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scheduler.load_state_dict(payload["scheduler_state_dict"])
        start_epoch = int(payload["epoch_index"])
        start_step_in_epoch = int(payload["step_in_epoch"])
        global_step = int(payload["global_step"])
        best_val_loss = float(payload["best_val_loss"])
        bad_epochs = int(payload["bad_epochs"])
        if args.override_learning_rate is not None:
            # Must come after load_state_dict, which restores the stored rate.
            for group in optimizer.param_groups:
                group["lr"] = args.override_learning_rate
            scheduler.num_bad_epochs = 0
            print(
                f"forced learning rate to {args.override_learning_rate:g} "
                f"and cleared the scheduler patience",
                flush=True,
            )

    train_dataset = LRS2MixDataset(args.data_root, "tr", segment_samples=32000)
    val_dataset = LRS2MixDataset(args.data_root, "cv", segment_samples=32000)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = parameter_count(model)
    metadata = {
        "implementation": "independent physical pruning reconstruction",
        "model": args.model,
        "initialization": "surviving weights inherited from baseline",
        "baseline_checkpoint": str(Path(args.baseline_checkpoint).resolve()),
        "masks": str(Path(args.masks).resolve()),
        "channels_kept": [int(mask.sum()) for mask in masks],
        "parameters": parameters,
        "epochs": args.epochs,
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
    append = bool(args.resume)
    started = time.monotonic()
    stop_reason = "epochs_completed"
    with training_path.open("a" if append else "w", newline="", encoding="utf-8") as train_stream, validation_path.open(
        "a" if append else "w", newline="", encoding="utf-8"
    ) as val_stream:
        train_writer = csv.DictWriter(
            train_stream,
            fieldnames=["global_step", "epoch", "step_in_epoch", "loss", "gradient_norm", "learning_rate", "elapsed_seconds"],
        )
        val_writer = csv.DictWriter(
            val_stream,
            fieldnames=["epoch", "global_step", "val_loss", "val_si_sdri_proxy", "learning_rate", "elapsed_seconds"],
        )
        if not append:
            train_writer.writeheader()
            val_writer.writeheader()

        should_stop = False
        model.train()
        for epoch in range(start_epoch, args.epochs):
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
                    raise RuntimeError(
                        f"Non-finite loss at epoch={epoch + 1}, step={batch_index + 1}: {loss}"
                    )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("Non-finite gradient norm")
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
                        f"pruned={args.model} epoch={epoch + 1}/{args.epochs} "
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
            # `improved` decides which weights are kept, so it stays strict: the
            # best checkpoint should be the genuinely best one. `meaningful`
            # decides whether progress is real enough to keep waiting, so it is
            # the one the threshold applies to.
            improved = val_loss < best_val_loss
            meaningful = val_loss < best_val_loss - args.plateau_threshold
            if improved:
                best_val_loss = val_loss
            bad_epochs = 0 if meaningful else bad_epochs + 1
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
                f"pruned={args.model} epoch={epoch + 1}/{args.epochs} "
                f"val_si_sdr={-val_loss:.4f} best={-best_val_loss:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g} bad_epochs={bad_epochs}",
                flush=True,
            )
            if bad_epochs >= args.early_stop_patience:
                stop_reason = "early_stopped"
                should_stop = True
                break

    result = {
        "status": "smoke_completed" if stop_reason == "max_steps_reached" else "completed",
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

