"""Evaluate one of the three unpruned SepPrune backbones on LRS2-2Mix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fast_bss_eval
import torch
from torch import Tensor

from .lrs2 import LRS2MixDataset
from .original_models import MODEL_SPECS, build_original_model, parameter_count


def si_sdr(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    projection = (estimate * target).sum(dim=-1, keepdim=True) * target
    projection = projection / (target.square().sum(dim=-1, keepdim=True) + eps)
    noise = estimate - projection
    return 10.0 * torch.log10(
        projection.square().sum(dim=-1) / (noise.square().sum(dim=-1) + eps) + eps
    )


def best_si_sdr(estimate: Tensor, target: Tensor) -> Tensor:
    direct = si_sdr(estimate[0], target[0]) + si_sdr(estimate[1], target[1])
    swapped = si_sdr(estimate[0], target[1]) + si_sdr(estimate[1], target[0])
    return 0.5 * torch.maximum(direct, swapped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--split", choices=["cv", "tt"], default="tt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sdr-filter-length", type=int, default=512)
    parser.add_argument("--output-dir", default="experiments/original_baselines_lrs2")
    args = parser.parse_args()

    device = torch.device(args.device)
    spec = MODEL_SPECS[args.model]
    if args.model == "tdanet":
        model = build_original_model("tdanet", tdanet_checkpoint=args.checkpoint)
        checkpoint_label = args.checkpoint or "JusperLee/TDANetBest-4ms-LRS2"
    else:
        if not args.checkpoint:
            raise ValueError(f"--checkpoint is required for {args.model}")
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if payload.get("model_name") != args.model:
            raise ValueError(f"Checkpoint is for {payload.get('model_name')}, not {args.model}")
        model = build_original_model(args.model)
        model.load_state_dict(payload["state_dict"], strict=True)
        checkpoint_label = str(Path(args.checkpoint).resolve())
    model = model.to(device).eval()

    dataset = LRS2MixDataset(args.data_root, args.split, segment_samples=None)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for index in range(count):
            mixture, sources, name = dataset[index]
            mixture = mixture.to(device)
            sources = sources.to(device)
            estimate = model(mixture.unsqueeze(0)).squeeze(0)
            repeated_mix = mixture.expand_as(sources)
            separated_si_sdr = best_si_sdr(estimate, sources)
            mixture_si_sdr = best_si_sdr(repeated_mix, sources)
            separated_sdr = -fast_bss_eval.sdr_pit_loss(
                sources, estimate, filter_length=args.sdr_filter_length
            ).mean()
            mixture_sdr = -fast_bss_eval.sdr_pit_loss(
                repeated_mix, sources, filter_length=args.sdr_filter_length
            ).mean()
            row = {
                "name": name,
                "sdr": float(separated_sdr.cpu()),
                "sdri": float((separated_sdr - mixture_sdr).cpu()),
                "si_sdr": float(separated_si_sdr.cpu()),
                "si_sdri": float((separated_si_sdr - mixture_si_sdr).cpu()),
            }
            rows.append(row)
            if index == 0 or (index + 1) % 10 == 0:
                print(
                    f"model={args.model} {index + 1}/{count} SDRi={row['sdri']:.3f} "
                    f"SI-SDRi={row['si_sdri']:.3f}",
                    flush=True,
                )

    metrics = ("sdr", "sdri", "si_sdr", "si_sdri")
    summary = {
        "model": args.model,
        "checkpoint": checkpoint_label,
        "split": args.split,
        "count": count,
        "parameters": parameter_count(model),
        "paper_original_lrs2_sdri": spec.paper_sdri,
        "paper_original_lrs2_si_sdri": spec.paper_si_sdri,
        **{metric: sum(float(row[metric]) for row in rows) / len(rows) for metric in metrics},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{args.model}_{args.split}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", *metrics])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{args.model}_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

