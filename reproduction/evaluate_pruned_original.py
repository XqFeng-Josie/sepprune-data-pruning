"""Evaluate a physically pruned A-FRCNN-12 or SuDoRM-RF checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fast_bss_eval
import torch

from .evaluate_original import best_si_sdr
from .lrs2 import LRS2MixDataset
from .original_models import MODEL_SPECS, parameter_count
from .train_pruned_original import build_inherited_pruned_model


PAPER_PRUNED = {
    "afrcnn12": {"parameters": 3_060_000, "sdri": 12.59, "si_sdri": 12.25},
    "sudormrf": {"parameters": 1_540_000, "sdri": 10.37, "si_sdri": 9.98},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["afrcnn12", "sudormrf"], required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--split", choices=["cv", "tt"], default="tt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sdr-filter-length", type=int, default=512)
    parser.add_argument("--label", default="seprune")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, _ = build_inherited_pruned_model(
        args.model, args.baseline_checkpoint, args.masks
    )
    checkpoint_label = "inherited_unfinetuned"
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if int(payload.get("parameters", -1)) != parameter_count(model):
            raise ValueError("Evaluation checkpoint has a different pruned structure")
        model.load_state_dict(payload["state_dict"], strict=True)
        checkpoint_label = str(Path(args.checkpoint).resolve())
    model = model.to(device).eval()

    dataset = LRS2MixDataset(args.data_root, args.split, segment_samples=None)
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    rows = []
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
                    f"pruned={args.model} {index + 1}/{count} "
                    f"SDRi={row['sdri']:.3f} SI-SDRi={row['si_sdri']:.3f}",
                    flush=True,
                )

    metrics = ("sdr", "sdri", "si_sdr", "si_sdri")
    spec = MODEL_SPECS[args.model]
    paper = PAPER_PRUNED[args.model]
    summary = {
        "model": args.model,
        "label": args.label,
        "checkpoint": checkpoint_label,
        "masks": str(Path(args.masks).resolve()),
        "split": args.split,
        "count": count,
        "parameters": parameter_count(model),
        "paper_original_sdri": spec.paper_sdri,
        "paper_original_si_sdri": spec.paper_si_sdri,
        "paper_pruned_parameters": paper["parameters"],
        "paper_pruned_sdri": paper["sdri"],
        "paper_pruned_si_sdri": paper["si_sdri"],
        **{metric: sum(float(row[metric]) for row in rows) / len(rows) for metric in metrics},
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.label}_{args.split}"
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", *metrics])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

