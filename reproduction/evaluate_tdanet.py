"""Evaluate original or physically pruned TDANet on LRS2-2Mix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import fast_bss_eval
import torch

from .lrs2 import LRS2MixDataset
from .tdanet_seprune import load_official_tdanet, physically_prune_ffn, si_sdr


def best_si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = si_sdr(estimate[0], target[0]) + si_sdr(estimate[1], target[1])
    swapped = si_sdr(estimate[0], target[1]) + si_sdr(estimate[1], target[0])
    return 0.5 * torch.maximum(direct, swapped)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/LRS2-2Mix")
    parser.add_argument("--split", choices=["cv", "tt"], default="tt")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mask", default=None, help="Optional mask.pt from mask learning")
    parser.add_argument(
        "--trained-state",
        default=None,
        help="Optional finetune/random-baseline checkpoint containing state_dict",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sdr-filter-length", type=int, default=512)
    parser.add_argument("--output-dir", default="experiments/tdanet_lrs2_eval")
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = LRS2MixDataset(args.data_root, args.split, segment_samples=None)
    model = load_official_tdanet(args.checkpoint)
    label = "original"
    if args.mask:
        mask = torch.load(args.mask, map_location="cpu", weights_only=True)
        model = physically_prune_ffn(model, mask)
        label = "pruned"
    if args.trained_state:
        payload = torch.load(args.trained_state, map_location="cpu", weights_only=True)
        state_dict = payload.get("state_dict", payload)
        model.load_state_dict(state_dict, strict=True)
        label = str(payload.get("label", "trained_pruned"))
    model = model.to(device).eval()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    count = min(len(dataset), args.limit) if args.limit else len(dataset)
    with torch.inference_mode():
        for index in range(count):
            mixture, sources, name = dataset[index]
            mixture = mixture.to(device)
            sources = sources.to(device)
            estimate = model(mixture.unsqueeze(0)).squeeze(0)
            repeated_mix = mixture.expand_as(sources)

            separated_si_sdr = best_si_sdr(estimate, sources)
            mixture_si_sdr = best_si_sdr(repeated_mix, sources)
            # Match the argument order in the released MetricsTracker exactly.
            separated_sdr = -fast_bss_eval.sdr_pit_loss(
                sources,
                estimate,
                filter_length=args.sdr_filter_length,
            ).mean()
            mixture_sdr = -fast_bss_eval.sdr_pit_loss(
                repeated_mix,
                sources,
                filter_length=args.sdr_filter_length,
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
                    f"{index + 1}/{count} SDRi={row['sdri']:.3f} "
                    f"SI-SDRi={row['si_sdri']:.3f}",
                    flush=True,
                )

    metrics = ("sdr", "sdri", "si_sdr", "si_sdri")
    summary = {
        "model": label,
        "split": args.split,
        "count": count,
        "paper_original_tdanet_lrs2_sdri": 12.74,
        "paper_original_tdanet_lrs2_si_sdri": 12.45,
        **{
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in metrics
        },
    }
    with (output_dir / f"{label}_{args.split}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", *metrics])
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{label}_{args.split}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
