"""Benchmark TDANet wall time and peak allocated CUDA memory."""

from __future__ import annotations

import argparse
import json
import time

import torch

from .tdanet_seprune import load_official_tdanet, physically_prune_ffn


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=["inference", "training"], default="inference")
    parser.add_argument("--samples", type=int, default=16000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_official_tdanet(args.checkpoint)
    label = "original"
    if args.mask:
        mask = torch.load(args.mask, map_location="cpu", weights_only=True)
        model = physically_prune_ffn(model, mask)
        label = "pruned"
    model = model.to(device)
    model.train(args.mode == "training")
    waveform = torch.randn(1, 1, args.samples, device=device)

    def step() -> None:
        if args.mode == "training":
            model.zero_grad(set_to_none=True)
            model(waveform).square().mean().backward()
        else:
            with torch.inference_mode():
                model(waveform)

    for _ in range(args.warmup):
        step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(args.repeats):
        step()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated(device) / 2**20
        if device.type == "cuda"
        else None
    )
    print(
        json.dumps(
            {
                "model": label,
                "mode": args.mode,
                "device": str(device),
                "samples": args.samples,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "milliseconds_per_iteration": elapsed * 1000 / args.repeats,
                "peak_allocated_mib": peak,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

