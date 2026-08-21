"""Measure TDANet parameters and MACs with the paper's ptflops protocol."""

from __future__ import annotations

import argparse
import json

import torch
from ptflops import get_model_complexity_info

from .tdanet_seprune import load_official_tdanet, physically_prune_ffn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mask", default=None)
    parser.add_argument("--samples", type=int, default=16000)
    args = parser.parse_args()

    model = load_official_tdanet(args.checkpoint).eval()
    label = "original"
    if args.mask:
        mask = torch.load(args.mask, map_location="cpu", weights_only=True)
        model = physically_prune_ffn(model, mask).eval()
        label = "pruned"
    macs, parameters = get_model_complexity_info(
        model,
        (1, args.samples),
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
    )
    print(
        json.dumps(
            {
                "model": label,
                "input_samples": args.samples,
                "parameters": int(parameters),
                "macs": int(macs),
                "gmac": macs / 1e9,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

