"""Structural and numerical checks for the reconstructed pruning operators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .original_models import build_original_model, parameter_count
from .seprune_originals import attach_original_masks, physically_prune_original
from .train_original import pit_negative_sdr, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["afrcnn12", "sudormrf"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples", type=int, default=3200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=True)
    model = build_original_model(args.model)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    original_parameters = parameter_count(model)

    mixture = torch.randn(1, 1, args.samples)
    with torch.inference_mode():
        reference = model(mixture)

    count = 4 if args.model == "afrcnn12" else 16
    channels = 512
    all_ones = [torch.ones(channels) for _ in range(count)]
    identical = physically_prune_original(model, args.model, all_ones).eval()
    with torch.inference_mode():
        identical_output = identical(mixture)
    max_all_kept_error = float((reference - identical_output).abs().max())
    if max_all_kept_error > 1e-6:
        raise RuntimeError(f"All-kept physical rebuild changed output: {max_all_kept_error}")
    if parameter_count(identical) != original_parameters:
        raise RuntimeError("All-kept physical rebuild changed parameter count")

    partial = []
    for index in range(count):
        keep = torch.zeros(channels)
        keep[index % 2 :: 2] = 1
        partial.append(keep)
    physically_pruned = physically_prune_original(model, args.model, partial).to(device).eval()
    probe = mixture.to(device)
    with torch.inference_mode():
        partial_output = physically_pruned(probe)
    if partial_output.shape != reference.shape or not torch.isfinite(partial_output).all():
        raise RuntimeError("Partially pruned forward failed shape/finite check")
    if parameter_count(physically_pruned) >= original_parameters:
        raise RuntimeError("Partial pruning did not reduce parameters")

    masked = build_original_model(args.model)
    masked.load_state_dict(payload["state_dict"], strict=True)
    masked = masked.to(device).eval()
    wrappers = attach_original_masks(
        masked,
        args.model,
        epsilon=0.7,
        temperature=1.0,
        seed=args.seed,
    )
    target = torch.randn(1, 2, args.samples, device=device)
    estimate = masked(probe)
    loss = pit_negative_sdr(estimate, target, kind="snr", threshold_byloss=True)
    loss.backward()
    mask_grads = [float(wrapper.mask.alpha.grad.norm().cpu()) for wrapper in wrappers]
    frozen_grads = [
        name for name, parameter in masked.named_parameters()
        if "mask.alpha" not in name and parameter.grad is not None
    ]
    if not torch.isfinite(torch.tensor(mask_grads)).all() or any(
        value == 0.0 for value in mask_grads
    ):
        raise RuntimeError(f"Invalid mask gradients: {mask_grads}")
    if frozen_grads:
        raise RuntimeError(f"Frozen parameters received gradients: {frozen_grads[:5]}")

    result = {
        "status": "passed",
        "model": args.model,
        "original_parameters": original_parameters,
        "half_channel_parameters": parameter_count(physically_pruned),
        "max_all_kept_output_error": max_all_kept_error,
        "output_shape": list(partial_output.shape),
        "partial_output_finite": bool(torch.isfinite(partial_output).all()),
        "mask_gradient_norms": mask_grads,
        "frozen_parameters_with_grad": frozen_grads,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
