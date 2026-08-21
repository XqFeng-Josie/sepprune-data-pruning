"""Minimal TDANet SepPrune loop.

This module reconstructs the TDANet path visible in the released SepPrune
scripts: learn a mask for the hidden channels of
``sm.unet.globalatt.mlp.fc1``, then physically slice fc1, its depth-wise
convolution and fc2. It is an independent reconstruction because the released
repository does not contain the authors' modified ``look2hear`` package.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.torch_version import TorchVersion


ROOT = Path(__file__).resolve().parents[1]
TDANET_ROOT = ROOT / "third_party" / "TDANet"
if not TDANET_ROOT.exists():
    raise RuntimeError(
        "Missing third_party/TDANet. Run scripts/bootstrap_reproduction.sh first."
    )
sys.path.insert(0, str(TDANET_ROOT))

import look2hear.models  # noqa: E402


DEFAULT_REPO = "JusperLee/TDANetBest-4ms-LRS2"
DEFAULT_REVISION = "d10e423ef25bc6f09f907455feb3f1030e9e3add"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_official_tdanet(
    checkpoint: str | None = None, *, load_weights: bool = True
) -> nn.Module:
    """Build TDANet from a safely loaded official checkpoint description.

    Setting ``load_weights=False`` uses the checkpoint only for architecture
    metadata and leaves the model at its constructor's random initialization.
    """

    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(
            DEFAULT_REPO, "pytorch_model.bin", revision=DEFAULT_REVISION
        )

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    # Old TDANet checkpoints store torch.__version__ as TorchVersion. Allow only
    # that harmless metadata class while retaining weights_only=True.
    with torch.serialization.safe_globals([TorchVersion]):
        payload: dict[str, Any] = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )

    model_cls = look2hear.models.get(payload["model_name"])
    model = model_cls(**payload["model_args"])
    if load_weights:
        model.load_state_dict(payload["state_dict"], strict=True)
    return model


class DifferentiableChannelMask(nn.Module):
    """Per-channel hard threshold with a clipped straight-through gradient."""

    def __init__(
        self,
        channels: int,
        epsilon: float = 0.7,
        temperature: float = 1.0,
        seed: int = 0,
        initial_probability_low: float = 0.55,
        initial_probability_high: float = 0.85,
    ) -> None:
        super().__init__()
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 < initial_probability_low < initial_probability_high < 1.0:
            raise ValueError("initial probability bounds must satisfy 0 < low < high < 1")
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)

        generator = torch.Generator().manual_seed(seed)
        # A broad initialization makes the smoke test exercise both branches.
        keep_probability = torch.empty(channels).uniform_(
            initial_probability_low,
            initial_probability_high,
            generator=generator,
        )
        logits = torch.logit(keep_probability.clamp(1e-4, 1 - 1e-4))
        self.alpha = nn.Parameter(logits)

    def probabilities(self) -> Tensor:
        return torch.sigmoid(self.alpha / self.temperature)

    def deterministic_mask(self) -> Tensor:
        return (self.probabilities() >= self.epsilon).to(self.alpha.dtype)

    def forward(self, x: Tensor, stochastic: bool = True) -> Tensor:
        if stochastic:
            uniform = torch.rand_like(self.alpha).clamp_(1e-6, 1 - 1e-6)
            logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
            probability = torch.sigmoid(
                (self.alpha + logistic_noise) / self.temperature
            )
        else:
            probability = self.probabilities()

        hard = (probability >= self.epsilon).to(probability.dtype)
        # Hard values in the forward pass and an identity-like, bounded gradient
        # through the probability in the backward pass.
        straight_through = hard + (probability - probability.detach())
        return x * straight_through.clamp(-1.0, 1.0).view(1, -1, 1)


class MaskedFFN(nn.Module):
    """Wrap the original TDANet FFN without changing its pretrained weights."""

    def __init__(
        self,
        base: nn.Module,
        epsilon: float,
        temperature: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.base = base
        channels = base.fc1.conv.out_channels
        self.mask = DifferentiableChannelMask(
            channels, epsilon=epsilon, temperature=temperature, seed=seed
        )
        self.stochastic = True

    def forward(self, x: Tensor) -> Tensor:
        x = self.base.fc1(x)
        x = self.mask(x, stochastic=self.stochastic)
        x = self.base.dwconv(x)
        x = self.base.act(x)
        x = self.base.drop(x)
        x = self.base.fc2(x)
        return self.base.drop(x)


def attach_mask(
    model: nn.Module, epsilon: float, temperature: float, seed: int
) -> MaskedFFN:
    base = model.sm.unet.globalatt.mlp
    model_device = next(model.parameters()).device
    masked = MaskedFFN(base, epsilon, temperature, seed).to(model_device)
    model.sm.unet.globalatt.mlp = masked
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    masked.mask.alpha.requires_grad_(True)
    return masked


def si_sdr(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    scale = (estimate * target).sum(dim=-1, keepdim=True)
    scale = scale / (target.square().sum(dim=-1, keepdim=True) + eps)
    projection = scale * target
    noise = estimate - projection
    return 10.0 * torch.log10(
        (projection.square().sum(dim=-1) + eps)
        / (noise.square().sum(dim=-1) + eps)
    )


def pit_negative_si_sdr(estimate: Tensor, target: Tensor) -> Tensor:
    if estimate.shape[1] != 2 or target.shape[1] != 2:
        raise ValueError("The minimal PIT implementation expects exactly two sources")
    direct = si_sdr(estimate[:, 0], target[:, 0]) + si_sdr(
        estimate[:, 1], target[:, 1]
    )
    swapped = si_sdr(estimate[:, 0], target[:, 1]) + si_sdr(
        estimate[:, 1], target[:, 0]
    )
    return -0.5 * torch.maximum(direct, swapped).mean()


def synthetic_batch(
    batch_size: int, samples: int, sample_rate: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    time = torch.arange(samples, device=device, dtype=torch.float32) / sample_rate
    sources: list[Tensor] = []
    for source_index in range(2):
        frequency = torch.empty(batch_size, 1, device=device).uniform_(
            120.0 + source_index * 170.0, 260.0 + source_index * 260.0
        )
        phase = torch.empty(batch_size, 1, device=device).uniform_(0.0, 2 * math.pi)
        amplitude = torch.empty(batch_size, 1, device=device).uniform_(0.4, 1.0)
        fundamental = amplitude * torch.sin(2 * math.pi * frequency * time + phase)
        harmonic = 0.25 * amplitude * torch.sin(
            4 * math.pi * frequency * time + 0.5 * phase
        )
        sources.append(fundamental + harmonic)
    target = torch.stack(sources, dim=1)
    mixture = target.sum(dim=1)
    peak = mixture.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    return (mixture / peak).unsqueeze(1), target / peak.unsqueeze(1)


def physically_prune_ffn(model: nn.Module, keep_mask: Tensor) -> nn.Module:
    """Slice the exact TDANet FFN dependency chain used by released scripts."""

    if isinstance(model.sm.unet.globalatt.mlp, MaskedFFN):
        model.sm.unet.globalatt.mlp = model.sm.unet.globalatt.mlp.base
    original = model.sm.unet.globalatt.mlp
    keep = keep_mask.detach().bool().cpu()
    if keep.sum() == 0:
        raise ValueError("Cannot create an FFN with zero hidden channels")

    pruned_model = copy.deepcopy(model).cpu()
    old = pruned_model.sm.unet.globalatt.mlp
    hidden = int(keep.sum())
    new = type(old)(old.fc1.conv.in_channels, hidden, drop=old.drop.p)

    with torch.no_grad():
        new.fc1.conv.weight.copy_(old.fc1.conv.weight[keep])
        new.fc1.norm.gamma.copy_(old.fc1.norm.gamma[keep])
        new.fc1.norm.beta.copy_(old.fc1.norm.beta[keep])
        new.dwconv.weight.copy_(old.dwconv.weight[keep])
        new.dwconv.bias.copy_(old.dwconv.bias[keep])
        new.fc2.conv.weight.copy_(old.fc2.conv.weight[:, keep, :])
        new.fc2.norm.load_state_dict(old.fc2.norm.state_dict())

    pruned_model.sm.unet.globalatt.mlp = new
    return pruned_model


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    model = load_official_tdanet(args.checkpoint).to(device).eval()
    original_parameters = parameter_count(model)
    masked = attach_mask(model, args.epsilon, args.temperature, args.seed)
    optimizer = torch.optim.Adam([masked.mask.alpha], lr=args.learning_rate)

    losses: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(args.iterations):
        mixture, target = synthetic_batch(
            args.batch_size, args.samples, args.sample_rate, device
        )
        optimizer.zero_grad(set_to_none=True)
        estimate = model(mixture)
        loss = pit_negative_si_sdr(estimate, target)
        loss.backward()
        gradient_norms.append(float(masked.mask.alpha.grad.norm().detach().cpu()))
        torch.nn.utils.clip_grad_norm_([masked.mask.alpha], max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    masked.stochastic = False
    keep_mask = masked.mask.deterministic_mask()
    kept = int(keep_mask.sum().item())
    total = int(keep_mask.numel())
    if kept == 0 or kept == total:
        raise RuntimeError(f"Degenerate learned mask: kept {kept}/{total} channels")

    pruned = physically_prune_ffn(model, keep_mask)
    pruned_parameters = parameter_count(pruned)
    pruned = pruned.to(device).eval()
    mixture, _ = synthetic_batch(1, args.samples, args.sample_rate, device)
    with torch.inference_mode():
        output = pruned(mixture)

    result = {
        "status": "passed",
        "scope": "synthetic smoke test; not a paper metric reproduction",
        "device": str(device),
        "seed": args.seed,
        "iterations": args.iterations,
        "epsilon": args.epsilon,
        "temperature": args.temperature,
        "losses": losses,
        "gradient_norms": gradient_norms,
        "channels_total": total,
        "channels_kept": kept,
        "channel_keep_ratio": kept / total,
        "original_parameters": original_parameters,
        "pruned_parameters": pruned_parameters,
        "parameter_reduction": 1.0 - pruned_parameters / original_parameters,
        "output_shape": list(output.shape),
        "output_finite": bool(torch.isfinite(output).all().item()),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(keep_mask.cpu(), output_dir / "tdanet_smoke_mask.pt")
    (output_dir / "tdanet_smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--epsilon", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--output-dir", default="experiments/smoke_tdanet")
    return parser


if __name__ == "__main__":
    outcome = run_smoke(build_parser().parse_args())
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
