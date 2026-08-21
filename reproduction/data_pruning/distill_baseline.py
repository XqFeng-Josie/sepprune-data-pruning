"""K0: the dense teacher used for distillation instead of for data selection.

The whole study rests on using the dense model as a scorer. A reviewer will
immediately ask why that model is not simply distilled into the pruned student,
which costs nothing extra here: the student's surviving weights were inherited
from exactly this teacher. K0 answers that question. It trains on the full
split, so it is not a data-pruning arm at all - it is the anchor that says how
much of any observed gain really comes from *selecting* data.

The objective adds an output-level distillation term to the unchanged task loss:

    L = (1 - alpha) * PIT-NegSNR(student, targets)
      +      alpha  * NegSNR(student, teacher | same permutation)

Two permutation details make this correct rather than approximately correct:

1. The teacher's two output channels are first reordered to match the ground
   truth, so "teacher source 0" and "target source 0" always mean the same
   speaker.
2. The distillation term reuses the permutation the task loss chose, instead of
   running its own PIT. Letting the two terms disagree would have the gradient
   pull the same output channel towards two different speakers. In practice the
   teacher is good enough that they almost always agree, but "almost always" is
   not a property worth relying on inside a loss function.

Nothing here defines its own training loop. `train_data_pruned.py` calls
`distillation_loss` when `--distill-alpha` is non-zero, so K0 and every data
arm run through byte-identical training code and differ only in the objective.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from ..original_models import build_original_model, parameter_count
from ..train_original import pairwise_negative_sdr

__all__ = ["build_teacher", "align_to_targets", "distillation_loss", "DistillationTerms"]


class DistillationTerms(dict):
    """Per-example task and distillation losses plus the blended objective."""

    @property
    def total(self) -> Tensor:
        return self["total"]


def build_teacher(
    model_name: str, checkpoint: str | Path, device: torch.device
) -> nn.Module:
    """Load the frozen dense teacher.

    Every parameter has `requires_grad` cleared as well as being wrapped in
    `no_grad` at call time, so a teacher tensor can never join the student's
    autograd graph even if the call site changes later.
    """

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("model_name") != model_name:
        raise ValueError(f"Teacher checkpoint is for {payload.get('model_name')}, not {model_name}")
    model = build_original_model(model_name)
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _permutation_losses(estimate: Tensor, target: Tensor, kind: str) -> tuple[Tensor, Tensor]:
    """Per-example loss of the two source assignments, `(direct, swapped)`.

    `pairwise_negative_sdr` returns `pair[b, i, j]` for estimate `i` against
    target `j`, so the direct assignment is the diagonal and the swapped one is
    the anti-diagonal.
    """

    pair = pairwise_negative_sdr(estimate, target, kind)
    direct = 0.5 * (pair[:, 0, 0] + pair[:, 1, 1])
    swapped = 0.5 * (pair[:, 0, 1] + pair[:, 1, 0])
    return direct, swapped


def align_to_targets(estimate: Tensor, target: Tensor, kind: str) -> Tensor:
    """Reorder `estimate`'s two sources to its best match against `target`."""

    direct, swapped = _permutation_losses(estimate, target, kind)
    swap = (swapped < direct).view(-1, 1, 1)
    return torch.where(swap, estimate.flip(1), estimate)


def distillation_loss(
    estimate: Tensor,
    target: Tensor,
    teacher_estimate: Tensor,
    *,
    alpha: float,
    kind: str,
) -> DistillationTerms:
    """Blend the PIT task loss with a permutation-consistent distillation term.

    `alpha == 0` reproduces the plain task loss exactly, which is what makes K0
    comparable to S0: the only difference between them is this one scalar.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if estimate.shape != target.shape or estimate.shape != teacher_estimate.shape:
        raise ValueError(
            "student, target and teacher must share a shape; got "
            f"{tuple(estimate.shape)}, {tuple(target.shape)}, {tuple(teacher_estimate.shape)}"
        )

    direct, swapped = _permutation_losses(estimate, target, kind)
    task = torch.minimum(direct, swapped)
    if alpha == 0.0:
        zero = torch.zeros_like(task)
        return DistillationTerms(task=task, distillation=zero, total=task)

    use_swapped = swapped < direct
    teacher_aligned = align_to_targets(teacher_estimate.detach(), target, kind)
    kd_direct, kd_swapped = _permutation_losses(estimate, teacher_aligned, kind)
    distillation = torch.where(use_swapped, kd_swapped, kd_direct)
    return DistillationTerms(
        task=task,
        distillation=distillation,
        total=(1.0 - alpha) * task + alpha * distillation,
    )


def describe_teacher(model: nn.Module, checkpoint: str | Path) -> dict:
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "parameters": parameter_count(model),
        "role": "frozen dense teacher, output-level distillation only",
    }
