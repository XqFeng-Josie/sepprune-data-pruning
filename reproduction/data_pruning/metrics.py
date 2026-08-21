"""Batched permutation-invariant SI-SDR built on the existing scalar helper.

`reproduction.evaluate_original.si_sdr` already reduces over the last axis and
broadcasts over every leading axis, so it is reused verbatim here rather than
reimplemented. Only the two-source permutation wrapper is new, and
`verify.py` asserts that it agrees with `evaluate_original.best_si_sdr`
element by element.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..evaluate_original import si_sdr

__all__ = ["pit_best_si_sdr", "SOURCES"]

SOURCES = 2


def pit_best_si_sdr(estimate: Tensor, target: Tensor) -> Tensor:
    """PIT-best SI-SDR averaged over the two sources.

    Accepts ``[2, T]`` (returns a scalar) or ``[batch, 2, T]`` (returns
    ``[batch]``). The arithmetic mirrors `evaluate_original.best_si_sdr`
    exactly: per-source SI-SDR is summed over the assignment, the better of the
    two assignments is kept, and the result is halved.
    """

    if estimate.shape != target.shape:
        raise ValueError(f"Shape mismatch: {tuple(estimate.shape)} vs {tuple(target.shape)}")
    if estimate.ndim not in (2, 3):
        raise ValueError(f"Expected [2, T] or [batch, 2, T], got {tuple(estimate.shape)}")
    squeeze = estimate.ndim == 2
    if squeeze:
        estimate = estimate.unsqueeze(0)
        target = target.unsqueeze(0)
    if estimate.shape[1] != SOURCES:
        raise ValueError(f"Expected {SOURCES} sources, got {estimate.shape[1]}")

    direct = si_sdr(estimate[:, 0], target[:, 0]) + si_sdr(estimate[:, 1], target[:, 1])
    swapped = si_sdr(estimate[:, 0], target[:, 1]) + si_sdr(estimate[:, 1], target[:, 0])
    best = 0.5 * torch.maximum(direct, swapped)
    return best.squeeze(0) if squeeze else best
