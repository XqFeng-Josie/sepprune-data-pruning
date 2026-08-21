"""Factories for the three unpruned SepPrune LRS2 backbone models."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

from torch import nn

from .tdanet_seprune import load_official_tdanet


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class OriginalModelSpec:
    name: str
    paper_parameters: int
    paper_sdri: float
    paper_si_sdri: float
    reported_training_epochs: int | None


MODEL_SPECS = {
    "tdanet": OriginalModelSpec("tdanet", 2_350_000, 12.74, 12.45, 493),
    "afrcnn12": OriginalModelSpec("afrcnn12", 5_130_000, 10.90, 10.50, 136),
    "sudormrf": OriginalModelSpec("sudormrf", 2_720_000, 11.43, 11.10, 86),
}


@lru_cache(maxsize=None)
def _load_source_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_original_model(name: str, *, tdanet_checkpoint: str | None = None) -> nn.Module:
    """Construct a paper-sized unpruned model.

    TDANet uses its public LRS2 checkpoint. A-FRCNN-12 and SuDoRM-RF use the
    official architecture sources and are randomly initialized for training.
    """

    if name == "tdanet":
        return load_official_tdanet(tdanet_checkpoint)
    if name == "afrcnn12":
        module = _load_source_module(
            "seprune_official_afrcnn",
            ROOT / "third_party" / "AFRCNN" / "AFRCNN.py",
        )
        return module.AFRCNN(
            out_channels=512,
            in_channels=512,
            num_blocks=12,
            upsampling_depth=4,
            enc_kernel_size=41,
            enc_num_basis=512,
            num_sources=2,
        )
    if name == "sudormrf":
        module = _load_source_module(
            "seprune_official_sudormrf",
            ROOT
            / "third_party"
            / "sudo_rm_rf"
            / "sudo_rm_rf"
            / "dnn"
            / "models"
            / "improved_sudormrf.py",
        )
        return module.SuDORMRF(
            out_channels=128,
            in_channels=512,
            num_blocks=16,
            upsampling_depth=5,
            enc_num_basis=512,
            enc_kernel_size=21,
            num_sources=2,
        )
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

