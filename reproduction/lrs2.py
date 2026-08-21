"""Portable LRS2-2Mix audio loader without PyTorch Lightning."""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset


def resolve_audio_root(path: str | Path) -> Path:
    """Find the directory directly containing tr/cv/tt subdirectories."""

    candidate = Path(path).expanduser().resolve()
    probes = [
        candidate,
        candidate / "wav16k" / "min",
        candidate / "audio" / "wav16k" / "min",
        candidate / "lrs2_rebuild" / "wav16k" / "min",
        candidate / "lrs2_rebuild" / "audio" / "wav16k" / "min",
    ]
    for probe in probes:
        if all((probe / split).is_dir() for split in ("tr", "cv", "tt")):
            return probe
    matches = [
        directory
        for directory in candidate.rglob("min")
        if all((directory / split).is_dir() for split in ("tr", "cv", "tt"))
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not uniquely locate a tr/cv/tt audio root under {candidate}; "
        f"candidates={matches}"
    )


class LRS2MixDataset(Dataset[tuple[Tensor, Tensor, str]]):
    def __init__(
        self,
        root: str | Path,
        split: str,
        sample_rate: int = 16000,
        segment_samples: int | None = None,
    ) -> None:
        if split not in {"tr", "cv", "tt"}:
            raise ValueError(f"Unknown split: {split}")
        self.root = resolve_audio_root(root)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_samples = segment_samples

        by_kind: dict[str, dict[str, Path]] = {}
        for kind in ("mix", "s1", "s2"):
            directory = self.root / split / kind
            by_kind[kind] = {path.name: path for path in directory.glob("*.wav")}
        names = set(by_kind["mix"])
        if names != set(by_kind["s1"]) or names != set(by_kind["s2"]):
            raise RuntimeError(f"Mismatched mix/s1/s2 filenames in {self.root / split}")
        self.names = sorted(names)
        self.paths = by_kind
        if not self.names:
            raise RuntimeError(f"No wav files found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.names)

    def _read(self, path: Path) -> Tensor:
        audio, rate = sf.read(path, dtype="float32", always_2d=False)
        if rate != self.sample_rate:
            raise RuntimeError(f"Expected {self.sample_rate} Hz, got {rate}: {path}")
        if audio.ndim != 1:
            raise RuntimeError(f"Expected mono audio, got shape {audio.shape}: {path}")
        return torch.from_numpy(audio)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, str]:
        name = self.names[index]
        mixture = self._read(self.paths["mix"][name])
        sources = torch.stack(
            [self._read(self.paths["s1"][name]), self._read(self.paths["s2"][name])]
        )
        length = min(mixture.shape[-1], sources.shape[-1])
        mixture = mixture[:length]
        sources = sources[:, :length]
        if self.segment_samples is not None:
            if length < self.segment_samples:
                raise RuntimeError(
                    f"{name} has {length} samples, shorter than {self.segment_samples}"
                )
            maximum_start = length - self.segment_samples
            start = (
                int(torch.randint(maximum_start + 1, ()).item())
                if maximum_start > 0
                else 0
            )
            stop = start + self.segment_samples
            mixture = mixture[start:stop]
            sources = sources[:, start:stop]
        return mixture.unsqueeze(0), sources, name

