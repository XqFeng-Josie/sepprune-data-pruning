"""Provenance helpers shared by every data-pruning component.

Every artefact this package writes must be traceable back to the exact
checkpoints and code that produced it, because a subset file is meaningless
without knowing which model scored it.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import time
from pathlib import Path
from typing import Any

__all__ = [
    "sha256_file",
    "sha256_text",
    "derive_seed",
    "seeded_rng",
    "git_commit",
    "utc_now",
    "write_json",
    "read_json",
    "atomic_write_text",
    "set_tf32",
    "tf32_state",
    "gpu_process_count",
]


def gpu_process_count() -> int | None:
    """How many processes are sharing the GPU right now.

    Batch-size-1 separation training is launch-latency bound, not compute bound,
    so a second job costs the incumbent only a few percent while a fifth costs
    it ~40%. Wall-clock is therefore only comparable between runs that saw the
    same load, which makes this a required field for the training-efficiency
    endpoint rather than a curiosity. Returns None when it cannot be determined.
    """

    import subprocess

    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return len([line for line in completed.stdout.splitlines() if line.strip()])


def tf32_state() -> dict[str, bool]:
    """Report the current TF32 flags without touching them.

    Reading matters as much as setting: on an A100 the defaults are asymmetric
    (cuDNN convolutions allow TF32, cuBLAS matmuls do not), so a well-meaning
    `set_tf32(True)` would silently enable matmul TF32 and change the numerics
    relative to every run this repository has already produced.
    """

    import torch

    return {
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def set_tf32(enabled: bool) -> dict[str, bool]:
    """Turn TF32 on or off for cuDNN convolutions and cuBLAS matmuls.

    Scoring must run with TF32 *off*. On an A100 the reduced mantissa moves a
    per-sample SI-SDR by up to 1.6e-3 dB depending on the batch size, and the
    20,000 training scores are spread over roughly 19 dB, so adjacent ranks sit
    about 1e-3 dB apart: TF32 noise is large enough to permute neighbouring
    percentile ranks and make a subset irreproducible. With TF32 off the
    batch-vs-single deviation drops to ~6e-6 dB.

    Only the legacy flags are used. PyTorch raises if the legacy and the newer
    per-operator `fp32_precision` APIs are mixed, and the legacy setter applies
    to every cuDNN operator at once, which is what we want here.
    """

    import torch

    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled
    return tf32_state()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed from arbitrary labels.

    `random.Random(str)` is also stable in CPython, but deriving the integer
    explicitly makes the guarantee independent of interpreter internals.
    """

    material = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def seeded_rng(*parts: object) -> random.Random:
    """Mersenne Twister seeded from `derive_seed`.

    Python's `random` is used instead of `torch.randperm` throughout this
    package because its algorithm is documented and stable across releases,
    which matters for subsets that must be reproducible months later.
    """

    return random.Random(derive_seed(*parts))


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)


def write_json(path: str | Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
