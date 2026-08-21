"""On-disk contract for per-sample data-pruning scores.

A score set is two files that must travel together:

    <stem>.jsonl        one row per training mixture, ordered by file name
    <stem>_meta.json    checkpoint hashes, code version, timing, row count

`load_score_set` re-validates every invariant on read, so a stale or
hand-edited score file fails loudly instead of silently producing a different
subset. Derived columns (`pct_*`, `gap_*`) are stored for auditability *and*
recomputed on load; a mismatch is an error, never a warning.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .common import atomic_write_text, read_json, sha256_text, utc_now, write_json

__all__ = [
    "SCHEMA_VERSION",
    "ROW_FIELDS",
    "ScoreSet",
    "parse_name",
    "percentile_ranks",
    "derive_columns",
    "write_score_set",
    "load_score_set",
    "DERIVED_TOLERANCE",
]

SCHEMA_VERSION = "data-pruning-scores/1"

ROW_FIELDS: tuple[str, ...] = (
    "name",
    "q_dense_sisdr",
    "q_pruned_sisdr",
    "mixture_sisdr_baseline",
    "pct_dense",
    "pct_pruned",
    "gap_rank",
    "gap_db",
    "snr_abs",
    "spk1",
    "spk2",
)

TEXT_FIELDS: frozenset[str] = frozenset({"name", "spk1", "spk2"})

DERIVED_TOLERANCE = 1e-9
"""Derived columns are pure float arithmetic, so they must round-trip exactly."""

SNR_SYMMETRY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ScoreSet:
    meta: dict
    rows: list[dict]

    @property
    def names(self) -> list[str]:
        return [row["name"] for row in self.rows]

    def by_name(self) -> dict[str, dict]:
        return {row["name"]: row for row in self.rows}

    def column(self, field: str) -> list[float]:
        if field not in ROW_FIELDS:
            raise KeyError(f"Unknown score field: {field}")
        return [row[field] for row in self.rows]


def parse_name(name: str) -> tuple[str, str, float]:
    """Return ``(speaker1, speaker2, |snr|)`` for an LRS2-2Mix file name.

    The layout is ``spk1_utt1_+g_spk2_utt2_-g.wav`` with the two source gains
    exactly opposite, which is what makes the mixture's own PIT-averaged SI-SDR
    vanish. The symmetry is asserted rather than assumed.
    """

    if not name.endswith(".wav"):
        raise ValueError(f"Expected a .wav file name, got {name!r}")
    parts = name[:-4].split("_")
    if len(parts) != 6:
        raise ValueError(f"Expected 6 underscore-separated fields in {name!r}, got {len(parts)}")
    gain1, gain2 = float(parts[2]), float(parts[5])
    if abs(gain1 + gain2) > SNR_SYMMETRY_TOLERANCE:
        raise ValueError(f"Source gains are not symmetric in {name!r}: {gain1} and {gain2}")
    return parts[0], parts[3], abs(gain1)


def percentile_ranks(values: Sequence[float], keys: Sequence[str]) -> list[float]:
    """Ordinal percentile rank in ``[0, 1]``, ties broken by ``keys``.

    Ordinal (not average) ranking keeps the mapping a strict bijection onto an
    evenly spaced grid, so ``pct_dense`` and ``pct_pruned`` are guaranteed to
    have identical marginals and the rank difference weights both axes equally.
    Ties in SI-SDR are essentially impossible with float32 audio, but the
    ``keys`` tie-break makes the output independent of input order regardless.
    """

    if len(values) != len(keys):
        raise ValueError("values and keys must have the same length")
    if len(values) < 2:
        raise ValueError("percentile ranks need at least two samples")
    order = sorted(range(len(values)), key=lambda i: (values[i], keys[i]))
    ranks = [0.0] * len(values)
    divisor = float(len(values) - 1)
    for position, index in enumerate(order):
        ranks[index] = position / divisor
    return ranks


def derive_columns(rows: Sequence[dict]) -> list[dict]:
    """Attach `pct_dense`, `pct_pruned`, `gap_rank`, `gap_db` to raw rows.

    The percentiles are defined over the *whole* score set, so this must be
    called once on the complete training split, never per shard.
    """

    names = [row["name"] for row in rows]
    pct_dense = percentile_ranks([row["q_dense_sisdr"] for row in rows], names)
    pct_pruned = percentile_ranks([row["q_pruned_sisdr"] for row in rows], names)
    enriched = []
    for row, dense, pruned in zip(rows, pct_dense, pct_pruned):
        enriched.append(
            {
                **row,
                "pct_dense": dense,
                "pct_pruned": pruned,
                "gap_rank": dense - pruned,
                "gap_db": row["q_dense_sisdr"] - row["q_pruned_sisdr"],
            }
        )
    return enriched


def _validate(meta: dict, rows: list[dict]) -> None:
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Score file schema is {meta.get('schema_version')!r}, expected {SCHEMA_VERSION!r}"
        )
    if not rows:
        raise ValueError("Score file contains no rows")
    if len(rows) != int(meta["count"]):
        raise ValueError(f"Meta declares {meta['count']} rows but the file has {len(rows)}")

    names = [row["name"] for row in rows]
    if len(set(names)) != len(names):
        raise ValueError("Score file contains duplicate file names")
    if names != sorted(names):
        raise ValueError("Score rows must be sorted by file name")

    for row in rows:
        missing = [field for field in ROW_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Row {row.get('name')!r} is missing fields: {missing}")
        for field in ROW_FIELDS:
            value = row[field]
            if field in TEXT_FIELDS:
                if not isinstance(value, str) or not value:
                    raise ValueError(f"Row {row['name']!r} has a non-string {field}={value!r}")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Row {row['name']!r} has a non-numeric {field}={value!r}")
            if not math.isfinite(value):
                raise ValueError(f"Row {row['name']!r} has non-finite {field}={value}")
        speaker1, speaker2, snr = parse_name(row["name"])
        if row["spk1"] != speaker1 or row["spk2"] != speaker2:
            raise ValueError(f"Row {row['name']!r} has speaker fields inconsistent with its name")
        if abs(row["snr_abs"] - snr) > SNR_SYMMETRY_TOLERANCE:
            raise ValueError(f"Row {row['name']!r} has snr_abs inconsistent with its name")

    recomputed = derive_columns(
        [
            {
                "name": row["name"],
                "q_dense_sisdr": row["q_dense_sisdr"],
                "q_pruned_sisdr": row["q_pruned_sisdr"],
            }
            for row in rows
        ]
    )
    for row, reference in zip(rows, recomputed):
        for field in ("pct_dense", "pct_pruned", "gap_rank", "gap_db"):
            if abs(row[field] - reference[field]) > DERIVED_TOLERANCE:
                raise ValueError(
                    f"Row {row['name']!r} has stale {field}: "
                    f"stored {row[field]!r}, recomputed {reference[field]!r}"
                )


def _rows_text(rows: Iterable[dict]) -> str:
    return "".join(
        json.dumps({field: row[field] for field in ROW_FIELDS}, ensure_ascii=False) + "\n"
        for row in rows
    )


def write_score_set(stem: str | Path, meta: dict, rows: Sequence[dict]) -> dict:
    """Write ``<stem>.jsonl`` plus ``<stem>_meta.json`` and return the final meta."""

    stem = Path(stem)
    ordered = sorted(rows, key=lambda row: row["name"])
    payload = dict(meta)
    payload["schema_version"] = SCHEMA_VERSION
    payload["count"] = len(ordered)
    payload["written_at_utc"] = utc_now()
    _validate(payload, list(ordered))

    text = _rows_text(ordered)
    payload["rows_sha256"] = sha256_text(text)
    atomic_write_text(stem.with_suffix(".jsonl"), text)
    write_json(stem.parent / f"{stem.name}_meta.json", payload)
    return payload


def load_score_set(stem: str | Path) -> ScoreSet:
    """Read and fully re-validate a score set written by `write_score_set`."""

    stem = Path(stem)
    if stem.suffix == ".jsonl":
        stem = stem.with_suffix("")
    rows_path = stem.with_suffix(".jsonl")
    meta_path = stem.parent / f"{stem.name}_meta.json"
    meta = read_json(meta_path)
    text = rows_path.read_text(encoding="utf-8")
    if sha256_text(text) != meta["rows_sha256"]:
        raise ValueError(
            f"{rows_path} does not match the sha256 recorded in {meta_path}; "
            "the score file was modified after it was written"
        )
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    _validate(meta, rows)
    return ScoreSet(meta=meta, rows=rows)
