"""Shared orchestration helpers for batch visualization workflows."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from scripts._visualization import atomic_write_json


def parse_indices(specification: str | None, total: int) -> list[int]:
    """Parse comma-separated indices and Python-style slices."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if specification is None or specification.strip().lower() in {"", "all", ":"}:
        return list(range(total))
    selected: set[int] = set()
    for token in specification.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            fields = token.split(":")
            if len(fields) > 3:
                raise ValueError(f"Invalid slice: {token}")
            try:
                value = slice(*(int(field) if field else None for field in fields))
                selected.update(range(*value.indices(total)))
            except (ValueError, ZeroDivisionError) as error:
                raise ValueError(f"Invalid slice: {token}") from error
        else:
            try:
                index = int(token)
            except ValueError as error:
                raise ValueError(f"Invalid index: {token}") from error
            if index < 0:
                index += total
            if not 0 <= index < total:
                raise ValueError(f"Index {token} is outside a dataset of size {total}")
            selected.add(index)
    return sorted(selected)


def contiguous_ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Collapse sorted indices into half-open contiguous ranges."""
    ranges: list[list[int]] = []
    for index in indices:
        if ranges and index == ranges[-1][1]:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index, index + 1])
    return [(start, stop) for start, stop in ranges]


def shard_indices(
    indices: list[int],
    *,
    node_rank: int,
    num_nodes: int,
    gpus: list[str],
) -> dict[str, list[int]]:
    """Deterministically shard ordered indices across nodes and local GPUs."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not 0 <= node_rank < num_nodes:
        raise ValueError(f"node_rank must be in [0, {num_nodes})")
    if not gpus or any(not gpu for gpu in gpus):
        raise ValueError("At least one non-empty GPU id is required")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU ids must be unique")
    world_size = num_nodes * len(gpus)
    return {
        gpu: indices[node_rank * len(gpus) + slot :: world_size]
        for slot, gpu in enumerate(gpus)
    }


def detect_gpus() -> list[str]:
    """Return visible GPU ids, respecting CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return [value.strip() for value in visible.split(",") if value.strip()]
    try:
        output = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise ValueError("Cannot detect GPUs; pass --gpus explicitly") from error
    return [
        str(index)
        for index in range(len(re.findall(r"^GPU \d+:", output, re.MULTILINE)))
    ]


def ensure_run_manifest(
    path: Path, expected: dict[str, Any], *, overwrite: bool = False
) -> None:
    """Create a shared run manifest once, or validate an existing one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.create")
    temporary.write_text(encoded, encoding="utf-8")
    try:
        os.link(temporary, path)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            if overwrite:
                atomic_write_json(path, expected)
                return
            raise ValueError(f"Existing run manifest has different parameters: {path}")
    finally:
        temporary.unlink(missing_ok=True)
