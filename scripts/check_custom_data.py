#!/usr/bin/env python3
"""Validate folder-per-sample custom datasets used by scripts/run.py."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

IMAGE_NAMES = ("input.png", "input.jpg", "input.jpeg", "input.webp")
EXPECTED_FORMATS = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
}


@dataclass(frozen=True)
class ValidationResult:
    root: Path
    sample_count: int
    errors: tuple[str, ...]


def _is_candidate_sample(directory: Path) -> bool:
    return any((directory / name).is_file() for name in (*IMAGE_NAMES, "prompt.txt"))


def discover_custom_roots(search_root: Path) -> list[Path]:
    """Find directories whose immediate children look like custom samples."""
    search_root = search_root.resolve()
    if not search_root.is_dir():
        raise ValueError(f"Search root does not exist: {search_root}")
    roots: set[Path] = set()
    for current, directories, files in os.walk(search_root):
        directories.sort()
        files = set(files)
        if files.intersection((*IMAGE_NAMES, "prompt.txt")):
            roots.add(Path(current).parent)
    return sorted(roots, key=lambda path: str(path))


def _validate_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            actual_format = image.format
            image.verify()
        with Image.open(path) as image:
            image.load()
            if image.width <= 0 or image.height <= 0:
                return "image has invalid dimensions"
    except (OSError, ValueError, UnidentifiedImageError) as error:
        return f"image cannot be decoded: {error}"
    expected_format = EXPECTED_FORMATS[path.suffix.lower()]
    if actual_format != expected_format:
        return f"extension expects {expected_format}, decoded format is {actual_format or 'unknown'}"
    return None


def validate_custom_root(root: Path) -> ValidationResult:
    root = root.resolve()
    if not root.is_dir():
        return ValidationResult(root, 0, (f"{root}: dataset root does not exist",))
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir() and _is_candidate_sample(path)),
        key=lambda path: path.name,
    )
    errors: list[str] = []
    if not candidates:
        errors.append(f"{root}: no custom samples found")
    for sample in candidates:
        images = [sample / name for name in IMAGE_NAMES if (sample / name).is_file()]
        if len(images) != 1:
            errors.append(f"{sample}: expected exactly one input image, found {len(images)}")
        else:
            image_error = _validate_image(images[0])
            if image_error:
                errors.append(f"{images[0]}: {image_error}")
        prompt = sample / "prompt.txt"
        if not prompt.is_file():
            errors.append(f"{sample}: missing prompt.txt")
        else:
            try:
                text = prompt.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                errors.append(f"{prompt}: invalid UTF-8: {error}")
            except OSError as error:
                errors.append(f"{prompt}: cannot be read: {error}")
            else:
                if not text.strip():
                    errors.append(f"{prompt}: prompt is empty")
    return ValidationResult(root, len(candidates), tuple(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--recursive", action="store_true", help="Discover and check every custom dataset below DATA_DIR")
    parser.add_argument("--list-roots", action="store_true", help="Print discovered roots instead of checking them")
    parser.add_argument("--null", action="store_true", help="With --list-roots, separate paths with NUL bytes")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.null and not args.list_roots:
            raise ValueError("--null requires --list-roots")
        roots = discover_custom_roots(args.data_dir) if (args.recursive or args.list_roots) else [args.data_dir.resolve()]
        if not roots:
            raise ValueError(f"No custom dataset roots found under {args.data_dir.resolve()}")
        if args.list_roots:
            separator = b"\0" if args.null else b"\n"
            sys.stdout.buffer.write(separator.join(os.fsencode(root) for root in roots) + separator)
            return 0
        failures = 0
        samples = 0
        for root in roots:
            result = validate_custom_root(root)
            samples += result.sample_count
            if result.errors:
                failures += 1
                print(f"[failed] {root}: {result.sample_count} candidate sample(s)")
                for error in result.errors:
                    print(f"  - {error}")
            else:
                print(f"[ok] {root}: {result.sample_count} sample(s)")
        print(f"[summary] {len(roots)} dataset(s), {samples} sample(s), {failures} failed dataset(s)")
        return int(bool(failures))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
