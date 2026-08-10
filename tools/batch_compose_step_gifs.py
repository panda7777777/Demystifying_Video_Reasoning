#!/usr/bin/env python3
"""Batch-create step progression GIFs from an existing experiment run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.compose_step_gif import compose_gif  # noqa: E402


def parse_indices(spec: str) -> list[int]:
    """Parse comma-separated indices and inclusive ranges, preserving order."""
    result: list[int] = []
    seen: set[int] = set()
    for token in spec.replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            values = range(start, end + (1 if end >= start else -1),
                           1 if end >= start else -1)
        else:
            values = (int(token),)
        for value in values:
            if value < 0:
                raise ValueError("indices must be non-negative")
            if value not in seen:
                seen.add(value)
                result.append(value)
    if not result:
        raise ValueError("no indices were provided")
    return result


def discover_samples(result_dir: Path) -> dict[int, Path]:
    """Map dataset source indices to sample directories."""
    samples_dir = result_dir / "samples"
    if not samples_dir.is_dir():
        raise ValueError(f"samples directory not found: {samples_dir}")
    discovered: dict[int, Path] = {}
    for sample_dir in sorted(path for path in samples_dir.iterdir() if path.is_dir()):
        metadata_path = sample_dir / "input" / "metadata.json"
        source_index = None
        if metadata_path.is_file():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                source_index = metadata.get("source", {}).get("index")
            except (json.JSONDecodeError, OSError):
                pass
        if source_index is None and sample_dir.name.isdigit():
            source_index = int(sample_dir.name)
        if isinstance(source_index, int):
            discovered[source_index] = sample_dir
    return discovered


def visualize_sample(sample_dir: Path, output: Path, args: argparse.Namespace) -> None:
    step_videos = list((sample_dir / "output" / "steps").glob("step_*/video.mp4"))
    if not step_videos:
        raise ValueError("no output/steps/step_*/video.mp4 files")
    initial_image = sample_dir / "input" / "initial_frame.png"
    prompt_path = sample_dir / "input" / "prompt.txt"
    if not initial_image.is_file():
        raise ValueError("input/initial_frame.png is missing")
    if not prompt_path.is_file():
        raise ValueError("input/prompt.txt is missing")
    compose_gif(
        video_paths=step_videos,
        output=output,
        prompt=prompt_path.read_text(encoding="utf-8").strip(),
        num_steps=args.num_steps,
        initial_image=initial_image,
        fps=args.fps,
        panel_width=args.panel_width,
        gap=args.gap,
        labels=args.show_labels,
        font_path=args.font_path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True,
                        help="Run directory containing samples/<sample_id>")
    parser.add_argument("--indices", required=True,
                        help="Dataset indices, e.g. '37,998,11' or '0-9'")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: <result-dir>/visualizations/step_gifs")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--panel-width", type=int, default=256)
    parser.add_argument("--gap", type=int, default=3,
                        help="Pixels between adjacent panels (default: 3)")
    parser.add_argument("--show-labels", action="store_true",
                        help="Show Initial frame / Step N titles; hidden by default")
    parser.add_argument("--font-path", type=Path,
                        help="Path to Times New Roman Bold (timesbd.ttf)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        indices = parse_indices(args.indices)
        samples = discover_samples(args.result_dir)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    if args.num_steps < 1 or args.fps <= 0 or args.panel_width < 32 or args.gap < 0:
        raise SystemExit("error: num-steps/fps must be positive, panel-width >= 32, gap >= 0")

    output_dir = args.output_dir or args.result_dir / "visualizations" / "step_gifs"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed, skipped, failed = 0, 0, []
    for position, index in enumerate(indices, start=1):
        sample_dir = samples.get(index)
        if sample_dir is None:
            failed.append((index, "dataset index not found in this run"))
            print(f"[{position}/{len(indices)}] index {index}: NOT FOUND", file=sys.stderr)
            continue
        output = output_dir / f"index_{index:06d}.gif"
        if output.exists() and not args.overwrite:
            skipped += 1
            print(f"[{position}/{len(indices)}] index {index}: skip existing {output.name}")
            continue
        try:
            visualize_sample(sample_dir, output, args)
        except Exception as error:
            failed.append((index, str(error)))
            print(f"[{position}/{len(indices)}] index {index}: FAILED: {error}", file=sys.stderr)
        else:
            completed += 1
            print(f"[{position}/{len(indices)}] index {index}: saved {output}")

    print(f"Done: {completed} created, {skipped} skipped, {len(failed)} failed")
    if failed:
        print("Failures:", file=sys.stderr)
        for index, reason in failed:
            print(f"  {index}: {reason}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
