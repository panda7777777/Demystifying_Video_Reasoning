#!/usr/bin/env python3
"""Inspect and render outputs from the Language-Table visualization workflow."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def sample_ids(root: Path, completed_only: bool) -> list[str]:
    directory = root / ("episodes" if completed_only else "inputs")
    if not directory.is_dir():
        raise ValueError(f"Result directory does not exist: {directory}")
    required = "metadata.json" if completed_only else "prompt.txt"
    return [
        path.name
        for path in sorted(directory.iterdir())
        if path.is_dir() and (path / required).is_file()
    ]


def find_prompt(root: Path, prompt: str) -> list[str]:
    normalized = prompt.strip()
    matches = []
    inputs = root / "inputs"
    if not inputs.is_dir():
        raise ValueError(f"Input directory does not exist: {inputs}")
    for path in sorted(inputs.iterdir()):
        prompt_path = path / "prompt.txt"
        if prompt_path.is_file() and prompt_path.read_text(
            encoding="utf-8"
        ).strip() == normalized:
            matches.append(path.name)
    return matches


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    if count <= 0 or count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return [
        round(index * (length - 1) / (count - 1)) for index in range(count)
    ]


def read_frames(video_path: Path, stride: int) -> list:
    import imageio.v2 as imageio
    import numpy as np

    frames = []
    reader = imageio.get_reader(str(video_path))
    try:
        for index, frame in enumerate(reader):
            if index % stride == 0:
                frames.append(np.asarray(frame))
    finally:
        reader.close()
    return frames


def build_montage(
    rows: list[tuple[int, list]],
    *,
    columns: int,
    thumbnail_width: int,
):
    from PIL import Image, ImageDraw, ImageFont

    nonempty = [(step, frames) for step, frames in rows if frames]
    if not nonempty:
        return None
    max_frames = max(len(frames) for _, frames in nonempty)
    column_indices = evenly_spaced_indices(max_frames, columns)
    sample = nonempty[0][1][0]
    height, width = sample.shape[:2]
    thumbnail_height = max(1, round(height * thumbnail_width / width))
    gap, left_margin, top_margin = 4, 76, 24
    canvas = Image.new(
        "RGB",
        (
            left_margin
            + len(column_indices) * thumbnail_width
            + (len(column_indices) + 1) * gap,
            top_margin
            + len(nonempty) * thumbnail_height
            + (len(nonempty) + 1) * gap,
        ),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, frame_index in enumerate(column_indices):
        x = left_margin + gap + column * (thumbnail_width + gap)
        draw.text((x, 5), f"frame {frame_index}", fill=(210, 210, 210), font=font)
    for row_index, (step, frames) in enumerate(nonempty):
        y = top_margin + gap + row_index * (thumbnail_height + gap)
        draw.text((6, y + thumbnail_height // 2), f"step {step}", fill=(120, 220, 120), font=font)
        for column, frame_index in enumerate(column_indices):
            selected = min(frame_index, len(frames) - 1)
            thumbnail = Image.fromarray(frames[selected]).resize(
                (thumbnail_width, thumbnail_height), Image.Resampling.BILINEAR
            )
            x = left_margin + gap + column * (thumbnail_width + gap)
            canvas.paste(thumbnail, (x, y))
    return canvas


def render_episode(
    root: Path,
    sample_id: str,
    *,
    stride: int,
    dump_frames: bool,
    make_montage: bool,
    columns: int,
    thumbnail_width: int,
    overwrite: bool,
) -> tuple[str, str, str]:
    episode_dir = root / "episodes" / sample_id
    step_files = sorted((episode_dir / "steps").glob("step_*.mp4"))
    if not step_files:
        return sample_id, "skipped", "no step videos"
    analysis_dir = root / "analysis" / sample_id
    metadata_path = analysis_dir / "metadata.json"
    expected = {
        "stride": stride,
        "dump_frames": dump_frames,
        "make_montage": make_montage,
        "columns": columns,
        "thumbnail_width": thumbnail_width,
        "steps": [path.name for path in step_files],
    }
    if metadata_path.is_file() and not overwrite:
        if json.loads(metadata_path.read_text(encoding="utf-8")) == expected:
            return sample_id, "skipped", "already rendered"
        return sample_id, "failed", "analysis parameters changed; use --overwrite"

    temporary = analysis_dir.with_name(f".{sample_id}.render.tmp")
    try:
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        rows = []
        written = 0
        for video_path in step_files:
            step = int(video_path.stem.removeprefix("step_"))
            frames = read_frames(video_path, stride)
            rows.append((step, frames))
            if dump_frames:
                frame_dir = temporary / "frames" / f"step_{step:03d}"
                frame_dir.mkdir(parents=True)
                for frame_index, frame in enumerate(frames):
                    from PIL import Image

                    Image.fromarray(frame).save(
                        frame_dir / f"frame_{frame_index:04d}.png"
                    )
                    written += 1
        if make_montage:
            montage = build_montage(
                rows, columns=columns, thumbnail_width=thumbnail_width
            )
            if montage is not None:
                montage.save(temporary / "montage.png")
        (temporary / "metadata.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if analysis_dir.exists():
            shutil.rmtree(analysis_dir)
        temporary.replace(analysis_dir)
        return sample_id, "completed", f"{len(step_files)} steps, {written} frames"
    except Exception as error:
        return sample_id, "failed", f"{type(error).__name__}: {error}"


def run_render(args: argparse.Namespace) -> int:
    ids = sample_ids(args.root, completed_only=True)
    if args.limit is not None:
        ids = ids[: args.limit]
    failures = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                render_episode,
                args.root,
                sample_id,
                stride=args.stride,
                dump_frames=not args.no_frames,
                make_montage=not args.no_montage,
                columns=args.montage_columns,
                thumbnail_width=args.thumbnail_width,
                overwrite=args.overwrite,
            )
            for sample_id in ids
        ]
        for future in as_completed(futures):
            sample_id, status, message = future.result()
            print(f"[{status}] {sample_id}: {message}")
            failures += status == "failed"
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    find_parser = commands.add_parser("find-prompt")
    find_parser.add_argument("prompt")

    sample_parser = commands.add_parser("sample")
    sample_parser.add_argument("count", type=int)
    sample_parser.add_argument("--seed", type=int)
    sample_parser.add_argument("--include-incomplete", action="store_true")

    render_parser = commands.add_parser("render")
    render_parser.add_argument("--workers", type=int, default=8)
    render_parser.add_argument("--stride", type=int, default=1)
    render_parser.add_argument("--montage-columns", type=int, default=8)
    render_parser.add_argument("--thumbnail-width", type=int, default=160)
    render_parser.add_argument("--limit", type=int)
    render_parser.add_argument("--overwrite", action="store_true")
    render_parser.add_argument("--no-frames", action="store_true")
    render_parser.add_argument("--no-montage", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "find-prompt":
            matches = find_prompt(args.root, args.prompt)
            print("\n".join(matches))
            return 0 if matches else 1
        if args.command == "sample":
            if args.count < 0:
                raise ValueError("count cannot be negative")
            ids = sample_ids(args.root, completed_only=not args.include_incomplete)
            if args.count > len(ids):
                raise ValueError(
                    f"Requested {args.count} samples, but only {len(ids)} are available"
                )
            print(
                json.dumps(
                    random.Random(args.seed).sample(ids, args.count),
                    ensure_ascii=False,
                )
            )
            return 0
        if args.workers <= 0 or args.stride <= 0 or args.thumbnail_width <= 0:
            raise ValueError("workers, stride, and thumbnail-width must be positive")
        return run_render(args)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
