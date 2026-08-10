#!/usr/bin/env python3
"""Compose an initial frame and sampled intermediate-step videos into a GIF.

Example:
    python tools/compose_step_gif.py \
        --videos output/sample/steps/*.mp4 \
        --prompt "Move the red block to the blue square." \
        --output output/sample/steps.gif
"""

from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path
from typing import Sequence

import imageio.v3 as iio
from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (248, 249, 251)
TEXT = (31, 41, 55)
MUTED = (82, 93, 110)
BORDER = (204, 211, 221)
ACCENT = (47, 96, 160)


def natural_key(path: Path) -> list[object]:
    """Sort step_2 before step_10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.as_posix())]


def video_label(path: Path) -> str:
    """Use the step directory for conventionally named ``video.mp4`` files."""
    raw = path.parent.name if path.stem.lower() == "video" else path.stem
    match = re.fullmatch(r"step[_-]?(\d+)", raw, flags=re.IGNORECASE)
    return f"Step {int(match.group(1))}" if match else raw.replace("_", " ")


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Return unique, endpoint-inclusive, approximately uniform indices."""
    if total < 1:
        raise ValueError("at least one video is required")
    if count < 1:
        raise ValueError("--num-steps must be at least 1")
    count = min(total, count)
    if count == 1:
        return [0]
    return [round(index * (total - 1) / (count - 1)) for index in range(count)]


def load_times_bold(size: int, font_path: Path | None = None) -> ImageFont.FreeTypeFont:
    """Load Times New Roman Bold without silently substituting another font."""
    candidates = ([font_path] if font_path else []) + [
        Path(__file__).resolve().parents[1] / "assets/fonts/Times_New_Roman_Bold.ttf",
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/TimesNewRomanBold.ttf"),
        Path.home() / ".fonts" / "timesbd.ttf",
        Path.home() / ".local/share/fonts" / "timesbd.ttf",
        Path("C:/Windows/Fonts/timesbd.ttf"),
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    raise RuntimeError(
        "Times New Roman Bold font was not found. Install timesbd.ttf or pass "
        "--font-path /path/to/timesbd.ttf"
    )


def read_video(path: Path) -> tuple[list[Image.Image], float]:
    try:
        array = iio.imread(path, index=None, plugin="FFMPEG")
        metadata = iio.immeta(path, plugin="FFMPEG")
    except Exception as error:
        raise RuntimeError(f"cannot decode video {path}: {error}") from error
    if array.ndim == 3:
        array = array[None, ...]
    frames = [Image.fromarray(frame[..., :3]).convert("RGB") for frame in array]
    if not frames:
        raise RuntimeError(f"video has no frames: {path}")
    return frames, float(metadata.get("fps") or 8.0)


def fit_frame(frame: Image.Image, width: int, height: int) -> Image.Image:
    """Letterbox without distorting the source aspect ratio."""
    frame = frame.copy()
    frame.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
    return canvas


def temporal_indices(frame_count: int, source_fps: float, output_fps: float,
                     output_count: int) -> list[int]:
    return [min(frame_count - 1, round(index * source_fps / output_fps))
            for index in range(output_count)]


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int,
                  canvas_width: int, font: ImageFont.ImageFont,
                  fill: tuple[int, int, int]) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    draw.text(((canvas_width - (box[2] - box[0])) / 2 - box[0], y - box[1]), text,
              font=font, fill=fill)


def compose_gif(
    video_paths: Sequence[Path], output: Path, prompt: str, num_steps: int = 5,
    initial_image: Path | None = None, fps: float = 8.0, panel_width: int = 256,
    gap: int = 3, labels: bool = False, font_path: Path | None = None,
) -> list[Path]:
    paths = sorted(video_paths, key=natural_key)
    selected = [paths[index] for index in evenly_spaced_indices(len(paths), num_steps)]
    loaded = [read_video(path) for path in selected]
    videos, source_fps = zip(*loaded)

    if initial_image:
        with Image.open(initial_image) as image:
            initial = image.convert("RGB").copy()
    else:
        initial = videos[0][0].copy()

    aspect = initial.height / initial.width
    panel_height = max(1, round(panel_width * aspect))
    margin_x, margin_top, margin_bottom = 4, 4, 4
    label_height = 28 if labels else 0
    # Keep the publication layout proportional when exporting at higher
    # resolution: 24 px at the default 256 px panel width, 60 px at 640 px.
    prompt_font_size = max(12, round(24 * panel_width / 256))
    label_font_size = max(20, round(panel_width * 0.09))
    prompt_font = load_times_bold(prompt_font_size, font_path)
    label_font = load_times_bold(label_font_size, font_path)
    columns = 1 + len(selected)
    canvas_width = margin_x * 2 + columns * panel_width + (columns - 1) * gap
    wrap_width = max(25, int(canvas_width / max(8, prompt_font_size * 0.55)))
    prompt_lines = textwrap.wrap(" ".join(prompt.split()), width=wrap_width) or [""]
    line_gap = max(2, round(prompt_font_size * 0.08))
    line_heights = [
        draw_box[3] - draw_box[1]
        for line in prompt_lines
        for draw_box in [ImageDraw.Draw(Image.new("L", (1, 1))).textbbox(
            (0, 0), line, font=prompt_font
        )]
    ]
    prompt_height = margin_top + sum(line_heights) + line_gap * (len(prompt_lines) - 1)
    canvas_height = margin_top + label_height + panel_height + prompt_height + margin_bottom

    durations = [len(frames) / rate for frames, rate in zip(videos, source_fps)]
    output_count = max(1, round(min(durations) * fps))
    indices = [temporal_indices(len(frames), rate, fps, output_count)
               for frames, rate in zip(videos, source_fps)]
    initial_panel = fit_frame(initial, panel_width, panel_height)
    rendered: list[Image.Image] = []

    for time_index in range(output_count):
        canvas = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
        draw = ImageDraw.Draw(canvas)
        panel_y = margin_top + label_height
        panels = [initial_panel] + [
            fit_frame(frames[index_list[time_index]], panel_width, panel_height)
            for frames, index_list in zip(videos, indices)
        ]
        for column, panel in enumerate(panels):
            x = margin_x + column * (panel_width + gap)
            canvas.paste(panel, (x, panel_y))
            draw.rounded_rectangle((x - 1, panel_y - 1, x + panel_width,
                                    panel_y + panel_height), radius=3,
                                   outline=BORDER, width=1)
            if labels:
                label = "Initial frame" if column == 0 else video_label(selected[column - 1])
                box = draw.textbbox((0, 0), label, font=label_font)
                draw.text((x + (panel_width - box[2] + box[0]) / 2, margin_top),
                          label, font=label_font,
                          fill=ACCENT if column else MUTED)

        prompt_y = panel_y + panel_height + margin_top
        for line, line_height in zip(prompt_lines, line_heights):
            draw_centered(draw, line, prompt_y, canvas_width, prompt_font, TEXT)
            prompt_y += line_height + line_gap
        rendered.append(canvas)

    output.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(
        output, save_all=True, append_images=rendered[1:],
        duration=round(1000 / fps), loop=0, disposal=2, optimize=False,
    )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", nargs="+", type=Path, required=True,
                        help="Candidate MP4 files; shell globs are supported")
    parser.add_argument("--output", type=Path, required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Input prompt shown below the panels")
    prompt.add_argument("--prompt-file", type=Path)
    parser.add_argument("--initial-image", type=Path,
                        help="Optional initial frame; defaults to video 1 frame 1")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="Number of uniformly sampled videos (default: 5)")
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--panel-width", type=int, default=256)
    parser.add_argument("--gap", type=int, default=3)
    parser.add_argument("--show-labels", action="store_true",
                        help="Show panel titles; hidden by default")
    parser.add_argument("--font-path", type=Path,
                        help="Path to Times New Roman Bold (timesbd.ttf)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing = [path for path in args.videos if not path.is_file()]
    if missing:
        raise SystemExit(f"error: video not found: {missing[0]}")
    if args.initial_image and not args.initial_image.is_file():
        raise SystemExit(f"error: initial image not found: {args.initial_image}")
    if args.fps <= 0 or args.panel_width < 32 or args.gap < 0:
        raise SystemExit("error: fps must be positive, panel-width >= 32, and gap >= 0")
    prompt = (args.prompt_file.read_text(encoding="utf-8").strip()
              if args.prompt_file else args.prompt.strip())
    if not prompt:
        raise SystemExit("error: prompt cannot be empty")
    try:
        selected = compose_gif(
            args.videos, args.output, prompt, args.num_steps,
            args.initial_image, args.fps, args.panel_width, args.gap,
            args.show_labels, args.font_path,
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"error: {error}") from error
    print(f"Selected {len(selected)} videos: {', '.join(path.name for path in selected)}")
    print(f"Saved GIF: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
