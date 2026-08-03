"""Canonical, publication-ready output rendering for inference runs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


def _frames(value: Any) -> list[Any]:
    """Normalize common DiffSynth callback containers to a PIL frame list."""
    if isinstance(value, tuple):
        value = value[0]
    if hasattr(value, "ndim") and value.ndim == 5:
        value = value[0]
    if isinstance(value, Sequence):
        return list(value)
    raise TypeError(f"Unsupported step-video value: {type(value).__name__}")


def _font(size: int, *, bold: bool = False):
    from PIL import ImageFont

    names = (
        ("DejaVuSans-Bold.ttf", "Arial Bold.ttf")
        if bold
        else ("DejaVuSans.ttf", "Arial.ttf")
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _sample_indices(length: int, count: int) -> list[int]:
    count = min(length, count)
    if count == 0:
        return []
    if count <= 1:
        return [0]
    return [round(index * (length - 1) / (count - 1)) for index in range(count)]


def render_overview(
    step_frames: list[tuple[int, int, list[Any]]],
    destination: Path,
    *,
    prompt: str,
    columns: int = 6,
) -> None:
    """Render a clean paper-style grid: one denoising step per row."""
    from PIL import Image, ImageDraw

    if not step_frames:
        return
    source = step_frames[0][2][0]
    ratio = source.height / source.width
    cell_w = 240
    cell_h = round(cell_w * ratio)
    label_w, gap, margin = 138, 12, 44
    title_h = 112
    row_gap = 28
    width = margin * 2 + label_w + columns * cell_w + (columns - 1) * gap
    height = title_h + margin + len(step_frames) * cell_h + (len(step_frames) - 1) * row_gap + margin
    canvas = Image.new("RGB", (width, height), "#FFFFFF")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 30), "Denoising trajectory", fill="#111827", font=_font(28, bold=True))
    summary = " ".join(prompt.split())
    if len(summary) > 150:
        summary = summary[:147] + "…"
    draw.text((margin, 70), summary, fill="#6B7280", font=_font(16))
    y = title_h + margin
    for step, total, frames in step_frames:
        draw.text((margin, y + 5), f"STEP {step + 1:02d}", fill="#2563EB", font=_font(16, bold=True))
        draw.text((margin, y + 30), f"of {total}", fill="#9CA3AF", font=_font(14))
        chosen = _sample_indices(len(frames), columns)
        for column in range(columns):
            frame = frames[chosen[min(column, len(chosen) - 1)]].convert("RGB")
            frame = frame.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = margin + label_w + column * (cell_w + gap)
            canvas.paste(frame, (x, y))
            draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline="#E5E7EB", width=1)
        y += cell_h + row_gap
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95)


def make_result_callback(
    output_dir: Path,
    *,
    prompt: str,
    fps: int,
    overview_columns: int = 6,
) -> Callable:
    """Save each step's video, every frame, and an evolving overview image."""
    from diffsynth.utils.data import save_video

    steps: list[tuple[int, int, list[Any]]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    def callback(step_index: int, total_steps: int, step_video: Any) -> None:
        frames = _frames(step_video)
        step_dir = output_dir / f"step_{step_index:03d}"
        frame_dir = step_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        save_video(frames, str(step_dir / "video.mp4"), fps=fps, quality=5)
        for frame_index, frame in enumerate(frames):
            frame.convert("RGB").save(frame_dir / f"frame_{frame_index:04d}.png")
        steps.append((step_index, total_steps, frames))
        render_overview(
            steps,
            output_dir.parent / "overview.png",
            prompt=prompt,
            columns=overview_columns,
        )
        print(f"  step {step_index + 1}/{total_steps} saved", flush=True)

    return callback
