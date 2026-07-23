#!/usr/bin/env python3
"""Run step visualization over a folder-per-sample custom dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts._visualization import GenerationConfig  # noqa: E402


VISUALIZATION_TOOL = REPO_ROOT / "tools" / "custom_step_visualization.py"
DEFAULT_DATASET = REPO_ROOT / "examples" / "custom_dataset"
IMAGE_NAMES = ("input.png", "input.jpg", "input.jpeg", "input.webp")
OVERRIDABLE_FIELDS = frozenset(
    {
        "model",
        "num_frames",
        "num_inference_steps",
        "max_denoising_steps",
        "visualization_steps",
        "seed",
        "fps",
        "max_size",
        "negative_prompt",
    }
)
DEFAULTS: dict[str, Any] = {
    "model": "wan2.2",
    "num_frames": 49,
    "num_inference_steps": 30,
    "max_denoising_steps": None,
    "visualization_steps": "all",
    "seed": 1,
    "fps": 16,
    "max_size": 832,
    "negative_prompt": None,
}


def find_image(sample_dir: Path) -> Path | None:
    return next(
        (sample_dir / name for name in IMAGE_NAMES if (sample_dir / name).is_file()),
        None,
    )


def list_samples(dataset_dir: Path) -> list[str]:
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    return [
        path.name
        for path in sorted(dataset_dir.iterdir())
        if path.is_dir()
        and find_image(path) is not None
        and (path / "prompt.txt").is_file()
    ]


def load_sample(dataset_dir: Path, sample_id: str) -> tuple[Path, str, dict]:
    sample_dir = dataset_dir / sample_id
    image_path = find_image(sample_dir)
    prompt_path = sample_dir / "prompt.txt"
    if image_path is None or not prompt_path.is_file():
        raise ValueError(f"Invalid sample directory: {sample_dir}")
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt is empty: {prompt_path}")

    metadata_path = sample_dir / "metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    unknown = sorted(set(metadata) - OVERRIDABLE_FIELDS)
    if unknown:
        raise ValueError(
            f"Unknown metadata fields in {metadata_path}: {', '.join(unknown)}"
        )
    return image_path, prompt, metadata


def prepare_image(source: Path, destination: Path, max_size: int) -> tuple[int, int]:
    if max_size <= 0:
        raise ValueError("max_size must be positive")
    image = Image.open(source)
    if image.mode != "RGB":
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, "white")
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        image = flattened

    width, height = image.size
    scale = min(1.0, max_size / max(width, height))
    width = max(16, round(width * scale))
    height = max(16, round(height * scale))
    width -= width % 16
    height -= height % 16
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return width, height


def resolve_config(
    args: argparse.Namespace, metadata: dict[str, Any]
) -> dict[str, Any]:
    resolved = {}
    for key, default in DEFAULTS.items():
        cli_value = getattr(args, key, None)
        resolved[key] = (
            cli_value
            if cli_value is not None
            else metadata.get(key, default)
        )
    return resolved


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_command(
    args: argparse.Namespace,
    config: dict[str, Any],
    image_path: Path,
    prompt: str,
    output_dir: Path,
) -> list[str]:
    command = [
        str(args.python),
        str(VISUALIZATION_TOOL),
        "--model",
        str(config["model"]),
        "--image",
        str(image_path),
        "--prompt",
        prompt,
        "--output-dir",
        str(output_dir),
        "--num-frames",
        str(config["num_frames"]),
        "--num-inference-steps",
        str(config["num_inference_steps"]),
        "--visualization-steps",
        str(config["visualization_steps"]),
        "--seed",
        str(config["seed"]),
        "--fps",
        str(config["fps"]),
        "--lora-alpha",
        str(args.lora_alpha),
    ]
    optional_values = {
        "--max-denoising-steps": config["max_denoising_steps"],
        "--negative-prompt": config["negative_prompt"],
        "--vbvr-model-path": args.vbvr_model_path,
        "--lora-path": args.lora_path,
        "--high-noise-lora-path": args.high_noise_lora_path,
        "--low-noise-lora-path": args.low_noise_lora_path,
    }
    for option, value in optional_values.items():
        if value is not None:
            command.extend((option, str(value)))
    return command


def run_sample(
    args: argparse.Namespace, sample_id: str
) -> tuple[str, str | None]:
    source, prompt, metadata = load_sample(args.dataset_dir, sample_id)
    config = resolve_config(args, metadata)
    generation = GenerationConfig(
        model=config["model"],
        num_frames=int(config["num_frames"]),
        num_inference_steps=int(config["num_inference_steps"]),
        max_denoising_steps=config["max_denoising_steps"],
        seed=int(config["seed"]),
        fps=int(config["fps"]),
        negative_prompt=config["negative_prompt"],
    )
    generation.validate()
    if int(config["max_size"]) <= 0:
        raise ValueError("max_size must be positive")
    input_dir = args.output_root / "inputs" / sample_id
    input_path = input_dir / "input.png"
    episode_dir = args.output_root / "episodes" / sample_id
    temporary = episode_dir.with_name(f".{sample_id}.{os.getpid()}.tmp")
    if args.dry_run:
        command = build_command(args, config, input_path, prompt, temporary)
        print(subprocess.list2cmdline(command))
        return "dry-run", None

    width, height = prepare_image(source, input_path, int(config["max_size"]))
    (input_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")

    expected_metadata = {
        "generation": {
            key: value for key, value in config.items() if key != "max_size"
        },
        "input": f"inputs/{sample_id}/input.png",
        "prompt": prompt,
        "size": [width, height],
    }
    metadata_path = episode_dir / "metadata.json"
    if metadata_path.is_file() and not args.overwrite:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in expected_metadata.items()):
            return "skipped", None
        return "failed", "existing result uses different parameters; use --overwrite"

    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    command = build_command(args, config, input_path, prompt, temporary)
    environment = dict(os.environ)
    if args.model_base_path:
        environment["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path)
        environment.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    result = subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False)
    if result.returncode:
        return "failed", f"visualization process exited with {result.returncode}"
    atomic_write_json(temporary / "metadata.json", expected_metadata)
    if episode_dir.exists():
        shutil.rmtree(episode_dir)
    temporary.replace(episode_dir)
    return "completed", None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=False)
    selection.add_argument("--sample", action="append")
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--list", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "output" / "step_visualization" / "custom",
    )
    parser.add_argument("--model", choices=("wan2.2", "wan2.1", "ltx2.3", "vbvr-wan2.2"))
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--max-denoising-steps", type=int)
    parser.add_argument("--visualization-steps")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--max-size", type=int)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model-base-path", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--vbvr-model-path", type=Path)
    parser.add_argument("--lora-path")
    parser.add_argument("--high-noise-lora-path")
    parser.add_argument("--low-noise-lora-path")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        available = list_samples(args.dataset_dir)
        if args.list:
            for sample_id in available:
                _, prompt, _ = load_sample(args.dataset_dir, sample_id)
                print(f"{sample_id}\t{prompt}")
            return 0
        selected = available if args.all else args.sample
        if not selected:
            raise ValueError("Select --sample ID, --all, or --list")
        unknown = sorted(set(selected) - set(available))
        if unknown:
            raise ValueError(f"Unknown samples: {', '.join(unknown)}")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: {error}") from error

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            args.output_root / "run.json",
            {"workflow": "custom", "dataset": str(args.dataset_dir.resolve())},
        )
    failures = []
    for sample_id in selected:
        try:
            status, error = run_sample(args, sample_id)
        except (OSError, ValueError, json.JSONDecodeError) as caught:
            status, error = "failed", str(caught)
        print(f"[{status}] {sample_id}" + (f": {error}" if error else ""))
        if status == "failed":
            failures.append(sample_id)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
