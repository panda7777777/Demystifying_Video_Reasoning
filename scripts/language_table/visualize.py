#!/usr/bin/env python3
"""Batch step visualization for Language-Table on multiple GPUs and nodes."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts._visualization import (  # noqa: E402
    GenerationConfig,
    WAN_MODELS,
    atomic_write_json,
    build_pipeline,
    default_negative_prompt,
    make_step_callback,
)


def resolve_builder_dir(data_dir: Path) -> Path:
    """Resolve a path to exactly one TFDS builder directory."""
    root = data_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    def is_builder(path: Path) -> bool:
        return (path / "dataset_info.json").is_file()

    if is_builder(root):
        return root
    candidates = sorted(
        path
        for depth in (1, 2)
        for path in root.glob("/".join("*" for _ in range(depth)))
        if path.is_dir() and is_builder(path)
    )
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"No TFDS builder directory found under {root}")
    listing = "\n  ".join(str(path) for path in candidates)
    raise ValueError(
        f"Dataset directory is ambiguous; pass one of:\n  {listing}"
    )


def read_episode_count(builder_dir: Path, split: str) -> int:
    info = json.loads(
        (builder_dir / "dataset_info.json").read_text(encoding="utf-8")
    )
    splits = info.get("splits", [])
    if isinstance(splits, dict):
        split_info = splits.get(split)
        if split_info is not None:
            return int(split_info.get("numExamples", split_info.get("num_examples", 0)))
        available = sorted(splits)
    else:
        for split_info in splits:
            if split_info.get("name") == split:
                lengths = split_info.get("shardLengths", [])
                return sum(int(length) for length in lengths)
        available = [item.get("name") for item in splits]
    raise ValueError(f"Split {split!r} not found; available splits: {available}")


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
                value = slice(
                    *(int(field) if field else None for field in fields)
                )
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
    encoded = (
        json.dumps(expected, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
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
            raise ValueError(
                f"Existing run manifest has different parameters: {path}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def input_complete(input_dir: Path) -> bool:
    return (
        (input_dir / "input.png").is_file()
        and (input_dir / "prompt.txt").is_file()
    )


def decode_instruction(value: Any) -> str:
    raw = value.numpy()
    if isinstance(raw, bytes):
        return raw.decode("utf-8").strip("\x00")
    values = raw.tolist() if hasattr(raw, "tolist") else raw
    return bytes(int(item) for item in values if 0 < int(item) < 256).decode(
        "utf-8"
    )


def extract_stage(args: argparse.Namespace) -> int:
    import tensorflow_datasets as tfds
    from PIL import Image

    indices = json.loads(args.assignment.read_text(encoding="utf-8"))["indices"]
    pending = [
        index
        for index in indices
        if not input_complete(args.output_root / "inputs" / f"{index:06d}")
    ]
    if not pending:
        return 0
    builder = tfds.builder_from_directory(str(args.builder_dir))
    for start, stop in contiguous_ranges(pending):
        dataset = builder.as_dataset(
            split=f"{args.split}[{start}:{stop}]", shuffle_files=False
        )
        for offset, episode in enumerate(dataset):
            index = start + offset
            first_step = next(iter(episode["steps"]))
            image = Image.fromarray(first_step["observation"]["rgb"].numpy()).convert(
                "RGB"
            )
            prompt = decode_instruction(first_step["observation"]["instruction"])
            final_dir = args.output_root / "inputs" / f"{index:06d}"
            temporary = final_dir.with_name(f".{final_dir.name}.{os.getpid()}.tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            image.save(temporary / "input.png")
            (temporary / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            if final_dir.exists():
                shutil.rmtree(final_dir)
            temporary.replace(final_dir)
            print(f"[extract] {index:06d}: {prompt!r}", flush=True)
    return 0


def inference_stage(args: argparse.Namespace) -> int:
    from PIL import Image

    generation = GenerationConfig(
        model=args.model,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        max_denoising_steps=args.max_denoising_steps,
        seed=args.seed,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )
    generation.validate()
    pipe = build_pipeline(
        args.model,
        vbvr_model_path=args.vbvr_model_path,
        lora_path=args.lora_path,
        high_noise_lora_path=args.high_noise_lora_path,
        low_noise_lora_path=args.low_noise_lora_path,
        lora_alpha=args.lora_alpha,
    )
    indices = json.loads(args.assignment.read_text(encoding="utf-8"))["indices"]
    failures = []
    for index in indices:
        sample_id = f"{index:06d}"
        input_dir = args.output_root / "inputs" / sample_id
        episode_dir = args.output_root / "episodes" / sample_id
        metadata = {
            "generation": generation.to_dict(),
            "input": f"inputs/{sample_id}/input.png",
            "prompt": (input_dir / "prompt.txt").read_text(encoding="utf-8").strip(),
        }
        metadata_path = episode_dir / "metadata.json"
        if metadata_path.is_file() and not args.overwrite:
            if json.loads(metadata_path.read_text(encoding="utf-8")) == metadata:
                print(f"[skip] {sample_id}", flush=True)
                continue
            failures.append((sample_id, "existing result uses different parameters"))
            continue

        temporary = episode_dir.with_name(f".{sample_id}.{os.getpid()}.tmp")
        try:
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            image = Image.open(input_dir / "input.png").convert("RGB")
            callback = make_step_callback(temporary / "steps", fps=args.fps)
            video = pipe(
                prompt=metadata["prompt"],
                negative_prompt=(
                    args.negative_prompt or default_negative_prompt(args.model)
                ),
                input_image=image,
                num_frames=args.num_frames,
                seed=args.seed,
                tiled=True,
                height=image.height,
                width=image.width,
                num_inference_steps=args.num_inference_steps,
                step_callback=callback,
                vis_steps=None,
                max_denoising_steps=args.max_denoising_steps,
            )
            from diffsynth.utils.data import save_video

            save_video(
                video,
                str(temporary / "generated.mp4"),
                fps=args.fps,
                quality=5,
            )
            atomic_write_json(temporary / "metadata.json", metadata)
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
            temporary.replace(episode_dir)
            print(f"[complete] {sample_id}", flush=True)
        except Exception as error:  # keep remaining assigned episodes running
            failures.append((sample_id, f"{type(error).__name__}: {error}"))
            print(f"[failed] {sample_id}: {failures[-1][1]}", file=sys.stderr)
    if failures:
        print(
            "Failed episodes: " + ", ".join(sample for sample, _ in failures),
            file=sys.stderr,
        )
    return 1 if failures else 0


def worker_arguments(args: argparse.Namespace, builder_dir: Path) -> list[str]:
    values = [
        "--builder-dir",
        str(builder_dir),
        "--split",
        args.split,
        "--output-root",
        str(args.output_root),
        "--model",
        args.model,
        "--num-frames",
        str(args.num_frames),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--max-denoising-steps",
        str(args.max_denoising_steps),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--lora-alpha",
        str(args.lora_alpha),
    ]
    for option, value in (
        ("--negative-prompt", args.negative_prompt),
        ("--vbvr-model-path", args.vbvr_model_path),
        ("--lora-path", args.lora_path),
        ("--high-noise-lora-path", args.high_noise_lora_path),
        ("--low-noise-lora-path", args.low_noise_lora_path),
    ):
        if value is not None:
            values.extend((option, str(value)))
    if args.overwrite:
        values.append("--overwrite")
    return values


def orchestrate(args: argparse.Namespace) -> int:
    builder_dir = resolve_builder_dir(args.data_dir)
    total = read_episode_count(builder_dir, args.split)
    indices = parse_indices(args.indices, total)
    if not indices:
        raise ValueError("No episodes selected")
    gpus = (
        [value.strip() for value in args.gpus.split(",") if value.strip()]
        if args.gpus
        else detect_gpus()
    )
    assignments = shard_indices(
        indices,
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        gpus=gpus,
    )
    generation = GenerationConfig(
        model=args.model,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        max_denoising_steps=args.max_denoising_steps,
        seed=args.seed,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )
    generation.validate()
    print(
        f"[plan] {len(indices)}/{total} episodes, {args.num_nodes} node(s), "
        f"{len(gpus)} GPU(s) per node"
    )
    for gpu, assigned in assignments.items():
        preview = ", ".join(str(index) for index in assigned[:6])
        print(f"[plan] GPU {gpu}: {len(assigned)} [{preview}]")
    if args.dry_run:
        return 0

    manifest = {
        "workflow": "language-table",
        "builder_dir": str(builder_dir),
        "split": args.split,
        "indices": indices,
        "num_nodes": args.num_nodes,
        "gpus_per_node": len(gpus),
        "generation": generation.to_dict(),
    }
    ensure_run_manifest(
        args.output_root / "run.json", manifest, overwrite=args.overwrite
    )
    runtime = args.output_root / "runtime"
    assignments_dir = runtime / "assignments"
    logs_dir = runtime / "logs"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    common = worker_arguments(args, builder_dir)
    node_indices = sorted(index for values in assignments.values() for index in values)

    missing = [
        index
        for index in node_indices
        if not input_complete(args.output_root / "inputs" / f"{index:06d}")
    ]
    if missing:
        if args.tfds_python is None:
            raise ValueError(
                f"{len(missing)} inputs need extraction; pass --tfds-python"
            )
        assignment = assignments_dir / f"node_{args.node_rank}_extract.json"
        atomic_write_json(assignment, {"indices": missing})
        subprocess.run(
            [
                str(args.tfds_python),
                str(Path(__file__).resolve()),
                "--stage",
                "extract",
                "--assignment",
                str(assignment),
                *common,
            ],
            check=True,
            cwd=REPO_ROOT,
        )
    if args.extract_only:
        return 0

    environment_base = dict(os.environ)
    if args.model_base_path:
        environment_base["DIFFSYNTH_MODEL_BASE_PATH"] = str(args.model_base_path)
        environment_base.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    processes = []
    inference_python = args.inference_python or Path(sys.executable)
    for gpu, assigned in assignments.items():
        if not assigned:
            continue
        assignment = assignments_dir / f"node_{args.node_rank}_gpu_{gpu}.json"
        atomic_write_json(assignment, {"indices": assigned})
        log_path = logs_dir / f"node_{args.node_rank}_gpu_{gpu}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        environment = dict(environment_base, CUDA_VISIBLE_DEVICES=gpu)
        process = subprocess.Popen(
            [
                str(inference_python),
                str(Path(__file__).resolve()),
                "--stage",
                "infer",
                "--assignment",
                str(assignment),
                *common,
            ],
            cwd=REPO_ROOT,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((gpu, process, log_handle, log_path))
        print(f"[launch] GPU {gpu}: PID {process.pid}, log {log_path}")

    failures = 0
    while processes:
        for entry in processes[:]:
            gpu, process, log_handle, log_path = entry
            return_code = process.poll()
            if return_code is None:
                continue
            log_handle.close()
            processes.remove(entry)
            failures += return_code != 0
            print(f"[worker] GPU {gpu}: exit {return_code}, log {log_path}")
        if processes:
            time.sleep(10)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--indices", default="all")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--model", choices=sorted(WAN_MODELS), default="wan2.2")
    parser.add_argument("--vbvr-model-path", type=Path)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--max-denoising-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--lora-path")
    parser.add_argument("--high-noise-lora-path")
    parser.add_argument("--low-noise-lora-path")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--gpus")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument(
        "--node-rank", type=int, default=int(os.environ.get("NODE_RANK", 0))
    )
    parser.add_argument("--tfds-python", type=Path)
    parser.add_argument("--inference-python", type=Path)
    parser.add_argument("--model-base-path", type=Path, default=REPO_ROOT / "models")
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("orchestrate", "extract", "infer"),
        default="orchestrate",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--assignment", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--builder-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.stage == "extract":
            return extract_stage(args)
        if args.stage == "infer":
            return inference_stage(args)
        if args.data_dir is None or args.output_root is None:
            raise ValueError("--data-dir and --output-root are required")
        return orchestrate(args)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
