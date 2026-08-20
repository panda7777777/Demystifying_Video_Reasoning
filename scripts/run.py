#!/usr/bin/env python3
"""Unified multi-dataset, multi-model video-reasoning inference entry point."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._batch_visualization import detect_gpus, shard_indices
from scripts._datasets import Sample, discover, prepare_image
from scripts._results import make_result_callback
from scripts._visualization import (
    WAN_MODELS,
    GenerationConfig,
    atomic_write_json,
    build_pipeline,
    default_negative_prompt,
    model_family,
    validate_model_source,
)

MODEL_SHORT = {"wan2.2": "wan22", "wan2.1": "wan21", "ltx2.3": "ltx23", "vbvr-wan2.2": "vbvr22", "lvp": "lvp"}
DATASET_SHORT = {"language-table": "ltable", "rmbench": "rmbench", "custom": "custom"}
IMMUTABLE_MANIFEST_KEYS = (
    "name", "dataset", "data_dir", "selection", "split", "generation",
    "adapters", "processing", "samples",
)


def generation_config(args: argparse.Namespace) -> GenerationConfig:
    max_steps = args.max_denoising_steps
    if max_steps is None and model_family(args.model) in WAN_MODELS:
        max_steps = 10
    config = GenerationConfig(
        model=args.model,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        max_denoising_steps=max_steps,
        seed=args.seed,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )
    config.validate()
    return config


def validate_adapter_paths(args: argparse.Namespace) -> None:
    for option, value in (
        ("--lora-path", args.lora_path),
        ("--high-noise-lora-path", args.high_noise_lora_path),
        ("--low-noise-lora-path", args.low_noise_lora_path),
        ("--lvp-checkpoint", args.lvp_checkpoint),
    ):
        if value is None:
            continue
        path = Path(value).expanduser()
        if not path.is_file():
            kind = "directory" if path.is_dir() else "missing path"
            raise ValueError(f"{option} must name a checkpoint file, got {kind}: {path}")
    if model_family(args.model) == "lvp":
        if args.lvp_base_model is None or args.lvp_checkpoint is None:
            raise ValueError("--model large-video-planner requires --lvp-base-model and --lvp-checkpoint")
        validate_model_source(args.lvp_base_model)


def validate_cuda_runtime(args: argparse.Namespace, gpu: str) -> None:
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
    check = subprocess.run(
        [
            str(args.python),
            "-c",
            "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        detail = (check.stderr or check.stdout).strip().splitlines()
        diagnostic = next(
            (line.strip() for line in detail if "Error " in line or "CUDA initialization" in line),
            detail[-1].strip() if detail else "",
        )
        suffix = f" Diagnostic: {diagnostic}" if diagnostic else ""
        raise ValueError(
            f"CUDA is unavailable to {args.python} with CUDA_VISIBLE_DEVICES={gpu}."
            f" Verify that the job/container has /dev/nvidia* devices and that the "
            f"NVIDIA driver supports this PyTorch CUDA build.{suffix}"
        )


def validate_task_name(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        raise ValueError("--task-name must not be empty")
    if value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError("--task-name must be a single safe path component")
    return value


def run_name(args: argparse.Namespace) -> str:
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M")
    parts = [stamp]
    if args.task_name is not None:
        parts.append(args.task_name)
    parts.extend((MODEL_SHORT[model_family(args.model)], DATASET_SHORT[args.dataset]))
    return "_".join(parts)


def make_manifest(
    args: argparse.Namespace,
    config: GenerationConfig,
    split: str,
    samples: list[Sample],
    name: str,
    gpus: list[str],
) -> dict:
    return {
        "name": name,
        "dataset": args.dataset,
        "data_dir": str(args.data_dir.resolve()),
        "selection": args.selection,
        "split": split,
        "generation": config.to_dict(),
        "adapters": {
            "lora": args.lora_path,
            "high_noise": args.high_noise_lora_path,
            "low_noise": args.low_noise_lora_path,
            "scale": args.lora_alpha,
            "lvp_base_model": args.lvp_base_model,
            "lvp_checkpoint": args.lvp_checkpoint,
        },
        "processing": {
            "max_size": args.max_size,
            "overview_columns": args.overview_columns,
            "python": str(args.python.resolve()),
        },
        "parallel": {
            "num_nodes": args.num_nodes,
            "gpus": gpus,
            "gpus_per_node": len(gpus),
            "batch_size_per_gpu": args.batch_size,
        },
        "samples": [sample.sample_id for sample in samples],
    }


def validate_resume_manifest(existing: dict, requested: dict, run_dir: Path) -> None:
    missing = [key for key in IMMUTABLE_MANIFEST_KEYS if key not in existing]
    if missing:
        raise ValueError(
            f"Cannot safely resume {run_dir}: run.json lacks immutable field(s): "
            + ", ".join(missing)
        )
    changed = [
        key for key in IMMUTABLE_MANIFEST_KEYS
        if existing[key] != requested[key]
    ]
    if changed:
        details = "; ".join(
            f"{key}: saved={existing[key]!r}, requested={requested[key]!r}"
            for key in changed
        )
        raise ValueError(
            "Resume configuration mismatch; only parallel parameters may change: "
            + details
        )


def completed_sample_ids(run_dir: Path, samples: list[Sample], config: GenerationConfig) -> set[str]:
    completed = set()
    expected_generation = config.to_dict()
    for sample in samples:
        metadata_path = run_dir / "samples" / sample.sample_id / "output" / "metadata.json"
        if not metadata_path.is_file():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid completed-sample metadata: {metadata_path}: {error}") from error
        if metadata.get("sample_id") != sample.sample_id or metadata.get("generation") != expected_generation:
            raise ValueError(f"Completed-sample metadata does not match run.json: {metadata_path}")
        completed.add(sample.sample_id)
    return completed


def persisted_samples(run_dir: Path, sample_ids: list[str]) -> list[Sample] | None:
    """Load immutable prepared inputs, or return None when preparation was interrupted."""
    samples = []
    for sample_id in sample_ids:
        if not isinstance(sample_id, str) or Path(sample_id).name != sample_id:
            raise ValueError(f"Unsafe sample id in run.json: {sample_id!r}")
        input_dir = run_dir / "samples" / sample_id / "input"
        image_path = input_dir / "initial_frame.png"
        prompt_path = input_dir / "prompt.txt"
        metadata_path = input_dir / "metadata.json"
        if not image_path.is_file() or not prompt_path.is_file() or not metadata_path.is_file():
            return None
        prompt = prompt_path.read_text(encoding="utf-8").strip()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source = metadata.get("source")
        if not prompt or not isinstance(source, dict):
            raise ValueError(f"Invalid persisted input metadata: {input_dir}")
        samples.append(Sample(sample_id, str(image_path), prompt, source))
    return samples


def worker_options(args: argparse.Namespace, run_dir: Path, assignment: Path) -> list[str]:
    values = [
        str(Path(__file__).resolve()), "--stage", "worker", "--run-dir", str(run_dir),
        "--assignment", str(assignment), "--model", args.model,
        "--num-frames", str(args.num_frames), "--num-inference-steps", str(args.num_inference_steps),
        "--seed", str(args.seed), "--fps", str(args.fps), "--overview-columns", str(args.overview_columns),
        "--lora-alpha", str(args.lora_alpha),
    ]
    for option, value in (
        ("--max-denoising-steps", generation_config(args).max_denoising_steps),
        ("--negative-prompt", args.negative_prompt),
        ("--lora-path", args.lora_path), ("--high-noise-lora-path", args.high_noise_lora_path),
        ("--low-noise-lora-path", args.low_noise_lora_path),
        ("--lvp-base-model", args.lvp_base_model), ("--lvp-checkpoint", args.lvp_checkpoint),
    ):
        if value is not None:
            values.extend((option, str(value)))
    return values


def worker(args: argparse.Namespace) -> int:
    config = generation_config(args)
    samples = [Sample.from_dict(value) for value in json.loads(args.assignment.read_text(encoding="utf-8"))["samples"]]
    pipe = build_pipeline(
        args.model, lora_path=args.lora_path,
        high_noise_lora_path=args.high_noise_lora_path, low_noise_lora_path=args.low_noise_lora_path,
        lora_alpha=args.lora_alpha,
        lvp_base_model=args.lvp_base_model, lvp_checkpoint=args.lvp_checkpoint,
    )
    failures = 0
    for sample in samples:
        sample_dir = args.run_dir / "samples" / sample.sample_id
        output_dir = sample_dir / "output"
        temporary = sample_dir / f".output.{os.getpid()}.tmp"
        try:
            if (output_dir / "metadata.json").is_file():
                print(f"[skip] {sample.sample_id}", flush=True)
                continue
            if output_dir.exists():
                preserved = sample_dir / f".output.incomplete.{int(time.time())}.{os.getpid()}"
                output_dir.replace(preserved)
                print(f"[preserve] {sample.sample_id}: {preserved.name}", flush=True)
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            image = Image.open(sample_dir / "input" / "initial_frame.png").convert("RGB")
            callback = make_result_callback(
                temporary / "steps", prompt=sample.prompt, fps=args.fps,
                overview_columns=args.overview_columns,
            )
            common = {
                "prompt": sample.prompt,
                "negative_prompt": args.negative_prompt or default_negative_prompt(args.model),
                "num_frames": args.num_frames, "seed": args.seed, "tiled": True,
                "height": image.height, "width": image.width,
                "num_inference_steps": args.num_inference_steps,
                "step_callback": callback, "vis_steps": None,
            }
            video_path = temporary / "video.mp4"
            if model_family(args.model) == "ltx2.3":
                video, audio = pipe(input_images=[image], input_images_indexes=[0], input_images_strength=1.0, **common)
                from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2
                write_video_audio_ltx2(video=video, audio=audio, output_path=str(video_path), fps=args.fps, audio_sample_rate=pipe.audio_vocoder.output_sampling_rate)
            else:
                video = pipe(input_image=image, max_denoising_steps=config.max_denoising_steps, **common)
                from diffsynth.utils.data import save_video
                save_video(video, str(video_path), fps=args.fps, quality=5)
            atomic_write_json(temporary / "metadata.json", {"sample_id": sample.sample_id, "source": sample.source, "generation": config.to_dict()})
            temporary.replace(output_dir)
            print(f"[complete] {sample.sample_id}", flush=True)
        except Exception as error:  # noqa: BLE001 -- isolate failures by sample
            failures += 1
            print(f"[failed] {sample.sample_id}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    return int(bool(failures))


def orchestrate(args: argparse.Namespace) -> int:
    config = generation_config(args)
    split = args.split or ("train" if args.dataset == "language-table" else "seen")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()] if args.gpus else detect_gpus()
    if not gpus:
        raise ValueError("No GPUs selected or detected")
    if not args.dry_run:
        validate_cuda_runtime(args, gpus[0])
    if args.resume_dir is not None:
        run_dir = args.resume_dir.resolve()
        manifest_path = run_dir / "run.json"
        if not run_dir.is_dir() or not manifest_path.is_file():
            raise ValueError(f"Resume directory is not a valid run folder: {run_dir}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        name = existing_manifest.get("name", run_dir.name)
        saved_sample_ids = existing_manifest.get("samples")
        if not isinstance(saved_sample_ids, list):
            raise ValueError(f"Invalid samples list in {manifest_path}")
        samples = persisted_samples(run_dir, saved_sample_ids)
        if samples is None:
            print("[resume] persisted inputs are incomplete; scanning the dataset", flush=True)
            samples = discover(args.dataset, args.data_dir.resolve(), args.selection, split)
    else:
        name = args.run_name or run_name(args)
        run_dir = args.output_dir.resolve() / name
        manifest_path = run_dir / "run.json"
        existing_manifest = None
        samples = discover(args.dataset, args.data_dir.resolve(), args.selection, split)

    manifest = make_manifest(args, config, split, samples, name, gpus)
    if existing_manifest is not None:
        validate_resume_manifest(existing_manifest, manifest, run_dir)
        completed = completed_sample_ids(run_dir, samples, config)
        pending_samples = [sample for sample in samples if sample.sample_id not in completed]
        incomplete = 0
        for sample in pending_samples:
            sample_dir = run_dir / "samples" / sample.sample_id
            if (sample_dir / "output").exists() or any(sample_dir.glob(".output.*")):
                incomplete += 1
        print(
            f"[resume] {len(samples)} total, {len(completed)} completed, "
            f"{incomplete} incomplete, {len(pending_samples)} pending"
        )
        samples_to_run = pending_samples
    else:
        completed = set()
        samples_to_run = samples

    # DiffSynth's video pipelines currently accept one sample per call.  Treat
    # batch size as process-level concurrency so each GPU can keep multiple
    # independent inference calls in flight without changing model internals.
    slots = [(gpu, slot) for gpu in gpus for slot in range(args.batch_size)]
    slot_names = [f"{gpu}_{slot}" for gpu, slot in slots]
    assignments = shard_indices(
        list(range(len(samples_to_run))),
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        gpus=slot_names,
    )
    print(f"[run] {run_dir}")
    print(
        f"[plan] {len(samples_to_run)} pending sample(s), {args.num_nodes} node(s), "
        f"{len(gpus)} GPU(s)/node, batch size {args.batch_size}/GPU"
    )
    for gpu, slot in slots:
        indices = assignments[f"{gpu}_{slot}"]
        print(f"[plan] GPU {gpu}, slot {slot}: {len(indices)} sample(s)")
    if args.dry_run:
        return 0
    if existing_manifest is not None and not samples_to_run:
        print("[resume] all samples are already complete")
        return 0
    if run_dir.exists() and not (run_dir / "run.json").is_file():
        raise ValueError(f"Refusing non-run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if existing_manifest is None and manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Run name collision with different parameters: {run_dir}")
    atomic_write_json(manifest_path, manifest)
    runtime = run_dir / "runtime"
    runtime.mkdir(exist_ok=True)
    runtime_suffix = (
        f"_resume_{datetime.now(ZoneInfo('Asia/Singapore')).strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        if existing_manifest is not None else ""
    )
    processes = []
    environment_base = dict(os.environ)
    for gpu, slot in slots:
        indices = assignments[f"{gpu}_{slot}"]
        if not indices:
            continue
        assigned = []
        for index in indices:
            sample = samples_to_run[index]
            input_dir = run_dir / "samples" / sample.sample_id / "input"
            image_path = input_dir / "initial_frame.png"
            saved = persisted_samples(run_dir, [sample.sample_id]) if existing_manifest is not None else None
            if saved is not None:
                assigned.append(saved[0])
                continue
            width, height = prepare_image(sample, image_path, args.max_size)
            (input_dir / "prompt.txt").write_text(sample.prompt + "\n", encoding="utf-8")
            source = {key: value for key, value in sample.source.items() if key != "rgb"}
            atomic_write_json(input_dir / "metadata.json", {"source": source, "size": [width, height]})
            assigned.append(Sample(sample.sample_id, str(image_path), sample.prompt, source))
        assignment = runtime / f"node_{args.node_rank}_gpu_{gpu}_slot_{slot}{runtime_suffix}.json"
        atomic_write_json(assignment, {"samples": [sample.to_dict() for sample in assigned]})
        log_path = runtime / f"node_{args.node_rank}_gpu_{gpu}_slot_{slot}{runtime_suffix}.log"
        log = log_path.open("w", encoding="utf-8")
        env = dict(environment_base, CUDA_VISIBLE_DEVICES=gpu)
        process = subprocess.Popen([str(args.python), *worker_options(args, run_dir, assignment)], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((gpu, slot, process, log, log_path))
        print(f"[launch] GPU {gpu}, slot {slot}: PID {process.pid}, log {log_path}")
    failed = 0
    while processes:
        for entry in processes[:]:
            gpu, slot, process, log, path = entry
            code = process.poll()
            if code is None:
                continue
            log.close(); processes.remove(entry); failed += code != 0
            print(f"[worker] GPU {gpu}, slot {slot}: exit {code}, log {path}")
            if code != 0:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                print(f"[worker] GPU {gpu}, slot {slot}: last log lines:", file=sys.stderr)
                for line in lines[-20:]:
                    print(f"  {line}", file=sys.stderr)
        if processes:
            time.sleep(5)
    return int(bool(failed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model", default="Wan-AI/Wan2.2-I2V-A14B",
        help="Absolute model directory or Hugging Face repository id",
    )
    parser.add_argument("--dataset", choices=tuple(DATASET_SHORT), default="custom")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "examples" / "custom_dataset")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "output")
    parser.add_argument(
        "--task-name",
        help="Optional task suffix inserted before the model suffix in automatic run names",
    )
    parser.add_argument("--resume-dir", type=Path, help="Resume an existing run directory")
    parser.add_argument("--selection", default="all", help="custom: ids; Language-Table: indices/slices; RMBench: tasks:episode-slice")
    parser.add_argument("--split", help="Defaults to train for Language-Table and seen for RMBench")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("NODE_RANK", "0")))
    parser.add_argument("--gpus", help="Comma-separated local GPU ids; auto-detected when omitted")
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Concurrent inference workers per GPU (DiffSynth pipelines are single-sample)",
    )
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--max-denoising-steps", type=int)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-size", type=int, default=832)
    parser.add_argument("--overview-columns", type=int, default=6)
    parser.add_argument("--negative-prompt")
    parser.add_argument("--lora-path")
    parser.add_argument("--high-noise-lora-path")
    parser.add_argument("--low-noise-lora-path")
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--lvp-base-model", help="Wan2.1-I2V-14B-480P directory used by LVP")
    parser.add_argument("--lvp-checkpoint", help="LVP Lightning .ckpt fine-tuned weights")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--run-name", help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=("orchestrate", "worker"), default="orchestrate", help=argparse.SUPPRESS)
    parser.add_argument("--run-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--assignment", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        args.model = validate_model_source(args.model)
        args.task_name = validate_task_name(args.task_name)
        model_family(args.model)
        validate_adapter_paths(args)
        if args.max_size <= 0 or args.overview_columns <= 0 or args.batch_size <= 0:
            raise ValueError("--max-size, --overview-columns, and --batch-size must be positive")
        if args.stage == "worker":
            if args.run_dir is None or args.assignment is None:
                raise ValueError("Internal worker arguments are incomplete")
            return worker(args)
        return orchestrate(args)
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
