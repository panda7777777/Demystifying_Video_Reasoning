#!/usr/bin/env python3
"""Batch step visualization for RMBench on multiple GPUs and nodes."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts._batch_visualization import (  # noqa: E402
    detect_gpus,
    ensure_run_manifest,
    parse_indices,
    shard_indices,
)
from scripts._visualization import (  # noqa: E402
    GenerationConfig,
    SUPPORTED_MODELS,
    WAN_MODELS,
    atomic_write_json,
    build_pipeline,
    default_negative_prompt,
    make_step_callback,
)


DEFAULT_DATA_DIR = Path("/mnt/umm/users/zuojing/code/RMBench/data/data")
EPISODE_PATTERN = re.compile(r"episode(\d+)$")


@dataclass(frozen=True)
class Sample:
    """One validated RMBench task episode."""

    task: str
    episode: int
    video_path: Path
    instruction_path: Path

    @property
    def sample_id(self) -> str:
        return f"{self.task}__episode_{self.episode:06d}"

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "episode": self.episode,
            "video_path": str(self.video_path),
            "instruction_path": str(self.instruction_path),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "Sample":
        return cls(
            task=str(payload["task"]),
            episode=int(payload["episode"]),
            video_path=Path(str(payload["video_path"])),
            instruction_path=Path(str(payload["instruction_path"])),
        )


def _episode_files(directory: Path, suffix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist: {directory}")
    for path in directory.iterdir():
        if not path.is_file() or path.suffix != suffix:
            continue
        match = EPISODE_PATTERN.fullmatch(path.stem)
        if match is None:
            continue
        episode = int(match.group(1))
        if episode in files:
            raise ValueError(f"Duplicate episode {episode} under {directory}")
        files[episode] = path.resolve()
    return files


def discover_tasks(data_dir: Path) -> dict[str, list[Sample]]:
    """Discover tasks with matching videos and instruction files."""
    root = data_dir.resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")
    tasks: dict[str, list[Sample]] = {}
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        setting = task_dir / "demo_clean"
        if not setting.is_dir():
            continue
        videos = _episode_files(setting / "video", ".mp4")
        instructions = _episode_files(setting / "instructions", ".json")
        if set(videos) != set(instructions):
            missing_videos = sorted(set(instructions) - set(videos))
            missing_instructions = sorted(set(videos) - set(instructions))
            details = []
            if missing_videos:
                details.append(f"missing videos {missing_videos}")
            if missing_instructions:
                details.append(f"missing instructions {missing_instructions}")
            raise ValueError(f"Incomplete task {task_dir.name}: {', '.join(details)}")
        if videos:
            tasks[task_dir.name] = [
                Sample(task_dir.name, episode, videos[episode], instructions[episode])
                for episode in sorted(videos)
            ]
    if not tasks:
        raise ValueError(f"No RMBench tasks found under {root}")
    return tasks


def parse_tasks(specification: str, available: list[str]) -> list[str]:
    """Select task names from a comma-separated list or ``all``."""
    if specification.strip().lower() == "all":
        return available
    selected = []
    for value in specification.split(","):
        task = value.strip()
        if task and task not in selected:
            selected.append(task)
    if not selected:
        raise ValueError("No tasks selected")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown tasks: {', '.join(unknown)}")
    return sorted(selected)


def select_samples(
    tasks: dict[str, list[Sample]],
    *,
    task_spec: str,
    episode_spec: str,
) -> list[Sample]:
    """Select episode ids independently within each requested task."""
    selected: list[Sample] = []
    for task in parse_tasks(task_spec, sorted(tasks)):
        samples = tasks[task]
        by_episode = {sample.episode: sample for sample in samples}
        if episode_spec.strip().lower() in {"", "all", ":"}:
            episode_ids = sorted(by_episode)
        else:
            upper_bound = max(by_episode) + 1
            episode_ids = parse_indices(episode_spec, upper_bound)
            missing = sorted(set(episode_ids) - set(by_episode))
            if missing:
                raise ValueError(f"Task {task} has no episodes: {missing}")
        selected.extend(by_episode[episode] for episode in episode_ids)
    if not selected:
        raise ValueError("No episodes selected")
    return selected


def load_prompt(sample: Sample, instruction_split: str) -> str:
    """Read the single global instruction selected for an episode."""
    payload = json.loads(sample.instruction_path.read_text(encoding="utf-8"))
    values = payload.get(instruction_split)
    if not isinstance(values, list) or len(values) != 1:
        raise ValueError(
            f"{sample.instruction_path} must contain exactly one "
            f"{instruction_split!r} instruction"
        )
    prompt = values[0]
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(
            f"{sample.instruction_path} contains an empty "
            f"{instruction_split!r} instruction"
        )
    return prompt.strip()


def source_metadata(sample: Sample, instruction_split: str, prompt: str) -> dict:
    return {
        "task": sample.task,
        "episode": sample.episode,
        "video": str(sample.video_path),
        "instruction": str(sample.instruction_path),
        "instruction_split": instruction_split,
        "prompt": prompt,
    }


def input_complete(input_dir: Path, expected: dict | None = None) -> bool:
    if not (
        (input_dir / "input.png").is_file()
        and (input_dir / "prompt.txt").is_file()
        and (input_dir / "metadata.json").is_file()
    ):
        return False
    if expected is None:
        return True
    try:
        return json.loads(
            (input_dir / "metadata.json").read_text(encoding="utf-8")
        ) == expected
    except (OSError, json.JSONDecodeError):
        return False


def read_first_frame(video_path: Path) -> Image.Image:
    """Decode the first frame of an RMBench head-camera video."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(video_path))
    try:
        frame = reader.get_data(0)
    finally:
        reader.close()
    image = Image.fromarray(frame).convert("RGB")
    if image.width <= 0 or image.height <= 0:
        raise ValueError(f"Invalid first frame dimensions in {video_path}")
    if image.width % 16 or image.height % 16:
        raise ValueError(
            f"First frame must be 16-pixel aligned, got "
            f"{image.width}x{image.height}: {video_path}"
        )
    return image


def read_assignment(path: Path) -> list[Sample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Sample.from_dict(item) for item in payload["samples"]]


def extract_stage(args: argparse.Namespace) -> int:
    failures = []
    for sample in read_assignment(args.assignment):
        input_dir = args.output_root / "inputs" / sample.sample_id
        try:
            prompt = load_prompt(sample, args.instruction_split)
            expected = source_metadata(sample, args.instruction_split, prompt)
            if input_complete(input_dir, expected) and not args.overwrite:
                print(f"[extract:skip] {sample.sample_id}", flush=True)
                continue
            if input_dir.exists() and not args.overwrite:
                raise ValueError(
                    "existing input is incomplete or uses different parameters; "
                    "use --overwrite"
                )
            temporary = input_dir.with_name(f".{input_dir.name}.{os.getpid()}.tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            image = read_first_frame(sample.video_path)
            image.save(temporary / "input.png")
            (temporary / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            atomic_write_json(temporary / "metadata.json", expected)
            if input_dir.exists():
                shutil.rmtree(input_dir)
            temporary.replace(input_dir)
            print(f"[extract] {sample.sample_id}: {prompt!r}", flush=True)
        except Exception as error:
            failures.append((sample.sample_id, f"{type(error).__name__}: {error}"))
            print(f"[extract:failed] {sample.sample_id}: {failures[-1][1]}", file=sys.stderr)
    return 1 if failures else 0


def resolved_generation(args: argparse.Namespace) -> GenerationConfig:
    max_steps = args.max_denoising_steps
    if max_steps is None and args.model in WAN_MODELS:
        max_steps = 10
    generation = GenerationConfig(
        model=args.model,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        max_denoising_steps=max_steps,
        seed=args.seed,
        fps=args.fps,
        negative_prompt=args.negative_prompt,
    )
    generation.validate()
    return generation


def inference_stage(args: argparse.Namespace) -> int:
    generation = resolved_generation(args)
    pipe = build_pipeline(
        args.model,
        vbvr_model_path=args.vbvr_model_path,
        lora_path=args.lora_path,
        high_noise_lora_path=args.high_noise_lora_path,
        low_noise_lora_path=args.low_noise_lora_path,
        lora_alpha=args.lora_alpha,
    )
    failures = []
    for sample in read_assignment(args.assignment):
        sample_id = sample.sample_id
        input_dir = args.output_root / "inputs" / sample_id
        episode_dir = args.output_root / "episodes" / sample_id
        try:
            source = json.loads(
                (input_dir / "metadata.json").read_text(encoding="utf-8")
            )
            metadata = {
                "generation": generation.to_dict(),
                "input": f"inputs/{sample_id}/input.png",
                "source": source,
            }
            metadata_path = episode_dir / "metadata.json"
            if metadata_path.is_file() and not args.overwrite:
                if json.loads(metadata_path.read_text(encoding="utf-8")) == metadata:
                    print(f"[skip] {sample_id}", flush=True)
                    continue
                raise ValueError(
                    "existing result uses different parameters; use --overwrite"
                )

            temporary = episode_dir.with_name(f".{sample_id}.{os.getpid()}.tmp")
            if temporary.exists():
                shutil.rmtree(temporary)
            temporary.mkdir(parents=True)
            image = Image.open(input_dir / "input.png").convert("RGB")
            callback = make_step_callback(temporary / "steps", fps=args.fps)
            negative_prompt = (
                args.negative_prompt or default_negative_prompt(args.model)
            )
            output_path = temporary / "generated.mp4"
            if args.model == "ltx2.3":
                from diffsynth.utils.data.media_io_ltx2 import (
                    write_video_audio_ltx2,
                )

                video, audio = pipe(
                    prompt=source["prompt"],
                    negative_prompt=negative_prompt,
                    input_images=[image],
                    input_images_indexes=[0],
                    input_images_strength=1.0,
                    num_frames=args.num_frames,
                    seed=args.seed,
                    tiled=True,
                    height=image.height,
                    width=image.width,
                    num_inference_steps=args.num_inference_steps,
                    step_callback=callback,
                    vis_steps=None,
                )
                write_video_audio_ltx2(
                    video=video,
                    audio=audio,
                    output_path=str(output_path),
                    fps=args.fps,
                    audio_sample_rate=pipe.audio_vocoder.output_sampling_rate,
                )
            else:
                from diffsynth.utils.data import save_video

                video = pipe(
                    prompt=source["prompt"],
                    negative_prompt=negative_prompt,
                    input_image=image,
                    num_frames=args.num_frames,
                    seed=args.seed,
                    tiled=True,
                    height=image.height,
                    width=image.width,
                    num_inference_steps=args.num_inference_steps,
                    step_callback=callback,
                    vis_steps=None,
                    max_denoising_steps=generation.max_denoising_steps,
                )
                save_video(
                    video,
                    str(output_path),
                    fps=args.fps,
                    quality=5,
                )
            atomic_write_json(temporary / "metadata.json", metadata)
            if episode_dir.exists():
                shutil.rmtree(episode_dir)
            temporary.replace(episode_dir)
            print(f"[complete] {sample_id}", flush=True)
        except Exception as error:
            failures.append((sample_id, f"{type(error).__name__}: {error}"))
            print(f"[failed] {sample_id}: {failures[-1][1]}", file=sys.stderr)
    return 1 if failures else 0


def worker_arguments(args: argparse.Namespace) -> list[str]:
    generation = resolved_generation(args)
    values = [
        "--output-root",
        str(args.output_root),
        "--instruction-split",
        args.instruction_split,
        "--model",
        args.model,
        "--num-frames",
        str(args.num_frames),
        "--num-inference-steps",
        str(args.num_inference_steps),
        "--seed",
        str(args.seed),
        "--fps",
        str(args.fps),
        "--lora-alpha",
        str(args.lora_alpha),
    ]
    for option, value in (
        ("--max-denoising-steps", generation.max_denoising_steps),
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


def write_assignment(path: Path, samples: list[Sample]) -> None:
    atomic_write_json(path, {"samples": [sample.to_dict() for sample in samples]})


def orchestrate(args: argparse.Namespace) -> int:
    tasks = discover_tasks(args.data_dir)
    samples = select_samples(
        tasks,
        task_spec=args.tasks,
        episode_spec=args.episodes,
    )
    # Validate every prompt before creating outputs or launching workers.
    for sample in samples:
        load_prompt(sample, args.instruction_split)
    gpus = (
        [value.strip() for value in args.gpus.split(",") if value.strip()]
        if args.gpus
        else detect_gpus()
    )
    assignments_by_index = shard_indices(
        list(range(len(samples))),
        node_rank=args.node_rank,
        num_nodes=args.num_nodes,
        gpus=gpus,
    )
    assignments = {
        gpu: [samples[index] for index in indices]
        for gpu, indices in assignments_by_index.items()
    }
    generation = resolved_generation(args)
    counts = Counter(sample.task for sample in samples)
    print(
        f"[plan] {len(samples)} episode(s) across {len(counts)} task(s), "
        f"{args.num_nodes} node(s), {len(gpus)} GPU(s) per node"
    )
    print("[plan] tasks: " + ", ".join(f"{task}={count}" for task, count in counts.items()))
    for gpu, assigned in assignments.items():
        preview = ", ".join(sample.sample_id for sample in assigned[:4])
        print(f"[plan] GPU {gpu}: {len(assigned)} [{preview}]")
    if args.dry_run:
        return 0

    manifest = {
        "workflow": "rmbench",
        "data_dir": str(args.data_dir.resolve()),
        "tasks": list(counts),
        "episodes": args.episodes,
        "sample_ids": [sample.sample_id for sample in samples],
        "instruction_split": args.instruction_split,
        "num_nodes": args.num_nodes,
        "gpus_per_node": len(gpus),
        "generation": generation.to_dict(),
    }
    ensure_run_manifest(
        args.output_root / "run.json",
        manifest,
        overwrite=args.overwrite,
    )
    runtime = args.output_root / "runtime"
    assignments_dir = runtime / "assignments"
    logs_dir = runtime / "logs"
    assignments_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    node_samples = [
        sample for assigned in assignments.values() for sample in assigned
    ]
    missing_inputs = []
    for sample in node_samples:
        prompt = load_prompt(sample, args.instruction_split)
        expected = source_metadata(sample, args.instruction_split, prompt)
        if not input_complete(
            args.output_root / "inputs" / sample.sample_id,
            expected,
        ):
            missing_inputs.append(sample)

    if missing_inputs or args.overwrite:
        extract_samples = node_samples if args.overwrite else missing_inputs
        assignment = assignments_dir / f"node_{args.node_rank}_extract.json"
        write_assignment(assignment, extract_samples)
        subprocess.run(
            [
                str(args.extraction_python or sys.executable),
                str(Path(__file__).resolve()),
                "--stage",
                "extract",
                "--assignment",
                str(assignment),
                *worker_arguments(args),
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
        write_assignment(assignment, assigned)
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
                *worker_arguments(args),
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
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--episodes", default="all")
    parser.add_argument(
        "--instruction-split",
        choices=("seen", "unseen"),
        default="seen",
    )
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default="wan2.2")
    parser.add_argument("--vbvr-model-path", type=Path)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument(
        "--max-denoising-steps",
        type=int,
        help="Defaults to 10 for Wan models and disabled for LTX",
    )
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
        "--node-rank",
        type=int,
        default=int(os.environ.get("NODE_RANK", 0)),
    )
    parser.add_argument("--extraction-python", type=Path)
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.stage == "extract":
            if args.assignment is None or args.output_root is None:
                raise ValueError("extract stage requires --assignment and --output-root")
            return extract_stage(args)
        if args.stage == "infer":
            if args.assignment is None or args.output_root is None:
                raise ValueError("infer stage requires --assignment and --output-root")
            return inference_stage(args)
        if args.output_root is None:
            raise ValueError("--output-root is required")
        return orchestrate(args)
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
