#!/usr/bin/env python3
"""Robust shared-filesystem coordinator for multi-node custom generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts._datasets import discover
from scripts._visualization import atomic_write_json
from scripts.check_custom_data import discover_custom_roots

def task_name(data_dir: Path) -> str:
    name = data_dir.name
    for prefix in ("T", "G-", "O-"):
        if name.startswith(prefix):
            value = name.split("_", 1)[0]
            if value[1:].isdigit() or value.startswith(("G-", "O-")):
                return value
    return name


def atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def complete(run_dir: Path) -> bool:
    try:
        manifest = read_json(run_dir / "run.json")
        samples = manifest["samples"]
        generation = manifest["generation"]
        for sample_id in samples:
            metadata = read_json(run_dir / "samples" / sample_id / "output" / "metadata.json")
            if metadata.get("sample_id") != sample_id or metadata.get("generation") != generation:
                return False
        return True
    except (KeyError, OSError, TypeError, json.JSONDecodeError):
        return False


def pending_ids(run_dir: Path | None, data_dir: Path, selection: str) -> list[str]:
    if run_dir is not None and (run_dir / "run.json").is_file():
        manifest = read_json(run_dir / "run.json")
        samples = manifest.get("samples", [])
        return [sample_id for sample_id in samples if not complete_sample(run_dir, sample_id, manifest.get("generation"))]
    return [sample.sample_id for sample in discover("custom", data_dir, selection, "seen")]


def complete_sample(run_dir: Path, sample_id: str, generation: dict | None) -> bool:
    try:
        metadata = read_json(run_dir / "samples" / sample_id / "output" / "metadata.json")
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get("sample_id") == sample_id and metadata.get("generation") == generation


def newest_resume(output_dir: Path, task: str, dates: list[str]) -> Path | None:
    candidates = []
    for date in dates:
        candidates.extend(output_dir.glob(f"{date}_*_{task}_wan22_custom"))
    candidates = [path for path in candidates if (path / "run.json").is_file()]
    return max(candidates, key=lambda path: path.name) if candidates else None


def session_id(args: argparse.Namespace) -> str:
    value = args.session_id or os.environ.get("BARRIER_ID") or os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID")
    if args.num_nodes > 1 and not value:
        raise ValueError("multi-node runs require a shared --session-id or BARRIER_ID/job id")
    return value or "single-node"


class Coordinator:
    def __init__(self, args: argparse.Namespace, sid: str):
        self.args = args
        self.sid = sid
        self.root = args.output_dir.resolve() / ".coordination" / sid
        self.root.mkdir(parents=True, exist_ok=True)

    def task_root(self, task: str) -> Path:
        path = self.root / "tasks" / task
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plan(self, task: str, run_dir: Path | None, pending: list[str], run_name: str) -> dict:
        payload = {"session_id": self.sid, "task": task, "run_dir": str(run_dir) if run_dir else None,
                   "run_name": run_name, "pending": pending, "num_nodes": self.args.num_nodes}
        payload["plan_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.task_root(task) / "plan.json"
        if self.args.node_rank == 0:
            atomic_write_json(path, payload)
        deadline = time.monotonic() + self.args.timeout
        while not path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for plan: {path}")
            time.sleep(2)
        saved = read_json(path)
        if saved.get("session_id") != self.sid or saved.get("num_nodes") != self.args.num_nodes:
            raise RuntimeError(f"plan/session mismatch for {task}")
        expected_hash = saved.get("plan_hash")
        unsigned = {key: value for key, value in saved.items() if key != "plan_hash"}
        actual_hash = hashlib.sha256(json.dumps(unsigned, sort_keys=True).encode()).hexdigest()
        if expected_hash != actual_hash:
            raise RuntimeError(f"plan hash mismatch for {task}")
        return saved

    def barrier(self, task: str, phase: str, status: str) -> None:
        directory = self.task_root(task) / phase
        directory.mkdir(parents=True, exist_ok=True)
        marker = directory / f"node_{self.args.node_rank}.json"
        atomic_write_json(marker, {"status": status, "rank": self.args.node_rank, "time": time.time()})
        deadline = time.monotonic() + self.args.timeout
        expected = [directory / f"node_{rank}.json" for rank in range(self.args.num_nodes)]
        while not all(path.is_file() for path in expected):
            if time.monotonic() >= deadline:
                missing = [str(path.name) for path in expected if not path.is_file()]
                raise TimeoutError(f"{task}/{phase} timed out; missing {missing}")
            time.sleep(2)
        states = [read_json(path).get("status") for path in expected]
        expected_status = "ready" if phase == "start" else "done"
        if any(state != expected_status for state in states):
            raise RuntimeError(f"{task}/{phase} failed: {states}")


def heartbeat(coordinator: Coordinator, task: str, stop: threading.Event) -> None:
    path = coordinator.task_root(task) / "heartbeat" / f"node_{coordinator.args.node_rank}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        atomic_write_json(path, {"rank": coordinator.args.node_rank, "task": task, "time": time.time()})
        stop.wait(coordinator.args.heartbeat)


def run_task(args: argparse.Namespace, coordinator: Coordinator, data_dir: Path, task: str) -> bool:
    resume = newest_resume(args.output_dir.resolve(), task, args.resume_dates)
    if resume and complete(resume):
        print(f"[skip] {task}: resume directory is already complete", flush=True)
        return True
    run_name = resume.name if resume else datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d_%H%M") + f"_{task}_wan22_custom"
    planned_dir = resume or (args.output_dir.resolve() / run_name)
    plan = coordinator.plan(task, planned_dir, pending_ids(resume, data_dir, args.selection), run_name)
    run_name = plan["run_name"]
    run_dir = Path(plan["run_dir"])
    coordinator.barrier(task, "start", "ready")
    command = [str(args.python), str(REPO_ROOT / "scripts" / "run.py"), "--model", args.model,
               "--dataset", "custom", "--data-dir", str(data_dir), "--task-name", task,
               "--output-dir", str(args.output_dir), "--selection", args.selection,
               "--run-name", run_name,
               "--num-nodes", str(args.num_nodes), "--node-rank", str(args.node_rank),
               "--gpus", args.gpus, "--batch-size", str(args.batch_size), "--num-frames", str(args.num_frames),
               "--num-inference-steps", str(args.num_inference_steps), "--seed", str(args.seed),
               "--fps", str(args.fps), "--max-size", str(args.max_size), "--overview-columns", str(args.overview_columns),
               "--python", str(args.python), "--max-denoising-steps", str(args.max_denoising_steps),
               "--lora-alpha", str(args.lora_alpha), "--high-noise-lora-path", args.high_noise_lora_path]
    if resume:
        command += ["--resume-dir", str(resume)]
    stop = threading.Event()
    thread = threading.Thread(target=heartbeat, args=(coordinator, task, stop), daemon=True)
    thread.start()
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, timeout=args.timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        print(f"[failed] {task}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        result = subprocess.CompletedProcess(command, 1)
    finally:
        stop.set()
        thread.join(timeout=2)
    coordinator.barrier(task, "done", "done" if result.returncode == 0 else "failed")
    return result.returncode == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--resume-dates", default="20260820")
    parser.add_argument("--session-id")
    parser.add_argument("--num-nodes", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("RANK", os.environ.get("NODE_RANK", "0"))))
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--selection", default="all")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--max-denoising-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--max-size", type=int, default=832)
    parser.add_argument("--overview-columns", type=int, default=6)
    parser.add_argument("--lora-alpha", type=float, default=1.0)
    parser.add_argument("--high-noise-lora-path", required=True)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--heartbeat", type=float, default=10)
    args = parser.parse_args()
    args.resume_dates = [value.strip() for value in args.resume_dates.split(",") if value.strip()]
    if args.num_nodes <= 0 or not 0 <= args.node_rank < args.num_nodes:
        raise ValueError("invalid node rank/world size")
    return args


def main() -> int:
    try:
        args = parse_args()
        sid = session_id(args)
        coordinator = Coordinator(args, sid)
        roots = discover_custom_roots(args.data_root)
        seen = set()
        for data_dir in roots:
            task = task_name(data_dir)
            if task in seen:
                raise ValueError(f"duplicate task name: {task}")
            seen.add(task)
            if not run_task(args, coordinator, data_dir, task):
                return 1
        return 0
    except (OSError, ValueError, RuntimeError, TimeoutError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
