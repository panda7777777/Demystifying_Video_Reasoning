"""Dataset adapters used by the single inference entry point."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from scripts._batch_visualization import parse_indices


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image: str
    prompt: str
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Sample:
        return cls(**value)


def _custom(data_dir: Path, selection: str) -> list[Sample]:
    names = ("input.png", "input.jpg", "input.jpeg", "input.webp")
    samples = []
    for directory in sorted(data_dir.iterdir()):
        image = next((directory / name for name in names if (directory / name).is_file()), None)
        prompt_file = directory / "prompt.txt"
        if directory.is_dir() and image and prompt_file.is_file():
            prompt = prompt_file.read_text(encoding="utf-8").strip()
            if prompt:
                samples.append(Sample(directory.name, str(image.resolve()), prompt, {"directory": str(directory.resolve())}))
    if ":" in selection and selection != ":":
        positions = set(parse_indices(selection, len(samples)))
        samples = [sample for position, sample in enumerate(samples) if position in positions]
    elif selection.lower() not in {"all", ":"}:
        wanted = {item.strip() for item in selection.split(",") if item.strip()}
        unknown = wanted - {sample.sample_id for sample in samples}
        if unknown:
            raise ValueError(f"Unknown custom samples: {', '.join(sorted(unknown))}")
        samples = [sample for sample in samples if sample.sample_id in wanted]
    return samples


def _rmbench(data_dir: Path, selection: str, split: str) -> list[Sample]:
    pattern = re.compile(r"episode(\d+)\.mp4$")
    task_spec, _, episode_spec = selection.partition(":")
    selected_tasks = None if task_spec.lower() in {"", "all"} else set(task_spec.split(","))
    samples = []
    for task_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if selected_tasks is not None and task_dir.name not in selected_tasks:
            continue
        setting = task_dir / "demo_clean"
        videos = []
        for video in sorted((setting / "video").glob("episode*.mp4")):
            match = pattern.fullmatch(video.name)
            if match:
                videos.append((int(match.group(1)), video))
        if episode_spec and episode_spec.lower() not in {"all", ":"}:
            positions = set(parse_indices(episode_spec, len(videos)))
            videos = [value for position, value in enumerate(videos) if position in positions]
        for episode, video in videos:
            instructions = setting / "instructions" / f"episode{episode}.json"
            payload = json.loads(instructions.read_text(encoding="utf-8"))
            values = payload.get(split)
            if not isinstance(values, list) or len(values) != 1 or not values[0].strip():
                raise ValueError(f"Expected one {split!r} instruction in {instructions}")
            samples.append(Sample(
                f"{task_dir.name}__episode_{episode:06d}",
                str(video.resolve()),
                values[0].strip(),
                {"task": task_dir.name, "episode": episode, "video": str(video.resolve()), "instructions": str(instructions.resolve())},
            ))
    if selected_tasks is not None:
        found = {sample.source["task"] for sample in samples}
        if selected_tasks - found:
            raise ValueError(f"Unknown or empty RMBench tasks: {', '.join(sorted(selected_tasks - found))}")
    return samples


def _language_table(data_dir: Path, selection: str, split: str) -> list[Sample]:
    try:
        import tensorflow_datasets as tfds
        from tqdm.auto import tqdm
    except ModuleNotFoundError as exc:
        if exc.name == "tensorflow_datasets":
            raise RuntimeError(
                "Language-Table support is not installed. "
                "Install it with: python -m pip install -e '.[language-table]'"
            ) from exc
        raise

    candidates = [data_dir] + sorted(data_dir.glob("*/dataset_info.json")) + sorted(data_dir.glob("*/*/dataset_info.json"))
    builders = [path if path.is_dir() else path.parent for path in candidates if (path if path.is_dir() else path.parent).joinpath("dataset_info.json").is_file()]
    builders = list(dict.fromkeys(path.resolve() for path in builders))
    if len(builders) != 1:
        raise ValueError(f"Expected exactly one TFDS builder under {data_dir}; found {len(builders)}")
    builder = tfds.builder_from_directory(str(builders[0]))
    total = int(builder.info.splits[split].num_examples)
    indices = parse_indices(selection, total)
    wanted = set(indices)
    remaining = set(wanted)
    samples = []
    dataset = builder.as_dataset(split=split, shuffle_files=False)
    scan_total = indices[-1] + 1
    with tqdm(total=scan_total, desc=f"Scanning Language-Table ({split})", unit="episode", dynamic_ncols=True) as progress:
        for index, episode in enumerate(dataset):
            progress.update()
            if index not in wanted:
                continue
            first = next(iter(episode["steps"]))
            rgb = first["observation"]["rgb"].numpy()
            raw = first["observation"]["instruction"].numpy()
            if isinstance(raw, bytes):
                prompt = raw.decode("utf-8").strip("\x00")
            else:
                prompt = bytes(int(x) for x in raw.tolist() if 0 < int(x) < 256).decode("utf-8")
            # TFDS frames have no durable standalone path; extraction writes this marker.
            samples.append(Sample(f"{index:06d}", "", prompt, {"builder": str(builders[0]), "split": split, "index": index, "rgb": rgb.tolist()}))
            remaining.remove(index)
            if not remaining:
                break
    return samples


def discover(dataset: str, data_dir: Path, selection: str, split: str) -> list[Sample]:
    if not data_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {data_dir}")
    if dataset == "custom":
        samples = _custom(data_dir, selection)
    elif dataset == "rmbench":
        samples = _rmbench(data_dir, selection, split)
    elif dataset == "language-table":
        samples = _language_table(data_dir, selection, split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if not samples:
        raise ValueError("No samples selected")
    return samples


def extract_image(sample: Sample) -> Image.Image:
    if sample.image:
        path = Path(sample.image)
        if path.suffix.lower() == ".mp4":
            import imageio.v2 as imageio

            reader = imageio.get_reader(str(path))
            try:
                return Image.fromarray(reader.get_data(0)).convert("RGB")
            finally:
                reader.close()
        return Image.open(path).convert("RGB")
    return Image.fromarray(__import__("numpy").asarray(sample.source["rgb"], dtype="uint8")).convert("RGB")


def prepare_image(sample: Sample, destination: Path, max_size: int) -> tuple[int, int]:
    image = extract_image(sample)
    scale = min(1.0, max_size / max(image.size))
    width = max(16, round(image.width * scale) // 16 * 16)
    height = max(16, round(image.height * scale) // 16 * 16)
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return image.size
