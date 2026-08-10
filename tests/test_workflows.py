from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts._batch_visualization import parse_indices, shard_indices
from scripts._datasets import Sample, discover, prepare_image
from scripts._results import render_overview
from scripts.run import completed_sample_ids, persisted_samples, validate_resume_manifest
from scripts._visualization import (
    GenerationConfig,
    model_family,
    parse_visualization_steps,
    validate_model_source,
)
from scripts.language_table.download import destination_path


class SharedWorkflowTests(unittest.TestCase):
    def test_generation_validation(self):
        GenerationConfig(max_denoising_steps=10).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(
                model="DiffSynth-Studio/LTX-2.3-Repackage",
                max_denoising_steps=1,
            ).validate()
        self.assertEqual(parse_visualization_steps("0-2,5"), {0, 1, 2, 5})

    def test_model_source_validation(self):
        self.assertEqual(
            validate_model_source("Wan-AI/Wan2.2-I2V-A14B"),
            "Wan-AI/Wan2.2-I2V-A14B",
        )
        self.assertEqual(model_family("Wan-AI/Wan2.1-I2V-14B-720P"), "wan2.1")
        self.assertEqual(validate_model_source("lvp"), "large-video-planner")
        self.assertEqual(model_family("large-video-planner"), "lvp")
        with self.assertRaises(ValueError):
            validate_model_source("wan2.2")
        with self.assertRaises(ValueError):
            validate_model_source("./models/Wan2.2-I2V-A14B")

    def test_selection_and_sharding(self):
        self.assertEqual(parse_indices("0:5:2,3,-1", 8), [0, 2, 3, 4, 7])
        indices = list(range(17))
        nodes = [shard_indices(indices, node_rank=rank, num_nodes=2, gpus=["0", "1"]) for rank in range(2)]
        assigned = [index for node in nodes for values in node.values() for index in values]
        self.assertEqual(sorted(assigned), indices)
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_custom_discovery_and_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            sample.mkdir()
            Image.new("RGBA", (101, 55), (255, 0, 0, 128)).save(sample / "input.png")
            (sample / "prompt.txt").write_text("move left\n", encoding="utf-8")
            found = discover("custom", root, "all", "seen")
            self.assertEqual([item.sample_id for item in found], ["sample"])
            output = root / "normalized.png"
            self.assertEqual(prepare_image(found[0], output, 80), (80, 32))
            with Image.open(output) as image:
                self.assertEqual(image.mode, "RGB")

    def test_rmbench_discovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task in ("alpha", "beta"):
                setting = root / task / "demo_clean"
                (setting / "video").mkdir(parents=True)
                (setting / "instructions").mkdir()
                for episode in range(3):
                    (setting / "video" / f"episode{episode}.mp4").write_bytes(b"video")
                    (setting / "instructions" / f"episode{episode}.json").write_text(json.dumps({"seen": ["move"], "unseen": ["go"]}))
            found = discover("rmbench", root, "alpha:0:2", "seen")
            self.assertEqual([item.sample_id for item in found], ["alpha__episode_000000", "alpha__episode_000001"])

    def test_publication_overview(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "overview.png"
            frames = [Image.new("RGB", (64, 40), (index * 20, 30, 60)) for index in range(5)]
            render_overview([(0, 2, frames), (1, 2, frames)], output, prompt="move the block", columns=4)
            with Image.open(output) as image:
                self.assertGreater(image.width, image.height)
                self.assertEqual(image.mode, "RGB")

    def test_language_table_utilities(self):
        self.assertEqual(destination_path("robotics/language_table/0.0.1/a", "language_table", "/data"), "/data/0.0.1/a")

    def test_resume_allows_only_parallel_changes(self):
        manifest = {
            "name": "run", "dataset": "custom", "data_dir": "/data",
            "selection": "all", "split": "seen", "generation": {"seed": 1},
            "adapters": {}, "processing": {"max_size": 832},
            "parallel": {"num_nodes": 1}, "samples": ["sample"],
        }
        changed_parallel = dict(manifest, parallel={"num_nodes": 2})
        validate_resume_manifest(manifest, changed_parallel, Path("/run"))
        changed_seed = dict(manifest, generation={"seed": 2})
        with self.assertRaisesRegex(ValueError, "only parallel parameters may change"):
            validate_resume_manifest(manifest, changed_seed, Path("/run"))

    def test_resume_completed_sample_detection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = Sample("sample", "", "move", {})
            config = GenerationConfig()
            output = root / "samples" / "sample" / "output"
            output.mkdir(parents=True)
            (output / "metadata.json").write_text(json.dumps({
                "sample_id": "sample", "source": {},
                "generation": config.to_dict(),
            }))
            self.assertEqual(completed_sample_ids(root, [sample], config), {"sample"})

    def test_resume_loads_persisted_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "samples" / "sample" / "input"
            input_dir.mkdir(parents=True)
            Image.new("RGB", (16, 16)).save(input_dir / "initial_frame.png")
            (input_dir / "prompt.txt").write_text("move\n")
            (input_dir / "metadata.json").write_text(json.dumps({"source": {"index": 1}, "size": [16, 16]}))
            samples = persisted_samples(root, ["sample"])
            self.assertIsNotNone(samples)
            self.assertEqual(samples[0].prompt, "move")
            self.assertEqual(samples[0].source, {"index": 1})


if __name__ == "__main__":
    unittest.main()
