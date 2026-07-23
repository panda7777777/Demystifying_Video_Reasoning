from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts._visualization import (
    GenerationConfig,
    parse_visualization_steps,
)
from scripts.custom.visualize import (
    DEFAULTS,
    list_samples,
    load_sample,
    prepare_image,
    resolve_config,
)
from scripts.language_table.download import destination_path
from scripts.language_table.results import evenly_spaced_indices
from scripts.language_table.visualize import (
    contiguous_ranges,
    ensure_run_manifest,
    orchestrate,
    parse_indices,
    read_episode_count,
    resolve_builder_dir,
    shard_indices,
)


class GenerationConfigTests(unittest.TestCase):
    def test_validates_step_limit(self):
        GenerationConfig(max_denoising_steps=1).validate()
        GenerationConfig(max_denoising_steps=30).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(max_denoising_steps=0).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(max_denoising_steps=31).validate()
        with self.assertRaises(ValueError):
            GenerationConfig(model="ltx2.3", max_denoising_steps=1).validate()

    def test_parses_visualization_steps(self):
        self.assertIsNone(parse_visualization_steps("all"))
        self.assertEqual(parse_visualization_steps("0-2,5"), {0, 1, 2, 5})
        with self.assertRaises(ValueError):
            parse_visualization_steps("3-1")


class LanguageTableSelectionTests(unittest.TestCase):
    def test_download_destination_avoids_primary_dataset_nesting(self):
        root = "/data/language-table"
        self.assertEqual(
            destination_path(
                "robotics/language_table/0.0.1/data.tfrecord",
                "language_table",
                root,
            ),
            "/data/language-table/0.0.1/data.tfrecord",
        )
        self.assertEqual(
            destination_path(
                "robotics/language_table_sim/0.0.1/data.tfrecord",
                "language_table_sim",
                root,
            ),
            "/data/language-table/language_table_sim/0.0.1/data.tfrecord",
        )

    def test_parse_indices(self):
        self.assertEqual(parse_indices("0:5:2,3,-1", 8), [0, 2, 3, 4, 7])
        self.assertEqual(parse_indices("-3:", 5), [2, 3, 4])
        self.assertEqual(parse_indices("all", 3), [0, 1, 2])
        with self.assertRaises(ValueError):
            parse_indices("10", 3)
        with self.assertRaises(ValueError):
            parse_indices("::0", 3)

    def test_contiguous_ranges(self):
        self.assertEqual(
            contiguous_ranges([0, 1, 2, 7, 8]), [(0, 3), (7, 9)]
        )
        self.assertEqual(contiguous_ranges([]), [])

    def test_sharding_is_complete_and_deterministic(self):
        indices = list(range(17))
        nodes = [
            shard_indices(indices, node_rank=rank, num_nodes=2, gpus=["0", "1"])
            for rank in range(2)
        ]
        assigned = [
            index
            for node in nodes
            for values in node.values()
            for index in values
        ]
        self.assertEqual(sorted(assigned), indices)
        self.assertEqual(len(assigned), len(set(assigned)))
        self.assertEqual(nodes[0]["0"], indices[0::4])
        with self.assertRaises(ValueError):
            shard_indices(indices, node_rank=2, num_nodes=2, gpus=["0"])
        with self.assertRaises(ValueError):
            shard_indices(indices, node_rank=0, num_nodes=1, gpus=[])

    def test_builder_resolution_and_episode_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder = root / "language_table" / "0.0.1"
            builder.mkdir(parents=True)
            (builder / "dataset_info.json").write_text(
                json.dumps(
                    {
                        "splits": [
                            {"name": "train", "shardLengths": ["2", "3"]}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_builder_dir(root), builder)
            self.assertEqual(read_episode_count(builder, "train"), 5)

            second = root / "sim" / "0.0.1"
            second.mkdir(parents=True)
            (second / "dataset_info.json").write_text(
                '{"splits": []}', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                resolve_builder_dir(root)

    def test_manifest_rejects_configuration_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.json"
            ensure_run_manifest(path, {"model": "wan2.2"})
            ensure_run_manifest(path, {"model": "wan2.2"})
            with self.assertRaises(ValueError):
                ensure_run_manifest(path, {"model": "wan2.1"})
            ensure_run_manifest(path, {"model": "wan2.1"}, overwrite=True)
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"model": "wan2.1"},
            )

    def test_orchestrator_dry_run_does_not_write_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            builder = root / "dataset"
            builder.mkdir()
            (builder / "dataset_info.json").write_text(
                '{"splits": [{"name": "train", "shardLengths": ["4"]}]}',
                encoding="utf-8",
            )
            output = root / "output"
            args = argparse.Namespace(
                data_dir=builder,
                output_root=output,
                split="train",
                indices="0:4",
                gpus="0,1",
                node_rank=0,
                num_nodes=1,
                model="wan2.2",
                num_frames=49,
                num_inference_steps=30,
                max_denoising_steps=10,
                seed=1,
                fps=16,
                negative_prompt=None,
                dry_run=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(orchestrate(args), 0)
            self.assertFalse(output.exists())


class CustomDatasetTests(unittest.TestCase):
    def test_discovers_and_loads_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            sample.mkdir()
            Image.new("RGBA", (33, 17), (255, 0, 0, 128)).save(
                sample / "input.png"
            )
            (sample / "prompt.txt").write_text("move left\n", encoding="utf-8")
            self.assertEqual(list_samples(root), ["sample"])
            image, prompt, metadata = load_sample(root, "sample")
            self.assertEqual(image, sample / "input.png")
            self.assertEqual(prompt, "move left")
            self.assertEqual(metadata, {})

    def test_rejects_unknown_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            sample.mkdir()
            Image.new("RGB", (16, 16)).save(sample / "input.png")
            (sample / "prompt.txt").write_text("prompt", encoding="utf-8")
            (sample / "metadata.json").write_text(
                '{"unknown": true}', encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_sample(root, "sample")

    def test_prepares_rgba_image_and_aligns_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = root / "source.png", root / "out" / "input.png"
            Image.new("RGBA", (101, 55), (0, 0, 0, 0)).save(source)
            size = prepare_image(source, destination, max_size=80)
            self.assertEqual(size, (80, 32))
            with Image.open(destination) as prepared:
                self.assertEqual(prepared.mode, "RGB")

    def test_cli_values_override_metadata_and_defaults(self):
        values = {key: None for key in DEFAULTS}
        values["seed"] = 9
        args = argparse.Namespace(**values)
        resolved = resolve_config(args, {"seed": 3, "fps": 12})
        self.assertEqual(resolved["seed"], 9)
        self.assertEqual(resolved["fps"], 12)
        self.assertEqual(resolved["model"], DEFAULTS["model"])


class ResultUtilityTests(unittest.TestCase):
    def test_even_frame_sampling(self):
        self.assertEqual(evenly_spaced_indices(0, 3), [])
        self.assertEqual(evenly_spaced_indices(5, 1), [0])
        self.assertEqual(evenly_spaced_indices(5, 3), [0, 2, 4])
        self.assertEqual(evenly_spaced_indices(3, 0), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
