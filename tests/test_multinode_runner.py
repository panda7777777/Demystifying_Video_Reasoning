from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.multinode_runner import complete, complete_sample, pending_ids


class MultiNodeRunnerTests(unittest.TestCase):
    def test_complete_requires_every_matching_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            generation = {"seed": 1}
            (run_dir / "samples" / "a" / "output").mkdir(parents=True)
            (run_dir / "samples" / "b" / "output").mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"samples": ["a", "b"], "generation": generation}),
                encoding="utf-8",
            )
            (run_dir / "samples" / "a" / "output" / "metadata.json").write_text(
                json.dumps({"sample_id": "a", "generation": generation}), encoding="utf-8"
            )
            self.assertFalse(complete(run_dir))
            (run_dir / "samples" / "b" / "output" / "metadata.json").write_text(
                json.dumps({"sample_id": "b", "generation": generation}), encoding="utf-8"
            )
            self.assertTrue(complete(run_dir))

    def test_pending_ids_preserve_manifest_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            generation = {"seed": 1}
            (run_dir / "samples" / "b" / "output").mkdir(parents=True)
            (run_dir / "run.json").write_text(
                json.dumps({"samples": ["a", "b"], "generation": generation}),
                encoding="utf-8",
            )
            (run_dir / "samples" / "b" / "output" / "metadata.json").write_text(
                json.dumps({"sample_id": "b", "generation": generation}), encoding="utf-8"
            )
            self.assertEqual(pending_ids(run_dir, Path("unused"), "all"), ["a"])
            self.assertTrue(complete_sample(run_dir, "b", generation))


if __name__ == "__main__":
    unittest.main()
