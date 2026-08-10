import unittest
from pathlib import Path

from tools.batch_compose_step_gifs import parse_indices
from tools.compose_step_gif import (
    evenly_spaced_indices,
    natural_key,
    temporal_indices,
    video_label,
)


class ComposeStepGifTests(unittest.TestCase):
    def test_evenly_spaced_indices_include_endpoints(self):
        self.assertEqual(evenly_spaced_indices(11, 5), [0, 2, 5, 8, 10])
        self.assertEqual(evenly_spaced_indices(3, 5), [0, 1, 2])
        self.assertEqual(evenly_spaced_indices(8, 1), [0])

    def test_natural_step_order(self):
        paths = [Path("step_10.mp4"), Path("step_2.mp4"), Path("step_1.mp4")]
        self.assertEqual(
            [path.name for path in sorted(paths, key=natural_key)],
            ["step_1.mp4", "step_2.mp4", "step_10.mp4"],
        )

    def test_temporal_resampling_is_bounded(self):
        self.assertEqual(temporal_indices(4, 4, 2, 4), [0, 2, 3, 3])

    def test_batch_indices_and_step_label(self):
        self.assertEqual(parse_indices("37, 11,8-10,11"), [37, 11, 8, 9, 10])
        self.assertEqual(video_label(Path("step_009/video.mp4")), "Step 9")


if __name__ == "__main__":
    unittest.main()
