import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from tools.results_dashboard import (
    DashboardServer,
    ResultIndex,
    parse_data_indices,
    positive_page_size,
)


class ResultDashboardTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "demo_run"
        (self.root / "samples").mkdir(parents=True)
        (self.root / "run.json").write_text(
            json.dumps({"samples": ["000002", "000001", "missing"]}),
            encoding="utf-8",
        )
        self._sample("000001", "first prompt", [0, 2])
        self._sample("000002", "second prompt", [0, 1, 2])
        self._sample("000003", None, [1])
        self.index = ResultIndex(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _sample(self, sample_id, prompt, steps):
        base = self.root / "samples" / sample_id
        (base / "input").mkdir(parents=True)
        if prompt is not None:
            (base / "input" / "prompt.txt").write_text(prompt, encoding="utf-8")
        (base / "input" / "initial_frame.png").write_bytes(b"fake-png")
        for step in steps:
            step_dir = base / "output" / "steps" / f"step_{step:03d}"
            step_dir.mkdir(parents=True)
            step_dir.joinpath("video.mp4").write_bytes(b"0123456789")

    def test_index_uses_manifest_order_then_discovered_samples(self):
        self.assertEqual(
            [sample.task_id for sample in self.index.samples],
            ["000002", "000001", "000003"],
        )
        self.assertEqual(self.index.step_names, ["step_000", "step_001", "step_002"])

    def test_page_serializes_current_slice_and_missing_media(self):
        page = self.index.page(2, 2)
        self.assertEqual(page["page"], 2)
        self.assertEqual(page["total_pages"], 2)
        self.assertEqual(page["items"][0]["task_id"], "000003")
        self.assertEqual(page["items"][0]["prompt"], "")
        self.assertNotIn("step_000", page["items"][0]["step_videos"])

    def test_page_clamps_high_page_and_rejects_invalid_values(self):
        self.assertEqual(self.index.page(99, 2)["page"], 2)
        for value in (0, 51):
            with self.assertRaises(ValueError):
                self.index.page(1, value)
        with self.assertRaises(ValueError):
            self.index.page(0, 5)

    def test_index_can_select_samples_by_zero_based_indices(self):
        selected = ResultIndex(self.root, data_indices=[2, 0])
        self.assertEqual(
            [sample.task_id for sample in selected.samples],
            ["000003", "000002"],
        )
        self.assertEqual(selected.page(1, 5)["total"], 2)

    def test_index_rejects_invalid_data_indices(self):
        for indices in ([], [-1], [3], [0, 0]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                ResultIndex(self.root, data_indices=indices)

    def test_media_resolution_rejects_escape_and_missing_file(self):
        with self.assertRaises(ValueError):
            self.index.resolve_media("../secret.txt")
        with self.assertRaises(FileNotFoundError):
            self.index.resolve_media("samples/000001/not-there.mp4")

    def test_page_size_parser(self):
        self.assertEqual(positive_page_size("5"), 5)
        for value in ("0", "51", "nope"):
            with self.assertRaises(Exception):
                positive_page_size(value)

    def test_data_indices_parser(self):
        self.assertIsNone(parse_data_indices(None))
        self.assertEqual(parse_data_indices(["0,2", "1"]), [0, 2, 1])
        for values in ([""], ["0,"], ["-1"], ["one"], ["1", "1"]):
            with self.subTest(values=values), self.assertRaises(Exception):
                parse_data_indices(values)

    def test_http_endpoints_and_byte_ranges(self):
        server = DashboardServer(("127.0.0.1", 0), self.index, 2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        open_direct = build_opener(ProxyHandler({})).open
        try:
            with open_direct(base + "/", timeout=3) as response:
                page_html = response.read()
                self.assertIn(b"Step Observatory", page_html)
                self.assertIn(b'id="page-number"', page_html)
            with open_direct(base + "/api/manifest", timeout=3) as response:
                manifest = json.load(response)
                self.assertEqual(manifest["total"], 3)
                self.assertEqual(manifest["default_page_size"], 2)
            with open_direct(base + "/api/samples?page=1&page_size=2", timeout=3) as response:
                payload = json.load(response)
                self.assertEqual(len(payload["items"]), 2)
            media = "/media/samples/000001/output/steps/step_000/video.mp4"
            request = Request(base + media, headers={"Range": "bytes=2-5"})
            with open_direct(request, timeout=3) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
                self.assertEqual(response.read(), b"2345")
            with self.assertRaises(HTTPError) as context:
                open_direct(base + "/api/samples?page=0&page_size=2", timeout=3)
            self.assertEqual(context.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
