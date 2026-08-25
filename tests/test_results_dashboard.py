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

    def test_page_filters_by_cos_and_success_status(self):
        self.index.save_annotation("000001", "Multi-Path", success=True)
        self.index.save_annotation("000002", "Memory", success=False)

        self.assertEqual(self.index.page(1, 5, cos_type="annotated")["total"], 2)
        self.assertEqual(self.index.page(1, 5, cos_type="unannotated")["total"], 1)
        self.assertEqual(self.index.page(1, 5, cos_type="Multi-Path")["total"], 1)
        self.assertEqual(self.index.page(1, 5, success="annotated")["total"], 2)
        self.assertEqual(self.index.page(1, 5, success="unannotated")["total"], 1)
        self.assertEqual(self.index.page(1, 5, success="true")["total"], 1)
        self.assertEqual(
            self.index.page(1, 5, cos_type="Memory", success="false")["total"],
            1,
        )
        with self.assertRaises(ValueError):
            self.index.page(1, 5, cos_type="invalid")
        with self.assertRaises(ValueError):
            self.index.page(1, 5, success="invalid")

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

    def test_annotation_round_trip_and_null_becomes_unknown(self):
        self.index.save_annotation("000001", "Multi-Path")
        annotation_path = self.root / "samples" / "000001" / "annotation.json"
        self.assertEqual(json.loads(annotation_path.read_text())["cos_type"], "Multi-Path")
        item = self.index.page(1, 2)["items"][1]
        self.assertEqual(item["cos_type"], "Multi-Path")
        self.assertIsNone(item["success"])

        self.index.save_annotation("000001", success=True)
        annotation = json.loads(annotation_path.read_text())
        self.assertEqual(annotation, {"cos_type": "Multi-Path", "success": True})

        self.index.save_annotation("000001", None)
        annotation = json.loads(annotation_path.read_text())
        self.assertEqual(annotation, {"cos_type": "Unknown", "success": True})
        item = self.index.page(1, 2)["items"][1]
        self.assertEqual(item["cos_type"], "Unknown")
        self.assertTrue(item["success"])

    def test_missing_annotation_is_unknown(self):
        item = self.index.page(1, 2)["items"][0]
        self.assertEqual(item["cos_type"], "Unknown")
        self.assertIsNone(item["success"])

    def test_annotation_rejects_invalid_type_and_unknown_sample(self):
        with self.assertRaises(ValueError):
            self.index.save_annotation("000001", "Other")
        with self.assertRaises(ValueError):
            self.index.save_annotation("000001", success="yes")
        with self.assertRaises(ValueError):
            self.index.save_annotation("missing", "Unknown")

    def test_clear_annotations_includes_samples_outside_selection(self):
        selected = ResultIndex(self.root, data_indices=[0])
        self.index.save_annotation("000001", "Memory")
        self.index.save_annotation("000003", "Unknown")
        self.assertEqual(selected.clear_annotations(), 2)
        for task_id in ("000001", "000002", "000003"):
            self.assertFalse((self.root / "samples" / task_id / "annotation.json").exists())

    def test_media_resolution_rejects_escape_and_missing_file(self):
        with self.assertRaises(ValueError):
            self.index.resolve_media("../secret.txt")
        with self.assertRaises(FileNotFoundError):
            self.index.resolve_media("samples/000001/not-there.mp4")

    def test_media_urls_change_when_file_content_changes(self):
        frame = self.root / "samples" / "000001" / "input" / "initial_frame.png"
        first_url = self.index.media_url(frame)
        self.assertIn("?v=", first_url)

        frame.write_bytes(b"different-fake-png")

        second_url = self.index.media_url(frame)
        self.assertNotEqual(first_url, second_url)
        self.assertEqual(first_url.split("?", 1)[0], second_url.split("?", 1)[0])

    def test_multiple_result_dirs_merge_samples_and_route_by_run(self):
        second_root = Path(self.temp_dir.name) / "second_run"
        sample = second_root / "samples" / "000001"
        (sample / "input").mkdir(parents=True)
        (sample / "input" / "prompt.txt").write_text("other run", encoding="utf-8")
        (sample / "input" / "initial_frame.png").write_bytes(b"other-png")
        step = sample / "output" / "steps" / "step_003"
        step.mkdir(parents=True)
        (step / "video.mp4").write_bytes(b"other-video")

        combined = ResultIndex([self.root, second_root])

        self.assertEqual(len(combined.samples), 4)
        self.assertEqual(combined.samples[0].sample_id, "demo_run/000002")
        self.assertEqual(combined.samples[-1].sample_id, "second_run/000001")
        self.assertEqual(
            combined.step_names,
            ["step_000", "step_001", "step_002", "step_003"],
        )
        item = combined.page(1, 10)["items"][-1]
        self.assertEqual(item["run_name"], "second_run")
        self.assertIn("/media/second_run/", item["initial_frame"])
        media_path = item["initial_frame"].split("/media/", 1)[1].split("?", 1)[0]
        self.assertEqual(combined.resolve_media(media_path).read_bytes(), b"other-png")
        run_page = combined.page(1, 10, run_name="second_run")
        self.assertEqual(run_page["total"], 1)
        self.assertEqual(run_page["items"][0]["run_name"], "second_run")
        with self.assertRaises(ValueError):
            combined.page(1, 10, run_name="missing_run")

        combined.save_annotation("second_run/000001", success=True)
        annotation = json.loads((sample / "annotation.json").read_text())
        self.assertTrue(annotation["success"])
        with self.assertRaisesRegex(ValueError, "sample_id"):
            combined.save_annotation("000001", success=False)

    def test_multiple_result_dirs_are_naturally_sorted_by_category(self):
        roots = []
        for run_name in (
            "20260822_0100_T010_wan22_custom",
            "20260820_0100_T002_wan22_custom",
            "20260821_0100_G-19_wan22_custom",
            "20260820_0200_G-3_wan22_custom",
        ):
            root = Path(self.temp_dir.name) / run_name
            (root / "samples").mkdir(parents=True)
            roots.append(root)

        index = ResultIndex(roots)

        self.assertEqual(
            [root.name for root in index.roots],
            [
                "20260820_0200_G-3_wan22_custom",
                "20260821_0100_G-19_wan22_custom",
                "20260820_0100_T002_wan22_custom",
                "20260822_0100_T010_wan22_custom",
            ],
        )

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
                self.assertIn("是否成功".encode(), page_html)
            with open_direct(base + "/api/manifest", timeout=3) as response:
                manifest = json.load(response)
                self.assertEqual(manifest["total"], 3)
                self.assertEqual(manifest["default_page_size"], 2)
                self.assertIn("Self-correction", manifest["cos_types"])
            with open_direct(base + "/api/samples?page=1&page_size=2", timeout=3) as response:
                payload = json.load(response)
                self.assertEqual(len(payload["items"]), 2)
            with open_direct(
                base + "/api/samples?page=1&page_size=2&cos_type=unannotated",
                timeout=3,
            ) as response:
                self.assertEqual(json.load(response)["total"], 3)
            media = "/media/samples/000001/output/steps/step_000/video.mp4"
            request = Request(base + media, headers={"Range": "bytes=2-5"})
            with open_direct(request, timeout=3) as response:
                self.assertEqual(response.status, 206)
                self.assertEqual(response.headers["Content-Range"], "bytes 2-5/10")
                self.assertEqual(response.read(), b"2345")
            with self.assertRaises(HTTPError) as context:
                open_direct(base + "/api/samples?page=0&page_size=2", timeout=3)
            self.assertEqual(context.exception.code, 400)

            annotation_request = Request(
                base + "/api/annotation",
                data=json.dumps(
                    {
                        "task_id": "000002",
                        "cos_type": "Perception Before Action",
                        "success": False,
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with open_direct(annotation_request, timeout=3) as response:
                self.assertTrue(json.load(response)["ok"])
            annotation = json.loads(
                (self.root / "samples" / "000002" / "annotation.json").read_text()
            )
            self.assertEqual(annotation["cos_type"], "Perception Before Action")
            self.assertFalse(annotation["success"])
            with open_direct(
                base + "/api/samples?page=1&page_size=2&success=false", timeout=3
            ) as response:
                filtered = json.load(response)
                self.assertEqual(filtered["total"], 1)
                self.assertEqual(filtered["items"][0]["task_id"], "000002")

            clear_request = Request(base + "/api/annotations", method="DELETE")
            with open_direct(clear_request, timeout=3) as response:
                self.assertEqual(json.load(response)["cleared"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
