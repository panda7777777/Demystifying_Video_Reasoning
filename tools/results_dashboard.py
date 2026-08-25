#!/usr/bin/env python3
"""Serve an offline dashboard for step-visualization results."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = REPO_ROOT / "output" / "20260804_1513_wan22_ltable"
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")
STEP_RE = re.compile(r"step_(\d+)$")
RUN_NAME_RE = re.compile(r"^\d{8}_\d{4}_(.+?)(?:_wan22_custom)?$")
ANNOTATION_FILENAME = "annotation.json"
COS_TYPES = (
    "Unknown",
    "Multi-Path",
    "Superposition",
    "Memory",
    "Self-correction",
    "Perception Before Action",
)
_UNSET = object()


@dataclass(frozen=True)
class Sample:
    run_name: str
    task_id: str
    path: Path
    root: Path
    sample_id: str


class ResultIndex:
    """Read a stable, lightweight index spanning one or more runs."""

    def __init__(
        self, result_dir: Path | list[Path], data_indices: list[int] | None = None
    ):
        requested_roots = [result_dir] if isinstance(result_dir, Path) else result_dir
        if not requested_roots:
            raise ValueError("结果目录列表不能为空")
        self.roots = sorted(
            (path.expanduser().resolve() for path in requested_roots),
            key=self._run_sort_key,
        )
        if len(self.roots) != len(set(self.roots)):
            raise ValueError("结果目录不能重复")
        run_names = [root.name for root in self.roots]
        if len(run_names) != len(set(run_names)):
            raise ValueError("多个结果目录不能使用相同的目录名")

        self.root = self.roots[0]  # Backwards-compatible convenience for one-run callers.
        self._roots_by_name: dict[str, Path] = {}
        self.all_samples: list[Sample] = []
        run_data_by_root: dict[Path, dict[str, object]] = {}
        for root in self.roots:
            if not root.is_dir():
                raise ValueError(f"结果目录不存在或不是目录: {root}")
            samples_dir = root / "samples"
            if not samples_dir.is_dir():
                raise ValueError(f"结果目录缺少 samples/: {root}")
            self._roots_by_name[root.name] = root
            run_data = self._read_run_data(root)
            run_data_by_root[root] = run_data
            ordered_ids = self._ordered_sample_ids(samples_dir, run_data)
            self.all_samples.extend(
                self._load_sample(root, samples_dir, task_id) for task_id in ordered_ids
            )

        selected_samples = self.all_samples
        if data_indices is not None:
            selected_samples = self._select_samples(self.all_samples, data_indices)
        self.samples = selected_samples
        self._samples_by_id = {sample.sample_id: sample for sample in self.all_samples}
        self._annotation_cache: dict[str, dict[str, object]] = {}
        self.step_names = self._discover_steps(run_data_by_root)

    @staticmethod
    def _select_samples(samples: list[Sample], data_indices: list[int]) -> list[Sample]:
        if not data_indices:
            raise ValueError("数据索引列表不能为空")
        if len(data_indices) != len(set(data_indices)):
            raise ValueError("数据索引列表不能包含重复值")
        invalid = [index for index in data_indices if index < 0 or index >= len(samples)]
        if invalid:
            valid_range = f"0 到 {len(samples) - 1}" if samples else "空"
            raise ValueError(
                f"数据索引越界: {', '.join(map(str, invalid))}（有效范围: {valid_range}）"
            )
        return [samples[index] for index in data_indices]

    @staticmethod
    def _read_run_data(root: Path) -> dict[str, object]:
        run_path = root / "run.json"
        if not run_path.is_file():
            return {}
        try:
            data = json.loads(run_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _ordered_sample_ids(
        self, samples_dir: Path, run_data: dict[str, object]
    ) -> list[str]:
        discovered = {path.name for path in samples_dir.iterdir() if path.is_dir()}
        ordered: list[str] = []
        run_samples = run_data.get("samples", [])
        if isinstance(run_samples, list):
            ordered.extend(
                sample_id
                for sample_id in run_samples
                if isinstance(sample_id, str) and sample_id in discovered
            )
        seen = set(ordered)
        ordered.extend(sorted(discovered - seen, key=self._natural_key))
        return ordered

    def _load_sample(self, root: Path, samples_dir: Path, task_id: str) -> Sample:
        sample_id = task_id if len(self.roots) == 1 else f"{root.name}/{task_id}"
        return Sample(
            run_name=root.name,
            task_id=task_id,
            path=samples_dir / task_id,
            root=root,
            sample_id=sample_id,
        )

    def _discover_steps(self, run_data_by_root: dict[Path, dict[str, object]]) -> list[str]:
        names: set[str] = set()
        # Runs use one shared step schedule. Checking a small prefix avoids thousands
        # of expensive network-filesystem directory reads while tolerating an
        # incomplete first sample.
        for root in self.roots:
            run_data = run_data_by_root[root]
            generation = run_data.get("generation", {})
            expected = (
                generation.get("max_denoising_steps")
                if isinstance(generation, dict)
                else None
            )
            expected = expected if isinstance(expected, int) and expected > 0 else None
            run_names: set[str] = set()
            run_samples = [sample for sample in self.samples if sample.root == root]
            for sample in run_samples[:20]:
                steps_dir = sample.path / "output" / "steps"
                if not steps_dir.is_dir():
                    continue
                run_names.update(
                    path.name
                    for path in steps_dir.iterdir()
                    if path.is_dir()
                    and STEP_RE.fullmatch(path.name)
                    and (path / "video.mp4").is_file()
                )
                if expected is not None and len(run_names) >= expected:
                    break
            names.update(run_names)
        return sorted(names, key=self._step_key)

    @staticmethod
    def _natural_key(value: str) -> tuple[int, int | str]:
        return (0, int(value)) if value.isdigit() else (1, value)

    @staticmethod
    def _run_category(run_name: str) -> str:
        match = RUN_NAME_RE.fullmatch(run_name)
        return match.group(1) if match else run_name

    @classmethod
    def _run_sort_key(cls, root: Path) -> tuple[tuple[tuple[int, object], ...], str]:
        category = cls._run_category(root.name)
        natural_category = tuple(
            (1, int(part)) if part.isdigit() else (0, part)
            for part in re.split(r"(\d+)", category)
            if part
        )
        return natural_category, root.name

    @staticmethod
    def _step_key(value: str) -> tuple[int, str]:
        match = STEP_RE.fullmatch(value)
        return (int(match.group(1)), value) if match else (10**9, value)

    def page(
        self,
        page: int,
        page_size: int,
        run_name: str | None = None,
        cos_type: str | None = None,
        success: str | None = None,
    ) -> dict[str, object]:
        if page < 1:
            raise ValueError("page 必须大于等于 1")
        if not 1 <= page_size <= 50:
            raise ValueError("page_size 必须在 1 到 50 之间")
        samples = self._filter_samples(run_name, cos_type, success)
        total = len(samples)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, total_pages)
        start = (page - 1) * page_size
        items = [self._serialize_sample(sample) for sample in samples[start : start + page_size]]
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "steps": self.step_names,
        }

    def _filter_samples(
        self,
        run_name: str | None,
        cos_type: str | None,
        success: str | None,
    ) -> list[Sample]:
        if run_name is not None and run_name not in self._roots_by_name:
            raise ValueError(f"无效的 run: {run_name}")
        valid_cos_filters = {"annotated", "unannotated", *COS_TYPES}
        if cos_type is not None and cos_type not in valid_cos_filters:
            raise ValueError(f"无效的 CoS 筛选值: {cos_type}")
        valid_success_filters = {"annotated", "unannotated", "true", "false"}
        if success is not None and success not in valid_success_filters:
            raise ValueError(f"无效的成功状态筛选值: {success}")

        filtered: list[Sample] = []
        for sample in self.samples:
            if run_name is not None and sample.run_name != run_name:
                continue
            if cos_type is None and success is None:
                filtered.append(sample)
                continue
            annotation = self._annotation_for(sample)
            sample_cos = annotation["cos_type"]
            sample_success = annotation["success"]
            if cos_type == "annotated" and sample_cos == "Unknown":
                continue
            if cos_type in {"unannotated", "Unknown"} and sample_cos != "Unknown":
                continue
            if cos_type not in {None, "annotated", "unannotated", "Unknown"}:
                if sample_cos != cos_type:
                    continue
            if success == "annotated" and sample_success is None:
                continue
            if success == "unannotated" and sample_success is not None:
                continue
            if success == "true" and sample_success is not True:
                continue
            if success == "false" and sample_success is not False:
                continue
            filtered.append(sample)
        return filtered

    def _serialize_sample(self, sample: Sample) -> dict[str, object]:
        prompt_path = sample.path / "input" / "prompt.txt"
        try:
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        except OSError:
            prompt = ""
        frame = sample.path / "input" / "initial_frame.png"
        step_videos: dict[str, str] = {}
        for name in self.step_names:
            video = sample.path / "output" / "steps" / name / "video.mp4"
            if video.is_file():
                step_videos[name] = self.media_url(video)
        annotation = self._annotation_for(sample)
        return {
            "sample_id": sample.sample_id,
            "run_name": sample.run_name,
            "task_id": sample.task_id,
            "prompt": prompt,
            "initial_frame": self.media_url(frame) if frame.is_file() else None,
            **annotation,
            "step_videos": step_videos,
        }

    @staticmethod
    def _read_annotation(sample: Sample) -> dict[str, object]:
        try:
            data = json.loads((sample.path / ANNOTATION_FILENAME).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        cos_type = data.get("cos_type") if isinstance(data, dict) else None
        success = data.get("success") if isinstance(data, dict) else None
        return {
            "cos_type": cos_type if cos_type in COS_TYPES else "Unknown",
            "success": success if isinstance(success, bool) else None,
        }

    def _annotation_for(self, sample: Sample) -> dict[str, object]:
        annotation = self._annotation_cache.get(sample.sample_id)
        if annotation is None:
            annotation = self._read_annotation(sample)
            self._annotation_cache[sample.sample_id] = annotation
        return annotation

    def save_annotation(
        self,
        sample_id: str,
        cos_type: str | None | object = _UNSET,
        success: bool | None | object = _UNSET,
    ) -> dict[str, object]:
        sample = self._sample_by_id(sample_id)
        annotation = self._annotation_for(sample)
        if cos_type is not _UNSET:
            cos_type = "Unknown" if cos_type is None else cos_type
            if cos_type not in COS_TYPES:
                raise ValueError(f"无效的 CoS Type: {cos_type}")
            annotation["cos_type"] = cos_type
        if success is not _UNSET:
            if success is not None and not isinstance(success, bool):
                raise ValueError("success 必须是布尔值或 null")
            annotation["success"] = success
        annotation_path = sample.path / ANNOTATION_FILENAME
        temporary_path = sample.path / f".{ANNOTATION_FILENAME}.tmp"
        temporary_path.write_text(
            json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(annotation_path)
        self._annotation_cache[sample.sample_id] = annotation
        return annotation

    def clear_annotations(self) -> int:
        cleared = 0
        for sample in self.all_samples:
            annotation_path = sample.path / ANNOTATION_FILENAME
            try:
                annotation_path.unlink()
                cleared += 1
            except FileNotFoundError:
                pass
            self._annotation_cache[sample.sample_id] = {
                "cos_type": "Unknown",
                "success": None,
            }
        return cleared

    def _sample_by_id(self, sample_id: str) -> Sample:
        sample = self._samples_by_id.get(sample_id)
        if sample is not None:
            return sample
        # Accept a bare task_id when it is unambiguous, preserving the old API.
        matches = [sample for sample in self.all_samples if sample.task_id == sample_id]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"样本 ID 在多个 run 中重复，请使用完整 sample_id: {sample_id}")
        raise ValueError(f"样本不存在: {sample_id}")

    def media_url(self, path: Path) -> str:
        resolved = path.resolve()
        root = next(
            (root for root in self.roots if resolved.is_relative_to(root)),
            None,
        )
        if root is None:
            raise ValueError(f"媒体路径不属于已加载的结果目录: {path}")
        relative = resolved.relative_to(root).as_posix()
        # Media paths repeat across runs (for example,
        # samples/00/input/initial_frame.png).  Include a file-version token so
        # switching the dashboard to another result directory cannot reuse a
        # cached image or video from the previous run.
        stat = path.stat()
        version = f"{stat.st_mtime_ns:x}-{stat.st_size:x}"
        if len(self.roots) > 1:
            relative = f"{quote(root.name, safe='')}/{quote(relative, safe='/')}"
        else:
            relative = quote(relative, safe="/")
        return "/media/" + relative + f"?v={version}"

    def resolve_media(self, raw_path: str) -> Path:
        decoded = unquote(raw_path)
        if len(self.roots) > 1:
            run_name, separator, decoded = decoded.partition("/")
            if not separator or run_name not in self._roots_by_name:
                raise ValueError("媒体路径缺少有效的 run")
            root = self._roots_by_name[run_name]
        else:
            root = self.root
        relative = Path(decoded)
        if relative.is_absolute():
            raise ValueError("媒体路径必须是相对路径")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("媒体路径越界") from exc
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], index: ResultIndex, page_size: int):
        self.index = index
        self.default_page_size = page_size
        self.annotation_lock = threading.Lock()
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/manifest":
            run_names = [root.name for root in self.server.index.roots]
            self._send_json(
                {
                    "run_name": run_names[0] if len(run_names) == 1 else f"{len(run_names)} runs",
                    "run_names": run_names,
                    "result_dir": str(self.server.index.root),
                    "result_dirs": [str(root) for root in self.server.index.roots],
                    "run_count": len(run_names),
                    "total": len(self.server.index.samples),
                    "steps": self.server.index.step_names,
                    "cos_types": list(COS_TYPES),
                    "default_page_size": self.server.default_page_size,
                }
            )
            return
        if parsed.path == "/api/samples":
            self._serve_samples(parsed.query)
            return
        if parsed.path.startswith("/media/"):
            self._serve_media(parsed.path[len("/media/") :])
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/media/"):
            self._serve_media(parsed.path[len("/media/") :], head_only=True)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("请求体大小无效")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("请求体必须是 JSON 对象")
            sample_id = payload.get("sample_id", payload.get("task_id"))
            if not isinstance(sample_id, str):
                raise ValueError("请求必须包含 sample_id")
            if "cos_type" not in payload and "success" not in payload:
                raise ValueError("请求必须包含 cos_type 或 success")
            cos_type = payload.get("cos_type", _UNSET)
            if cos_type is not None and not isinstance(cos_type, str):
                if cos_type is not _UNSET:
                    raise ValueError("cos_type 必须是字符串或 null")
            success = payload.get("success", _UNSET)
            if success is not _UNSET and success is not None and not isinstance(success, bool):
                raise ValueError("success 必须是布尔值或 null")
            with self.server.annotation_lock:
                annotation = self.server.index.save_annotation(
                    sample_id, cos_type=cos_type, success=success
                )
        except (ValueError, TypeError, json.JSONDecodeError, OSError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json({"ok": True, **annotation})

    def do_DELETE(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/annotations":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            with self.server.annotation_lock:
                cleared = self.server.index.clear_annotations()
        except OSError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json({"ok": True, "cleared": cleared})

    def _serve_samples(self, query: str) -> None:
        values = parse_qs(query)
        try:
            page = int(values.get("page", ["1"])[0])
            page_size = int(
                values.get("page_size", [str(self.server.default_page_size)])[0]
            )
            run_name = values.get("run", [None])[0]
            cos_type = values.get("cos_type", [None])[0]
            success = values.get("success", [None])[0]
            payload = self.server.index.page(
                page,
                page_size,
                run_name=run_name,
                cos_type=cos_type,
                success=success,
            )
        except (ValueError, TypeError) as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        self._send_json(payload)

    def _serve_media(self, raw_path: str, head_only: bool = False) -> None:
        try:
            path = self.server.index.resolve_media(raw_path)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        size = path.stat().st_size
        start, end = 0, max(0, size - 1)
        status = HTTPStatus.OK
        range_header = self.headers.get("Range")
        if range_header:
            parsed_range = self._parse_range(range_header, size)
            if parsed_range is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            start, end = parsed_range
            status = HTTPStatus.PARTIAL_CONTENT

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        length = end - start + 1 if size else 0
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "public, max-age=3600")
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        if head_only or not size:
            return
        with path.open("rb") as file:
            file.seek(start)
            remaining = length
            while remaining:
                chunk = file.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    @staticmethod
    def _parse_range(value: str, size: int) -> tuple[int, int] | None:
        match = RANGE_RE.fullmatch(value.strip())
        if not match or size <= 0:
            return None
        first, last = match.groups()
        if not first:
            if not last:
                return None
            suffix = int(last)
            if suffix <= 0:
                return None
            return max(0, size - suffix), size - 1
        start = int(first)
        end = int(last) if last else size - 1
        if start >= size or end < start:
            return None
        return start, min(end, size - 1)

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[dashboard] {self.address_string()} - {format % args}")


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Video Reasoning · Step Observatory</title>
  <style>
    :root { --ink:#132238; --muted:#607086; --paper:#f5f2ea; --card:#fffefa; --line:#d8d4ca; --accent:#ff6846; --blue:#3b65dc; --id-w:132px; --prompt-w:360px; --frame-w:264px; }
    body.multi-run { --id-w:260px; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:radial-gradient(circle at 8% 5%,#fff6d9 0,transparent 28%),linear-gradient(135deg,#eef2fb,#f7f3ea 56%,#ebf3ee); font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif; min-height:100vh; }
    header { padding:30px 38px 22px; display:flex; align-items:flex-end; justify-content:space-between; gap:24px; }
    h1 { margin:0; font-size:clamp(30px,3vw,52px); line-height:.95; font-weight:950; letter-spacing:-.05em; }
    .eyebrow { color:var(--accent); font-size:14px; font-weight:900; letter-spacing:.16em; text-transform:uppercase; margin-bottom:10px; }
    #run-info { max-width:650px; color:var(--muted); font-size:16px; font-weight:750; text-align:right; overflow-wrap:anywhere; }
    .toolbar { margin:0 24px 16px; padding:14px 18px; background:rgba(255,255,255,.82); border:1px solid rgba(255,255,255,.95); border-radius:18px; box-shadow:0 12px 35px rgba(25,43,68,.1); display:flex; align-items:center; justify-content:space-between; gap:18px; backdrop-filter:blur(16px); }
    .pager,.page-size,.annotation-controls,.filters,.filter { display:flex; align-items:center; gap:10px; font-weight:850; font-size:17px; }
    .filters { flex-wrap:wrap; }
    .filter select { min-width:180px; padding:7px 10px; }
    button,input,select { font:inherit; font-weight:850; border:2px solid var(--ink); background:#fff; color:var(--ink); border-radius:11px; min-height:44px; }
    button { padding:8px 16px; cursor:pointer; box-shadow:3px 3px 0 var(--ink); transition:transform .12s,box-shadow .12s; }
    button:hover:not(:disabled) { transform:translate(-1px,-1px); box-shadow:5px 5px 0 var(--ink); }
    button:disabled { opacity:.35; cursor:not-allowed; box-shadow:none; }
    input { width:76px; padding:7px 10px; text-align:center; }
    #page-label { min-width:220px; text-align:center; }
    #page-number { width:74px; margin:0 5px; }
    .keys { color:var(--muted); font-size:14px; font-weight:800; }
    kbd { border:1px solid #adb5c1; border-bottom-width:3px; border-radius:6px; padding:2px 7px; background:white; color:var(--ink); }
    .table-shell { margin:0 24px 26px; overflow:auto; border-radius:18px; border:1px solid var(--line); background:var(--card); box-shadow:0 18px 50px rgba(25,43,68,.14); max-height:calc(100vh - 290px); }
    table { border-collapse:separate; border-spacing:0; min-width:100%; table-layout:fixed; }
    th,td { border-right:1px solid var(--line); border-bottom:1px solid var(--line); padding:12px; vertical-align:middle; background:var(--card); }
    th { position:sticky; top:0; z-index:8; background:#172842; color:white; height:58px; font-size:17px; font-weight:950; text-transform:uppercase; letter-spacing:.04em; }
    tr:last-child td { border-bottom:0; }
    th.id,td.id { position:sticky; left:0; width:var(--id-w); min-width:var(--id-w); z-index:6; }
    th.prompt,td.prompt { position:sticky; left:var(--id-w); width:var(--prompt-w); min-width:var(--prompt-w); z-index:6; }
    th.frame,td.frame { position:sticky; left:calc(var(--id-w) + var(--prompt-w)); width:var(--frame-w); min-width:var(--frame-w); z-index:6; box-shadow:8px 0 15px rgba(31,44,61,.09); }
    th.id,th.prompt,th.frame { z-index:10; background:#172842; }
    td.id,td.prompt,td.frame { background:#fffdf7; }
    td.id { color:var(--blue); font-size:22px; font-weight:950; font-variant-numeric:tabular-nums; }
    .run-name { color:var(--muted); font-size:13px; line-height:1.25; margin-bottom:7px; overflow-wrap:anywhere; }
    td.prompt { font-size:14px; line-height:1.25; font-weight:700; overflow-wrap:anywhere; }
    th.annotation,td.annotation { width:250px; min-width:250px; }
    th.success,td.success { width:150px; min-width:150px; }
    td.annotation { font-size:18px; font-weight:900; text-align:center; }
    .annotation-value { display:inline-block; padding:9px 12px; border-radius:10px; background:#e7edff; color:#284aa7; }
    .annotation-value.unset { background:#eeeae1; color:var(--muted); }
    .annotation-value.success { background:#dff4e6; color:#17653a; }
    .annotation-value.failure { background:#fbe4df; color:#9c3025; }
    .annotation-select { width:100%; padding:8px 10px; }
    .annotation-select.saving { opacity:.55; }
    .annotation-error { color:#a62f25; font-size:13px; margin-top:7px; }
    th.step,td.step { width:310px; min-width:310px; }
    img,video { display:block; width:100%; height:auto; object-fit:contain; border-radius:11px; background:#101722; }
    video { box-shadow:inset 0 0 0 1px rgba(255,255,255,.12); }
    .missing { width:100%; aspect-ratio:20/11; display:grid; place-items:center; border:2px dashed #c7c2b8; border-radius:11px; color:#7b8491; background:#f1eee7; font-weight:900; }
    .empty,.error { padding:60px; font-size:22px; font-weight:900; text-align:center; }
    .error { color:#a62f25; }
    .loading { opacity:.55; pointer-events:none; }
    @media (max-width:900px) { :root { --id-w:108px; --prompt-w:280px; --frame-w:220px; } header { padding:22px 20px 16px; align-items:flex-start; flex-direction:column; } #run-info { text-align:left; } .toolbar { margin-inline:12px; flex-wrap:wrap; } .table-shell { margin-inline:12px; max-height:none; } .keys { display:none; } th.step,td.step { width:270px; min-width:270px; } }
  </style>
</head>
<body>
  <header>
    <div><div class="eyebrow">Denoising timeline</div><h1>Step Observatory</h1></div>
    <div id="run-info">正在读取结果目录…</div>
  </header>
  <section class="toolbar">
    <div class="pager">
      <button id="prev" type="button">← 上一页</button>
      <label id="page-label">第 <input id="page-number" type="number" min="1" step="1" aria-label="跳转页码"> / <span id="total-pages">—</span> 页</label>
      <button id="next" type="button">下一页 →</button>
    </div>
    <div class="keys"><kbd>A</kbd> / <kbd>←</kbd> 上一页　<kbd>D</kbd> / <kbd>→</kbd> 下一页</div>
    <label class="page-size">每页 <input id="page-size" type="number" min="1" max="50" step="1"> 条</label>
  </section>
  <section class="toolbar">
    <div class="filters">
      <label class="filter">Run <select id="run-filter"><option value="">全部 Run</option></select></label>
      <label class="filter">CoS <select id="cos-filter"><option value="">全部 CoS</option><option value="annotated">已有 CoS</option><option value="unannotated">没有 CoS</option></select></label>
      <label class="filter">是否成功 <select id="success-filter"><option value="">全部状态</option><option value="annotated">已标注</option><option value="unannotated">未标注</option><option value="true">成功</option><option value="false">失败</option></select></label>
    </div>
    <div class="keys">筛选条件可以组合使用</div>
  </section>
  <section class="toolbar">
    <div class="annotation-controls">
      <button id="toggle-mode" type="button">进入标注模式</button>
      <button id="clear-annotations" type="button">清空所有标注</button>
      <button id="toggle-annotations" type="button">隐藏标注</button>
    </div>
    <div class="keys" id="mode-hint">当前为查看模式</div>
  </section>
  <main class="table-shell" id="shell"><div class="empty">正在加载可视化结果…</div></main>
  <script>
    const state={page:1,pageSize:5,totalPages:1,loading:false,annotationMode:false,annotationsHidden:false,multiRun:false,cosTypes:[],filters:{run:'',cosType:'',success:''},data:null};
    const shell=document.querySelector('#shell'), prev=document.querySelector('#prev'), next=document.querySelector('#next'), pageInput=document.querySelector('#page-number'), totalPagesLabel=document.querySelector('#total-pages'), sizeInput=document.querySelector('#page-size'), runFilter=document.querySelector('#run-filter'), cosFilter=document.querySelector('#cos-filter'), successFilter=document.querySelector('#success-filter'), modeButton=document.querySelector('#toggle-mode'), clearButton=document.querySelector('#clear-annotations'), annotationsButton=document.querySelector('#toggle-annotations'), modeHint=document.querySelector('#mode-hint');
    const missing=label=>{const d=document.createElement('div');d.className='missing';d.textContent=label;return d};
    function mediaCell(kind,url,label){const td=document.createElement('td');td.className=kind==='video'?'step':'frame';if(!url){td.append(missing('暂无'+label));return td}const el=document.createElement(kind);el.src=url;el.preload='metadata';if(kind==='video'){el.autoplay=true;el.muted=true;el.loop=true;el.playsInline=true;el.controls=true;}else{el.alt=label;el.loading='lazy';}el.addEventListener('error',()=>el.replaceWith(missing(label+'加载失败')));td.append(el);return td}
    async function saveAnnotation(item,select,cell,field){select.disabled=true;select.classList.add('saving');cell.querySelector('.annotation-error')?.remove();const previous=field==='cos_type'?item.cos_type:item.success;let next=select.value;if(field==='success')next=next==='true'?true:next==='false'?false:null;try{const response=await fetch('/api/annotation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample_id:item.sample_id,[field]:next})}),data=await response.json();if(!response.ok)throw new Error(data.error||'保存失败');item.cos_type=data.cos_type;item.success=data.success;await load(state.page);}catch(error){select.value=field==='cos_type'?previous:(previous===true?'true':previous===false?'false':'');const message=document.createElement('div');message.className='annotation-error';message.textContent='保存失败：'+error.message;cell.append(message);}finally{select.disabled=false;select.classList.remove('saving')}}
    function annotationCell(item){const td=document.createElement('td');td.className='annotation';if(state.annotationMode){const select=document.createElement('select');select.className='annotation-select';select.setAttribute('aria-label',`${item.task_id} 的 CoS Type`);state.cosTypes.forEach(label=>{const option=document.createElement('option');option.value=label;option.textContent=label;option.selected=item.cos_type===label;select.append(option)});select.addEventListener('change',()=>saveAnnotation(item,select,td,'cos_type'));td.append(select);}else{const value=document.createElement('span');value.className='annotation-value';value.textContent=item.cos_type;td.append(value)}return td}
    function successCell(item){const td=document.createElement('td');td.className='annotation success';if(state.annotationMode){const select=document.createElement('select');select.className='annotation-select';select.setAttribute('aria-label',`${item.task_id} 是否成功`);[['','未标注'],['true','成功'],['false','失败']].forEach(([optionValue,label])=>{const option=document.createElement('option');option.value=optionValue;option.textContent=label;option.selected=(item.success===true?'true':item.success===false?'false':'')===optionValue;select.append(option)});select.addEventListener('change',()=>saveAnnotation(item,select,td,'success'));td.append(select);}else{const value=document.createElement('span');value.className='annotation-value '+(item.success===true?'success':item.success===false?'failure':'unset');value.textContent=item.success===true?'成功':item.success===false?'失败':'未标注';td.append(value)}return td}
    function render(data){state.data=data;const table=document.createElement('table'),head=document.createElement('thead'),hr=document.createElement('tr');[[(state.multiRun?'Run / ':'')+'任务 ID','id'],['提示词','prompt'],['初始帧','frame']].forEach(([text,cls])=>{const th=document.createElement('th');th.className=cls;th.textContent=text;hr.append(th)});if(!state.annotationsHidden){[['CoS Type','annotation'],['是否成功','annotation success']].forEach(([text,cls])=>{const th=document.createElement('th');th.className=cls;th.textContent=text;hr.append(th)})}data.steps.forEach(step=>{const th=document.createElement('th');th.className='step';th.textContent='Step '+Number(step.slice(5));hr.append(th)});head.append(hr);table.append(head);const body=document.createElement('tbody');data.items.forEach(item=>{const tr=document.createElement('tr'),id=document.createElement('td'),prompt=document.createElement('td');id.className='id';if(state.multiRun){const run=document.createElement('div');run.className='run-name';run.textContent=item.run_name;id.append(run,document.createTextNode(item.task_id));}else{id.textContent=item.task_id;}prompt.className='prompt';prompt.textContent=item.prompt||'暂无提示词';tr.append(id,prompt,mediaCell('img',item.initial_frame,'初始帧'));if(!state.annotationsHidden)tr.append(annotationCell(item),successCell(item));data.steps.forEach(step=>tr.append(mediaCell('video',item.step_videos[step],step)));body.append(tr)});table.append(body);shell.replaceChildren(data.items.length?table:Object.assign(document.createElement('div'),{className:'empty',textContent:'没有可展示的样本'}));document.querySelectorAll('video').forEach(v=>v.play().catch(()=>{}));}
    async function load(page=state.page){if(state.loading)return;state.loading=true;shell.classList.add('loading');try{const params=new URLSearchParams({page:String(page),page_size:String(state.pageSize)});if(state.filters.run)params.set('run',state.filters.run);if(state.filters.cosType)params.set('cos_type',state.filters.cosType);if(state.filters.success)params.set('success',state.filters.success);const response=await fetch('/api/samples?'+params);const data=await response.json();if(!response.ok)throw new Error(data.error||'请求失败');state.page=data.page;state.totalPages=data.total_pages;render(data);pageInput.value=data.page;pageInput.max=data.total_pages;totalPagesLabel.textContent=`${data.total_pages} · 共 ${data.total} 条`;prev.disabled=data.page<=1;next.disabled=data.page>=data.total_pages;history.replaceState(null,'',`#page=${data.page}`);}catch(error){shell.innerHTML=`<div class="error"></div>`;shell.firstChild.textContent='加载失败：'+error.message;}finally{state.loading=false;shell.classList.remove('loading')}}
    function changePage(delta){const target=Math.max(1,Math.min(state.totalPages,state.page+delta));if(target!==state.page)load(target)}
    function jumpToPage(){const value=Number(pageInput.value);if(Number.isInteger(value)&&value>=1){load(Math.min(value,state.totalPages))}else pageInput.value=state.page}
    prev.addEventListener('click',()=>changePage(-1));next.addEventListener('click',()=>changePage(1));pageInput.addEventListener('change',jumpToPage);pageInput.addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();pageInput.blur()}});sizeInput.addEventListener('change',()=>{const value=Number(sizeInput.value);if(Number.isInteger(value)&&value>=1&&value<=50){state.pageSize=value;load(1)}else sizeInput.value=state.pageSize});
    runFilter.addEventListener('change',()=>{state.filters.run=runFilter.value;load(1)});cosFilter.addEventListener('change',()=>{state.filters.cosType=cosFilter.value;load(1)});successFilter.addEventListener('change',()=>{state.filters.success=successFilter.value;load(1)});
    modeButton.addEventListener('click',()=>{state.annotationMode=!state.annotationMode;modeButton.textContent=state.annotationMode?'返回查看模式':'进入标注模式';modeHint.textContent=state.annotationMode?'当前为标注模式，选择后自动保存':'当前为查看模式';if(state.data)render(state.data)});
    annotationsButton.addEventListener('click',()=>{state.annotationsHidden=!state.annotationsHidden;annotationsButton.textContent=state.annotationsHidden?'显示标注':'隐藏标注';if(state.data)render(state.data)});
    clearButton.addEventListener('click',async()=>{if(!confirm('确定删除已加载结果目录下所有样本的 annotation.json 吗？此操作不可撤销。'))return;clearButton.disabled=true;try{const response=await fetch('/api/annotations',{method:'DELETE'}),data=await response.json();if(!response.ok)throw new Error(data.error||'清空失败');await load(state.page);alert(`已清空 ${data.cleared} 个标注文件`);}catch(error){alert('清空标注失败：'+error.message)}finally{clearButton.disabled=false}});
    document.addEventListener('keydown',event=>{if(event.altKey||event.ctrlKey||event.metaKey||event.shiftKey)return;const tag=event.target.tagName;if(['INPUT','TEXTAREA','SELECT','BUTTON','VIDEO'].includes(tag)||event.target.isContentEditable)return;if(event.key==='a'||event.key==='A'||event.key==='ArrowLeft'){event.preventDefault();changePage(-1)}if(event.key==='d'||event.key==='D'||event.key==='ArrowRight'){event.preventDefault();changePage(1)}});
    (async()=>{try{const manifest=await fetch('/api/manifest').then(r=>r.json());state.pageSize=manifest.default_page_size;state.cosTypes=manifest.cos_types;state.multiRun=manifest.run_count>1;document.body.classList.toggle('multi-run',state.multiRun);sizeInput.value=state.pageSize;manifest.run_names.forEach(name=>{const option=document.createElement('option');option.value=name;option.textContent=name;runFilter.append(option)});manifest.cos_types.filter(name=>name!=='Unknown').forEach(name=>{const option=document.createElement('option');option.value=name;option.textContent=name;cosFilter.append(option)});document.querySelector('#run-info').textContent=`${manifest.run_name} · ${manifest.total} 条样本 · ${manifest.steps.length} 个 Step`;const requested=Number(new URLSearchParams(location.hash.slice(1)).get('page'));load(Number.isInteger(requested)&&requested>0?requested:1)}catch(error){shell.innerHTML='<div class="error">无法连接到看板服务</div>'}})();
  </script>
</body>
</html>'''


def positive_page_size(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是整数") from exc
    if not 1 <= parsed <= 50:
        raise argparse.ArgumentTypeError("必须在 1 到 50 之间")
    return parsed


def parse_data_indices(values: list[str] | None) -> list[int] | None:
    """Parse comma- and/or whitespace-separated zero-based data indices."""
    if values is None:
        return None
    raw_indices = [part.strip() for value in values for part in value.split(",")]
    if not raw_indices or any(not part for part in raw_indices):
        raise argparse.ArgumentTypeError("--indices 必须是以逗号或空格分隔的整数列表")
    try:
        indices = [int(part) for part in raw_indices]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--indices 必须是以逗号或空格分隔的整数列表") from exc
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("--indices 不能包含负数")
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("--indices 不能包含重复值")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="浏览逐步去噪可视化结果")
    parser.add_argument(
        "--result-dir",
        type=Path,
        nargs="+",
        default=[DEFAULT_RESULT_DIR],
        help="一个或多个结果目录（支持由 shell 展开的通配符）",
    )
    parser.add_argument("--page-size", type=positive_page_size, default=5)
    parser.add_argument(
        "--indices",
        nargs="+",
        metavar="INDEX",
        help="只展示指定的 0-based 数据索引，支持逗号或空格分隔（如 0,2,5）",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("error: --port 必须在 1 到 65535 之间")
    try:
        data_indices = parse_data_indices(args.indices)
        index = ResultIndex(args.result_dir, data_indices=data_indices)
    except (ValueError, argparse.ArgumentTypeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    server = DashboardServer((args.host, args.port), index, args.page_size)
    print(f"[dashboard] runs: {len(index.roots)}")
    for root in index.roots:
        print(f"[dashboard]   {root.name}")
    print(f"[dashboard] samples: {len(index.samples)}, steps: {len(index.step_names)}")
    print(f"[dashboard] open: http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
