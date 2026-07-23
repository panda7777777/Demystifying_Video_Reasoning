#!/usr/bin/env python3
"""
Language-Table Dataset Downloader
=================================

Downloads the Google Language-Table datasets
(https://github.com/google-research/language-table#datasets) from the public
GCS bucket ``gs://gresearch/robotics/`` over plain HTTPS.

Pure standard library -- no gsutil / gcloud / tensorflow required.
Downloads are resumable: already-complete files (matching size) are skipped,
partial files are re-downloaded, so the script can simply be re-run.

The primary ``language_table`` dataset is written directly under ``out_dir`` so
that TFDS can read a version directory without a redundant dataset-name level:

    {out_dir}/{version}/...
    e.g. tfds.builder_from_directory('{out_dir}/0.0.1')

Other datasets keep their dataset-name directory to avoid version collisions.

Usage:
    # Inspect dataset sizes without downloading anything
    python scripts/language_table/download.py --list

    # Download everything (the full language-table release, several TB!)
    python scripts/language_table/download.py --datasets all --yes

    # Download only the real-robot dataset
    python scripts/language_table/download.py --datasets language_table

    # Custom destination / more parallel connections
    python scripts/language_table/download.py \
        --datasets language_table_sim \
        --out-dir ./data/language-table \
        --workers 16
"""

import argparse
import base64
import hashlib
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BUCKET = "gresearch"
BUCKET_PREFIX = "robotics"
API_ROOT = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA_ROOT = f"https://storage.googleapis.com/{BUCKET}"

# All datasets listed at https://github.com/google-research/language-table#datasets
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = [
    "language_table",
    "language_table_sim",
    "language_table_blocktoblock_sim",
    "language_table_blocktoblock_4block_sim",
    "language_table_blocktoblock_oracle_sim",
    "language_table_blocktoblockrelative_oracle_sim",
    "language_table_blocktoabsolute_oracle_sim",
    "language_table_blocktorelative_oracle_sim",
    "language_table_separate_oracle_sim",
]

RETRIES = 5
RETRY_BACKOFF = 2.0  # seconds, doubled per attempt
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


# ── GCS JSON API helpers ─────────────────────────────────────────────────────

def _api_get(params):
    """One GET against the GCS JSON API with retries. Returns parsed JSON."""
    url = API_ROOT + "?" + urllib.parse.urlencode(params)
    last_err = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"GCS listing failed after {RETRIES} attempts: {url}\n{last_err}")


def list_objects(prefix):
    """Yield {'name': ..., 'size': int, 'md5Hash': ...} for every object under prefix."""
    page_token = None
    while True:
        params = {
            "prefix": prefix,
            "maxResults": 1000,
            "fields": "items(name,size,md5Hash),nextPageToken",
        }
        if page_token:
            params["pageToken"] = page_token
        data = _api_get(params)
        for item in data.get("items", []):
            yield {"name": item["name"], "size": int(item["size"]),
                   "md5Hash": item.get("md5Hash")}
        page_token = data.get("nextPageToken")
        if not page_token:
            return


def discover_versions(dataset):
    """Return sorted version directory names (e.g. ['0.0.1']) for a dataset."""
    data = _api_get({
        "prefix": f"{BUCKET_PREFIX}/{dataset}/",
        "delimiter": "/",
        "maxResults": 1000,
        "fields": "prefixes",
    })
    versions = []
    for p in data.get("prefixes", []):
        # 'robotics/language_table/0.0.1/' -> '0.0.1'
        versions.append(p.rstrip("/").rsplit("/", 1)[-1])
    return sorted(versions)


# ── Download ─────────────────────────────────────────────────────────────────

def destination_path(object_name, dataset, out_dir):
    """Map a GCS object to the normalized local dataset layout."""
    prefix = f"{BUCKET_PREFIX}/{dataset}/"
    if not object_name.startswith(prefix):
        raise ValueError(f"Object {object_name!r} is outside {prefix!r}")
    relative = object_name[len(prefix):]
    dataset_root = (
        out_dir
        if dataset == "language_table"
        else os.path.join(out_dir, dataset)
    )
    return os.path.join(dataset_root, relative)


def download_object(obj, dataset, out_dir, verify_md5=False):
    """Download one object into the normalized local dataset layout.

    Returns (status, bytes) where status is 'skipped' | 'downloaded'.
    """
    dest = destination_path(obj["name"], dataset, out_dir)
    if os.path.exists(dest) and os.path.getsize(dest) == obj["size"]:
        return "skipped", obj["size"]

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{MEDIA_ROOT}/{urllib.parse.quote(obj['name'])}"
    part = dest + ".part"

    last_err = None
    for attempt in range(RETRIES):
        try:
            md5 = hashlib.md5() if verify_md5 else None
            with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    if md5:
                        md5.update(chunk)
            if os.path.getsize(part) != obj["size"]:
                raise IOError(f"size mismatch: got {os.path.getsize(part)}, "
                              f"want {obj['size']}")
            if md5 and obj.get("md5Hash"):
                got = base64.b64encode(md5.digest()).decode("ascii")
                if got != obj["md5Hash"]:
                    raise IOError(f"md5 mismatch: got {got}, want {obj['md5Hash']}")
            os.replace(part, dest)  # atomic
            return "downloaded", obj["size"]
        except (urllib.error.URLError, TimeoutError, ConnectionError, IOError) as e:
            last_err = e
            time.sleep(RETRY_BACKOFF * (2 ** attempt))
    raise RuntimeError(f"Failed after {RETRIES} attempts: {obj['name']}\n{last_err}")


def human(n):
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Language-Table datasets from gs://gresearch/robotics/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=str,
        default=os.path.join(REPO_ROOT, "data", "language-table"),
        help="Destination directory",
    )
    parser.add_argument("--datasets", type=str, nargs="+", default=["all"],
                        help=f"'all' or any of: {' '.join(DATASETS)}")
    parser.add_argument("--version", type=str, default=None,
                        help="Dataset version (e.g. 0.0.1). Default: all versions found")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel download connections")
    parser.add_argument("--list", action="store_true",
                        help="Only list datasets/versions/sizes, download nothing")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Resolve the full file list and report totals, no download")
    parser.add_argument("--verify-md5", dest="verify_md5", action="store_true",
                        help="Verify md5 of every downloaded file (slower)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="Skip the size confirmation prompt")
    args = parser.parse_args()

    if args.datasets == ["all"]:
        datasets = DATASETS
    else:
        unknown = [d for d in args.datasets if d not in DATASETS]
        if unknown:
            parser.error(f"Unknown dataset(s): {unknown}. Choose from: {DATASETS}")
        datasets = args.datasets

    # Resolve (dataset, version) -> object list
    print(f"Listing objects in gs://{BUCKET}/{BUCKET_PREFIX}/ ...")
    plan = []          # (dataset, version, [objects])
    total_bytes = 0
    total_files = 0
    for ds in datasets:
        versions = discover_versions(ds)
        if not versions:
            print(f"  WARNING: no versions found for '{ds}', skipping")
            continue
        if args.version:
            if args.version not in versions:
                print(f"  WARNING: '{ds}' has no version {args.version} "
                      f"(found: {versions}), skipping")
                continue
            versions = [args.version]
        for ver in versions:
            objs = list(list_objects(f"{BUCKET_PREFIX}/{ds}/{ver}/"))
            size = sum(o["size"] for o in objs)
            plan.append((ds, ver, objs))
            total_bytes += size
            total_files += len(objs)
            print(f"  {ds}/{ver}: {len(objs)} files, {human(size)}")

    print(f"\nTotal: {total_files} files, {human(total_bytes)}")
    if args.list or args.dry_run:
        return

    if not args.yes:
        reply = input(f"Download {human(total_bytes)} to {args.out_dir}? [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    # Flatten and download
    all_objs = [(dataset, obj) for dataset, _, objects in plan for obj in objects]
    done = {"files": 0, "bytes": 0, "skipped": 0}
    lock = threading.Lock()
    t0 = time.time()
    failed = []

    def _one(dataset, obj):
        status, size = download_object(
            obj, dataset, args.out_dir, verify_md5=args.verify_md5
        )
        with lock:
            done["files"] += 1
            done["bytes"] += size
            if status == "skipped":
                done["skipped"] += 1
            if done["files"] % 50 == 0 or done["files"] == total_files:
                elapsed = time.time() - t0
                print(f"  [{done['files']}/{total_files}] "
                      f"{human(done['bytes'])} / {human(total_bytes)} "
                      f"({done['skipped']} skipped, {elapsed:.0f}s)", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_one, dataset, obj): obj
            for dataset, obj in all_objs
        }
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                failed.append((futures[fut]["name"], str(e)))
                print(f"  ERROR: {e}", file=sys.stderr, flush=True)

    print(f"\nDone: {done['files'] - len(failed)}/{total_files} files OK, "
          f"{done['skipped']} were already present.")
    if failed:
        print(f"{len(failed)} file(s) FAILED -- re-run the script to retry:",
              file=sys.stderr)
        for name, err in failed[:20]:
            print(f"  {name}: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
