
#!/usr/bin/env python3
"""
Databrary filtered-pretraining downloader.

Commands:
    python track1_download.py check
    python track1_download.py list VOLUME [FOLDER ...]
    python track1_download.py get VOLUME [FOLDER ...]
    python track1_download.py extract SRC_DIR --out DIR
    python track1_download.py merge-devcv SRC_DIR --out DIR

Features:
- Authenticated Playwright/Chromium downloads
- Folder and file selection
- Original downloaded filenames
- Strict verification for non-video files
- Transcoded-video size/SHA-1 compatibility
- Download SHA-1 sidecar records
- Safe restart and skip verification
- File-level retries
- No overwriting existing files
- Preserve failed temporary files
- Separate MJPEG extraction and DevCV merging
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import threading
import time
import uuid

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError


HERE = Path(__file__).resolve().parent

REQUIRED = [
    "DATABRARY_USERNAME",
    "DATABRARY_PASSWORD",
]

RETRIES = 8
STOP = threading.Event()

DEVCV_FOLDER_IDS = {
    "78234", "78235", "78236",
    "78237", "78238", "78239",
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv",
    ".webm", ".m4v",
}


# ============================================================
# Utilities
# ============================================================

def log(message):
    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}",
        flush=True,
    )


def load_env(path=HERE / ".env"):
    if not path.exists():
        return

    for line in path.read_text().splitlines():
        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)

        os.environ.setdefault(
            key.strip(),
            value.strip().strip("'\""),
        )


def human(n):
    n = float(n)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1000


def safe_name(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .strip("_")
        or "unnamed"
    )


def valid_filename(name):
    return (
        isinstance(name, str)
        and bool(name)
        and name not in (".", "..")
        and "/" not in name
        and "\\" not in name
        and "\x00" not in name
    )


def safe_relative_path(name):
    """
    Validate a relative path from an MJPEG index.
    """
    if not isinstance(name, str) or not name:
        raise RuntimeError("Invalid relative path")

    normalized = name.replace("\\", "/")
    path = Path(normalized)

    if (
        path.is_absolute()
        or any(
            part in ("", ".", "..")
            for part in normalized.split("/")
        )
        or "\x00" in normalized
        or ":" in normalized.split("/")[0]
    ):
        raise RuntimeError(
            f"Unsafe relative path: {name!r}"
        )

    return path


def sha1_of(path):
    h = hashlib.sha1()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(16 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def is_video_file(name):
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


def sidecar_path(dest):
    return dest.with_name(dest.name + ".download.json")


def load_sidecar(dest):
    path = sidecar_path(dest)

    if path.is_symlink() or not path.is_file():
        return None

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    return data


def check_file(
    path,
    size,
    expected_sha1=None,
    verify=True,
    file_id=None,
    volume=None,
    folder=None,
):
    """
    Validate an existing download.

    Non-video:
        Strict metadata size and SHA-1 checks.

    Video:
        Prefer recorded actual size and SHA-1.
        Without a sidecar, only accept an exact
        original-metadata match.
    """
    path = Path(path)

    if path.is_symlink() or not path.is_file():
        return False

    actual_size = path.stat().st_size

    if is_video_file(path.name):
        record = load_sidecar(path)

        if record is not None:
            try:
                recorded_size = int(record["actual_size"])
                recorded_sha1 = str(record["actual_sha1"])
            except (KeyError, TypeError, ValueError):
                return False

            if record.get("filename") != path.name:
                return False

            if file_id is not None:
                if str(record.get("file_id")) != str(file_id):
                    return False

            if volume is not None:
                if str(record.get("volume")) != str(volume):
                    return False

            if folder is not None:
                if str(record.get("folder")) != str(folder):
                    return False

            if actual_size != recorded_size:
                return False

            if verify:
                return (
                    sha1_of(path).lower()
                    == recorded_sha1.lower()
                )

            return True

        # No sidecar: accept if size matches the download-link size.
        if actual_size != size:
            return False

        return True

    if actual_size != size:
        return False

    if verify and expected_sha1:
        return (
            sha1_of(path).lower()
            == str(expected_sha1).lower()
        )

    return True


def publish_new_file(temp, dest):
    """
    Publish without replacing an existing file.

    Both paths must be on the same filesystem.
    """
    if dest.exists() or dest.is_symlink():
        raise FileExistsError(
            f"Destination already exists: {dest}"
        )

    os.link(temp, dest)
    temp.unlink()


def write_sidecar(dest, record):
    """
    Write metadata without overwriting an existing
    sidecar. Uses an exclusive temporary file and
    atomic hard-link publication.
    """
    path = sidecar_path(dest)

    if path.exists() or path.is_symlink():
        raise FileExistsError(
            f"Sidecar already exists: {path}"
        )

    temp = path.parent / (
        f".{path.name}.{uuid.uuid4().hex}.meta-tmp"
    )

    try:
        with open(temp, "x", encoding="utf-8") as f:
            json.dump(
                record,
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")

        publish_new_file(temp, path)

    except BaseException:
        if temp.exists():
            log(
                f"  Temporary metadata retained: {temp}"
            )
        raise


def ensure_separate_output(src, out):
    """
    Prevent derived output from being written
    inside raw input or its ancestors.
    """
    src = Path(src).resolve()
    out = Path(out).resolve()

    if (
        src == out
        or src in out.parents
        or out in src.parents
    ):
        raise RuntimeError(
            "Derived output must be in a separate "
            "directory outside the source directory tree."
        )


def check_disk_space(directory, size):
    free = shutil.disk_usage(directory).free

    if free < size + 1_000_000_000:
        raise RuntimeError(
            f"Not enough free space: "
            f"{human(size)} required plus 1 GB reserve; "
            f"{human(free)} available"
        )


# ============================================================
# Client
# ============================================================

class Client:
    """Authenticated Databrary client using Playwright."""

    def __init__(self):
        from databrary_auth import DatabarySession

        self._db = DatabarySession()

    def close(self):
        self._db.close()

    def json(self, path):
        return self._db.api_json(path)

    def pages(self, path):
        yield from self._db.api_pages(path)

    def folders(self, volume):
        return list(
            self.pages(f"volumes/{volume}/folders/")
        )

    def files(self, volume, folder):
        return list(
            self.pages(
                f"volumes/{volume}/folders/"
                f"{folder}/files/"
            )
        )

    def browser_download(self, trigger, destination):
        return self._db.browser_download(
            trigger,
            destination,
        )


# ============================================================
# Browser download
# ============================================================

def download_file(
    client,
    volume,
    folder,
    meta,
    out_dir,
    workers=1,
    verify=True,
):
    """
    Download through Chromium.

    Browser downloads are sequential.
    workers is retained for CLI compatibility.

    Transcoded videos may differ from original
    metadata in both size and SHA-1.
    """

    api_path = (
        f"volumes/{volume}/folders/{folder}/"
        f"files/{meta['id']}/download-link/"
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = client.json(api_path)

    name = info.get("fileName") or meta["name"]
    size = int(info.get("fileSize") or meta["size"])
    original_size = int(meta["size"])
    expected_sha1 = meta.get("sha1")

    if not valid_filename(name):
        raise RuntimeError(
            f"Unsafe filename: {name!r}"
        )

    dest = out_dir / name
    is_video = is_video_file(name)

    # Never overwrite an existing file.
    if dest.exists() or dest.is_symlink():
        if check_file(
            dest,
            size,
            expected_sha1,
            verify,
            file_id=meta["id"],
            volume=volume,
            folder=folder,
        ):
            log(
                f"Skipping {dest} "
                "(already complete)"
            )
            return dest

        raise RuntimeError(
            f"Existing file failed validation: {dest}. "
            "File preserved without modification."
        )

    # Prevent conflicts with existing metadata.
    if is_video:
        record_path = sidecar_path(dest)

        if record_path.exists() or record_path.is_symlink():
            raise RuntimeError(
                f"Metadata sidecar already exists: "
                f"{record_path}. "
                "Existing metadata preserved."
            )

    check_disk_space(out_dir, size)

    log(
        f"Downloading {name} "
        f"({human(size)}) -> {dest}"
    )

    last_error = None

    for attempt in range(1, RETRIES + 1):
        if STOP.is_set():
            raise KeyboardInterrupt

        temp = out_dir / (
            f".{name}.{uuid.uuid4().hex}.browser-tmp"
        )

        try:
            # Refresh signed URL for every attempt.
            info = client.json(api_path)
            download_url = info["downloadUrl"]

            def trigger(page):
                try:
                    page.goto(
                        download_url,
                        wait_until="commit",
                        timeout=30000,
                    )
                except PlaywrightError as exc:
                    if "Download is starting" not in str(exc):
                        raise

            t0 = time.perf_counter()

            client.browser_download(
                trigger,
                temp,
            )

            elapsed = time.perf_counter() - t0
            actual_size = temp.stat().st_size

            log(
                f"  Browser download + save_as: "
                f"{elapsed:.2f}s "
                f"({actual_size / max(elapsed, 1e-6) / 1e6:.2f} MB/s)"
            )

            if actual_size <= 0:
                raise RuntimeError(
                    f"Empty downloaded file: {name}"
                )

            actual_sha1 = None
            sha1_matched = False

            if is_video:
                # Videos can be transcoded.
                # Always calculate their actual SHA-1
                # for later restart verification.
                log(
                    f"  calculating downloaded SHA-1 "
                    f"of {name}..."
                )

                actual_sha1 = sha1_of(temp)

                sha1_matched = (
                    bool(expected_sha1)
                    and actual_sha1.lower()
                    == str(expected_sha1).lower()
                )

                if (
                    actual_size != original_size
                    or not sha1_matched
                ):
                    log(
                        "  Video differs from original "
                        "metadata (likely transcoded): "
                        f"original {human(original_size)}, "
                        f"downloaded {human(actual_size)}"
                    )

                status = (
                    "verified"
                    if (
                        sha1_matched
                        and actual_size == original_size
                    )
                    else "transcoded/unverified"
                )

            else:
                # Strict validation for non-video files.
                if actual_size != size:
                    raise RuntimeError(
                        f"Size mismatch for {name}: "
                        f"expected {size}, "
                        f"got {actual_size}"
                    )

                if verify and expected_sha1:
                    log(
                        f"  verifying SHA-1 of {name}..."
                    )

                    actual_sha1 = sha1_of(temp)

                    sha1_matched = (
                        actual_sha1.lower()
                        == str(expected_sha1).lower()
                    )

                    if not sha1_matched:
                        raise RuntimeError(
                            f"SHA-1 mismatch for {name}"
                        )

                status = (
                    "verified"
                    if verify and expected_sha1
                    else "not checked"
                )

            # Publish atomically without overwriting.
            publish_new_file(temp, dest)

            # Record the actual downloaded video.
            if is_video:
                record = {
                    "schema_version": 1,
                    "volume": str(volume),
                    "folder": str(folder),
                    "file_id": str(meta["id"]),
                    "filename": name,
                    "original_size": original_size,
                    "download_link_size": size,
                    "actual_size": actual_size,
                    "original_sha1": expected_sha1,
                    "actual_sha1": actual_sha1,
                    "sha1_matches_original": sha1_matched,
                    "verification_status": status,
                    "downloaded_at": datetime.now().isoformat(),
                }

                try:
                    write_sidecar(dest, record)
                except Exception as exc:
                    log(
                        "  WARNING: Video saved, but "
                        f"metadata recording failed: {exc}"
                    )
                    log(
                        "  Existing video preserved. "
                        "A later restart may require "
                        "manual validation."
                    )

            log(
                f"Saved {dest} "
                f"({human(actual_size)}, "
                f"SHA-1 {status})"
            )

            if is_video:
                log(
                    f"  Download SHA-1: {actual_sha1}"
                )

            return dest

        except KeyboardInterrupt:
            log(f"Interrupted: {name}")
            raise

        except Exception as exc:
            last_error = exc

            # Preserve unexpected temporary data.
            if temp.exists():
                log(
                    f"  Temporary file retained: {temp}"
                )

            if isinstance(exc, FileExistsError):
                raise RuntimeError(
                    f"Destination already exists: {dest}. "
                    "No file was overwritten."
                ) from exc

            log(
                f"  Attempt {attempt}/{RETRIES} failed: "
                f"{type(exc).__name__}: {exc}"
            )

            if attempt < RETRIES:
                wait = min(
                    60,
                    5 * 2 ** (attempt - 1),
                )

                log(f"  Retrying in {wait}s...")

                if STOP.wait(wait):
                    raise KeyboardInterrupt

    raise RuntimeError(
        f"Failed to download {name} "
        f"after {RETRIES} attempts: {last_error}"
    )


# ============================================================
# MJPEG extraction
# ============================================================

def iter_index(index):
    version = index.get("schema_version")

    if version == 1:
        yield index["video"], index["frames"]

    elif version == 2:
        for chunk in index["chunks"]:
            yield chunk["video"], chunk["frames"]

    else:
        raise RuntimeError(
            f"Unsupported index schema_version: {version}"
        )


def find_mjpeg(folder, video_name):
    """
    Locate the MJPEG file, accepting .mjpeg and Databrary's .mjpg names.
    """
    relative = safe_relative_path(video_name)

    candidates = [
        folder / relative,
        folder / relative.with_suffix(""),
    ]

    if relative.suffix.lower() in (".mjpeg", ".mjpg"):
        alternate = ".mjpg" if relative.suffix.lower() == ".mjpeg" else ".mjpeg"
        candidates.append(folder / relative.with_suffix(alternate))

    root = folder.resolve()

    for candidate in candidates:
        if (
            candidate.is_file()
            and candidate.resolve().is_relative_to(root)
        ):
            return candidate

    raise FileNotFoundError(
        f"Cannot find {video_name!r} in {folder}"
    )


def extract_frame(task):
    """
    Extract one JPEG without overwriting existing files.
    """
    mjpeg, offset, length, dest, sha256 = task

    if STOP.is_set():
        raise KeyboardInterrupt

    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() or not dest.is_file():
            raise RuntimeError(
                f"Existing frame is not a regular file: {dest}"
            )

        if sha256:
            h = hashlib.sha256()

            with open(dest, "rb") as f:
                for block in iter(
                    lambda: f.read(1024 * 1024),
                    b"",
                ):
                    h.update(block)

            if h.hexdigest().lower() != sha256.lower():
                raise RuntimeError(
                    f"Existing frame failed SHA-256: {dest}"
                )

        elif dest.stat().st_size != length:
            raise RuntimeError(
                f"Existing frame has unexpected size: {dest}"
            )

        return "skipped"

    with open(mjpeg, "rb") as f:
        f.seek(offset)
        data = f.read(length)

    if len(data) != length:
        raise RuntimeError(
            f"Short read at offset {offset} in {mjpeg.name}"
        )

    if sha256:
        actual = hashlib.sha256(data).hexdigest()

        if actual.lower() != sha256.lower():
            raise RuntimeError(
                f"SHA-256 mismatch for {dest.name}"
            )

    dest.parent.mkdir(parents=True, exist_ok=True)

    temp = dest.parent / (
        f".{dest.name}.{uuid.uuid4().hex}.extract-tmp"
    )

    try:
        with open(temp, "xb") as f:
            f.write(data)

        publish_new_file(temp, dest)

    except FileExistsError:
        if temp.exists():
            log(
                f"  Temporary frame retained: {temp}"
            )
        raise RuntimeError(
            f"Frame destination already exists: {dest}"
        )

    except BaseException:
        if temp.exists():
            log(
                f"  Temporary frame retained: {temp}"
            )
        raise

    return "completed"


def cmd_extract(args):
    src = Path(args.src).resolve()

    if not src.is_dir():
        raise FileNotFoundError(src)

    if not args.out:
        raise RuntimeError(
            "Extraction requires --out DIR. "
            "Choose a separate derived-data directory."
        )

    out = Path(args.out).resolve()

    ensure_separate_output(src, out)

    index_files = sorted(src.glob("*.index.json"))

    if not index_files:
        raise RuntimeError(
            f"No *.index.json found in {src}"
        )

    if len(index_files) > 1:
        raise RuntimeError(
            f"Multiple index files found in {src}"
        )

    index = json.loads(
        index_files[0].read_text(encoding="utf-8")
    )

    tasks = []

    for video_name, frames in iter_index(index):
        mjpeg = find_mjpeg(src, video_name)

        for frame in frames:
            relative = safe_relative_path(frame["path"])

            dest = out / relative

            if not dest.resolve().is_relative_to(out):
                raise RuntimeError(
                    f"Frame path escapes output root: "
                    f"{relative}"
                )

            sha256 = (
                None
                if args.no_verify
                else frame.get("sha256")
            )

            tasks.append((
                mjpeg,
                int(frame["offset"]),
                int(frame["length"]),
                dest,
                sha256,
            ))

    log(
        f"Extracting {len(tasks):,} frames "
        f"from {src.name}"
    )

    completed = 0
    skipped = 0

    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as pool:

        for start in range(0, len(tasks), 10_000):
            if STOP.is_set():
                raise KeyboardInterrupt

            batch = tasks[start:start + 10_000]

            futures = [
                pool.submit(extract_frame, task)
                for task in batch
            ]

            for future in as_completed(futures):
                result = future.result()

                if result == "completed":
                    completed += 1
                else:
                    skipped += 1

    log(
        f"Extraction done: "
        f"{completed:,} completed, "
        f"{skipped:,} skipped"
    )

    log(f"Derived frames: {out}")


# ============================================================
# DevCV merge
# ============================================================

def cmd_merge_devcv(args):
    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if not src.is_dir():
        raise FileNotFoundError(src)

    ensure_separate_output(src, out)

    devcv_dirs = sorted(
        d for d in src.iterdir()
        if (
            d.is_dir()
            and d.name.split("_")[0]
            in DEVCV_FOLDER_IDS
        )
    )

    if not devcv_dirs:
        log("No DevCV folders found.")
        return

    by_task_split = {}

    for directory in devcv_dirs:
        for file in sorted(directory.iterdir()):
            if (
                not file.is_file()
                or file.name.endswith(".index.json")
                or file.stat().st_size > 50 * 1024 * 1024
            ):
                continue

            try:
                data = json.loads(
                    file.read_text(encoding="utf-8")
                )
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                ValueError,
            ):
                continue

            if (
                not isinstance(data, list)
                or not data
                or not isinstance(data[0], dict)
                or "conversations" not in data[0]
            ):
                continue

            transformed = []

            for entry in data:
                if not isinstance(entry, dict):
                    raise RuntimeError(
                        f"Unexpected entry in {file}"
                    )

                new_entry = dict(entry)
                imgs = new_entry.get("image", [])

                if isinstance(imgs, str):
                    new_entry["image"] = (
                        f"{directory.name}/{imgs}"
                    )

                elif isinstance(imgs, list):
                    new_entry["image"] = [
                        f"{directory.name}/{img}"
                        for img in imgs
                    ]

                transformed.append(new_entry)

            by_task_split.setdefault(
                file.name, []
            ).extend(transformed)

    if not by_task_split:
        log("No DevCV task JSON files found.")
        return

    out.mkdir(parents=True, exist_ok=True)

    for filename, entries in sorted(
        by_task_split.items()
    ):
        if not valid_filename(filename):
            raise RuntimeError(
                f"Unsafe merged filename: {filename}"
            )

        dest = out / filename

        if dest.exists() or dest.is_symlink():
            raise RuntimeError(
                f"Output already exists: {dest}. "
                "No file was overwritten."
            )

        temp = out / (
            f".{filename}.{uuid.uuid4().hex}.merge-tmp"
        )

        try:
            with open(temp, "x", encoding="utf-8") as f:
                json.dump(
                    entries,
                    f,
                    ensure_ascii=False,
                )

            publish_new_file(temp, dest)

        except BaseException:
            if temp.exists():
                log(
                    f"  Temporary merge file retained: {temp}"
                )
            raise

        log(
            f"Merged {filename}: "
            f"{len(entries):,} entries"
        )

    log(f"DevCV merge done: {out}")


# ============================================================
# Commands
# ============================================================

def cmd_check(client, args):
    info = client.json("volumes/1981/")

    log(
        "OK — connected to Databrary. "
        f"Volume 1981: "
        f"{info.get('name', '(no name)')}"
    )


def cmd_list(client, args):
    folders = client.folders(args.volume)

    if args.folders:
        folders = [
            f for f in folders
            if str(f["id"]) in args.folders
        ]

    grand = 0

    for folder in folders:
        files = client.files(
            args.volume,
            folder["id"],
        )

        total = sum(
            f["size"] for f in files
        )

        grand += total

        print(
            f"\nFolder {folder['id']} "
            f"{folder['name']} "
            f"({len(files)} files, {human(total)})"
        )

        for file in files:
            print(
                f"  {str(file['id']):>10} "
                f" {human(file['size']):>12} "
                f" {file['name']}"
            )

    print(f"\nTotal: {human(grand)}")


def cmd_get(client, args):
    folders = client.folders(args.volume)

    if args.folders:
        folders = [
            f for f in folders
            if str(f["id"]) in args.folders
        ]

        found = {
            str(f["id"]) for f in folders
        }

        missing = set(args.folders) - found

        if missing:
            raise RuntimeError(
                "Folder(s) not found: "
                + ", ".join(sorted(missing))
            )

    if args.files and len(folders) != 1:
        raise RuntimeError(
            "--files requires exactly one folder."
        )

    for folder in folders:
        if STOP.is_set():
            raise KeyboardInterrupt

        files = client.files(
            args.volume,
            folder["id"],
        )

        if args.files:
            files = [
                f for f in files
                if str(f["id"]) in args.files
            ]

            found_files = {
                str(f["id"]) for f in files
            }

            missing = set(args.files) - found_files

            if missing:
                raise RuntimeError(
                    "File(s) not found: "
                    + ", ".join(sorted(missing))
                )

        out_dir = (
            Path(args.out)
            / f"{folder['id']}_{safe_name(folder['name'])}"
        )

        total = sum(
            f["size"] for f in files
        )

        log(
            f"Folder {folder['name']}: "
            f"{len(files)} file(s), "
            f"{human(total)}"
        )

        for meta in sorted(
            files,
            key=lambda f: f["size"],
        ):
            if STOP.is_set():
                raise KeyboardInterrupt

            download_file(
                client,
                args.volume,
                folder["id"],
                meta,
                out_dir,
                args.workers,
                not args.no_verify,
            )

    log("All downloads done.")


# ============================================================
# CLI
# ============================================================

def on_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    sub.add_parser("check")

    # List
    s = sub.add_parser("list")

    s.add_argument("volume")

    s.add_argument(
        "folders",
        nargs="*",
        metavar="FOLDER",
    )

    # Get
    s = sub.add_parser("get")

    s.add_argument("volume")

    s.add_argument(
        "folders",
        nargs="*",
        metavar="FOLDER",
    )

    s.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
    )

    s.add_argument(
        "--out",
        default="downloads",
    )

    s.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Compatibility option; "
            "browser downloads are sequential"
        ),
    )

    s.add_argument(
        "--no-verify",
        action="store_true",
    )

    s.add_argument(
        "--no-extract",
        action="store_true",
        help=(
            "Compatibility option; "
            "extraction is always separate"
        ),
    )

    # Extract
    s = sub.add_parser("extract")

    s.add_argument("src")

    s.add_argument(
        "--out",
        required=True,
    )

    s.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    s.add_argument(
        "--no-verify",
        action="store_true",
    )

    # Merge DevCV
    s = sub.add_parser("merge-devcv")

    s.add_argument("src")

    s.add_argument(
        "--out",
        required=True,
    )

    args = parser.parse_args()

    signal.signal(
        signal.SIGTERM,
        on_sigterm,
    )

    if hasattr(args, "workers") and args.workers < 1:
        parser.error("--workers must be >= 1")

    # Offline processing commands.
    if args.cmd in ("extract", "merge-devcv"):
        try:
            if args.cmd == "extract":
                cmd_extract(args)
            else:
                cmd_merge_devcv(args)

        except KeyboardInterrupt:
            log("Interrupted.")
            sys.exit(130)

        except (
            RuntimeError,
            FileNotFoundError,
            FileExistsError,
        ) as exc:
            log(f"ERROR: {exc}")
            sys.exit(1)

        return

    load_env()

    client = Client()

    try:
        {
            "check": cmd_check,
            "list": cmd_list,
            "get": cmd_get,
        }[args.cmd](client, args)

    except KeyboardInterrupt:
        log("Interrupted.")
        sys.exit(130)

    except (
        RuntimeError,
        FileNotFoundError,
        FileExistsError,
    ) as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    main()
