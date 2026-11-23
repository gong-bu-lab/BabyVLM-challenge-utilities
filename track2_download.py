
#!/usr/bin/env python3
"""
Databrary browser-based bulk downloader.

Commands:
    python track2_download.py check
    python track2_download.py list VOLUME --sessions
    python track2_download.py get VOLUME --sessions --flat --out DIR
    python track2_download.py get VOLUME SESSION_ID FILE_ID --sessions --flat --out DIR

Downloads use authenticated Playwright/Chromium, not requests.

Features:
- Volume / session / folder enumeration
- Select a single session or individual file IDs
- Original filenames
- File size and optional SHA-1 verification
- File-level retries and restart
- No overwriting existing files
- Preserves old .part and .progress.json files

Requirements:
- databrary_auth.py with DatabarySession.browser_download()
- Chromium launched with accept_downloads=True
"""

import argparse
import json
import hashlib
import os
import shutil
import signal
import sys
import threading
import time
import uuid

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


def sha1_of(path):
    h = hashlib.sha1()

    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(16 * 1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def is_video(name):
    return Path(name).suffix.lower() in VIDEO_EXTENSIONS


def record_path(path):
    return path.with_name(path.name + ".download.json")


def check_file(path, size, expected_sha1=None, verify=True):
    """Never trust metadata SHA-1/size for a transcoded video."""
    if not path.is_file() or path.is_symlink():
        return False

    if is_video(path.name):
        sidecar = record_path(path)
        if not sidecar.is_file() or sidecar.is_symlink():
            return False
        try:
            record = json.loads(sidecar.read_text(encoding="utf-8"))
            if record.get("downloaded_size") != path.stat().st_size:
                return False
            if verify:
                return sha1_of(path).lower() == record["downloaded_sha1"].lower()
            return True
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    if path.stat().st_size != size:
        return False
    if verify and expected_sha1:
        return sha1_of(path).lower() == str(expected_sha1).lower()
    return True


def validate_video_container(path):
    """Minimal MP4 signature check; not a full decode/integrity proof."""
    if path.suffix.lower() != ".mp4":
        return
    with path.open("rb") as f:
        header = f.read(32)
    if len(header) < 12 or header[4:8] != b"ftyp":
        raise RuntimeError(f"Invalid MP4 header: {path.name}")


def write_record_exclusive(path, record):
    """Never overwrite an existing record, including a dangling symlink."""
    with path.open("x", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")


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

    def containers(self, volume, kind="folders"):
        return list(
            self.pages(
                f"volumes/{volume}/{kind}/"
            )
        )

    def container_files(self, volume, kind, container_id):
        return list(
            self.pages(
                f"volumes/{volume}/{kind}/"
                f"{container_id}/files/"
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
    kind,
    container_id,
    meta,
    out_dir,
    workers,
    verify=True,
):
    """
    Download a file through Chromium.

    Browser downloads are sequential.
    The workers argument is kept for CLI compatibility.
    """

    api_path = (
        f"volumes/{volume}/{kind}/{container_id}/"
        f"files/{meta['id']}/download-link/"
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve download metadata through authenticated browser.
    info = client.json(api_path)

    name = info.get("fileName") or meta["name"]
    size = int(info.get("fileSize") or meta["size"])

    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
    ):
        raise RuntimeError(
            f"Unsafe filename: {name!r}"
        )

    dest = out_dir / name
    expected_sha1 = meta.get("sha1")
    if is_video(name) and (record_path(dest).exists() or record_path(dest).is_symlink()) and not (dest.exists() or dest.is_symlink()):
        raise RuntimeError(f"Existing sidecar without destination: {record_path(dest)}. Preserved.")

    # Never overwrite an existing file.
    if dest.exists() or dest.is_symlink():
        if check_file(
            dest,
            size,
            expected_sha1,
            verify,
        ):
            log(f"Skipping {dest} (already complete)")
            return dest

        raise RuntimeError(
            f"Existing file failed validation: {dest}. "
            "File preserved without modification."
        )

    # Check available disk space.
    free = shutil.disk_usage(out_dir).free

    if free < size + 1_000_000_000:
        raise RuntimeError(
            f"Not enough free space for {name} "
            f"({human(size)} required, "
            f"{human(free)} available)"
        )

    log(
        f"Downloading {name} "
        f"({human(size)}) -> {dest}"
    )

    last_error = None

    for attempt in range(1, RETRIES + 1):
        if STOP.is_set():
            raise KeyboardInterrupt

        # Each attempt uses a new temporary filename.
        # Existing .part and .progress.json are untouched.
        temp = out_dir / (
            f".{name}.{uuid.uuid4().hex}.browser-tmp"
        )

        try:
            # Refresh signed URL for each attempt.
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
                    # Expected when navigation initiates a download.
                    if "Download is starting" not in str(exc):
                        raise

            t0 = time.perf_counter()

            client.browser_download(
                trigger,
                temp,
            )

            t1 = time.perf_counter()

            actual_size = temp.stat().st_size
            log(
                f"  Browser download + save_as: {t1 - t0:.2f}s "
                f"({actual_size / max(t1 - t0, 1e-9) / 1e6:.2f} MB/s)"
            )

            # Databrary can return a transcoded video with different
            # size and SHA-1 than the original API metadata.
            actual_size = temp.stat().st_size
            if actual_size == 0:
                raise RuntimeError(f"Empty download: {name}")

            video = is_video(name)
            if not video and actual_size != size:
                raise RuntimeError(
                    f"Size mismatch for {name}: expected {size}, got {actual_size}"
                )
            if video:
                validate_video_container(temp)

            log(f"  calculating downloaded SHA-1 of {name}...")
            actual_sha1 = sha1_of(temp)
            sha1_matched = bool(
                expected_sha1 and actual_sha1.lower() == str(expected_sha1).lower()
            )

            if not video and verify and expected_sha1 and not sha1_matched:
                raise RuntimeError(f"SHA-1 mismatch for {name}")

            if video and (actual_size != size or (expected_sha1 and not sha1_matched)):
                log(
                    f"  Video differs from original metadata (likely transcoded): "
                    f"original {human(size)}, downloaded {human(actual_size)}"
                )

            # Publish atomically without overwriting existing data.
            # The new temporary file is unlinked only after a successful hard link.
            os.link(temp, dest)
            temp.unlink()

            if video:
                record = {
                    "volume_id": str(volume),
                    "container_type": kind,
                    "container_id": str(container_id),
                    "file_id": str(meta["id"]),
                    "filename": name,
                    "original_size": size,
                    "original_sha1": expected_sha1,
                    "downloaded_size": actual_size,
                    "downloaded_sha1": actual_sha1,
                    "original_sha1_matched": sha1_matched,
                    "verification": "download_sha1_recorded_mp4_header_checked"
                    if name.lower().endswith(".mp4") else "download_sha1_recorded",
                }
                write_record_exclusive(record_path(dest), record)
                status = "original SHA-1 matched" if sha1_matched else "transcoded/unverified"
            else:
                status = "verified" if verify and expected_sha1 else "not checked"

            log(f"Saved {dest} ({human(actual_size)}, SHA-1 {status})")
            log(f"  Download SHA-1: {actual_sha1}")

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

            if isinstance(exc, FileExistsError) and dest.exists():
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
# Commands
# ============================================================

def cmd_check(client, args):
    info = client.json("volumes/1981/")

    log(
        "OK — connected to Databrary. "
        f"Volume 1981: {info.get('name', '(no name)')}"
    )


def cmd_list(client, args):
    kind = (
        "sessions"
        if args.sessions
        else "folders"
    )

    label = (
        "Session"
        if args.sessions
        else "Folder"
    )

    containers = client.containers(
        args.volume,
        kind,
    )

    if args.container:
        containers = [
            c for c in containers
            if (
                str(c["id"]) == args.container
                or c["name"] == args.container
            )
        ]

    if args.container and not containers:
        sys.exit(f"{label} {args.container} not found in volume {args.volume}")

    grand = 0

    for c in containers:
        files = client.container_files(
            args.volume,
            kind,
            c["id"],
        )

        total = sum(
            f["size"] for f in files
        )

        grand += total

        print(
            f"\n{label} {c['id']}  {c['name']} "
            f"({len(files)} files, {human(total)})"
        )

        for f in files:
            print(
                f"  {str(f['id']):>10} "
                f" {human(f['size']):>12} "
                f" {f['name']}"
            )

    print(
        f"\nTotal: {human(grand)}"
    )


def cmd_get(client, args):
    kind = (
        "sessions"
        if args.sessions
        else "folders"
    )

    label = (
        "Session"
        if args.sessions
        else "Folder"
    )

    containers = client.containers(
        args.volume,
        kind,
    )

    if args.container:
        containers = [
            c for c in containers
            if (
                str(c["id"]) == args.container
                or c["name"] == args.container
            )
        ]

        if not containers:
            sys.exit(
                f"{label} {args.container} "
                f"not found in volume {args.volume}"
            )

    for c in containers:
        if STOP.is_set():
            raise KeyboardInterrupt

        files = client.container_files(
            args.volume,
            kind,
            c["id"],
        )

        if args.files:
            files = [
                f for f in files
                if str(f["id"]) in args.files
            ]

        if args.flat:
            out_dir = Path(args.out)
        else:
            out_dir = (
                Path(args.out)
                / f"{c['id']}_{safe_name(c['name'])}"
            )

        total = sum(
            f["size"] for f in files
        )

        log(
            f"{label} {c['name']}: "
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
                kind,
                c["id"],
                meta,
                out_dir,
                args.workers,
                not args.no_verify,
            )

    log("All done.")


# ============================================================
# CLI
# ============================================================

def on_sigterm(signum, frame):
    raise KeyboardInterrupt


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = p.add_subparsers(
        dest="cmd",
        required=True,
    )

    sub.add_parser("check")

    s = sub.add_parser("list")

    s.add_argument("volume")

    s.add_argument(
        "container",
        nargs="?",
        help="folder/session ID or name",
    )

    s.add_argument(
        "--sessions",
        action="store_true",
        help="use sessions API",
    )

    s = sub.add_parser("get")

    s.add_argument("volume")

    s.add_argument(
        "container",
        nargs="?",
        help="folder/session ID or name",
    )

    s.add_argument(
        "files",
        nargs="*",
        help="specific file IDs",
    )

    s.add_argument(
        "--out",
        default="downloads",
        help="output directory",
    )

    s.add_argument(
        "--workers",
        type=int,
        default=1,
        help="compatibility option; browser downloads are sequential",
    )

    s.add_argument(
        "--no-verify",
        action="store_true",
        help="skip SHA-1 verification",
    )

    s.add_argument(
        "--sessions",
        action="store_true",
        help="use sessions API",
    )

    s.add_argument(
        "--flat",
        action="store_true",
        help="save directly in output directory",
    )

    args = p.parse_args()

    signal.signal(
        signal.SIGTERM,
        on_sigterm,
    )

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

    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)

    finally:
        client.close()


if __name__ == "__main__":
    main()
