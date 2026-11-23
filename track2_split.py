#!/usr/bin/env python3
"""Split downloaded raw videos into train/val/test clips.

Reads a predefined split JSON and uses ffmpeg to cut each video into clips
assigned to train, val, or test.

Usage:
    python track2_split.py saycam   --video-dir /path/to/saycam_videos   --out /path/to/output
    python track2_split.py babyview --video-dir /path/to/babyview_videos --out /path/to/output

Split JSONs are downloaded automatically from Databrary (volume 1981, folder 78269)
and cached in ~/.cache/databrary/splits/. Requires DATABRARY_USERNAME and
DATABRARY_PASSWORD (env vars or .env file next to this script).

If you already have the split JSONs locally, pass --splits-dir to skip the download:
    python track2_split.py saycam --splits-dir /path/to/splits --video-dir ... --out ...
Expected filenames: saycam_split.json and babyview_split.json.

Split JSON format:
    {"train": {"video_id": [[start, end], ...], ...},
     "val":   {...},
     "test":  {...}}

Inclusive/exclusive rules:
    - Train range [a, b]:       extract seconds a..b inclusive → ffmpeg -ss a -t (b-a+1)
    - Val/test range [a, b]:    extract seconds (a+1)..(b-1) exclusive → ffmpeg -ss (a+1) -t (b-a-1)
                                Skip if b - a <= 1 (empty after exclusion).
    - Single-element [s]:       always extract exactly 1 second → ffmpeg -ss s -t 1
                                (no boundary to exclude; the second is explicitly assigned here)
    - Degenerate [a, a]:        val/test only; empty after exclusion → skip.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Databrary volume/folder containing the split JSONs
SPLITS_VOLUME = 1981
SPLITS_FOLDER = 78269
SPLIT_FILE_IDS = {
    "saycam":   760106,
    "babyview": 760105,
}
SPLITS_CACHE = Path.home() / ".cache" / "databrary" / "splits"

SPLITS = ["train", "val", "test"]


def load_env(path=HERE / ".env"):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def fetch_split_json(dataset, splits_dir=None):
    """Return the split dict, from splits_dir, cache, or Databrary."""
    load_env()
    if splits_dir is not None:
        p = Path(splits_dir) / f"{dataset}_split.json"
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")
        log(f"Using split JSON: {p}")
        with open(p) as f:
            return json.load(f)

    cache_path = SPLITS_CACHE / f"{dataset}_split.json"
    if cache_path.exists():
        log(f"Using cached split JSON: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    log(f"Downloading {dataset} split JSON from Databrary...")
    try:
        from databrary_auth import DatabarySession
    except ImportError:
        sys.exit("ERROR: databrary_auth.py not found. Run from the databrary package directory.")

    missing = [k for k in ("DATABRARY_USERNAME", "DATABRARY_PASSWORD") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing credentials: {', '.join(missing)} (put them in {HERE / '.env'})")

    file_id = SPLIT_FILE_IDS[dataset]
    db = DatabarySession()
    try:
        info = db.api_json(f"volumes/{SPLITS_VOLUME}/folders/{SPLITS_FOLDER}/files/{file_id}/")
        download_url = info["downloadUrl"]
        content = db.download_session().get(download_url, timeout=60).text
    finally:
        db.close()

    SPLITS_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(content)
    log(f"Cached to {cache_path}")
    return json.loads(content)


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def clip_range(split, entry):
    """Return (ss, duration_secs) for ffmpeg, or None to skip.

    ss and duration are floats in seconds.
    """
    if len(entry) == 1:
        # Single-element: always extract that 1 second
        return (entry[0], 1)

    a, b = entry[0], entry[1]
    if split == "train":
        # Inclusive: extract seconds a through b (b+1 exclusive in ffmpeg terms)
        duration = b - a + 1
        if duration <= 0:
            return None
        return (a, duration)
    else:
        # Exclusive: drop second a and second b, extract a+1 through b-1
        ss = a + 1
        end = b          # ffmpeg -to b = last included second is b-1
        duration = end - ss
        if duration <= 0:
            return None
        return (ss, duration)


def find_video(video_dir, video_id):
    """Locate the video file. SAYCam keys include .mp4; BabyView don't."""
    p = Path(video_dir) / video_id
    if p.exists():
        return p
    p = Path(video_dir) / (video_id + ".mp4")
    if p.exists():
        return p
    # Try stripping extension from key and re-adding .mp4
    stem = Path(video_id).stem
    p = Path(video_dir) / (stem + ".mp4")
    if p.exists():
        return p
    return None


def output_path(out_dir, dataset, split, video_id, ss, duration):
    stem = Path(video_id).stem
    end = ss + duration
    name = f"{stem}__{int(ss):07d}_{int(end):07d}.mp4"
    return Path(out_dir) / split / dataset / name


def run_ffmpeg(input_path, out_path, ss, duration):
    """Extract clip using fast-seek + stream copy. Returns 'ok', 'skipped', or 'error'."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(ss),
        "-i", str(input_path),
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=600)

    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        return "error"

    if not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return "error"

    tmp.rename(out_path)
    return "ok"


def process_dataset(dataset, video_dir, out_dir, dry_run, splits_dir=None):
    splits = fetch_split_json(dataset, splits_dir)

    # Collect all (video_id, split, entry) triples
    tasks = []
    for split in SPLITS:
        for video_id, ranges in splits[split].items():
            for entry in ranges:
                r = clip_range(split, entry)
                if r is not None:
                    tasks.append((video_id, split, entry, r[0], r[1]))

    log(f"{dataset}: {len(tasks):,} clips to extract across {sum(len(splits[s]) for s in SPLITS):,} video×split pairs")

    counters = defaultdict(int)
    missing_videos = set()

    for i, (video_id, split, entry, ss, duration) in enumerate(tasks, 1):
        if video_id in missing_videos:
            counters["missing"] += 1
            continue

        video_path = find_video(video_dir, video_id)
        if video_path is None:
            if video_id not in missing_videos:
                log(f"  WARNING: not found: {video_id}")
                missing_videos.add(video_id)
            counters["missing"] += 1
            continue

        out = output_path(out_dir, dataset, split, video_id, ss, duration)

        if dry_run:
            print(f"  [{split}] {video_id} {entry} → ss={ss} dur={duration}s → {out.name}")
            counters["dry_run"] += 1
            continue

        status = run_ffmpeg(video_path, out, ss, duration)
        counters[status] += 1

        if status == "error":
            log(f"  ERROR: {split} {video_id} {entry} ss={ss} dur={duration}")
        elif i % 500 == 0:
            log(f"  {i:,}/{len(tasks):,} — ok={counters['ok']} skipped={counters['skipped']} "
                f"error={counters['error']} missing={counters['missing']}")

    return counters


def main():
    parser = argparse.ArgumentParser(description="Split raw videos into train/val/test clips")
    parser.add_argument("dataset", choices=["saycam", "babyview"], help="Dataset to split")
    parser.add_argument("--video-dir", required=True, help="Directory of downloaded raw videos")
    parser.add_argument("--out", required=True, help="Output root; clips go into {out}/{split}/{dataset}/")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be extracted without running ffmpeg")
    parser.add_argument("--splits-dir", help="Directory containing saycam_split.json / babyview_split.json (skips Databrary download)")
    args = parser.parse_args()

    if not Path(args.video_dir).is_dir():
        sys.exit(f"ERROR: --video-dir does not exist: {args.video_dir}")

    counters = process_dataset(args.dataset, args.video_dir, args.out, args.dry_run, args.splits_dir)

    log("Done.")
    for k, v in sorted(counters.items()):
        log(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
