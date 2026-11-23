#!/usr/bin/env python3
"""Extract frames from MJPEG folders downloaded manually from the Databrary UI.

Use this if you downloaded Track 1 data by hand from databrary.org instead of
using track1_download.py. Point it at a single downloaded folder (the one
containing the *.mjpeg and *.index.json files) and it will extract all frames.

Usage:
    python extract_frames.py SRC_DIR --out OUT_DIR [--workers N] [--no-verify]

Arguments:
    SRC_DIR     directory containing the downloaded *.mjpeg + *.index.json files
    --out       destination directory for extracted frames (must be separate from SRC_DIR)
    --workers   parallel extraction workers (default: 4)
    --no-verify skip SHA-256 frame verification
"""
import sys
from track1_download import main

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "extract"] + sys.argv[1:]
    main()
