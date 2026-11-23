#!/usr/bin/env python3
"""Download and merge the DevCV toolbox evaluation data from Databrary volume 1981.

Downloads all six finetune folders (SAYCam + BabyView, train/val/test), extracts
frames from each, then writes merged_devcv/ with one JSON per task×split combining
both sources.

Usage:
    python sft_download.py --out /path/to/storage
"""
import sys
from track1_download import main

DEVCV_FOLDERS = ["78234", "78235", "78236", "78237", "78238", "78239"]

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "get", "1981"] + DEVCV_FOLDERS + sys.argv[1:]
    main()
