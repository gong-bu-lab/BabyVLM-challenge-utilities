# BabyVLM dataset downloader

Downloads data from Databrary for use with BabyLLaVA-V2 training.

## Table of contents

- [Setup](#setup)
- [Track 1 — Filtered pretraining data](#track-1--filtered-pretraining-data)
  - [Downloading](#downloading) 
  - [Using the data for training](#using-the-data-for-training)
- [Track 2 — Raw pretraining data](#track-2--raw-pretraining-data)
  - [Step 1 — Download raw metadata](#track-2--raw-pretraining-data)
  - [Step 2 — Download SAYCam videos](#track-2--raw-pretraining-data)
  - [Step 3 — Download BabyView videos](#track-2--raw-pretraining-data)
  - [Step 4 — IMPORTANT: Split into train/val/test](#track-2--raw-pretraining-data)
- [Download SFT data (all tracks!)](#download-sft-data-all-tracks)

## Setup

Requires Python 3.9+.

```bash
conda create -n databrary python=3.11
conda activate databrary
pip install -r requirements.txt
playwright install chromium
```

On first run the scripts open a headless browser and prompt for your Databrary
email and password. If your account has two-factor authentication enabled you
will also be prompted for the code once. After that the session is saved to
`~/.databrary-browser-profile/` and subsequent runs require no interaction.

**Optional:** to skip the prompts, save your credentials in a `.env` file next
to the scripts (never commit it):

```
DATABRARY_USERNAME=your@email.com
DATABRARY_PASSWORD=yourpassword
```
<!-- 
```bash
chmod 600 .env
``` -->

---

## Track 1 — Filtered pretraining data

### Downloading

Total size is ~555 GB; make sure your destination has at least 600 GB free.

**Option A — API (recommended)**

`track1_download.py` downloads and extracts all frames automatically.

```bash
python track1_download.py get 1981 --out /path/to/storage
```

Progress is saved continuously — re-run the same command to resume if interrupted.
For long runs, use `tmux` or `nohup` so it survives logout:

```bash
nohup python track1_download.py get 1981 --out /path/to/storage > download.log 2>&1 &
tail -f download.log
```

**Option B — Manual download from Databrary UI**

Download the folders from [databrary.org/volume/1981](https://databrary.org/volume/1981)
by hand. Each folder contains `.mjpeg` and `.index.json` files. Then run
`track1_extract_frames.py` on each downloaded folder to extract the individual frames:

```bash
python track1_extract_frames.py /path/to/downloaded_folder --out /path/to/storage --workers 4
```

Repeat for each folder. Already-extracted frames are skipped, so re-running is safe.

### Using the data for training

The training scripts are in the [BabyLLaVA-V2](https://github.com/ShawnKing98/BabyLLaVA-V2) repo.
Clone that repo, then edit its training scripts to point at this data.

`--image_folder` is the same directory you passed to `--out` above.
The JSON files reference image paths relative to `--image_folder`, so both
values must be set for images to load correctly.

**Phase 1** uses `merged_single_image.json`:

```bash
--image_folder /path/to/storage \
--data_path /path/to/storage/78233_merged_pretrain/merged_single_image.json
```

**Phase 2** uses `merged_mix_format.json`:

```bash
--image_folder /path/to/storage \
--data_path /path/to/storage/78233_merged_pretrain/merged_mix_format.json
```

Per-dataset JSONs are also downloaded into `78177_SAYCam_pretrain/` and
`78187_BabyView_pretrain/` (each with `saycam_`/`babyview_` prefixed versions of
`single_image`, `multi_image`, `interleaved`, and `mix_format`) for experimentation
with individual data sources.

---

## Track 2 — Raw pretraining data

`track2_download.py` downloads the raw utterance JSONs and unfiltered
video files for both SAYCam and BabyView.

**Step 1 — Download raw metadata** (volume 1981, folder 78240, ~3.5 GB):

```bash
python track2_download.py get 1981 78240 --out /path/to/storage
```

Downloads into `/path/to/storage/78240_raw_metadata/`:

- `vtc2_whisper` — VTC/WhisperX transcripts
- `brouhaha` — Brouhaha voice activity and speech metadata
- `metadata` — participant metadata
- `README` — PDF describing each file in detail

**Step 2 — Download SAYCam videos** (volume 564, 463 sessions, ~450 GB):

```bash
python track2_download.py get 564 --sessions --flat --out /path/to/saycam_videos
```

The `--flat` flag saves all files directly into `/path/to/saycam_videos/` with no
subdirectory, so each file is `/path/to/saycam_videos/{video_id}.mp4`. This lets
`saycam_utterances.json` index directly into the downloaded files:

```python
video_path = Path("/path/to/saycam_videos") / (record["video_id"] + ".mp4")
```

Progress is saved continuously — re-run the same command to resume if interrupted.

**Step 3 — Download BabyView videos** (volume 1882, 51 sessions, 16,303 files, ~342 GB):

```bash
python track2_download.py get 1882 --sessions --flat --out /path/to/babyview_videos
```

With `--flat`, each file lands at `/path/to/babyview_videos/{rec_id}.mp4` and can
be looked up from `babyview_utterances.json` via the `video_id` field.

**Step 4 — Split into train/val/test**

`track2_split.py` cuts the raw videos into short clips using the predefined challenge
splits. Split JSONs are downloaded automatically from Databrary on first run and
cached in `~/.cache/databrary/splits/`.

```bash
python track2_split.py saycam   --video-dir /path/to/saycam_videos  --out /path/to/clips
python track2_split.py babyview --video-dir /path/to/babyview_videos --out /path/to/clips
```

If you downloaded the videos manually from the Databrary UI, point `--video-dir` at
the flat folder containing the `.mp4` files (the same layout steps 2 and 3 produce
with `--flat`).

If you also downloaded the split JSONs manually (from volume 1981, folder 78269),
pass `--splits-dir` to skip the automatic Databrary download. The folder must contain
`saycam_split.json` and `babyview_split.json`:

```bash
python track2_split.py saycam --splits-dir /path/to/splits --video-dir /path/to/saycam_videos --out /path/to/clips
```

Clips are saved to `{out}/{train|val|test}/{saycam|babyview}/{video_stem}__{start}_{end}.mp4`.
The script is resumable — already-extracted clips are skipped on re-run.
Use `--dry-run` to preview what would be extracted without running ffmpeg.

---

## Download SFT data (all tracks!)

The SFT fine-tuning data ([DevCV Toolbox](https://github.com/ShawnKing98/DevCV-Toolbox))
is used for both tracks. It lives in volume 1981 as six finetune folders (SAYCam +
BabyView, train/val/test).

```bash
python sft_download.py --out /path/to/storage
```

Each folder is extracted into a `{folder_id}_{folder_name}/` subdirectory under `--out`.
After downloading, the script automatically creates `merged_devcv/` under `--out` with one
JSON per task×split (e.g. `compare_real_train.json`) combining SAYCam and BabyView entries,
with image paths prefixed by their source folder so they resolve relative to `--out`.
