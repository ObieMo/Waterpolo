# Annotator

This folder contains the annotation helper used to align event rows from an Excel sheet with timestamps in a match video.

## Purpose

The annotator scans a match video, reads the on-screen game clock with OCR, and matches those clock readings to event rows from the Excel event log.

The main script is:

- `Annotator/main.py`

There is also a small ROI selection helper:

- `Annotator/picker.py`

## Inputs

`main.py` expects these files in the current working directory:

- `match.mp4`
  - the source video to scan
- `match.xlsx`
  - the Excel file containing the event log

The script currently reads:

- sheet name: `Event_Log_For_Annotation`

The Excel sheet is expected to contain event rows with enough information to build:

- period
- game clock
- event code
- suggested label
- team
- player
- cap number

## Process

The annotation flow is:

1. Load the event rows from `match.xlsx`.
2. Ask you for the start and end time of each quarter in video time using `HH:MM:SS`.
3. Open the video and ask you to select the scoreboard clock ROI.
4. Sample the video second by second inside the quarter windows.
5. For each sampled second, run OCR multiple times on nearby frames.
6. Vote on the most likely clock reading.
7. Match OCR clock readings against pending event rows for the same period and clock.
8. Save matched rows as JSON annotations.
9. Save progress state so the scan can be resumed later.

Hotkeys during scanning:

- `r`
  - repick the clock ROI
- `q`
  - stop safely and save progress

## Outputs

`main.py` writes:

- `annotations.json`
  - matched annotations with event metadata and video timestamps
- `scan_state.json`
  - saved progress, quarter windows, next scan index, and ROI

Each annotation entry includes fields such as:

- `row_id`
- `period`
- `clock`
- `code`
- `suggested_label`
- `video_time_sec`
- `video_time_hhmmss`
- `frame_idx`
- `ocr_text`
- `vote_hits`
- `valid_reads`

## Conversion For The Pipeline

The generated `annotations.json` file is an intermediate annotation file, not the final format expected by the spotting pipeline.

Why conversion is needed:

- `annotations.json` contains a lot of extra fields used for annotation and OCR debugging
- the training pipeline only needs a compact `Labels.json`-style structure
- the current spotting setup only uses two labels:
  - `GOAL`
  - `MissedShot`

To convert the annotator output into pipeline-ready JSON, use:

- `Annotator/Converter/convert.py`

What `convert.py` does:

1. Reads the annotation rows from input JSON files.
2. Keeps only rows whose event code belongs to one of the supported classes.
3. Maps raw event codes to the 2 pipeline labels.
4. Drops all unused metadata fields.
5. Writes a compact output JSON with:
   - `gameTime`
   - `label`
   - `position`
   - `timeSec`
   - `code`

Current code mapping in `convert.py`:

- `GA`, `GE`, `GC`, `5` -> `GOAL`
- `SA`, `MX`, `BR` -> `MissedShot`

Everything else is ignored.

The output format is:

```json
{
  "annotations": [
    {
      "gameTime": "00:12:34",
      "label": "GOAL",
      "position": "12345",
      "timeSec": 754,
      "code": "GA"
    }
  ]
}
```

This converted file is the one you should then use as the label file for the spotting pipeline.

## Running

Run from the `Annotator` folder:

```bash
python main.py
```

If you only want to pick and inspect the clock ROI on the first frame:

```bash
python picker.py
```

To convert a folder of annotator JSON files into pipeline-ready JSON files:

```bash
cd Converter
python convert.py
```

By default, `convert.py` reads from:

- `input_jsons`

and writes to:

- `output_jsons`

## Notes

- The OCR model is loaded with `torch.hub` from the PARSeq repository.
- The annotator resumes automatically if `scan_state.json` already exists.
- The script does not directly write final `Labels.json` files for training; it produces intermediate matched annotations in `annotations.json`, which should be converted with `Annotator/Converter/convert.py`.
