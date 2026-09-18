# Testing Instructions

Always test against `output/downloads/source_video.mp4` unless the user explicitly requests another source. The video argument is optional because this path is the built-in default.

Use `configs/layouts/source_video.json` for this broadcast. For a new broadcast style, add a separate layout under `configs/layouts/`; never overwrite the current calibration.

Interpreter: `c:/Users/17602/Desktop/Madden-Scout/.venv/Scripts/python.exe`

## Batch Regression Run

Batch mode is the accuracy reference because it votes across multiple frames per spot:

```powershell
c:/Users/17602/Desktop/Madden-Scout/.venv/Scripts/python.exe src/video_intake.py --start-seconds 0 --max-duration-seconds 420 --sample-every-n-frames 90 --max-frames 140 --ocr-workers 4
```

After the fresh scan writes `frame_states.json`, append `--reuse-state-cache` for fast logic-only iterations.

Expected artifacts:

- `output/football_play_data.csv`
- `output/per_play_records.json`
- `output/frame_metadata.csv`
- `output/play_boundaries.json`
- `output/calibration_report.csv`
- `output/calibration_report.json`

## Live Monitor Run

```powershell
c:/Users/17602/Desktop/Madden-Scout/.venv/Scripts/python.exe src/video_intake.py --live --sample-every-n-frames 30
```

Live mode displays scanned regions and parsed state, appends completed plays to the CSV, and stops when `q` is pressed. It is forward-only and therefore noisier than batch mode. Use batch output for final analysis.

## Calibration Policy

- Truth set: `configs/calibration/source_video_first_14.json`
- Use labels only to generate expected-vs-observed reports and guide detector improvements.
- Never copy truth values into `football_play_data.csv` or `per_play_records.json`.
- Track matched, mismatched, and missing fields separately.
- Preserve unlabeled values as unlabeled rather than treating them as failures.

## Performance And Diagnostics

Tesseract is the bottleneck. Keep smoke tests at 60 frames or fewer. Lowering `--sample-every-n-frames` improves coverage but increases runtime.

- `tools/roi_probe.py <frame.jpg>` probes individual OCR regions.
- `tools/state_dump.py` prints parsed state for sampled frames.
