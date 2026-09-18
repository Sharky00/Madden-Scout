# Madden Scout Repository Instructions

## Product Contract

Build an evidence-driven Madden NFL 27 video-analysis pipeline that emits one football play per CSV/JSON record. Preserve the relationship between pre-play situation, offensive and defensive calls, result, and resulting situation.

Do not hardcode labeled answers into production output. Use `configs/calibration/source_video_first_14.json` only to measure expected-vs-observed accuracy and guide improvements.

Do not store matchup-specific team abbreviations, controlled-team names, formations, or play answers in production code or layout configs. Team names must come from scoreboard OCR. Possession must come from the scoreboard possession marker and football state transitions. Calibration labels may appear only under `configs/calibration/` and tests.

## Default Source

- Video: `output/downloads/source_video.mp4` (the CLI default)
- Layout: `configs/layouts/source_video.json`
- Calibration: `configs/calibration/source_video_first_14.json`
- Python: `.venv/Scripts/python.exe`
- Tesseract: `C:/Program Files/Tesseract-OCR/tesseract.exe`

## Football Semantics

- Calibration phrases such as `7+`, `6+`, `5+`, and `4+ minutes` refer to the in-game football clock, not video-file timestamps.
- `start_timestamp_seconds` and `end_timestamp_seconds` are absolute source-video positions. Nonzero `--start-seconds` runs must preserve this absolute timeline.
- Stream side can switch between `offense` and `defense` during a game.
- On a defensive stream, offensive personnel is visible only if the offense selects first.
- The `PREVIOUS PLAY` banner can disappear on a possession change, PAT, or kickoff. Absence is evidence, not an OCR error to fill with an invented call.
- The call shown at resulting spot `i` belongs to the play from spot `i-1` to spot `i`.
- Track previous calls separately from the current play's calls.
- Include special teams plays (PAT and kickoff), scoring, possession changes, stream side, personnel, formation, field side (`own`/`opponent`), and possession team in the schema even when unavailable.
- Never calculate signed yardage from `abs(end_yard-start_yard)` without knowing field side and direction. Across midfield or possession changes, that calculation can be wrong.

## Current Broadcast OCR Facts

- Bottom scoreboard strip is `y=0.9..1.0`; exact subregions are in `build_element_regions`.
- Quarter is best parsed from ordinal suffix: `ST=1`, `ND=2`, `RD=3`, `TH=4`.
- Clock OCR can add a leading digit (`17:51` for `7:51`); accept a valid trailing clock with minutes <= 15.
- Down/distance can read `1st & 10` as `157&10`; use the leading down digit and number after `&`.
- Yard-line OCR currently captures the number but not reliably the field side or team territory.
- OCR the offensive and defensive halves of the previous-play panel separately.
- The small black dot between team name and score identifies possession; detect its away/home region rather than assuming which team the streamer controls.

## Madden 27 Play-Name Validation

- Validate OCR-derived offensive and defensive call names against the AceMadden Madden 27 catalog: `https://acemadden.com/playbooks/madden27`.
- Runtime validation must use the offline cache at `configs/reference/acemadden_madden27_plays.json`; refresh it with `tools/refresh_acemadden_catalog.py` rather than depending on network access during video processing.
- Only validate or canonicalize a call when the `PREVIOUS PLAY` banner was visibly present in the source frames.
- If the banner was absent, emit `PREVIOUS PLAY BANNER NOT SHOWN`; do not infer or backfill a call from the catalog.
- Preserve raw OCR and validation status. Only replace OCR with a catalog name when the match is exact or unambiguously strong; ambiguous text stays unverified.
- The AceMadden catalog is a vocabulary constraint, not evidence that a specific play occurred.

## Validation

After changes, run the narrowest relevant parser/runtime check, then a batch regression when extraction behavior changes. A valid improvement raises calibration matches or reduces mismatches/missing fields without source-truth substitution.

Batch reference:

```powershell
.venv/Scripts/python.exe src/video_intake.py --start-seconds 0 --max-duration-seconds 420 --sample-every-n-frames 90 --max-frames 140 --ocr-workers 4
```

The first 14 labeled plays span approximately the first seven minutes of source video. Use a fresh full scan to create `frame_states.json`, then add `--reuse-state-cache` for fast segmentation/result iterations.

Use `--live --sample-every-n-frames 30` for monitoring, but treat batch output as the accuracy reference.
