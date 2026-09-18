---
description: "Use when editing the Madden video ingestion, OCR, play segmentation, football state, CSV/JSON output, live monitor, or calibration code."
applyTo: "src/**/*.py,configs/**/*.json,tools/**/*.py"
---

# Madden Video Pipeline Instructions

## Data Integrity

Keep extracted data and labeled truth separate. Production records must come from video evidence. Calibration labels may produce reports but may not patch, backfill, renumber, or replace observed records.

Retain explicit unknown values for fields that are not visually observable. Do not convert OCR noise into confident football facts.

Never hardcode a controlled team or matchup-specific team alias. Read both abbreviations from the scoreboard and identify possession from the away/home possession dot. Use transitions only to bridge frames where the dot is temporarily hidden.

## Stream Perspective

A game can alternate between offensive and defensive stream perspectives. Model `stream_side` per play, not globally.

- Offensive stream: the streamer selects the offensive play.
- Defensive stream: the streamer selects the defensive play; offensive personnel is visible only if the offense selects first.
- Previous-play banners may be absent at a possession change or special-teams transition.

Treat a missing banner at these boundaries as valid state. Detect possession changes from game-state transitions and special-teams/scoring context, not from banner presence alone.

Validate visible previous-play OCR against the offline AceMadden Madden 27 play-name cache. If the banner was not visible, report `PREVIOUS PLAY BANNER NOT SHOWN` instead of inventing a call. Preserve unverified raw OCR and never use the catalog as evidence that a play occurred.

## Play Construction

- Treat game-clock labels and video timestamps as separate time domains. Locate labeled plays by scoreboard sequence, not by interpreting `7+` as seven minutes into the file.
- Group sampled frames into stable football spots using down, distance, yard line, clock, score, and possession evidence.
- Pair the previous spot with the new spot; the new screen's `PREVIOUS PLAY` calls describe the play that just ended.
- Preserve previous offensive/defensive calls separately from current offensive/defensive calls.
- Identify run, pass, PAT, kickoff, touchdown, incomplete, turnover on downs, and possession-change states when evidence supports them.
- Signed yardage requires possession direction plus own/opponent territory. Do not use absolute yard-line difference as a universal yardage formula.

## Required Output Fields

Keep CSV and JSON aligned and include: play number, timestamps, quarter/clock, teams/scores, possession team, stream side, own/opponent field side, down/distance, yard line, personnel, formation, previous calls, current calls, play type, result, signed yards, possession change, and end situation.

## Source Truth Regression

Use `configs/calibration/source_video_first_14.json` as the first-drive regression set. Write `output/calibration_report.json` and `.csv` with `match`, `mismatch`, and `missing` per field. Improve the producer when the report exposes an error.

For repeated calibration work, build `frame_states.json` once and use `--reuse-state-cache` when sampled-frame signatures match.
