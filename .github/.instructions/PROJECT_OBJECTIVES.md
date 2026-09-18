# Project Objectives

## Mission

Build a Madden play-call intelligence system that converts Madden NFL 27 gameplay video into a structured, queryable database of game situations, play calls, and outcomes.

## Project Scope Clarification

The application is designed specifically for Madden NFL 27 gameplay recordings.

Input will generally be a recorded Madden NFL 27 game from a supported source such as YouTube, Twitch, or a directly supplied video file.

Because the source material is Madden NFL 27 gameplay rather than real-world NFL broadcast footage, the system should be optimized around the game's consistent user interface, scoreboard, field presentation, play-selection screens, animations, and other visual elements.

The application should use these consistent elements to reconstruct game state and associate each play with the corresponding play call and play outcome.

## Primary Objectives

- Ingest Madden NFL 27 gameplay video from supported sources.
- Detect and segment individual plays across a full game recording.
- Reconstruct game-state context for each play (clock, down, distance, field position, score, etc.).
- Identify or infer the offensive play call and key descriptive attributes.
- Capture the play result and link it to the associated game state and play call.
- Store all extracted data in a query-friendly structure for trend and tendency analysis.

## Madden-Specific Data To Capture

For each play, the system should attempt to capture:

- Offensive team
- Defensive team
- Quarter
- Game clock
- Down
- Distance
- Score
- Ball position
- Previous play
- Previous play result
- Current play call
- Play result
- Yards gained or lost
- Turnovers
- Penalties
- Timeouts (where relevant)
- Drive information
- Formation (where identifiable)
- Personnel (where identifiable)
- Motion (where identifiable)
- Play type
- Relevant Madden gameplay metadata

The system should leverage Madden's on-screen interface to extract information that is typically difficult to obtain from real football broadcasts.

## Madden Play-Call Analysis

The core purpose is to build a database of Madden NFL 27 play calls organized by game situation.

Example situation:

- 3rd and 6
- Ball on opponent 35-yard line
- 2:14 remaining
- Down by 4
- Previous play: 2-yard run
- Offensive team: Team A

Example observed play:

- Formation: Shotgun
- Personnel: 11
- Play: Inside Zone
- Result: 5-yard gain

The database should preserve these relationships so the same situation can be compared across hundreds or thousands of plays.

## Success Criteria

- Produce per-play records with consistent game-state, play-call, and result linkage.
- Support retrieval and comparison of plays by situation (for example, down-distance-score-time contexts).
- Enable tendency analysis across teams, players, previous plays, formations, and outcomes.
- Provide enough accuracy and coverage to support practical scouting and decision support.

## Long-Term Objective

The resulting dataset should make it possible to analyze Madden play-calling behavior based on game state.

Example queries include:

- What play is most commonly called on 3rd and 6?
- What does the offense call after an incomplete pass?
- What plays are called when leading late in the fourth quarter?
- What plays are called in the red zone?
- What formations are most common in specific situations?
- What does a particular opponent or player tend to call?
- What play follows a particular previous play?
- What is the success rate of each play in a given situation?
- What play is most likely to be called given current game state and previous play?

The long-term goal is a Madden play-call intelligence database that can be queried to identify tendencies and relationships between game situations, previous plays, and subsequent play calls.

## Constraints

- Initial scope is limited to Madden NFL 27 gameplay captured from video.
- Real NFL broadcast analysis is explicitly out of scope for the initial release.
- The pipeline should be tuned for Madden NFL 27's visual environment and presentation style.
- Stream perspective can switch between offense and defense during the same game.
- A defensive stream may expose offensive personnel only when the offense selects its play first.
- The previous-play banner may be absent during changes of possession, PATs, and kickoffs.
- Missing visual evidence must remain missing or low-confidence; labeled calibration data must never silently overwrite extracted output.

## Calibration Objective

Use `configs/calibration/source_video_first_14.json` as the labeled source of truth for the first 14 observed states in the current source video. Every batch run should compare extracted records against this file and write an expected-vs-observed report.

Calibration is for measuring and improving the OCR, state tracking, play segmentation, stream-side detection, and play-result logic. It is not an output override. A change is successful only when the detector independently reproduces more labeled fields.

## Scope Boundary

The initial system will focus only on Madden NFL 27 gameplay captured from video.

It will not initially attempt to analyze real NFL broadcasts.

This restriction is intentional and allows the computer-vision and data-extraction pipeline to be optimized for the Madden NFL 27 environment.

Future versions may support additional Madden versions or real-world football footage.
