import csv
import json
import os
import re
from typing import Any

DIRECT_FIELDS = (
    "stream_side",
    "possession_team",
    "field_side",
    "quarter",
    "game_clock_bucket",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "down",
    "distance",
    "yard_line",
    "personnel",
    "formation",
    "previous_offensive_play_call",
    "previous_defensive_play_call",
    "offensive_play_call",
    "defensive_play_call",
    "play_type",
    "yards_gained",
    "result",
    "possession_change",
    "end_down_distance",
    "end_yard_line",
    "end_possession_team",
)


def load_calibration(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as calibration_file:
        payload = json.load(calibration_file)
    return list(payload.get("plays", []))


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Z0-9]+", " ", str(value).upper()).strip()


def values_match(field: str, expected: Any, observed: Any) -> bool:
    if expected is None:
        return True
    if observed is None:
        return False
    if field in {
        "formation",
        "previous_offensive_play_call",
        "previous_defensive_play_call",
        "offensive_play_call",
        "defensive_play_call",
        "result",
    }:
        expected_text = normalize_text(expected)
        observed_text = normalize_text(observed)
        expected_compact = expected_text.replace(" ", "")
        observed_compact = observed_text.replace(" ", "")
        return (
            expected_text == observed_text
            or expected_text in observed_text
            or observed_text in expected_text
            or expected_compact == observed_compact
        )
    return expected == observed


def expected_value(play: dict[str, Any], field: str) -> Any:
    if field == "game_clock_bucket":
        return play.get("clock_bucket")
    if field == "distance" and play.get("distance_label") == "inches":
        return 0
    if field != "end_down_distance":
        return play.get(field)

    down = play.get("end_down")
    distance = play.get("end_distance")
    label = play.get("end_distance_label")
    if down is None:
        return None
    if label:
        return f"{down} & {label}"
    if distance is not None:
        return f"{down} & {distance}"
    return None


def observed_value(play: dict[str, Any], field: str) -> Any:
    if field != "game_clock_bucket":
        return play.get(field)
    clock = play.get("game_clock")
    match = re.match(r"^(\d{1,2}):\d{2}$", str(clock or ""))
    return f"{int(match.group(1))}+" if match else None


def play_alignment_score(expected: dict[str, Any], observed: dict[str, Any]) -> float:
    weights = {
        "quarter": 1.0,
        "away_score": 2.0,
        "home_score": 2.0,
        "down": 4.0,
        "distance": 3.0,
        "yard_line": 4.0,
        "offensive_play_call": 3.0,
        "defensive_play_call": 3.0,
    }
    score = 0.0
    compared = 0
    for field, weight in weights.items():
        expected_field = expected.get(field)
        observed_field = observed.get(field)
        if expected_field is None or observed_field is None:
            continue
        compared += 1
        score += (
            weight
            if values_match(field, expected_field, observed_field)
            else -weight * 0.75
        )
    return score if compared >= 2 else -3.0


def align_plays(
    expected_plays: list[dict[str, Any]],
    observed_plays: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    expected_count = len(expected_plays)
    observed_count = len(observed_plays)
    gap_penalty = -3.0
    scores = [[0.0] * (observed_count + 1) for _ in range(expected_count + 1)]
    moves = [[""] * (observed_count + 1) for _ in range(expected_count + 1)]

    for expected_index in range(1, expected_count + 1):
        scores[expected_index][0] = expected_index * gap_penalty
        moves[expected_index][0] = "skip_expected"
    for observed_index in range(1, observed_count + 1):
        scores[0][observed_index] = observed_index * gap_penalty
        moves[0][observed_index] = "skip_observed"

    for expected_index in range(1, expected_count + 1):
        for observed_index in range(1, observed_count + 1):
            candidates = {
                "match": scores[expected_index - 1][observed_index - 1]
                + play_alignment_score(
                    expected_plays[expected_index - 1],
                    observed_plays[observed_index - 1],
                ),
                "skip_expected": scores[expected_index - 1][observed_index]
                + gap_penalty,
                "skip_observed": scores[expected_index][observed_index - 1]
                + gap_penalty,
            }
            move, best_score = max(candidates.items(), key=lambda item: item[1])
            scores[expected_index][observed_index] = best_score
            moves[expected_index][observed_index] = move

    aligned: dict[int, dict[str, Any]] = {}
    expected_index = expected_count
    observed_index = observed_count
    while expected_index > 0 or observed_index > 0:
        move = moves[expected_index][observed_index]
        if move == "match":
            expected = expected_plays[expected_index - 1]
            observed = observed_plays[observed_index - 1]
            if play_alignment_score(expected, observed) > 0:
                aligned[int(expected["play_number"])] = observed
            expected_index -= 1
            observed_index -= 1
        elif move == "skip_expected":
            expected_index -= 1
        else:
            observed_index -= 1
    return aligned


def build_calibration_report(
    observed_plays: list[dict[str, Any]],
    expected_plays: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_by_number = align_plays(expected_plays, observed_plays)

    comparisons: list[dict[str, Any]] = []
    matched = 0
    mismatched = 0
    missing = 0
    ignored = 0

    for expected_play in expected_plays:
        play_number = int(expected_play["play_number"])
        observed_play = observed_by_number.get(play_number)

        for field in DIRECT_FIELDS:
            expected = expected_value(expected_play, field)
            if expected is None:
                ignored += 1
                continue

            observed = observed_value(observed_play, field) if observed_play else None
            if observed_play is None or observed is None:
                status = "missing"
                missing += 1
            elif values_match(field, expected, observed):
                status = "match"
                matched += 1
            else:
                status = "mismatch"
                mismatched += 1

            comparisons.append(
                {
                    "play_number": play_number,
                    "observed_play_number": observed_play.get("play_number")
                    if observed_play
                    else None,
                    "stream_side": expected_play.get("stream_side"),
                    "possession_team": expected_play.get("possession_team"),
                    "field": field,
                    "expected": expected,
                    "observed": observed,
                    "status": status,
                }
            )

    comparable = matched + mismatched + missing
    return {
        "summary": {
            "expected_plays": len(expected_plays),
            "observed_plays": len(observed_plays),
            "matched_fields": matched,
            "mismatched_fields": mismatched,
            "missing_fields": missing,
            "ignored_unlabeled_fields": ignored,
            "field_accuracy": round(matched / comparable, 4) if comparable else 0.0,
        },
        "comparisons": comparisons,
    }


def write_calibration_report(
    observed_plays: list[dict[str, Any]],
    calibration_path: str,
    output_dir: str,
) -> tuple[str, str, dict[str, Any]] | None:
    expected_plays = load_calibration(calibration_path)
    if not expected_plays:
        return None

    report = build_calibration_report(observed_plays, expected_plays)
    json_path = os.path.join(output_dir, "calibration_report.json")
    csv_path = os.path.join(output_dir, "calibration_report.csv")

    with open(json_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=(
                "play_number",
                "observed_play_number",
                "stream_side",
                "possession_team",
                "field",
                "expected",
                "observed",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows(report["comparisons"])

    return json_path, csv_path, report
