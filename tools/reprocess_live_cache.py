"""Rebuild live-monitor outputs from a saved raw frame-state cache."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import video_intake as vi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--video",
        default="output/downloads/source_video.mp4",
    )
    parser.add_argument(
        "--layout-config",
        default="configs/layouts/source_video.json",
    )
    parser.add_argument(
        "--calibration-file",
        default="configs/calibration/source_video_first_14.json",
    )
    parser.add_argument("--no-refine-calls", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vi.configure_tesseract(None)
    cache_path = args.output_dir / "frame_states.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    if cache.get("version") != vi.FRAME_STATE_CACHE_VERSION:
        raise RuntimeError(
            f"Unsupported state cache version {cache.get('version')}; "
            f"expected {vi.FRAME_STATE_CACHE_VERSION}"
        )

    states = [dict(state) for state in cache.get("states", [])]
    summaries = vi.build_situation_summaries(states)
    if not args.no_refine_calls:
        layout = vi.load_layout_config(args.layout_config)
        if layout is None:
            raise RuntimeError(f"Could not load layout config: {args.layout_config}")
        regions = vi.build_element_regions(
            layout["scoreboard"], layout["play_call_banner"]
        )
        vi.refine_summary_calls(args.video, summaries, regions)

    plays = [
        play
        for play in vi.plays_from_summaries(summaries)
        if vi.has_game_state_anchor(play)
    ]
    vi.write_football_csv(plays, str(args.output_dir), finalized=True)
    with (args.output_dir / "per_play_records.json").open(
        "w", encoding="utf-8"
    ) as output_file:
        json.dump(plays, output_file, indent=2)
    vi.write_calibration_report(
        plays,
        args.calibration_file,
        str(args.output_dir),
    )
    print(f"Reprocessed {len(states)} states into {len(plays)} plays")


if __name__ == "__main__":
    main()
