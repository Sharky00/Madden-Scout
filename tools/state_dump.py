"""Dump per-frame extracted state for every sampled frame, using the real pipeline."""

import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import video_intake as vi


def main() -> None:
    vi.configure_tesseract(None)
    layout = vi.load_layout_config("configs/layouts/source_video.json")
    regions = vi.build_element_regions(layout["scoreboard"], layout["play_call_banner"])

    frames = sorted(glob.glob("output/frames/frame_*.jpg"))
    for path in frames:
        s = vi.extract_frame_state(path, regions)
        name = os.path.basename(path)
        print(
            f"{name} dd={s['down']}&{s['distance']} yd={s['yard_line']} "
            f"clk={s['game_clock']} q={s['quarter']} "
            f"off={s['prev_off_call']!r} def={s['prev_def_call']!r} "
            f"sc={s['away_score']}-{s['home_score']}"
        )


if __name__ == "__main__":
    main()
