"""Diagnostic: crop candidate scoreboard sub-regions from a frame and OCR them.

Grounds ROI fractions against a known frame so extraction is fact-based, not guessed.
Run: python tools/roi_probe.py output/layout_debug/frame_0501.jpg
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import video_intake as vi

# Candidate regions as full-frame fractions (x, y, w, h), grounded on the 5:01 frame.
CANDIDATES = {
    "team_away": (0.292, 0.902, 0.056, 0.046),
    "score_away": (0.340, 0.900, 0.040, 0.055),
    "score_home": (0.412, 0.900, 0.040, 0.055),
    "team_home": (0.463, 0.902, 0.050, 0.046),
    "quarter": (0.663, 0.902, 0.038, 0.048),
    "game_clock": (0.700, 0.900, 0.052, 0.055),
    "play_clock": (0.752, 0.900, 0.046, 0.055),
    "down_distance": (0.815, 0.900, 0.100, 0.055),
    "yard_line": (0.918, 0.900, 0.062, 0.055),
    "prev_play_def": (0.715, 0.175, 0.128, 0.045),
    "prev_play_off": (0.845, 0.175, 0.128, 0.045),
}

WHITELISTS = {
    "team_away": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "team_home": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "score_away": "0123456789",
    "score_home": "0123456789",
    "quarter": "1234",
    "game_clock": "0123456789:",
    "play_clock": "0123456789:",
    "down_distance": "0123456789STNDRDTH&AND ",
    "yard_line": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
    "prev_play_def": None,
    "prev_play_off": None,
}

PSMS = {
    "score_away": 10,
    "score_home": 10,
    "quarter": 10,
    "game_clock": 7,
    "play_clock": 10,
    "down_distance": 7,
    "yard_line": 7,
    "team_away": 8,
    "team_home": 8,
    "prev_play_def": 7,
    "prev_play_off": 7,
}


def main() -> None:
    frame_path = (
        sys.argv[1] if len(sys.argv) > 1 else "output/layout_debug/frame_0501.jpg"
    )
    vi.configure_tesseract(None)
    out_dir = "output/roi_probe"
    os.makedirs(out_dir, exist_ok=True)

    frame = cv2.imread(frame_path)
    print(f"frame: {frame_path} shape={None if frame is None else frame.shape}")

    for name, (x, y, w, h) in CANDIDATES.items():
        region = vi.Region(x=x, y=y, w=w, h=h)
        crop = vi.crop_region(frame, region)
        cv2.imwrite(os.path.join(out_dir, f"{name}.png"), crop)
        text = vi.ocr_text_from_crop(
            crop,
            whitelist=WHITELISTS.get(name),
            psm=PSMS.get(name, 7),
        )
        print(f"{name:16s} -> '{text}'")


if __name__ == "__main__":
    main()
