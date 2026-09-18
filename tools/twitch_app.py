"""Download or record Twitch streams, then run the Madden analysis pipeline."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import video_intake as vi


def is_twitch_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return host == "twitch.tv" or host.endswith(".twitch.tv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "urls", nargs="+", help="Twitch VOD, clip, or live-channel URLs"
    )
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--record-live", action="store_true")
    parser.add_argument("--clip-seconds", type=int, default=None)
    parser.add_argument(
        "--cookies-from-browser",
        choices=(
            "brave",
            "chrome",
            "chromium",
            "edge",
            "firefox",
            "opera",
            "safari",
            "vivaldi",
            "whale",
        ),
        default=None,
    )
    parser.add_argument(
        "--layout-config",
        default="configs/layouts/source_video.json",
    )
    parser.add_argument(
        "--calibration-file",
        default="configs/calibration/source_video_first_14.json",
    )
    parser.add_argument("--sample-every-n-frames", type=int, default=30)
    parser.add_argument("--ocr-workers", type=int, default=4)
    parser.add_argument("--tesseract-cmd", default=None)
    parser.add_argument("--no-refine-calls", action="store_true")
    args = parser.parse_args()

    invalid_urls = [url for url in args.urls if not is_twitch_url(url)]
    if invalid_urls:
        parser.error(f"Only Twitch URLs are accepted: {', '.join(invalid_urls)}")
    if args.clip_seconds is not None and args.clip_seconds <= 0:
        parser.error("--clip-seconds must be greater than 0")
    if args.sample_every_n_frames <= 0:
        parser.error("--sample-every-n-frames must be greater than 0")
    if args.ocr_workers <= 0:
        parser.error("--ocr-workers must be greater than 0")
    return args


def analysis_command(args: argparse.Namespace, video_path: str) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "src" / "video_intake.py"),
        video_path,
        "--output-dir",
        args.output_dir,
        "--layout-config",
        args.layout_config,
        "--calibration-file",
        args.calibration_file,
        "--sample-every-n-frames",
        str(args.sample_every_n_frames),
        "--ocr-workers",
        str(args.ocr_workers),
    ]
    if args.tesseract_cmd:
        command.extend(("--tesseract-cmd", args.tesseract_cmd))
    if args.no_refine_calls:
        command.append("--no-refine-calls")
    if getattr(args, "show_analysis_window", False):
        command.append("--live")
    if getattr(args, "embed_analysis_preview", False):
        preview_path = str(Path(args.output_dir) / "live_monitor_preview.png")
        command.extend(
            ("--live", "--no-live-window", "--live-preview-path", preview_path)
        )
    return command


def main() -> None:
    args = parse_args()
    vi.initialize_team_database(args.output_dir)
    for position, url in enumerate(args.urls, start=1):
        action = "Recording" if args.record_live else "Downloading"
        print(f"[{position}/{len(args.urls)}] {action}: {url}")
        video_path = vi.download_video(
            url,
            args.output_dir,
            clip_seconds=args.clip_seconds,
            record_live=args.record_live,
            cookies_from_browser=args.cookies_from_browser,
        )
        print(f"Saved recording: {video_path}")
        subprocess.run(analysis_command(args, video_path), check=True)

    print(
        f"Completed {len(args.urls)} Twitch job(s). Final CSVs are in "
        f"{Path(args.output_dir) / vi.GAME_DATA_DIRECTORY}; team copies are in "
        f"{Path(args.output_dir) / vi.TEAM_DATA_DIRECTORY}."
    )


if __name__ == "__main__":
    main()
