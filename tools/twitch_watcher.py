"""Watch Twitch channels, record live broadcasts, and analyze completed games."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from twitch_app import analysis_command

import video_intake as vi

TOKEN_URL = "https://id.twitch.tv/oauth2/token"
STREAMS_URL = "https://api.twitch.tv/helix/streams"


class QuietYdlLogger:
    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        pass

    def error(self, message: str) -> None:
        pass


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    channels = config.get("channels")
    if not isinstance(channels, list) or not channels:
        raise RuntimeError("Watcher config must contain a non-empty channels list")
    normalized_channels = []
    for channel in channels:
        normalized = str(channel).strip().lower()
        if normalized and normalized not in normalized_channels:
            normalized_channels.append(normalized)
    if not normalized_channels:
        raise RuntimeError("Watcher config does not contain any valid channels")
    config["channels"] = normalized_channels
    poll_seconds = int(config.get("poll_seconds", 60))
    if poll_seconds < 15:
        raise RuntimeError("poll_seconds must be at least 15")
    config["poll_seconds"] = poll_seconds
    return config


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"streams": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state.get("streams"), dict):
        raise RuntimeError(f"Invalid watcher state: {path}")
    return state


def save_state(path: Path, state: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        temporary_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary_path.replace(path)


def request_json(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(url, method=method, headers=headers or {})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def acquire_app_token(client_id: str, client_secret: str) -> tuple[str, float]:
    query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        }
    )
    payload = request_json(f"{TOKEN_URL}?{query}", method="POST")
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("Twitch did not return an application access token")
    expires_in = int(payload.get("expires_in", 3600))
    return str(token), time.time() + max(60, expires_in - 300)


def fetch_live_streams(
    channels: list[str], client_id: str, access_token: str
) -> dict[str, dict[str, Any]]:
    live_streams: dict[str, dict[str, Any]] = {}
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Client-Id": client_id,
    }
    for offset in range(0, len(channels), 100):
        query = urllib.parse.urlencode(
            [("user_login", channel) for channel in channels[offset : offset + 100]]
        )
        payload = request_json(f"{STREAMS_URL}?{query}", headers=headers)
        for stream in payload.get("data", []):
            login = str(stream.get("user_login", "")).lower()
            stream_id = str(stream.get("id", ""))
            if login and stream_id:
                live_streams[login] = stream
    return live_streams


def fetch_public_live_streams(channels: list[str]) -> dict[str, dict[str, Any]]:
    live_streams: dict[str, dict[str, Any]] = {}
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "logger": QuietYdlLogger(),
    }
    for channel in channels:
        try:
            with YoutubeDL(options) as downloader:
                info = downloader.extract_info(
                    f"https://www.twitch.tv/{channel}", download=False
                )
        except DownloadError as error:
            message = str(error).lower()
            if "not currently live" in message or "offline" in message:
                continue
            raise
        if not info or not info.get("is_live"):
            continue
        started_at = info.get("release_timestamp") or info.get("timestamp")
        source_id = info.get("id") or channel
        stream_id = f"public-{channel}-{source_id}-{started_at or 'live'}"
        live_streams[channel] = {
            "id": stream_id,
            "user_login": channel,
            "title": info.get("title"),
            "started_at": started_at,
        }
    return live_streams


def worker_args(config: dict[str, Any]) -> Namespace:
    return Namespace(
        output_dir=str(config.get("output_dir", "output")),
        layout_config=str(
            config.get("layout_config", "configs/layouts/source_video.json")
        ),
        calibration_file=str(
            config.get(
                "calibration_file",
                "configs/calibration/source_video_first_14.json",
            )
        ),
        sample_every_n_frames=int(config.get("sample_every_n_frames", 30)),
        ocr_workers=int(config.get("ocr_workers", 4)),
        tesseract_cmd=config.get("tesseract_cmd"),
        no_refine_calls=bool(config.get("no_refine_calls", False)),
        show_analysis_window=bool(config.get("show_analysis_window", True)),
        embed_analysis_preview=bool(config.get("embed_analysis_preview", True)),
    )


def record_and_analyze(
    channel: str,
    stream_id: str,
    config: dict[str, Any],
    state: dict[str, Any],
    state_path: Path,
    state_lock: threading.Lock,
    analysis_lock: threading.Lock,
) -> None:
    stream_state = state["streams"][stream_id]
    try:
        video_path = vi.download_video(
            f"https://www.twitch.tv/{channel}",
            str(config.get("output_dir", "output")),
            clip_seconds=None,
            record_live=True,
            cookies_from_browser=config.get("cookies_from_browser"),
        )
    except Exception as error:
        stream_state["status"] = "recording_failed"
        stream_state["error"] = str(error)
        stream_state["failed_at"] = datetime.now(timezone.utc).isoformat()
        save_state(state_path, state, state_lock)
        return

    stream_state["recording_path"] = video_path
    stream_state["recording_completed_at"] = datetime.now(timezone.utc).isoformat()
    stream_state["status"] = "awaiting_analysis"
    save_state(state_path, state, state_lock)

    try:
        with analysis_lock:
            stream_state["status"] = "analyzing"
            save_state(state_path, state, state_lock)
            subprocess.run(
                analysis_command(worker_args(config), video_path), check=True
            )
    except Exception as error:
        stream_state["status"] = "analysis_failed"
        stream_state["error"] = str(error)
        stream_state["failed_at"] = datetime.now(timezone.utc).isoformat()
        stream_state["video_retained"] = True
        save_state(state_path, state, state_lock)
        return

    stream_state["analysis_completed_at"] = datetime.now(timezone.utc).isoformat()
    try:
        Path(video_path).unlink()
    except Exception as error:
        stream_state["status"] = "cleanup_failed"
        stream_state["cleanup_error"] = str(error)
        stream_state["video_retained"] = True
        save_state(state_path, state, state_lock)
        return

    stream_state["status"] = "completed"
    stream_state["video_retained"] = False
    stream_state["video_deleted_at"] = datetime.now(timezone.utc).isoformat()
    stream_state["completed_at"] = datetime.now(timezone.utc).isoformat()
    save_state(state_path, state, state_lock)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/twitch_watcher.json", type=Path)
    parser.add_argument(
        "--state", default="output/twitch_watcher_state.json", type=Path
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once, report live channels without recording, and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    state = load_state(args.state)
    client_id = os.environ.get("TWITCH_CLIENT_ID")
    client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
    use_helix = bool(client_id and client_secret)

    output_dir = str(config.get("output_dir", "output"))
    vi.initialize_team_database(output_dir)
    state_lock = threading.Lock()
    analysis_lock = threading.Lock()
    active_jobs: dict[str, threading.Thread] = {}
    access_token = ""
    token_expires_at = 0.0

    poll_method = "Twitch Helix" if use_helix else "public yt-dlp polling"
    print(f"Watching {len(config['channels'])} Twitch channel(s) via {poll_method}")
    try:
        while True:
            try:
                if use_helix and time.time() >= token_expires_at:
                    access_token, token_expires_at = acquire_app_token(
                        client_id, client_secret
                    )
                if use_helix:
                    live_streams = fetch_live_streams(
                        config["channels"], client_id, access_token
                    )
                else:
                    live_streams = fetch_public_live_streams(config["channels"])
            except Exception as error:
                if args.once:
                    raise
                token_expires_at = 0.0
                print(f"Twitch poll failed; retrying: {error}", file=sys.stderr)
                time.sleep(config["poll_seconds"])
                continue
            if args.once:
                if live_streams:
                    for channel, stream in live_streams.items():
                        print(f"LIVE: {channel} (stream {stream['id']})")
                else:
                    print("No configured channels are currently live")
                break
            for channel, stream in live_streams.items():
                stream_id = str(stream["id"])
                if stream_id in state["streams"]:
                    continue
                state["streams"][stream_id] = {
                    "channel": channel,
                    "title": stream.get("title"),
                    "started_at": stream.get("started_at"),
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                    "status": "recording",
                }
                save_state(args.state, state, state_lock)
                job = threading.Thread(
                    target=record_and_analyze,
                    args=(
                        channel,
                        stream_id,
                        config,
                        state,
                        args.state,
                        state_lock,
                        analysis_lock,
                    ),
                    name=f"twitch-{channel}-{stream_id}",
                    daemon=False,
                )
                active_jobs[stream_id] = job
                job.start()
                print(f"Started recording {channel} (stream {stream_id})")

            active_jobs = {
                stream_id: job
                for stream_id, job in active_jobs.items()
                if job.is_alive()
            }
            time.sleep(config["poll_seconds"])
    except KeyboardInterrupt:
        print("Watcher stopping; active recordings will finish before shutdown")
    finally:
        for job in active_jobs.values():
            job.join()


if __name__ == "__main__":
    main()
