import json
import os
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import twitch_watcher
from twitch_watcher_gui import (
    active_analysis_channel,
    should_start_recording,
    split_channel_statuses,
)


class TwitchWatcherTests(unittest.TestCase):
    def test_gui_resumes_stale_recording_after_restart(self) -> None:
        state = {"streams": {"stream-1": {"status": "recording"}}}

        self.assertTrue(should_start_recording("stream-1", state, {}))

    def test_gui_does_not_duplicate_active_recording(self) -> None:
        active_job = MagicMock()
        active_job.is_alive.return_value = True
        state = {"streams": {"stream-1": {"status": "recording"}}}

        self.assertFalse(
            should_start_recording("stream-1", state, {"stream-1": active_job})
        )

    def test_gui_monitor_selects_only_actively_analyzing_video(self) -> None:
        channel = active_analysis_channel(
            {
                "streams": {
                    "one": {"channel": "first", "status": "recording"},
                    "two": {
                        "channel": "second",
                        "status": "analyzing",
                    },
                    "three": {"channel": "third", "status": "completed"},
                }
            }
        )

        self.assertEqual("second", channel)

    def test_gui_monitor_is_none_without_active_analysis(self) -> None:
        channel = active_analysis_channel(
            {"streams": {"one": {"channel": "first", "status": "recording"}}}
        )

        self.assertIsNone(channel)

    def test_gui_splits_streaming_and_offline_channels_in_config_order(self) -> None:
        streaming, offline = split_channel_statuses(
            ["first", "second", "third"],
            {"second": {"id": "stream-2"}},
        )

        self.assertEqual(["second"], streaming)
        self.assertEqual(["first", "third"], offline)

    def run_recording_worker(
        self,
        directory: str,
        video_path: Path,
        run_side_effect=None,
    ) -> dict:
        state = {"streams": {"stream-1": {"status": "recording"}}}
        state_path = Path(directory) / "state.json"
        config = {"output_dir": directory}
        with (
            patch("twitch_watcher.vi.download_video", return_value=str(video_path)),
            patch("twitch_watcher.subprocess.run", side_effect=run_side_effect),
        ):
            twitch_watcher.record_and_analyze(
                "example",
                "stream-1",
                config,
                state,
                state_path,
                threading.Lock(),
                threading.Lock(),
            )
        return state["streams"]["stream-1"]

    def test_successful_analysis_deletes_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "recording.mp4"
            video_path.write_bytes(b"video")

            stream_state = self.run_recording_worker(directory, video_path)

            self.assertFalse(video_path.exists())
        self.assertEqual("completed", stream_state["status"])
        self.assertFalse(stream_state["video_retained"])
        self.assertIn("video_deleted_at", stream_state)

    def test_failed_analysis_retains_recording(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video_path = Path(directory) / "recording.mp4"
            video_path.write_bytes(b"video")

            stream_state = self.run_recording_worker(
                directory, video_path, RuntimeError("analysis failed")
            )

            self.assertTrue(video_path.exists())
        self.assertEqual("analysis_failed", stream_state["status"])
        self.assertTrue(stream_state["video_retained"])

    def test_load_config_normalizes_and_deduplicates_channels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watcher.json"
            path.write_text(
                json.dumps(
                    {
                        "channels": [" Example ", "example", "Second"],
                        "poll_seconds": 30,
                    }
                ),
                encoding="utf-8",
            )

            config = twitch_watcher.load_config(path)

        self.assertEqual(["example", "second"], config["channels"])
        self.assertEqual(30, config["poll_seconds"])

    def test_load_config_rejects_polling_faster_than_fifteen_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watcher.json"
            path.write_text(
                json.dumps({"channels": ["example"], "poll_seconds": 10}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "at least 15"):
                twitch_watcher.load_config(path)

    @patch("twitch_watcher.request_json")
    def test_fetch_live_streams_indexes_results_by_login(self, request_json) -> None:
        request_json.return_value = {
            "data": [
                {
                    "id": "stream-1",
                    "user_login": "Example",
                    "title": "Madden",
                }
            ]
        }

        streams = twitch_watcher.fetch_live_streams(
            ["example", "offline"], "client", "token"
        )

        self.assertEqual("stream-1", streams["example"]["id"])
        requested_url = request_json.call_args.args[0]
        self.assertIn("user_login=example", requested_url)
        self.assertIn("user_login=offline", requested_url)

    @patch("twitch_watcher.YoutubeDL")
    def test_public_poll_returns_stable_live_stream_id(self, youtube_dl) -> None:
        downloader = youtube_dl.return_value.__enter__.return_value
        downloader.extract_info.return_value = {
            "id": "example",
            "is_live": True,
            "release_timestamp": 123456,
            "title": "Madden",
        }

        streams = twitch_watcher.fetch_public_live_streams(["example"])

        self.assertEqual("public-example-example-123456", streams["example"]["id"])

    @patch("twitch_watcher.YoutubeDL")
    def test_public_poll_ignores_offline_channel(self, youtube_dl) -> None:
        downloader = youtube_dl.return_value.__enter__.return_value
        downloader.extract_info.side_effect = twitch_watcher.DownloadError(
            "The channel is not currently live"
        )

        streams = twitch_watcher.fetch_public_live_streams(["example"])

        self.assertEqual({}, streams)

    def test_state_round_trip_is_atomic_and_preserves_stream_ids(self) -> None:
        state = {"streams": {"stream-1": {"status": "recording"}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"

            twitch_watcher.save_state(path, state, threading.Lock())
            loaded = twitch_watcher.load_state(path)

        self.assertEqual(state, loaded)

    @patch.dict(
        os.environ,
        {"TWITCH_CLIENT_ID": "client", "TWITCH_CLIENT_SECRET": "secret"},
    )
    @patch("twitch_watcher.threading.Thread")
    @patch("twitch_watcher.fetch_live_streams")
    @patch("twitch_watcher.acquire_app_token")
    @patch("twitch_watcher.vi.initialize_team_database")
    @patch("twitch_watcher.load_state")
    @patch("twitch_watcher.load_config")
    @patch("twitch_watcher.parse_args")
    def test_once_reports_status_without_starting_recording(
        self,
        parse_args,
        load_config,
        load_state,
        initialize_team_database,
        acquire_app_token,
        fetch_live_streams,
        thread_class,
    ) -> None:
        parse_args.return_value = Namespace(
            config=Path("config.json"), state=Path("state.json"), once=True
        )
        load_config.return_value = {
            "channels": ["example"],
            "poll_seconds": 60,
            "output_dir": "output",
        }
        load_state.return_value = {"streams": {}}
        acquire_app_token.return_value = ("token", float("inf"))
        fetch_live_streams.return_value = {
            "example": {"id": "stream-1", "user_login": "example"}
        }
        initialize_team_database.return_value = []
        thread_class.return_value = MagicMock()

        twitch_watcher.main()

        thread_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
