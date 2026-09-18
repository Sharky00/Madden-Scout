import os
import sys
import unittest
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import twitch_app


class TwitchAppTests(unittest.TestCase):
    def test_twitch_url_accepts_channels_vods_and_clips(self) -> None:
        self.assertTrue(twitch_app.is_twitch_url("https://www.twitch.tv/channel"))
        self.assertTrue(twitch_app.is_twitch_url("https://www.twitch.tv/videos/123456"))
        self.assertTrue(twitch_app.is_twitch_url("https://clips.twitch.tv/example"))
        self.assertFalse(twitch_app.is_twitch_url("https://example.com/video"))

    def test_analysis_command_runs_local_video_through_pipeline(self) -> None:
        args = Namespace(
            output_dir="output",
            layout_config="configs/layouts/source_video.json",
            calibration_file="configs/calibration/source_video_first_14.json",
            sample_every_n_frames=30,
            ocr_workers=4,
            tesseract_cmd=None,
            no_refine_calls=False,
        )

        command = twitch_app.analysis_command(args, "output/downloads/vod.mp4")

        self.assertEqual(sys.executable, command[0])
        self.assertTrue(command[1].endswith(os.path.join("src", "video_intake.py")))
        self.assertEqual("output/downloads/vod.mp4", command[2])
        self.assertIn("--output-dir", command)
        self.assertIn("--layout-config", command)

    def test_analysis_command_enables_visible_monitor_when_requested(self) -> None:
        args = Namespace(
            output_dir="output",
            layout_config="configs/layouts/source_video.json",
            calibration_file="configs/calibration/source_video_first_14.json",
            sample_every_n_frames=30,
            ocr_workers=4,
            tesseract_cmd=None,
            no_refine_calls=False,
            show_analysis_window=True,
        )

        command = twitch_app.analysis_command(args, "output/downloads/vod.mp4")

        self.assertIn("--live", command)

    def test_analysis_command_embeds_monitor_without_separate_window(self) -> None:
        args = Namespace(
            output_dir="output",
            layout_config="configs/layouts/source_video.json",
            calibration_file="configs/calibration/source_video_first_14.json",
            sample_every_n_frames=30,
            ocr_workers=4,
            tesseract_cmd=None,
            no_refine_calls=False,
            show_analysis_window=False,
            embed_analysis_preview=True,
        )

        command = twitch_app.analysis_command(args, "output/downloads/vod.mp4")

        self.assertIn("--live", command)
        self.assertIn("--no-live-window", command)
        preview_index = command.index("--live-preview-path") + 1
        self.assertEqual(
            os.path.join("output", "live_monitor_preview.png"),
            command[preview_index],
        )


if __name__ == "__main__":
    unittest.main()
