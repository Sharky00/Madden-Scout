import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

from discord_twitch_discovery import (
    add_channels_to_config,
    extract_twitch_channels,
    valid_discord_channel_url,
)


class DiscordTwitchDiscoveryTests(unittest.TestCase):
    def test_extracts_unique_channels_and_ignores_reserved_paths(self) -> None:
        channels = extract_twitch_channels(
            [
                "Live: https://www.twitch.tv/Example_Player",
                "https://twitch.tv/example_player https://twitch.tv/videos/123",
                "Embed https://www.twitch.tv/SecondOne",
            ]
        )

        self.assertEqual(["example_player", "secondone"], channels)

    def test_add_channels_persists_only_new_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "watcher.json"
            path.write_text(json.dumps({"channels": ["Existing"]}), encoding="utf-8")

            added = add_channels_to_config(path, ["existing", "new_channel", "another"])
            config = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(["new_channel", "another"], added)
        self.assertEqual(["Existing", "new_channel", "another"], config["channels"])

    def test_parses_supplied_discord_channel_url(self) -> None:
        guild_id, channel_id = valid_discord_channel_url(
            "https://discordapp.com/channels/746087307748048916/746090210994946149"
        )

        self.assertEqual(746087307748048916, guild_id)
        self.assertEqual(746090210994946149, channel_id)


if __name__ == "__main__":
    unittest.main()
