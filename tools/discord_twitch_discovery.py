"""Discover Twitch channels from an authorized Discord bot connection."""

import argparse
import asyncio
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import discord

TWITCH_URL_PATTERN = re.compile(
    r"https?://(?:www\.)?twitch\.tv/([A-Za-z0-9_]+)", re.IGNORECASE
)
TWITCH_RESERVED_PATHS = {
    "directory",
    "downloads",
    "jobs",
    "p",
    "search",
    "settings",
    "subscriptions",
    "turbo",
    "videos",
    "wallet",
}


def extract_twitch_channels(texts: list[str]) -> list[str]:
    channels: list[str] = []
    for text in texts:
        for match in TWITCH_URL_PATTERN.finditer(text or ""):
            channel = match.group(1).lower()
            if channel not in TWITCH_RESERVED_PATHS and channel not in channels:
                channels.append(channel)
    return channels


def message_texts(message: discord.Message) -> list[str]:
    texts = [message.content]
    for embed in message.embeds:
        texts.extend(
            value
            for value in (
                embed.url,
                embed.title,
                embed.description,
                embed.author.url,
            )
            if value
        )
        texts.extend(field.value for field in embed.fields if field.value)
    return texts


def add_channels_to_config(config_path: Path, channels: list[str]) -> list[str]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    existing = config.get("channels")
    if not isinstance(existing, list):
        raise RuntimeError("Watcher config channels must be a list")
    existing_normalized = {str(channel).lower() for channel in existing}
    added = [
        channel for channel in channels if channel.lower() not in existing_normalized
    ]
    if not added:
        return []
    config["channels"].extend(added)
    temporary_path = config_path.with_suffix(f"{config_path.suffix}.tmp")
    temporary_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(config_path)
    return added


def valid_discord_channel_url(value: str) -> tuple[int, int]:
    parsed = urlparse(value)
    parts = [part for part in parsed.path.split("/") if part]
    if parsed.netloc not in {"discord.com", "discordapp.com"} or len(parts) != 3:
        raise ValueError(f"Invalid Discord channel URL: {value}")
    if parts[0] != "channels":
        raise ValueError(f"Invalid Discord channel URL: {value}")
    return int(parts[1]), int(parts[2])


class KarenDiscoveryClient(discord.Client):
    def __init__(
        self,
        config_path: Path,
        channel_id: int,
        author_name: str | None,
        on_channels_added: Callable[[list[str]], None] | None = None,
        backfill_hours: int | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.config_path = config_path
        self.channel_id = channel_id
        self.author_name = author_name.lower() if author_name else None
        self.on_channels_added = on_channels_added
        self.backfill_hours = backfill_hours

    async def on_ready(self) -> None:
        print(f"Discord discovery connected as {self.user}")
        if self.backfill_hours is not None:
            await self.backfill_history(self.backfill_hours)
            await self.close()

    def message_is_watched(self, message: discord.Message) -> bool:
        if message.channel.id != self.channel_id:
            return False
        return (
            not self.author_name
            or self.author_name in message.author.display_name.lower()
        )

    def discover_from_message(self, message: discord.Message) -> list[str]:
        if not self.message_is_watched(message):
            return []
        channels = extract_twitch_channels(message_texts(message))
        return add_channels_to_config(self.config_path, channels)

    async def backfill_history(self, hours: int) -> None:
        channel = self.get_channel(self.channel_id)
        if channel is None:
            channel = await self.fetch_channel(self.channel_id)
        if not isinstance(channel, discord.TextChannel):
            raise RuntimeError(
                f"Discord channel {self.channel_id} is not a text channel"
            )
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        discovered: list[str] = []
        async for message in channel.history(after=cutoff, oldest_first=True):
            for twitch_channel in self.discover_from_message(message):
                if twitch_channel not in discovered:
                    discovered.append(twitch_channel)
        print(
            f"Backfilled {hours} hour(s); added {len(discovered)} Twitch channel(s): "
            f"{', '.join(discovered) if discovered else 'none'}"
        )

    async def on_message(self, message: discord.Message) -> None:
        added = self.discover_from_message(message)
        if added:
            print(f"Discovered Twitch channel(s): {', '.join(added)}")
            if self.on_channels_added:
                self.on_channels_added(added)


def run_discovery_client(
    config_path: Path,
    token: str,
    on_channels_added: Callable[[list[str]], None] | None = None,
    backfill_hours: int | None = None,
) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    channel_id = int(config["discord_channel_id"])
    author_name = config.get("discord_author_name")
    client = KarenDiscoveryClient(
        config_path,
        channel_id,
        author_name,
        on_channels_added,
        backfill_hours,
    )
    client.run(token, log_handler=None)


def start_discovery_thread(
    config_path: Path,
    token: str,
    on_channels_added: Callable[[list[str]], None] | None = None,
) -> threading.Thread:
    thread = threading.Thread(
        target=run_discovery_client,
        args=(config_path, token, on_channels_added),
        name="discord-twitch-discovery",
        daemon=True,
    )
    thread.start()
    return thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/twitch_watcher.json", type=Path)
    parser.add_argument("--backfill-hours", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("Set DISCORD_BOT_TOKEN before starting Discord discovery")
    if args.backfill_hours is not None and args.backfill_hours <= 0:
        raise RuntimeError("--backfill-hours must be greater than 0")
    try:
        run_discovery_client(args.config, token, backfill_hours=args.backfill_hours)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    main()
