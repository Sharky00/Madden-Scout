"""Desktop dashboard for the always-on Twitch recording watcher."""

import argparse
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from discord_twitch_discovery import start_discovery_thread
from twitch_watcher import (
    acquire_app_token,
    fetch_live_streams,
    fetch_public_live_streams,
    load_config,
    load_state,
    record_and_analyze,
    save_state,
)

import video_intake as vi


def split_channel_statuses(
    channels: list[str], live_streams: dict[str, dict[str, Any]]
) -> tuple[list[str], list[str]]:
    streaming = [channel for channel in channels if channel in live_streams]
    offline = [channel for channel in channels if channel not in live_streams]
    return streaming, offline


def active_analysis_channel(state: dict[str, Any]) -> str | None:
    for stream in state.get("streams", {}).values():
        if stream.get("status") == "analyzing":
            return str(stream.get("channel") or "Unknown channel")
    return None


def should_start_recording(
    stream_id: str,
    state: dict[str, Any],
    active_jobs: dict[str, threading.Thread],
) -> bool:
    if stream_id in active_jobs and active_jobs[stream_id].is_alive():
        return False
    existing = state.get("streams", {}).get(stream_id)
    return existing is None or existing.get("status") == "recording"


class WatcherDashboard:
    def __init__(self, root: tk.Tk, config_path: Path, state_path: Path) -> None:
        self.root = root
        self.config_path = config_path
        self.config = load_config(config_path)
        self.state_path = state_path
        self.state = load_state(state_path)
        self.state_lock = threading.Lock()
        self.analysis_lock = threading.Lock()
        self.active_jobs: dict[str, threading.Thread] = {}
        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.poll_in_progress = False
        self.access_token = ""
        self.token_expires_at = 0.0
        self.client_id = os.environ.get("TWITCH_CLIENT_ID")
        self.client_secret = os.environ.get("TWITCH_CLIENT_SECRET")
        self.use_helix = bool(self.client_id and self.client_secret)
        self.last_live_streams: dict[str, dict[str, Any]] = {}
        self.discord_thread: threading.Thread | None = None
        self.preview_path = (
            Path(str(self.config.get("output_dir", "output")))
            / "live_monitor_preview.png"
        )
        self.preview_image: tk.PhotoImage | None = None
        self.preview_mtime_ns: int | None = None

        vi.initialize_team_database(str(self.config.get("output_dir", "output")))
        self.build_window()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(100, self.process_events)
        self.root.after(250, self.start_poll)
        self.root.after(1000, self.refresh_jobs)
        self.root.after(200, self.refresh_monitor)
        discord_token = os.environ.get("DISCORD_BOT_TOKEN")
        if discord_token and self.config.get("discord_channel_id"):
            self.discord_thread = start_discovery_thread(
                self.config_path,
                discord_token,
                lambda channels: self.events.put(("channels_added", channels)),
            )

    def build_window(self) -> None:
        self.root.title("Madden Scout Twitch Watcher")
        self.root.geometry("1380x640")
        self.root.minsize(1080, 520)
        self.root.configure(background="#101820")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background="#101820")
        style.configure(
            "Header.TLabel",
            background="#101820",
            foreground="#f4f1de",
            font=("Segoe UI Semibold", 22),
        )
        style.configure(
            "Meta.TLabel",
            background="#101820",
            foreground="#a8b5bd",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Panel.TLabelframe",
            background="#17242e",
            foreground="#f4f1de",
            bordercolor="#324653",
            relief="solid",
        )
        style.configure(
            "Panel.TLabelframe.Label",
            background="#101820",
            foreground="#f4f1de",
            font=("Segoe UI Semibold", 12),
        )
        style.configure(
            "Channel.Treeview",
            background="#17242e",
            fieldbackground="#17242e",
            foreground="#edf2f4",
            rowheight=32,
            borderwidth=0,
            font=("Segoe UI", 11),
        )
        style.configure(
            "Channel.Treeview.Heading",
            background="#243743",
            foreground="#edf2f4",
            relief="flat",
            font=("Segoe UI Semibold", 10),
        )
        style.map("Channel.Treeview", background=[("selected", "#2f6f73")])

        outer = ttk.Frame(self.root, padding=22, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(header, text="Twitch Watcher", style="Header.TLabel").pack(
            side="left"
        )
        self.health_label = ttk.Label(header, text="Starting...", style="Meta.TLabel")
        self.health_label.pack(side="right", pady=(10, 0))

        lists = ttk.Frame(outer, style="App.TFrame")
        lists.pack(fill="both", expand=True)
        lists.columnconfigure(0, weight=1)
        lists.columnconfigure(1, weight=1)
        lists.columnconfigure(2, weight=2)
        lists.rowconfigure(0, weight=1)

        self.streaming_tree = self.create_channel_panel(
            lists, "Streaming", 0, "No watched channels are live"
        )
        self.offline_tree = self.create_channel_panel(
            lists, "Offline", 1, "Waiting for next poll"
        )
        self.create_monitor_panel(lists, 2)

        footer = ttk.Frame(outer, style="App.TFrame")
        footer.pack(fill="x", pady=(18, 0))
        self.queue_label = ttk.Label(
            footer, text="Recordings: 0 | Analysis queue: idle", style="Meta.TLabel"
        )
        self.queue_label.pack(side="left")
        method = "Twitch Helix" if self.use_helix else "public Twitch polling"
        self.method_label = ttk.Label(
            footer,
            text=f"{method} every {self.config['poll_seconds']} seconds",
            style="Meta.TLabel",
        )
        self.method_label.pack(side="right")

    def create_channel_panel(
        self, parent: ttk.Frame, title: str, column: int, empty_text: str
    ) -> ttk.Treeview:
        panel = ttk.LabelFrame(
            parent, text=title, padding=12, style="Panel.TLabelframe"
        )
        panel.grid(
            row=0,
            column=column,
            sticky="nsew",
            padx=(0, 8) if column == 0 else (8, 8),
        )
        tree = ttk.Treeview(
            panel,
            columns=("channel", "detail"),
            show="headings",
            style="Channel.Treeview",
        )
        tree.heading("channel", text="CHANNEL")
        tree.heading("detail", text="STATUS")
        tree.column("channel", width=170, anchor="w")
        tree.column("detail", width=220, anchor="w")
        tree.pack(fill="both", expand=True)
        tree.insert("", "end", values=(empty_text, ""), tags=("placeholder",))
        return tree

    def create_monitor_panel(self, parent: ttk.Frame, column: int) -> None:
        panel = ttk.LabelFrame(
            parent, text="OCR Monitor", padding=12, style="Panel.TLabelframe"
        )
        panel.grid(row=0, column=column, sticky="nsew", padx=(8, 0))
        self.monitor_canvas = tk.Canvas(
            panel,
            background="#000000",
            borderwidth=0,
            highlightthickness=0,
        )
        self.monitor_canvas.pack(fill="both", expand=True)
        self.monitor_canvas.bind("<Configure>", lambda _event: self.draw_monitor())

    def start_poll(self) -> None:
        if self.poll_in_progress:
            return
        self.config = load_config(self.config_path)
        self.poll_in_progress = True
        self.health_label.configure(text="Checking Twitch...")
        threading.Thread(target=self.poll_worker, daemon=True).start()

    def poll_worker(self) -> None:
        try:
            if self.use_helix:
                if time.time() >= self.token_expires_at:
                    self.access_token, self.token_expires_at = acquire_app_token(
                        str(self.client_id), str(self.client_secret)
                    )
                streams = fetch_live_streams(
                    self.config["channels"], str(self.client_id), self.access_token
                )
            else:
                streams = fetch_public_live_streams(self.config["channels"])
            self.events.put(("poll_success", streams))
        except Exception as error:
            self.token_expires_at = 0.0
            self.events.put(("poll_error", str(error)))

    def process_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "poll_success":
                self.poll_in_progress = False
                self.apply_poll(payload)
                self.root.after(self.config["poll_seconds"] * 1000, self.start_poll)
            elif event == "poll_error":
                self.poll_in_progress = False
                self.health_label.configure(text=f"Poll failed: {payload}")
                self.root.after(self.config["poll_seconds"] * 1000, self.start_poll)
            elif event == "channels_added":
                self.config = load_config(self.config_path)
                self.health_label.configure(
                    text=f"Karen discovered: {', '.join(payload)}"
                )
                if not self.poll_in_progress:
                    self.root.after(0, self.start_poll)
        self.root.after(250, self.process_events)

    def apply_poll(self, live_streams: dict[str, dict[str, Any]]) -> None:
        self.last_live_streams = live_streams
        streaming, offline = split_channel_statuses(
            self.config["channels"], live_streams
        )
        streaming_rows = [
            (channel, live_streams[channel].get("title") or "Live now")
            for channel in streaming
        ]
        offline_rows = [(channel, "Offline") for channel in offline]
        self.fill_tree(
            self.streaming_tree,
            streaming_rows,
            "No watched channels are live",
        )
        self.fill_tree(self.offline_tree, offline_rows, "Everyone is streaming")
        now = datetime.now().strftime("%I:%M:%S %p")
        self.health_label.configure(text=f"Live | Last checked {now}")
        for channel in streaming:
            self.start_recording(channel, live_streams[channel])

    def fill_tree(
        self, tree: ttk.Treeview, rows: list[tuple[str, str]], empty_text: str
    ) -> None:
        for item in tree.get_children():
            tree.delete(item)
        if not rows:
            tree.insert("", "end", values=(empty_text, ""))
            return
        for channel, detail in rows:
            tree.insert("", "end", values=(channel, detail))

    def start_recording(self, channel: str, stream: dict[str, Any]) -> None:
        stream_id = str(stream["id"])
        if not should_start_recording(stream_id, self.state, self.active_jobs):
            return
        if stream_id not in self.state["streams"]:
            self.state["streams"][stream_id] = {
                "channel": channel,
                "title": stream.get("title"),
                "started_at": stream.get("started_at"),
                "detected_at": datetime.now().astimezone().isoformat(),
                "status": "recording",
            }
        else:
            self.state["streams"][stream_id]["resumed_at"] = (
                datetime.now().astimezone().isoformat()
            )
        save_state(self.state_path, self.state, self.state_lock)
        job = threading.Thread(
            target=record_and_analyze,
            args=(
                channel,
                stream_id,
                self.config,
                self.state,
                self.state_path,
                self.state_lock,
                self.analysis_lock,
            ),
            name=f"twitch-{channel}-{stream_id}",
            daemon=False,
        )
        self.active_jobs[stream_id] = job
        job.start()

    def refresh_jobs(self) -> None:
        self.active_jobs = {
            stream_id: job
            for stream_id, job in self.active_jobs.items()
            if job.is_alive()
        }
        recording_count = sum(
            1
            for stream in self.state["streams"].values()
            if stream.get("status") == "recording"
        )
        queued_count = sum(
            1
            for stream in self.state["streams"].values()
            if stream.get("status") == "awaiting_analysis"
        )
        analyzing_count = sum(
            1
            for stream in self.state["streams"].values()
            if stream.get("status") == "analyzing"
        )
        analysis_text = "idle"
        if analyzing_count:
            analysis_text = "running"
        elif queued_count:
            analysis_text = f"{queued_count} waiting"
        self.queue_label.configure(
            text=f"Recordings: {recording_count} | Analysis queue: {analysis_text}"
        )
        self.root.after(1000, self.refresh_jobs)

    def refresh_monitor(self) -> None:
        channel = active_analysis_channel(self.state)
        if channel and self.preview_path.exists():
            mtime_ns = self.preview_path.stat().st_mtime_ns
            if mtime_ns != self.preview_mtime_ns:
                try:
                    self.preview_image = tk.PhotoImage(file=str(self.preview_path))
                    self.preview_mtime_ns = mtime_ns
                except tk.TclError:
                    pass
        else:
            self.preview_image = None
            self.preview_mtime_ns = None
        self.draw_monitor()
        self.root.after(200, self.refresh_monitor)

    def draw_monitor(self) -> None:
        self.monitor_canvas.delete("all")
        width = max(1, self.monitor_canvas.winfo_width())
        height = max(1, self.monitor_canvas.winfo_height())
        channel = active_analysis_channel(self.state)
        if channel and self.preview_image:
            source_width = self.preview_image.width()
            source_height = self.preview_image.height()
            factor = max(
                1,
                (source_width + width - 1) // width,
                (source_height + height - 1) // height,
            )
            display_image = self.preview_image.subsample(factor, factor)
            self.monitor_canvas.display_image = display_image
            self.monitor_canvas.create_image(
                width // 2, height // 2, image=display_image, anchor="center"
            )
            self.monitor_canvas.create_text(
                12,
                12,
                text=f"OCR: {channel}",
                fill="#f4f1de",
                anchor="nw",
                font=("Segoe UI Semibold", 11),
            )
            return
        self.monitor_canvas.display_image = None
        self.monitor_canvas.create_text(
            width // 2,
            height // 2,
            text="NONE",
            fill="#84939c",
            anchor="center",
            font=("Segoe UI Semibold", 28),
        )

    def close(self) -> None:
        active_count = sum(1 for job in self.active_jobs.values() if job.is_alive())
        if active_count and not messagebox.askyesno(
            "Stop watcher?",
            f"{active_count} recording or analysis job(s) are active. Stop displaying "
            "the dashboard? Active jobs will finish before the process exits.",
        ):
            return
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/twitch_watcher.json", type=Path)
    parser.add_argument(
        "--state", default="output/twitch_watcher_state.json", type=Path
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    dashboard = WatcherDashboard(root, args.config, args.state)
    root.mainloop()
    for job in dashboard.active_jobs.values():
        job.join()


if __name__ == "__main__":
    main()
