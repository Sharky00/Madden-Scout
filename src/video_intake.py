import argparse
import csv
import difflib
import glob
import json
import os
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2
from yt_dlp import YoutubeDL

from calibration import write_calibration_report

FRAME_STATE_CACHE_VERSION = 5
GAME_DATA_DIRECTORY = "Game_Data"
TEAM_DATA_DIRECTORY = "Team_Data"
TEAM_REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "reference",
    "nfl_team_abbreviations.json",
)
PLAY_CATALOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "configs",
    "reference",
    "acemadden_madden27_plays.json",
)
CALL_BANNER_NOT_SHOWN = "PREVIOUS PLAY BANNER NOT SHOWN"

try:
    import pytesseract
except ImportError:
    pytesseract = None


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_seconds: float
    output_path: str


@dataclass
class BoundaryRecord:
    frame_index: int
    timestamp_seconds: float
    diff_score: float


@dataclass
class Region:
    x: float
    y: float
    w: float
    h: float


def is_tesseract_ready() -> bool:
    if pytesseract is None:
        return False

    try:
        _ = pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def configure_tesseract(tesseract_cmd: str | None) -> None:
    if pytesseract is None:
        return

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        return

    common_paths = [
        "C:/Program Files/Tesseract-OCR/tesseract.exe",
        "C:/Program Files (x86)/Tesseract-OCR/tesseract.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def region_to_rect(
    region: Region, frame_width: int, frame_height: int
) -> tuple[int, int, int, int]:
    x1 = int(clamp01(region.x) * frame_width)
    y1 = int(clamp01(region.y) * frame_height)
    x2 = int(clamp01(region.x + region.w) * frame_width)
    y2 = int(clamp01(region.y + region.h) * frame_height)
    if x2 <= x1:
        x2 = min(frame_width, x1 + 1)
    if y2 <= y1:
        y2 = min(frame_height, y1 + 1)
    return x1, y1, x2, y2


def crop_region(frame: Any, region: Region) -> Any:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = region_to_rect(region, frame_width, frame_height)
    return frame[y1:y2, x1:x2]


def region_to_dict(region: Region) -> dict[str, float]:
    return {
        "x": round(region.x, 6),
        "y": round(region.y, 6),
        "w": round(region.w, 6),
        "h": round(region.h, 6),
    }


def expand_region(region: Region, pad_x: float, pad_y: float) -> Region:
    return Region(
        x=clamp01(region.x - pad_x),
        y=clamp01(region.y - pad_y),
        w=clamp01(region.w + (2 * pad_x)),
        h=clamp01(region.h + (2 * pad_y)),
    )


def dict_to_region(data: dict[str, Any], default: Region) -> Region:
    try:
        return Region(
            x=float(data.get("x", default.x)),
            y=float(data.get("y", default.y)),
            w=float(data.get("w", default.w)),
            h=float(data.get("h", default.h)),
        )
    except Exception:
        return default


def auto_locate_text_region(
    frame: Any,
    y_start: float,
    y_end: float,
    min_aspect_ratio: float,
    min_width_ratio: float,
    center_tolerance: float | None = None,
) -> Region | None:
    height, width = frame.shape[:2]
    band_top = int(height * y_start)
    band_bottom = int(height * y_end)
    if band_bottom <= band_top:
        return None
    band = frame[band_top:band_bottom, :]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    _, binary = cv2.threshold(abs_grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_rect = None
    best_score = -1.0
    for contour in contours:
        x, y, width_value, height_value = cv2.boundingRect(contour)
        if width_value < width * min_width_ratio or height_value < band.shape[0] * 0.08:
            continue
        aspect_ratio = width_value / max(1, height_value)
        if aspect_ratio < min_aspect_ratio:
            continue
        area_score = float(width_value * height_value)
        center_offset = abs((x + width_value / 2) - width / 2) / (width / 2)
        if center_tolerance is not None and center_offset > center_tolerance:
            continue
        score = area_score * (0.8 + 0.2 * (1.0 - center_offset))
        if score > best_score:
            best_score = score
            best_rect = (x, y, width_value, height_value)
    if best_rect is None:
        return None
    x, y, width_value, height_value = best_rect
    return Region(
        x=clamp01(x / width),
        y=clamp01((band_top + y) / height),
        w=clamp01(width_value / width),
        h=clamp01(height_value / height),
    )


def discover_layout_from_frame(frame: Any) -> dict[str, Region]:
    default_scoreboard = Region(x=0.10, y=0.00, w=0.80, h=0.17)
    default_playcall = Region(x=0.20, y=0.72, w=0.60, h=0.20)
    discovered_scoreboard = auto_locate_text_region(
        frame,
        y_start=0.00,
        y_end=0.28,
        min_aspect_ratio=2.0,
        min_width_ratio=0.10,
    )
    if discovered_scoreboard is not None:
        discovered_scoreboard = expand_region(
            discovered_scoreboard, pad_x=0.03, pad_y=0.01
        )
    discovered_playcall = auto_locate_text_region(
        frame,
        y_start=0.55,
        y_end=0.95,
        min_aspect_ratio=1.5,
        min_width_ratio=0.40,
        center_tolerance=0.25,
    )
    return {
        "scoreboard": discovered_scoreboard or default_scoreboard,
        "play_call_banner": discovered_playcall or default_playcall,
    }


def median_region(regions: list[Region], fallback: Region) -> Region:
    if not regions:
        return fallback
    return Region(
        x=float(statistics.median(r.x for r in regions)),
        y=float(statistics.median(r.y for r in regions)),
        w=float(statistics.median(r.w for r in regions)),
        h=float(statistics.median(r.h for r in regions)),
    )


def discover_layout_from_records(records: list[FrameRecord]) -> dict[str, Region]:
    default_layout = {
        "scoreboard": Region(x=0.10, y=0.00, w=0.80, h=0.17),
        "play_call_banner": Region(x=0.20, y=0.72, w=0.60, h=0.20),
    }
    if not records:
        return default_layout

    sample_count = min(6, len(records))
    sample_indexes = sorted(
        {
            int(i * (len(records) - 1) / max(1, sample_count - 1))
            for i in range(sample_count)
        }
    )

    scoreboard_regions: list[Region] = []
    playcall_regions: list[Region] = []

    for idx in sample_indexes:
        frame = cv2.imread(records[idx].output_path)
        if frame is None:
            continue
        discovered = discover_layout_from_frame(frame)
        scoreboard_regions.append(discovered["scoreboard"])
        playcall_regions.append(discovered["play_call_banner"])

    return {
        "scoreboard": median_region(scoreboard_regions, default_layout["scoreboard"]),
        "play_call_banner": median_region(
            playcall_regions, default_layout["play_call_banner"]
        ),
    }


def load_layout_config(layout_path: str) -> dict[str, Region] | None:
    if not layout_path:
        return None
    if not os.path.exists(layout_path):
        return None

    with open(layout_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    regions = data.get("regions", {})
    scoreboard_default = Region(x=0.10, y=0.00, w=0.80, h=0.17)
    playcall_default = Region(x=0.20, y=0.72, w=0.60, h=0.20)
    return {
        "scoreboard": dict_to_region(regions.get("scoreboard", {}), scoreboard_default),
        "play_call_banner": dict_to_region(
            regions.get("play_call_banner", {}), playcall_default
        ),
    }


def save_layout_config(layout_path: str, layout_regions: dict[str, Region]) -> None:
    layout_dir = os.path.dirname(layout_path)
    if layout_dir:
        os.makedirs(layout_dir, exist_ok=True)
    payload = {
        "layout_name": os.path.splitext(os.path.basename(layout_path))[0],
        "regions": {
            name: region_to_dict(region) for name, region in layout_regions.items()
        },
    }
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_layout_debug_images(
    output_dir: str, sample_frame_path: str, layout_regions: dict[str, Region]
) -> None:
    frame = cv2.imread(sample_frame_path)
    if frame is None:
        return

    debug_dir = os.path.join(output_dir, "layout_debug")
    os.makedirs(debug_dir, exist_ok=True)

    for region_name, region in layout_regions.items():
        crop = crop_region(frame, region)
        if crop is None or crop.size == 0:
            continue
        cv2.imwrite(os.path.join(debug_dir, f"{region_name}.jpg"), crop)


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def download_video(
    url: str,
    output_dir: str,
    clip_seconds: int | None,
    record_live: bool = False,
    cookies_from_browser: str | None = None,
) -> str:
    downloads_dir = os.path.join(output_dir, "downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    outtmpl = os.path.join(
        downloads_dir,
        "%(extractor)s_%(id)s_%(upload_date)s.%(ext)s",
    )
    ydl_options = {
        "outtmpl": outtmpl,
        "format": "best[ext=mp4]/best",
        "noplaylist": True,
        "quiet": False,
    }

    if clip_seconds is not None:
        ydl_options["download_sections"] = f"*0-{clip_seconds}"
    if record_live:
        ydl_options["live_from_start"] = True
    if cookies_from_browser:
        ydl_options["cookiesfrombrowser"] = (cookies_from_browser,)

    with YoutubeDL(ydl_options) as ydl:
        info = ydl.extract_info(url, download=True)
        prepared_path = ydl.prepare_filename(info)

    if os.path.exists(prepared_path):
        return prepared_path

    base_path, _ = os.path.splitext(prepared_path)
    candidates = glob.glob(f"{base_path}.*")
    if not candidates:
        raise RuntimeError("Download completed but local video file was not found")

    return candidates[0]


def extract_frames(
    video_path: str,
    output_dir: str,
    sample_every_n_frames: int,
    max_frames: int | None,
    start_seconds: float,
    max_duration_seconds: float | None,
) -> list[FrameRecord]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    if start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)

    start_frame_index = int(round(start_seconds * fps))

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    records: list[FrameRecord] = []
    relative_frame_index = 0
    saved_count = 0

    while True:
        success, frame = capture.read()
        if not success:
            break

        if relative_frame_index % sample_every_n_frames == 0:
            timestamp_seconds = start_seconds + (relative_frame_index / fps)
            if (
                max_duration_seconds is not None
                and timestamp_seconds - start_seconds > max_duration_seconds
            ):
                break

            absolute_frame_index = start_frame_index + relative_frame_index
            filename = f"frame_{absolute_frame_index:06d}.jpg"
            frame_path = os.path.join(frames_dir, filename)
            cv2.imwrite(frame_path, frame)

            records.append(
                FrameRecord(
                    frame_index=absolute_frame_index,
                    timestamp_seconds=timestamp_seconds,
                    output_path=frame_path,
                )
            )
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        relative_frame_index += 1

    capture.release()
    return records


def read_gray(frame_path: str) -> Any:
    frame = cv2.imread(frame_path)
    if frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def compute_frame_diff_score(
    previous_frame_path: str, current_frame_path: str
) -> float:
    prev_gray = read_gray(previous_frame_path)
    curr_gray = read_gray(current_frame_path)
    if prev_gray is None or curr_gray is None:
        return 0.0

    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean())


def detect_play_boundaries(
    records: list[FrameRecord],
    diff_threshold: float,
    min_boundary_gap_samples: int,
) -> list[BoundaryRecord]:
    boundaries: list[BoundaryRecord] = []
    last_boundary_idx = -10_000

    for sample_idx in range(1, len(records)):
        previous_record = records[sample_idx - 1]
        current_record = records[sample_idx]
        diff_score = compute_frame_diff_score(
            previous_record.output_path,
            current_record.output_path,
        )

        if diff_score < diff_threshold:
            continue

        if sample_idx - last_boundary_idx < min_boundary_gap_samples:
            continue

        boundaries.append(
            BoundaryRecord(
                frame_index=current_record.frame_index,
                timestamp_seconds=current_record.timestamp_seconds,
                diff_score=diff_score,
            )
        )
        last_boundary_idx = sample_idx

    return boundaries


def run_region_ocr(
    frame_path: str, region: Region, whitelist: str | None, psm: int = 6
) -> dict[str, Any]:
    ocr_ready = is_tesseract_ready()
    frame = cv2.imread(frame_path)
    if frame is None:
        return {"ocr_available": ocr_ready, "raw_text": ""}

    crop = crop_region(frame, region)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    raw_text = ""
    if ocr_ready:
        try:
            raw_text = ocr_text_from_crop(threshold, whitelist=whitelist, psm=psm)
        except Exception:
            raw_text = ""
    return {
        "ocr_available": ocr_ready,
        "raw_text": " ".join(raw_text.upper().split()),
    }


def ocr_text_from_crop(crop: Any, whitelist: str | None, psm: int) -> str:
    if pytesseract is None:
        return ""

    base_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    upscaled = cv2.resize(
        base_gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )
    blur = cv2.GaussianBlur(upscaled, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(otsu)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )

    best_text = ""
    for variant in (otsu, inv, adaptive):
        config = f"--psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        text = pytesseract.image_to_string(variant, config=config)
        text = " ".join(text.upper().split())
        if len(text) > len(best_text):
            best_text = text
    return best_text


def ocr_region_from_frame(
    frame: Any,
    region: Region,
    whitelist: str | None,
    psm: int,
    variants: tuple[str, ...] = ("otsu", "inv"),
) -> str:
    if pytesseract is None:
        return ""
    crop = crop_region(frame, region)
    if crop is None or crop.size == 0:
        return ""

    base_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    upscaled = cv2.resize(
        base_gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
    )
    blur = cv2.GaussianBlur(upscaled, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    prepared = {"otsu": otsu, "inv": cv2.bitwise_not(otsu)}
    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    for name in variants:
        image = prepared.get(name)
        if image is None:
            continue
        try:
            text = " ".join(
                pytesseract.image_to_string(image, config=config).upper().split()
            )
        except Exception:
            text = ""
        if text:
            return text
    return ""


def subregion(
    parent: Region, rel_x: float, rel_y: float, rel_w: float, rel_h: float
) -> Region:
    return Region(
        x=clamp01(parent.x + (parent.w * rel_x)),
        y=clamp01(parent.y + (parent.h * rel_y)),
        w=clamp01(parent.w * rel_w),
        h=clamp01(parent.h * rel_h),
    )


def estimate_scoreboard_visibility(frame_path: str, scoreboard_region: Region) -> float:
    frame = cv2.imread(frame_path)
    if frame is None:
        return 0.0

    strip = crop_region(frame, scoreboard_region)
    if strip is None or strip.size == 0:
        return 0.0

    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 60, 180)

    dark_ratio = float((binary == 0).mean())
    bright_ratio = float((binary == 255).mean())
    edge_density = float((edges > 0).mean())

    right_state_region = subregion(scoreboard_region, 0.64, 0.0, 0.36, 0.58)
    right_text = run_region_ocr(frame_path, right_state_region, whitelist=None, psm=7)[
        "raw_text"
    ]
    has_clock = bool(re.search(r"\b[0-5]?\d:[0-5]\d\b", right_text))
    has_down_distance = bool(
        re.search(r"\b[1-4][A-Z]{0,2}\s*&\s*\d{1,2}\b", right_text)
    )

    score = 0.0
    score += min(1.0, edge_density / 0.12) * 0.4
    score += min(1.0, dark_ratio / 0.45) * 0.2
    score += min(1.0, bright_ratio / 0.35) * 0.2
    if has_clock:
        score += 0.2
    if has_down_distance:
        score += 0.2

    return max(0.0, min(1.0, score))


def filter_records_by_scoreboard_visibility(
    records: list[FrameRecord],
    scoreboard_region: Region,
    min_visibility_score: float,
) -> tuple[list[FrameRecord], int]:
    kept: list[FrameRecord] = []
    skipped = 0

    for record in records:
        visibility = estimate_scoreboard_visibility(
            record.output_path, scoreboard_region
        )
        if visibility >= min_visibility_score:
            kept.append(record)
        else:
            skipped += 1

    return kept, skipped


def clock_to_seconds(clock_value: str | None) -> int | None:
    if not clock_value:
        return None
    match = re.match(r"^([0-5]?\d):([0-5]\d)$", clock_value)
    if not match:
        return None
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    return (minutes * 60) + seconds


def seconds_to_clock(total_seconds: int) -> str:
    total_seconds = max(0, min(15 * 60, total_seconds))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


def truncate_at_possession_change(plays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not plays:
        return plays

    first_team: str | None = None
    for play in plays:
        team = play.get("offense")
        if team:
            first_team = team
            break

    if not first_team:
        return plays

    result: list[dict[str, Any]] = []
    for play in plays:
        team = play.get("offense")
        if team and team != first_team:
            break
        result.append(play)

    for idx, play in enumerate(result):
        play["play_number"] = idx

    return result


# --- Broadcast element extraction (calibrated for the source_video Madden layout) ---
# Each element is expressed relative to its parent config region so recalibrating the
# scoreboard/banner region moves the whole set together for a different broadcast.


def build_element_regions(
    scoreboard_region: Region, banner_region: Region
) -> dict[str, Region]:
    sr = scoreboard_region
    br = banner_region
    return {
        "team_away": subregion(sr, 0.292, 0.02, 0.056, 0.46),
        "team_home": subregion(sr, 0.463, 0.02, 0.050, 0.46),
        "possession_away": Region(0.333, 0.917, 0.010, 0.020),
        "possession_home": Region(0.438, 0.917, 0.010, 0.020),
        "score_away": subregion(sr, 0.340, 0.00, 0.040, 0.55),
        "score_home": subregion(sr, 0.412, 0.00, 0.040, 0.55),
        "quarter": subregion(sr, 0.663, 0.02, 0.038, 0.48),
        "game_clock": subregion(sr, 0.700, 0.00, 0.052, 0.55),
        "play_clock": subregion(sr, 0.752, 0.00, 0.046, 0.55),
        "down_distance": subregion(sr, 0.815, 0.00, 0.100, 0.55),
        "yard_line": Region(0.952, 0.900, 0.045, 0.055),
        "field_side_indicator": Region(0.940, 0.915, 0.018, 0.035),
        "play_call_banner": br,
        "prev_def": subregion(br, 0.00, 0.57, 0.50, 0.25),
        "prev_off": subregion(br, 0.50, 0.57, 0.50, 0.25),
        "defense_header": Region(0.790, 0.245, 0.155, 0.035),
        "defense_formation": Region(0.735, 0.277, 0.205, 0.035),
        "defense_personnel": Region(0.748, 0.302, 0.170, 0.035),
        "offense_formation": Region(0.105, 0.440, 0.135, 0.030),
    }


def parse_quarter_text(text: str) -> int | None:
    t = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not t:
        return None
    for suffix, quarter in (("ND", 2), ("RD", 3), ("TH", 4), ("ST", 1)):
        if t.endswith(suffix) or suffix in t:
            return quarter
    match = re.search(r"[1-4]", t)
    return int(match.group()) if match else None


def parse_clock_text(text: str) -> str | None:
    for minutes, seconds in re.findall(r"(\d{1,2}):(\d{2})", text):
        if int(minutes) <= 15 and int(seconds) < 60:
            return f"{int(minutes)}:{seconds}"
    # OCR sometimes prefixes a phantom digit (e.g. "17:51" for 7:51); recover MM:SS.
    for minutes, seconds in re.findall(r"(\d):(\d{2})", text):
        if int(seconds) < 60:
            return f"{minutes}:{seconds}"
    return None


def parse_down_distance_text(text: str) -> tuple[int | None, int | None]:
    t = text.upper()
    match = re.search(r"([1-4]).*?&\s*(\d{1,2})", t)
    if match is None:
        match = re.search(r"([1-4]).*?AND\s*(\d{1,2})", t)
    if match is None:
        inches_match = re.search(r"([1-4]).*?(?:&|AND).*?IN(?:CH(?:ES)?)?", t)
        if inches_match:
            return int(inches_match.group(1)), 0
        return None, None
    down = int(match.group(1))
    distance = int(match.group(2))
    if not (0 <= distance <= 40):
        return down, None
    return down, distance


def parse_special_state(text: str) -> str | None:
    normalized = normalize_text(text)
    if "KICKOFF" in normalized:
        return "kickoff"
    if re.search(r"\bPAT\b", normalized):
        return "extra_point"
    return None


def format_down_distance(down: int | None, distance: int | None) -> str | None:
    if down is None or distance is None:
        return None
    distance_text = "inches" if distance == 0 else str(distance)
    return f"{down} & {distance_text}"


def parse_yard_text(text: str) -> int | None:
    numbers = re.findall(r"\d{1,2}", text)
    if not numbers:
        return None
    value = int(numbers[-1])
    return value if 1 <= value <= 50 else None


def parse_int_text(text: str) -> int | None:
    match = re.search(r"\d{1,3}", text)
    return int(match.group()) if match else None


def parse_team_text(text: str) -> str | None:
    letters = re.sub(r"[^A-Z]", "", text.upper())
    if len(letters) < 2:
        return None
    return letters[:3]


def ocr_team_from_frame(frame: Any, region: Region) -> str | None:
    if pytesseract is None:
        return None
    crop = crop_region(frame, region)
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidates: list[str] = []
    for image in (otsu, cv2.bitwise_not(otsu)):
        text = pytesseract.image_to_string(
            image,
            config="--psm 10 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        )
        parsed = parse_team_text(text)
        if parsed:
            candidates.append(parsed)
    return _mode_non_null(candidates)


def clean_call_text(text: str) -> str | None:
    t = text.upper()
    t = re.sub(r"^[^A-Z0-9]+", "", t)
    t = re.sub(r"[^A-Z0-9]+$", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or None


@lru_cache(maxsize=1)
def load_play_catalog(path: str = PLAY_CATALOG_PATH) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as catalog_file:
            payload = json.load(catalog_file)
    except (OSError, json.JSONDecodeError):
        return {"offense": [], "defense": [], "plays": []}
    return {
        "source": payload.get("source", path),
        "counts": dict(payload.get("counts", {})),
        "offense": list(payload.get("offense", [])),
        "defense": list(payload.get("defense", [])),
        "plays": list(payload.get("plays", [])),
    }


def validate_play_call(
    ocr_text: str | None,
    side: str,
    catalog: dict[str, Any] | None = None,
    formation: str | None = None,
) -> tuple[str | None, str]:
    if not ocr_text:
        return None, "unreadable"
    candidates = (catalog or load_play_catalog()).get(side, [])
    if not candidates:
        return None, "catalog_unavailable"

    compact_ocr = re.sub(r"[^A-Z0-9]", "", ocr_text.upper())
    exact_matches = {
        candidate
        for candidate in candidates
        if re.sub(r"[^A-Z0-9]", "", candidate.upper()) == compact_ocr
    }
    if len(exact_matches) == 1:
        return exact_matches.pop().upper(), "verified"
    if len(compact_ocr) < 3:
        return None, "unverified"

    constrained_candidates = formation_catalog_names(
        side,
        formation,
        catalog or load_play_catalog(),
    )
    if constrained_candidates:
        candidates = constrained_candidates

    ranked: list[tuple[float, str]] = []
    for candidate in candidates:
        compact_candidate = re.sub(r"[^A-Z0-9]", "", candidate.upper())
        score = difflib.SequenceMatcher(None, compact_ocr, compact_candidate).ratio()
        ranked.append((score, candidate))

    ranked.sort(reverse=True)
    best_score, best_name = ranked[0]
    next_score = ranked[1][0] if len(ranked) > 1 else 0.0
    if best_score >= 0.90 or (best_score >= 0.80 and best_score - next_score >= 0.08):
        return best_name.upper(), "verified_fuzzy"
    return None, "unverified"


def formation_catalog_names(
    side: str,
    formation: str | None,
    catalog: dict[str, Any],
) -> list[str]:
    if side != "offense" or not formation:
        return []
    formation_key = re.sub(r"[^A-Z0-9]", "", formation.upper())
    matches = {
        str(play.get("play_name"))
        for play in catalog.get("plays", [])
        if play.get("side") == side
        and re.sub(
            r"[^A-Z0-9]",
            "",
            f"{play.get('formation_family', '')}{play.get('formation', '')}".upper(),
        )
        == formation_key
    }
    return sorted(matches, key=str.casefold)


def resolve_summary_calls(
    summary: dict[str, Any],
    catalog: dict[str, Any],
    formation: str | None = None,
) -> tuple[str | None, str | None, str, str]:
    if summary.get("_banner_shown") is not True:
        return (
            CALL_BANNER_NOT_SHOWN,
            CALL_BANNER_NOT_SHOWN,
            "banner_not_shown",
            "banner_not_shown",
        )
    offensive_call, offensive_status = validate_play_call(
        summary.get("_prev_off"), "offense", catalog, formation
    )
    defensive_call, defensive_status = validate_play_call(
        summary.get("_prev_def"), "defense", catalog
    )
    return offensive_call, defensive_call, offensive_status, defensive_status


def catalog_source_url(
    matched_name: str | None,
    side: str,
    catalog: dict[str, Any],
) -> str | None:
    normalized_name = normalize_text(matched_name)
    for play in catalog.get("plays", []):
        if (
            play.get("side") == side
            and normalize_text(play.get("play_name")) == normalized_name
        ):
            return play.get("source_url")
    return None


def catalog_play_type(
    matched_name: str | None,
    formation: str | None,
    catalog: dict[str, Any],
) -> str | None:
    normalized_name = normalize_text(matched_name)
    if not normalized_name:
        return None
    records = [
        play
        for play in catalog.get("plays", [])
        if play.get("side") == "offense"
        and normalize_text(play.get("play_name")) == normalized_name
    ]
    formation_names = formation_catalog_names("offense", formation, catalog)
    if formation_names:
        formation_key = re.sub(r"[^A-Z0-9]", "", (formation or "").upper())
        formation_records = [
            play
            for play in records
            if re.sub(
                r"[^A-Z0-9]",
                "",
                f"{play.get('formation_family', '')}{play.get('formation', '')}".upper(),
            )
            == formation_key
        ]
        formation_types = {play.get("play_type") for play in formation_records}
        if len(formation_types) == 1:
            return formation_types.pop()
    play_types = {play.get("play_type") for play in records}
    return play_types.pop() if len(play_types) == 1 else None


def catalog_unique_formation(
    matched_name: str | None,
    catalog: dict[str, Any],
) -> str | None:
    normalized_name = normalize_text(matched_name)
    formations = {
        (str(play.get("formation_family")), str(play.get("formation")))
        for play in catalog.get("plays", [])
        if play.get("side") == "offense"
        and normalize_text(play.get("play_name")) == normalized_name
    }
    if len(formations) != 1:
        return None
    family, formation = formations.pop()
    return f"{family} - {formation}"


def parse_personnel_text(text: str) -> str | None:
    normalized = normalize_text(text)
    match = re.search(r"(\d)\s*RB.*?(\d)\s*TE.*?(\d)\s*WR", normalized)
    if not match:
        return None
    running_backs, tight_ends, receivers = match.groups()
    return f"{running_backs} RB, {tight_ends} TE, {receivers} WR"


def canonicalize_formation(text: str) -> str | None:
    normalized = normalize_text(text)
    normalized = normalized.replace("HORMAL", "NORMAL").replace("CIOSE", "CLOSE")
    normalized = normalized.replace("STOT", "SLOT")
    normalized = normalized.replace("YOFFCLOSE", "Y OFF CLOSE")
    if not normalized:
        return None
    family_match = re.search(
        r"\b(GUN|SINGLEBACK|PISTOL|SHOTGUN|I FORM|STRONG|WEAK)\b", normalized
    )
    if family_match:
        normalized = normalized[family_match.start() :]
    elif "Y OFF CLOSE" not in normalized:
        return None

    token_names = {
        "GUN": "Gun",
        "SINGLEBACK": "Singleback",
        "PISTOL": "Pistol",
        "SHOTGUN": "Shotgun",
        "STRONG": "Strong",
        "WEAK": "Weak",
        "NORMAL": "Normal",
        "ACE": "Ace",
        "SLOT": "Slot",
        "TRIPS": "Trips",
        "WING": "Wing",
        "CLOSE": "Close",
        "OFF": "Off",
        "TE": "TE",
        "Y": "Y",
        "I": "I",
        "FORM": "Form",
    }
    words = [token_names.get(word, word.title()) for word in normalized.split()]
    if (
        words
        and words[0] in {"Gun", "Singleback", "Pistol", "Shotgun", "Strong", "Weak"}
        and len(words) > 1
    ):
        return f"{words[0]} - {' '.join(words[1:])}"
    return " ".join(words)


def parse_field_side(frame: Any, region: Region) -> str | None:
    crop = crop_region(frame, region)
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    row_counts = (gray < 100).sum(axis=1)
    nonempty_rows = [index for index, count in enumerate(row_counts) if count > 0]
    if len(nonempty_rows) < 4:
        return None
    first, last = nonempty_rows[0], nonempty_rows[-1]
    midpoint = (first + last + 1) // 2
    top = int(row_counts[first:midpoint].sum())
    bottom = int(row_counts[midpoint : last + 1].sum())
    if top > bottom * 1.15:
        return "own"
    if bottom > top * 1.35:
        return "opponent"
    return None


def parse_possession_team(
    frame: Any,
    regions: dict[str, Region],
    away_team: str | None,
    home_team: str | None,
) -> str | None:
    ratios: dict[str, float] = {}
    for side in ("away", "home"):
        crop = crop_region(frame, regions[f"possession_{side}"])
        if crop is None or crop.size == 0:
            ratios[side] = 0.0
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ratios[side] = float((gray < 60).mean())

    away_ratio = ratios["away"]
    home_ratio = ratios["home"]
    if away_team and away_ratio >= 0.08 and away_ratio > home_ratio * 1.5:
        return away_team
    if home_team and home_ratio >= 0.08 and home_ratio > away_ratio * 1.5:
        return home_team
    return None


def is_previous_play_panel_visible(frame: Any, banner_region: Region) -> bool:
    crop = crop_region(frame, banner_region)
    if crop is None or crop.size == 0:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    dark_ratio = float((gray < 55).mean())
    edge_density = float((cv2.Canny(gray, 60, 180) > 0).mean())
    return dark_ratio >= 0.55 and edge_density >= 0.04


def read_previous_play_heading(frame: Any, banner_region: Region) -> str | None:
    heading_region = subregion(banner_region, 0.0, 0.0, 1.0, 0.5)
    text = clean_call_text(
        ocr_text_from_crop(crop_region(frame, heading_region), None, 11)
    )
    if text and "PREVIOUS" in text and "PLAY" in text:
        return text
    return None


def extract_state_from_frame(frame: Any, regions: dict[str, Region]) -> dict[str, Any]:
    def ocr(
        name: str,
        whitelist: str | None,
        psm: int,
        variants: tuple[str, ...] = ("otsu", "inv"),
    ) -> str:
        return ocr_region_from_frame(frame, regions[name], whitelist, psm, variants)

    down_distance_text = ocr(
        "down_distance", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ& ", 7
    )
    down, distance = parse_down_distance_text(down_distance_text)
    panel_visible = is_previous_play_panel_visible(frame, regions["play_call_banner"])
    stream_side = None
    personnel = None
    formation = None
    defense_header = normalize_text(ocr("defense_header", None, 7, ("otsu",)))
    is_defense = "DEFEN" in defense_header or (
        "PICK" in defense_header and "PLA" in defense_header
    )
    if is_defense:
        defense_personnel = ocr("defense_personnel", None, 7, ("otsu",))
        personnel = parse_personnel_text(defense_personnel)
        stream_side = "defense"
        formation = canonicalize_formation(ocr("defense_formation", None, 7, ("otsu",)))
    else:
        offense_formation = canonicalize_formation(
            ocr("offense_formation", None, 7, ("otsu",))
        )
        if offense_formation is not None:
            stream_side = "offense"
            formation = offense_formation
    away_team = ocr_team_from_frame(frame, regions["team_away"])
    home_team = ocr_team_from_frame(frame, regions["team_home"])
    return {
        "quarter": parse_quarter_text(ocr("quarter", None, 7)),
        "game_clock": parse_clock_text(ocr("game_clock", "0123456789:", 7, ("otsu",))),
        "down": down,
        "distance": distance,
        "special_state": parse_special_state(down_distance_text),
        "yard_line": parse_yard_text(
            ocr("yard_line", "0123456789", 7, ("otsu", "inv"))
        ),
        "field_side": parse_field_side(frame, regions["field_side_indicator"]),
        "stream_side": stream_side,
        "personnel": personnel,
        "formation": formation,
        "away_team": away_team,
        "home_team": home_team,
        "possession_team": parse_possession_team(frame, regions, away_team, home_team),
        "away_score": parse_int_text(
            ocr("score_away", "0123456789", 10, ("otsu", "inv"))
        ),
        "home_score": parse_int_text(
            ocr("score_home", "0123456789", 10, ("otsu", "inv"))
        ),
        "previous_play_banner_shown": panel_visible,
        "prev_def_call": clean_call_text(ocr("prev_def", None, 7, ("otsu",)))
        if panel_visible
        else None,
        "prev_off_call": clean_call_text(ocr("prev_off", None, 7, ("otsu",)))
        if panel_visible
        else None,
    }


def extract_frame_state(frame_path: str, regions: dict[str, Region]) -> dict[str, Any]:
    frame = cv2.imread(frame_path)
    if frame is None:
        return {
            "quarter": None,
            "game_clock": None,
            "down": None,
            "distance": None,
            "special_state": None,
            "yard_line": None,
            "field_side": None,
            "stream_side": None,
            "personnel": None,
            "formation": None,
            "away_team": None,
            "home_team": None,
            "possession_team": None,
            "away_score": None,
            "home_score": None,
            "previous_play_banner_shown": False,
            "prev_def_call": None,
            "prev_off_call": None,
        }
    return extract_state_from_frame(frame, regions)


def _mode_non_null(values: list[Any]) -> Any:
    counts: dict[Any, int] = {}
    for value in values:
        if value in (None, ""):
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def fuzzy_text_consensus(values: list[Any]) -> str | None:
    texts = [normalize_text(str(value)) for value in values if value not in (None, "")]
    texts = [text for text in texts if text]
    if not texts:
        return None

    def agreement(candidate: str) -> float:
        similarity = sum(
            difflib.SequenceMatcher(
                None, candidate.replace(" ", ""), other.replace(" ", "")
            ).ratio()
            for other in texts
        )
        football_tokens = {
            "PA",
            "HB",
            "COVER",
            "TAMPA",
            "BLAST",
            "DIVE",
            "STRETCH",
            "SAIL",
            "CLOUD",
            "SKY",
            "ZONE",
            "FIRE",
            "DOG",
            "BASE",
        }
        token_bonus = sum(
            1.5 for token in candidate.split() if token in football_tokens
        )
        length_bonus = min(len(candidate), 24) * 0.03
        return similarity + token_bonus + length_bonus

    return max(texts, key=lambda candidate: (agreement(candidate), len(candidate)))


FOOTBALL_CALL_TOKENS = {
    "PA",
    "HB",
    "COVER",
    "TAMPA",
    "BLAST",
    "DIVE",
    "STRETCH",
    "SAIL",
    "CLOUD",
    "SKY",
    "ZONE",
    "FIRE",
    "DOG",
    "BASE",
    "CURL",
    "POST",
}


def call_quality(call: str | None) -> int:
    tokens = normalize_text(call).split()
    return sum(1 for token in tokens if token in FOOTBALL_CALL_TOKENS)


def refine_summary_calls(
    video_path: str,
    summaries: list[dict[str, Any]],
    regions: dict[str, Region],
) -> None:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        return

    catalog = load_play_catalog()
    for index, summary in enumerate(summaries):
        offensive_call = summary.get("_prev_off")
        defensive_call = summary.get("_prev_def")
        formation = summaries[index - 1].get("_formation") if index > 0 else None
        _, offensive_status = validate_play_call(
            offensive_call, "offense", catalog, formation
        )
        _, defensive_status = validate_play_call(defensive_call, "defense", catalog)
        needs_offense = offensive_status not in {"verified", "verified_fuzzy"}
        needs_defense = defensive_status not in {"verified", "verified_fuzzy"}
        if not needs_offense and not needs_defense:
            continue

        offensive_candidates = (
            [offensive_call] if needs_offense and offensive_call else []
        )
        defensive_candidates = (
            [defensive_call] if needs_defense and defensive_call else []
        )
        heading_evidence: list[str] = []
        start = float(summary["start_timestamp_seconds"])
        end = min(float(summary["end_timestamp_seconds"]) + 1.0, start + 12.0)
        timestamp = start
        while timestamp <= end:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if ok and is_previous_play_panel_visible(
                frame, regions["play_call_banner"]
            ):
                heading = read_previous_play_heading(frame, regions["play_call_banner"])
                if heading:
                    heading_evidence.append(heading)
                if needs_offense:
                    offensive_candidates.extend(
                        ocr_call_candidates(frame, regions["prev_off"])
                    )
                if needs_defense:
                    defensive_candidates.extend(
                        ocr_call_candidates(frame, regions["prev_def"])
                    )
            timestamp += 1.0

        refined_offensive = fuzzy_text_consensus(offensive_candidates)
        refined_defensive = fuzzy_text_consensus(defensive_candidates)
        voted_offensive = catalog_candidate_consensus(
            offensive_candidates, "offense", catalog, formation
        )
        voted_defensive = catalog_candidate_consensus(
            defensive_candidates, "defense", catalog
        )
        if voted_offensive:
            summary["_prev_off"] = voted_offensive
        elif len(offensive_candidates) >= 3 and call_quality(
            refined_offensive
        ) >= call_quality(offensive_call):
            summary["_prev_off"] = refined_offensive
        if voted_defensive:
            summary["_prev_def"] = voted_defensive
        elif len(defensive_candidates) >= 3 and call_quality(
            refined_defensive
        ) >= call_quality(defensive_call):
            summary["_prev_def"] = refined_defensive
        summary["_prev_off_evidence"] = sorted(set(offensive_candidates))
        summary["_prev_def_evidence"] = sorted(set(defensive_candidates))
        summary["_banner_heading_evidence"] = sorted(set(heading_evidence))

    capture.release()


def ocr_call_candidates(frame: Any, region: Region) -> list[str]:
    crop = crop_region(frame, region)
    if crop is None or crop.size == 0:
        return []
    candidates: list[str] = []
    for psm in (6, 7, 13):
        candidate = clean_call_text(ocr_text_from_crop(crop, None, psm))
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def catalog_candidate_consensus(
    candidates: list[str | None],
    side: str,
    catalog: dict[str, Any],
    formation: str | None = None,
) -> str | None:
    evidence: dict[str, list[tuple[str, str]]] = {}
    for candidate in candidates:
        matched, status = validate_play_call(candidate, side, catalog, formation)
        if matched and status in {"verified", "verified_fuzzy"}:
            compact_candidate = re.sub(r"[^A-Z0-9]", "", (candidate or "").upper())
            evidence.setdefault(matched, []).append((status, compact_candidate))
    if not evidence:
        return None
    match, observations = max(evidence.items(), key=lambda item: len(item[1]))
    if len(observations) < 2:
        return None
    if any(status == "verified" for status, _ in observations):
        return match
    distinct_variants = {variant for _, variant in observations}
    return match if len(distinct_variants) >= 2 else None


def stabilize_score_field(states: list[dict[str, Any]], field: str) -> None:
    raw_values = [state.get(field) for state in states]
    stable = 0
    non_null = [
        (index, value)
        for index, value in enumerate(raw_values)
        if isinstance(value, int)
    ]
    transitions: dict[int, int] = {}
    for position, (index, candidate) in enumerate(non_null):
        score_delta = candidate - stable
        if candidate < stable or score_delta not in {0, 1, 2, 3, 6, 7, 8}:
            continue
        if score_delta == 1:
            nearby_states = states[max(0, index - 12) : index + 13]
            if not any(
                state.get("special_state") == "extra_point" for state in nearby_states
            ):
                continue
        next_values = [value for _, value in non_null[position + 1 : position + 4]]
        corroborated = candidate == stable or candidate in next_values
        followed_by_legal_growth = any(
            value > candidate and value - candidate in {1, 2, 3, 6, 7, 8}
            for value in next_values
        )
        if corroborated or followed_by_legal_growth:
            stable = candidate
            transitions[index] = stable

    stable = 0
    for index, state in enumerate(states):
        if index in transitions:
            stable = transitions[index]
        state[field] = stable


def stabilize_frame_states(states: list[dict[str, Any]]) -> None:
    stabilize_score_field(states, "away_score")
    stabilize_score_field(states, "home_score")
    stabilize_situation_fields(states)


def stabilize_situation_fields(states: list[dict[str, Any]]) -> None:
    def compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_yard = left.get("yard_line")
        right_yard = right.get("yard_line")
        if left_yard is None or right_yard is None or left_yard != right_yard:
            return False
        left_down = left.get("down")
        right_down = right.get("down")
        return left_down is None or right_down is None or left_down == right_down

    supported = [False] * len(states)
    for index in range(1, len(states)):
        if compatible(states[index - 1], states[index]):
            supported[index - 1] = True
            supported[index] = True

    ordinary_anchors = [
        index
        for index, state in enumerate(states)
        if state.get("yard_line") is not None or state.get("down") is not None
    ]
    if ordinary_anchors:
        supported[ordinary_anchors[0]] = True
        supported[ordinary_anchors[-1]] = True

    for index, state in enumerate(states):
        if state.get("special_state") is not None or supported[index]:
            continue
        state["down"] = None
        state["distance"] = None
        state["yard_line"] = None


def summarize_group(frames: list[dict[str, Any]]) -> dict[str, Any]:
    def vote(field: str) -> Any:
        return _mode_non_null([f.get(field) for f in frames])

    down = vote("down")
    distance = vote("distance")
    clocks = [frame.get("game_clock") for frame in frames]
    valid_clocks = [clock for clock in clocks if clock_to_seconds(clock) is not None]
    game_clock = (
        min(valid_clocks, key=lambda clock: clock_to_seconds(clock) or 0)
        if valid_clocks
        else None
    )
    return {
        "start_timestamp_seconds": round(frames[0]["timestamp"], 3),
        "end_timestamp_seconds": round(frames[-1]["timestamp"], 3),
        "quarter": vote("quarter"),
        "game_clock": game_clock,
        "down": down,
        "distance": distance,
        "down_distance": format_down_distance(down, distance),
        "special_state": vote("special_state"),
        "yard_line": vote("yard_line"),
        "field_side": vote("field_side"),
        "_field_side_observed": vote("field_side"),
        "_stream_side": vote("stream_side"),
        "_personnel": vote("personnel"),
        "_formation": vote("formation"),
        "away_team": vote("away_team"),
        "home_team": vote("home_team"),
        "possession_team": vote("possession_team"),
        "_possession_team_observed": vote("possession_team"),
        "away_score": vote("away_score"),
        "home_score": vote("home_score"),
        "_banner_shown": any(
            frame.get("previous_play_banner_shown") is True for frame in frames
        ),
        "_prev_def": fuzzy_text_consensus(
            [frame.get("prev_def_call") for frame in frames]
        ),
        "_prev_off": fuzzy_text_consensus(
            [frame.get("prev_off_call") for frame in frames]
        ),
    }


def build_situation_plays(
    records: list[FrameRecord],
    scoreboard_region: Region,
    banner_region: Region,
    state_cache_path: str | None = None,
    reuse_state_cache: bool = False,
    ocr_workers: int = 4,
    video_path: str | None = None,
    refine_calls: bool = True,
) -> list[dict[str, Any]]:
    regions = build_element_regions(scoreboard_region, banner_region)

    record_signature = [
        [record.frame_index, round(record.timestamp_seconds, 3)] for record in records
    ]
    states: list[dict[str, Any]] | None = None
    if reuse_state_cache and state_cache_path and os.path.exists(state_cache_path):
        with open(state_cache_path, "r", encoding="utf-8") as cache_file:
            cached = json.load(cache_file)
        if (
            cached.get("version") == FRAME_STATE_CACHE_VERSION
            and cached.get("records") == record_signature
        ):
            states = list(cached.get("states", []))

    if states is None:
        states = []

        def extract_record(record: FrameRecord) -> dict[str, Any]:
            extracted = extract_frame_state(record.output_path, regions)
            extracted["timestamp"] = record.timestamp_seconds
            return extracted

        with ThreadPoolExecutor(max_workers=ocr_workers) as executor:
            extracted_states = executor.map(extract_record, records)
            for index, state in enumerate(extracted_states, start=1):
                states.append(state)
                if index % 10 == 0 or index == len(records):
                    print(f"OCR states: {index}/{len(records)}")
        if state_cache_path:
            with open(state_cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(
                    {
                        "version": FRAME_STATE_CACHE_VERSION,
                        "records": record_signature,
                        "states": states,
                    },
                    cache_file,
                    indent=2,
                )

    summaries = build_situation_summaries(states)
    if refine_calls and video_path:
        refine_summary_calls(video_path, summaries, regions)
    return plays_from_summaries(summaries)


def build_situation_summaries(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stabilize_frame_states(states)

    # Group frames into ball "spots". A new spot begins when the yard line changes,
    # or when the down changes at the same yard line (incomplete pass / penalty).
    # Mid-play frames whose scoreboard is unreadable attach to the current spot.
    spots: list[list[dict[str, Any]]] = []
    cur_yard: int | None = None
    spot_down: int | None = None
    spot_special: str | None = None
    spot_score: tuple[int | None, int | None] | None = None
    pending_states: list[dict[str, Any]] = []
    for state in states:
        yard = state["yard_line"]
        down = state["down"]
        special = state.get("special_state")
        score = (state.get("away_score"), state.get("home_score"))

        if yard is None and down is None and special is None:
            if spots:
                pending_states.append(state)
            continue

        new_spot = not spots
        same_special_state = special is not None and special == spot_special
        if same_special_state:
            new_spot = False
        elif spots and yard is not None and cur_yard is not None and yard != cur_yard:
            new_spot = True
        elif (
            spots
            and yard is not None
            and yard == cur_yard
            and down is not None
            and spot_down is not None
            and down != spot_down
        ):
            new_spot = True
        elif spots and spot_special is not None and special != spot_special:
            new_spot = True
        elif (
            spots
            and score[0] is not None
            and score[1] is not None
            and spot_score is not None
            and score != spot_score
        ):
            new_spot = True

        if new_spot:
            spots.append([*pending_states, state])
            spot_down = down
            spot_special = special
            spot_score = score if None not in score else None
        else:
            spots[-1].extend(pending_states)
            spots[-1].append(state)
        pending_states = []

        if yard is not None:
            cur_yard = yard
        if down is not None and spot_down is None:
            spot_down = down
        if special is not None and spot_special is None:
            spot_special = special
        if None not in score and spot_score is None:
            spot_score = score

    summaries = [summarize_group(spot) for spot in spots if spot]
    summaries = [
        summary
        for summary in summaries
        if summary["yard_line"] is not None
        or summary["down"] is not None
        or summary["special_state"] is not None
    ]
    collapse_same_clock_summaries(summaries)
    collapse_pre_special_states(summaries)
    backfill_static_and_scores(summaries)
    repair_down_progression(summaries)
    resolve_stream_sides(summaries)
    resolve_possession_teams(summaries)
    propagate_field_sides(summaries)
    propagate_game_clocks(summaries)
    repair_quarter_and_clock_progression(summaries)
    collapse_same_clock_summaries(summaries)
    remove_unsupported_situation_transitions(summaries)
    repair_special_team_scores(summaries)
    return summaries


def collapse_same_clock_summaries(summaries: list[dict[str, Any]]) -> None:
    index = 1
    while index < len(summaries):
        previous = summaries[index - 1]
        current = summaries[index]
        same_situation_clock = (
            previous.get("special_state") is None
            and current.get("special_state") is None
            and previous.get("quarter") == current.get("quarter")
            and previous.get("game_clock") is not None
            and previous.get("game_clock") == current.get("game_clock")
        )
        if not same_situation_clock:
            index += 1
            continue

        previous_duration = previous.get("end_timestamp_seconds", 0) - previous.get(
            "start_timestamp_seconds", previous.get("end_timestamp_seconds", 0)
        )
        current_duration = current.get("end_timestamp_seconds", 0) - current.get(
            "start_timestamp_seconds", current.get("end_timestamp_seconds", 0)
        )
        winner = current if current_duration > previous_duration else previous
        loser = previous if winner is current else current
        winner["start_timestamp_seconds"] = min(
            previous.get("start_timestamp_seconds", current["start_timestamp_seconds"]),
            current["start_timestamp_seconds"],
        )
        winner["end_timestamp_seconds"] = max(
            previous["end_timestamp_seconds"], current["end_timestamp_seconds"]
        )
        for field in (
            "_stream_side",
            "_personnel",
            "_formation",
            "away_team",
            "home_team",
            "possession_team",
            "field_side",
        ):
            if winner.get(field) is None:
                winner[field] = loser.get(field)
        winner["_prev_off"] = fuzzy_text_consensus(
            [previous.get("_prev_off"), current.get("_prev_off")]
        )
        winner["_prev_def"] = fuzzy_text_consensus(
            [previous.get("_prev_def"), current.get("_prev_def")]
        )
        winner["_banner_shown"] = bool(
            previous.get("_banner_shown") or current.get("_banner_shown")
        )
        summaries[index - 1] = winner
        del summaries[index]


def transition_has_play_evidence(
    previous: dict[str, Any],
    current: dict[str, Any],
    catalog: dict[str, Any] | None = None,
) -> bool:
    return bool(transition_evidence_reasons(previous, current, catalog))


def transition_evidence_reasons(
    previous: dict[str, Any],
    current: dict[str, Any],
    catalog: dict[str, Any] | None = None,
) -> list[str]:
    reasons: list[str] = []
    if (
        previous.get("special_state") is not None
        or current.get("special_state") is not None
    ):
        reasons.append("special_state")

    score_changed = any(
        previous.get(field) != current.get(field)
        for field in ("away_score", "home_score")
        if previous.get(field) is not None and current.get(field) is not None
    )
    if score_changed:
        reasons.append("score_change")

    previous_down = previous.get("down")
    current_down = current.get("down")
    if isinstance(previous_down, int) and isinstance(current_down, int):
        if current_down == previous_down + 1:
            reasons.append("legal_down_progression")
        elif current_down == 1:
            reasons.append("first_down_or_possession_transition")
        return reasons

    if current.get("_banner_shown") is True:
        offensive_call, defensive_call, _, _ = resolve_summary_calls(
            current, catalog or load_play_catalog()
        )
        if offensive_call is not None or defensive_call is not None:
            reasons.append("validated_previous_play_banner")
    return reasons


def remove_unsupported_situation_transitions(
    summaries: list[dict[str, Any]],
) -> None:
    catalog = load_play_catalog()
    index = 1
    while index < len(summaries):
        previous = summaries[index - 1]
        current = summaries[index]
        if transition_has_play_evidence(previous, current, catalog):
            index += 1
            continue

        previous["end_timestamp_seconds"] = max(
            previous.get("end_timestamp_seconds", 0),
            current.get("end_timestamp_seconds", 0),
        )
        previous["_prev_off"] = fuzzy_text_consensus(
            [previous.get("_prev_off"), current.get("_prev_off")]
        )
        previous["_prev_def"] = fuzzy_text_consensus(
            [previous.get("_prev_def"), current.get("_prev_def")]
        )
        previous["_banner_shown"] = bool(
            previous.get("_banner_shown") or current.get("_banner_shown")
        )
        del summaries[index]


def collapse_pre_special_states(summaries: list[dict[str, Any]]) -> None:
    index = 1
    while index < len(summaries) - 1:
        previous = summaries[index - 1]
        current = summaries[index]
        following = summaries[index + 1]
        if (
            previous.get("down") is not None
            and current.get("down") is None
            and current.get("special_state") is None
            and following.get("special_state") in {"extra_point", "kickoff"}
        ):
            del summaries[index]
            continue
        index += 1


def plays_from_summaries(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:

    # Pair consecutive spots into completed plays: the play that moved the ball from
    # spot[i-1] to spot[i] is the one shown on spot[i]'s "previous play" panel.
    plays: list[dict[str, Any]] = []
    for i in range(1, len(summaries)):
        if (
            summaries[i - 1].get("down") is None
            and summaries[i - 1].get("special_state") is None
        ):
            continue
        play = pair_to_play(summaries[i - 1], summaries[i])
        play["play_number"] = len(plays)
        play["description"] = describe_play(play)
        plays.append(play)

    return plays


def repair_down_progression(summaries: list[dict[str, Any]]) -> None:
    for index in range(1, len(summaries) - 1):
        previous = summaries[index - 1]
        current = summaries[index]
        following = summaries[index + 1]
        previous_down = previous.get("down")
        current_down = current.get("down")
        following_down = following.get("down")
        if (
            isinstance(previous_down, int)
            and 1 <= previous_down < 4
            and isinstance(current_down, int)
            and current_down not in {1, previous_down + 1}
            and following_down == 1
            and previous.get("special_state") is None
            and current.get("special_state") is None
        ):
            current["down"] = previous_down + 1
            current["down_distance"] = format_down_distance(
                current["down"],
                current.get("distance"),
            )


def opposite_stream_side(stream_side: str | None) -> str | None:
    if stream_side == "offense":
        return "defense"
    if stream_side == "defense":
        return "offense"
    return None


def apply_stream_transitions(summaries: list[dict[str, Any]]) -> None:
    for index in range(1, len(summaries)):
        previous = summaries[index - 1]
        current = summaries[index]
        turnover_on_downs = previous.get("down") == 4 and current.get("down") == 1
        after_kickoff = previous.get("special_state") == "kickoff"
        if turnover_on_downs or after_kickoff:
            flipped = opposite_stream_side(previous.get("_stream_side"))
            if flipped is not None:
                current["_stream_side"] = flipped


def resolve_stream_sides(summaries: list[dict[str, Any]]) -> None:
    current_side: str | None = None
    for index, summary in enumerate(summaries):
        if index > 0:
            previous = summaries[index - 1]
            possession_changed = (
                previous.get("down") == 4 and summary.get("down") == 1
            ) or previous.get("special_state") == "kickoff"
            if possession_changed:
                current_side = opposite_stream_side(current_side)

        observed_side = summary.get("_stream_side")
        if observed_side is not None:
            current_side = observed_side
        summary["_stream_side"] = current_side


def propagate_field_sides(summaries: list[dict[str, Any]]) -> None:
    current_side: str | None = None
    for summary in summaries:
        observed_side = summary.get("field_side")
        if observed_side is not None:
            current_side = observed_side
        else:
            summary["field_side"] = current_side


def propagate_game_clocks(summaries: list[dict[str, Any]]) -> None:
    current_clock: str | None = None
    for summary in summaries:
        observed_clock = summary.get("game_clock")
        if observed_clock is not None:
            current_clock = observed_clock
        else:
            summary["game_clock"] = current_clock


def repair_quarter_and_clock_progression(summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return

    stable_quarter = summaries[0].get("quarter") or 1
    stable_clock = clock_to_seconds(summaries[0].get("game_clock"))
    summaries[0]["quarter"] = stable_quarter

    for summary in summaries[1:]:
        observed_quarter = summary.get("quarter")
        observed_clock = clock_to_seconds(summary.get("game_clock"))

        if observed_quarter == stable_quarter + 1:
            clock_restarted = (
                observed_clock is not None
                and stable_clock is not None
                and observed_clock >= 5 * 60
                and stable_clock <= 45
            )
            if clock_restarted:
                stable_quarter = observed_quarter
                stable_clock = observed_clock
                continue
        elif observed_quarter not in (None, stable_quarter):
            summary["quarter"] = stable_quarter

        if (
            observed_clock is not None
            and stable_clock is not None
            and observed_clock >= 5 * 60
            and stable_clock <= 45
            and stable_quarter < 4
        ):
            stable_quarter += 1
            summary["quarter"] = stable_quarter
            stable_clock = observed_clock
            continue

        summary["quarter"] = stable_quarter
        if (
            observed_clock is not None
            and stable_clock is not None
            and observed_clock > stable_clock
        ):
            summary["game_clock"] = seconds_to_clock(stable_clock)
        elif observed_clock is not None:
            stable_clock = observed_clock


def other_team(
    team: str | None,
    away_team: str | None,
    home_team: str | None,
) -> str | None:
    if team == away_team:
        return home_team
    if team == home_team:
        return away_team
    return None


def resolve_possession_teams(summaries: list[dict[str, Any]]) -> None:
    current_team: str | None = None
    for index, summary in enumerate(summaries):
        if index > 0:
            previous = summaries[index - 1]
            possession_changed = (
                previous.get("down") == 4 and summary.get("down") == 1
            ) or previous.get("special_state") == "kickoff"
            if possession_changed:
                current_team = other_team(
                    current_team,
                    summary.get("away_team"),
                    summary.get("home_team"),
                )
        observed_team = summary.get("possession_team")
        scoreboard_teams = {summary.get("away_team"), summary.get("home_team")}
        if observed_team in scoreboard_teams:
            current_team = observed_team
        summary["possession_team"] = current_team


def repair_special_team_scores(summaries: list[dict[str, Any]]) -> None:
    for index in range(len(summaries) - 1):
        pat = summaries[index]
        kickoff = summaries[index + 1]
        if (
            pat.get("special_state") != "extra_point"
            or kickoff.get("special_state") != "kickoff"
        ):
            continue
        scoring_team = pat.get("possession_team")
        score_field = None
        if scoring_team == pat.get("away_team"):
            score_field = "away_score"
        elif scoring_team == pat.get("home_team"):
            score_field = "home_score"
        if score_field is None:
            continue

        previous_score = None
        if index > 0:
            previous_score = summaries[index - 1].get(score_field)
        if not isinstance(previous_score, int):
            continue

        touchdown_score = previous_score + 6
        final_score = None
        for following in summaries[index + 1 : index + 13]:
            candidate = following.get(score_field)
            if candidate in {touchdown_score, touchdown_score + 1}:
                final_score = candidate
                break

        pat[score_field] = touchdown_score
        kickoff[score_field] = final_score or touchdown_score


def pair_to_play(
    prev: dict[str, Any],
    cur: dict[str, Any],
) -> dict[str, Any]:
    possession_team = prev.get("possession_team")
    end_possession_team = cur.get("possession_team")
    catalog = load_play_catalog()
    banner_shown = cur.get("_banner_shown") is True
    raw_offensive_call = cur.get("_prev_off")
    raw_defensive_call = cur.get("_prev_def")
    offensive_match, defensive_match, offensive_validation, defensive_validation = (
        resolve_summary_calls(cur, catalog, prev.get("_formation"))
    )
    possession_boundary = (
        possession_team is not None
        and end_possession_team is not None
        and possession_team != end_possession_team
    ) or (prev.get("down") == 4 and cur.get("down") == 1)
    away_delta = (cur.get("away_score") or 0) - (prev.get("away_score") or 0)
    home_delta = (cur.get("home_score") or 0) - (prev.get("home_score") or 0)
    scoring_boundary = away_delta >= 3 or home_delta >= 3
    unsupported_statuses = {"unverified", "unreadable", "catalog_unavailable"}
    if (
        (possession_boundary or scoring_boundary)
        and offensive_validation in unsupported_statuses
        and defensive_validation in unsupported_statuses
    ):
        banner_shown = False
        offensive_match = CALL_BANNER_NOT_SHOWN
        defensive_match = CALL_BANNER_NOT_SHOWN
        offensive_validation = "banner_not_shown"
        defensive_validation = "banner_not_shown"
    previous_offensive_match, previous_defensive_match, _, _ = resolve_summary_calls(
        prev, catalog
    )
    observed_formation = prev.get("_formation")
    resolved_formation = observed_formation or catalog_unique_formation(
        offensive_match, catalog
    )
    play = {
        "start_timestamp_seconds": prev["start_timestamp_seconds"],
        "end_timestamp_seconds": cur["start_timestamp_seconds"],
        "quarter": prev["quarter"],
        "game_clock": prev["game_clock"],
        "away_team": prev["away_team"],
        "away_score": prev["away_score"],
        "home_team": prev["home_team"],
        "home_score": prev["home_score"],
        "down": prev["down"],
        "distance": prev["distance"],
        "down_distance": prev["down_distance"],
        "yard_line": prev["yard_line"],
        "yard_line_observed": prev["yard_line"],
        "yard_line_provenance": "scoreboard_ocr_consensus",
        "stream_side": prev["_stream_side"],
        "possession_team": possession_team,
        "possession_team_observed": prev.get("_possession_team_observed"),
        "possession_team_provenance": (
            "possession_marker"
            if prev.get("_possession_team_observed") == possession_team
            else "football_state_transition"
        ),
        "field_side": prev["field_side"],
        "field_side_observed": prev.get("_field_side_observed"),
        "field_side_provenance": (
            "scoreboard_direction_indicator"
            if prev.get("_field_side_observed") is not None
            else "forward_propagation"
        ),
        "personnel": prev["_personnel"],
        "formation_observed": observed_formation,
        "formation": resolved_formation,
        "formation_provenance": (
            "play_selection_ocr"
            if observed_formation
            else "acemadden_unique_play_formation"
            if resolved_formation
            else "unavailable"
        ),
        "previous_offensive_play_call": previous_offensive_match,
        "previous_defensive_play_call": previous_defensive_match,
        "previous_play_banner_shown": banner_shown,
        "offensive_play_call_ocr": raw_offensive_call,
        "defensive_play_call_ocr": raw_defensive_call,
        "offensive_play_call_evidence_count": len(cur.get("_prev_off_evidence", [])),
        "defensive_play_call_evidence_count": len(cur.get("_prev_def_evidence", [])),
        "offensive_play_call_evidence": json.dumps(cur.get("_prev_off_evidence", [])),
        "defensive_play_call_evidence": json.dumps(cur.get("_prev_def_evidence", [])),
        "previous_play_heading_evidence": json.dumps(
            cur.get("_banner_heading_evidence", [])
        ),
        "offensive_play_call_validation": offensive_validation,
        "defensive_play_call_validation": defensive_validation,
        "offensive_play_call_catalog_match": offensive_match,
        "defensive_play_call_catalog_match": defensive_match,
        "offensive_play_call_catalog_url": catalog_source_url(
            offensive_match, "offense", catalog
        ),
        "defensive_play_call_catalog_url": catalog_source_url(
            defensive_match, "defense", catalog
        ),
        "offensive_play_call": offensive_match,
        "defensive_play_call": defensive_match,
        "end_down_distance": cur["down_distance"],
        "end_yard_line": cur["yard_line"],
        "end_away_score": cur.get("away_score"),
        "end_home_score": cur.get("home_score"),
        "end_special_state": cur["special_state"],
        "end_possession_team": end_possession_team,
        "transition_evidence": transition_evidence_reasons(prev, cur, catalog),
    }
    play["play_type"] = catalog_play_type(
        offensive_match,
        prev.get("_formation"),
        catalog,
    ) or infer_play_type(play["offensive_play_call"])
    play["possession_change"] = possession_boundary
    if prev.get("special_state") == "extra_point":
        play["play_type"] = "extra_point"
        play["offensive_play_call"] = "PAT Kick"
        play["offensive_play_call_validation"] = "special_state"
        play["offensive_play_call_catalog_match"] = None
        play["offensive_play_call_catalog_url"] = None
        play["defensive_play_call"] = None
        play["defensive_play_call_validation"] = "not_applicable"
        play["defensive_play_call_catalog_match"] = None
        play["defensive_play_call_catalog_url"] = None
    elif prev.get("special_state") == "kickoff":
        play["play_type"] = "kickoff"
        play["offensive_play_call"] = "Kickoff"
        play["defensive_play_call"] = None
        play["offensive_play_call_validation"] = "special_state"
        play["defensive_play_call_validation"] = "not_applicable"
        play["offensive_play_call_catalog_match"] = None
        play["defensive_play_call_catalog_match"] = None
        play["offensive_play_call_catalog_url"] = None
        play["defensive_play_call_catalog_url"] = None
        play["possession_change"] = True
    compute_play_result(play, prev, cur)
    add_play_quality(play, prev, cur)
    return play


def add_play_quality(
    play: dict[str, Any],
    previous: dict[str, Any],
    current: dict[str, Any],
) -> None:
    situation_fields = [
        "quarter",
        "game_clock",
        "away_team",
        "home_team",
        "away_score",
        "home_score",
        "possession_team",
    ]
    if play.get("play_type") not in {"extra_point", "kickoff"}:
        situation_fields.extend(["down", "distance", "yard_line"])
    observed = sum(play.get(field) is not None for field in situation_fields)
    situation_confidence = observed / len(situation_fields)

    status_scores = {
        "verified": 1.0,
        "verified_fuzzy": 0.9,
        "special_state": 1.0,
        "not_applicable": 1.0,
        "banner_not_shown": 1.0,
        "unverified": 0.2,
        "unreadable": 0.0,
        "catalog_unavailable": 0.0,
    }
    call_statuses = [
        play.get("offensive_play_call_validation"),
        play.get("defensive_play_call_validation"),
    ]
    call_confidence = (
        sum(status_scores.get(status, 0.0) for status in call_statuses) / 2
    )

    away_delta = (current.get("away_score") or 0) - (previous.get("away_score") or 0)
    home_delta = (current.get("home_score") or 0) - (previous.get("home_score") or 0)
    if play.get("play_type") in {"extra_point", "kickoff"} or away_delta or home_delta:
        result_confidence = 1.0
        result_provenance = "scoreboard_score+special_state"
    elif previous.get("down") is not None and current.get("down") is not None:
        result_confidence = 0.85
        result_provenance = "scoreboard_down_distance_transition"
    elif play.get("previous_play_banner_shown"):
        result_confidence = 0.65
        result_provenance = "yard_line+validated_previous_play_banner"
    else:
        result_confidence = 0.4
        result_provenance = "yard_line_transition_only"

    play["situation_confidence"] = round(situation_confidence, 3)
    play["call_confidence"] = round(call_confidence, 3)
    play["result_confidence"] = round(result_confidence, 3)
    play["overall_confidence"] = round(
        (situation_confidence + call_confidence + result_confidence) / 3,
        3,
    )
    play["situation_provenance"] = "scoreboard_ocr+stabilized_state_transition"
    play["call_provenance"] = (
        "previous_play_banner+acemadden_catalog"
        if play.get("previous_play_banner_shown")
        else "previous_play_banner_not_shown"
    )
    play["result_provenance"] = result_provenance


def infer_play_type(offensive_play_call: str | None) -> str | None:
    call = normalize_text(offensive_play_call)
    if not call:
        return None
    if "PAT" in call:
        return "extra_point"
    if "KICK" in call:
        return "kickoff"
    if any(
        token in call
        for token in ("HB", "DIVE", "STRETCH", "BASE", "ZONE", "POWER", "COUNTER")
    ):
        return "run"
    if any(
        token in call
        for token in ("PA ", "POST", "SAIL", "CURL", "DIG", "SLANT", "VERT", "FLAT")
    ):
        return "pass"
    return None


def normalize_text(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", (value or "").upper()).strip()


def backfill_static_and_scores(plays: list[dict[str, Any]]) -> None:
    if not plays:
        return

    away_team = _mode_non_null([p.get("away_team") for p in plays])
    home_team = _mode_non_null([p.get("home_team") for p in plays])

    last_quarter = 1
    last_away = 0
    last_home = 0
    for play in plays:
        play["away_team"] = away_team
        play["home_team"] = home_team

        if play.get("quarter") is None:
            play["quarter"] = last_quarter
        else:
            last_quarter = play["quarter"]

        away_score = play.get("away_score")
        if (
            away_score is None
            or away_score < last_away
            or away_score - last_away not in {0, 1, 2, 3, 6, 7, 8}
        ):
            play["away_score"] = last_away
        else:
            last_away = away_score

        home_score = play.get("home_score")
        if (
            home_score is None
            or home_score < last_home
            or home_score - last_home not in {0, 1, 2, 3, 6, 7, 8}
        ):
            play["home_score"] = last_home
        else:
            last_home = home_score


def compute_play_result(
    play: dict[str, Any], prev: dict[str, Any], cur: dict[str, Any]
) -> None:
    away_delta = (cur.get("away_score") or 0) - (prev.get("away_score") or 0)
    home_delta = (cur.get("home_score") or 0) - (prev.get("home_score") or 0)

    down = prev.get("down")
    distance = prev.get("distance")
    next_down = cur.get("down")
    next_distance = cur.get("distance")
    start_yard = prev.get("yard_line")
    end_yard = cur.get("yard_line")

    gained = None
    # Sequential down-distance is more precise than integer yard markers, especially for inches.
    if (
        down is not None
        and next_down == down + 1
        and distance is not None
        and next_distance is not None
    ):
        gained = distance - next_distance
    elif start_yard is not None and end_yard is not None:
        gained = abs(end_yard - start_yard)
    play["yards_gained"] = gained

    if play.get("play_type") == "extra_point":
        play["yards_gained"] = None
        play["result"] = (
            "extra point good"
            if away_delta == 1 or home_delta == 1
            else "extra point attempt"
        )
        return
    if play.get("play_type") == "kickoff":
        play["yards_gained"] = None
        side = cur.get("field_side")
        destination = cur.get("yard_line")
        if side and destination is not None:
            play["result"] = f"kickoff returned to {side} {destination}"
        else:
            play["result"] = "kickoff"
        return
    if play.get("possession_change") and down == 4 and next_down == 1:
        play["yards_gained"] = 0 if start_yard == end_yard else gained
        play["result"] = "turnover on downs"
        return

    if away_delta >= 6 or home_delta >= 6:
        if play.get("field_side") == "opponent" and start_yard is not None:
            play["yards_gained"] = start_yard
        elif play.get("field_side") == "own" and start_yard is not None:
            play["yards_gained"] = 100 - start_yard
        play["result"] = "touchdown"
        return
    if away_delta in (2, 3) or home_delta in (2, 3):
        play["yards_gained"] = None
        play["result"] = "field goal good"
        return

    if next_down == 1 and (next_distance == 10 or next_distance is None):
        play["result"] = f"{gained}-yard gain / first down" if gained else "first down"
        return

    if down is not None and next_down == down + 1:
        play_type = play.get("play_type")
        if gained is None:
            play["result"] = "short gain"
        elif gained > 0:
            if play_type == "pass":
                play["result"] = f"{gained}-yard completion"
            elif play_type == "run":
                play["result"] = f"{gained}-yard gain"
            else:
                play["result"] = f"gain of {gained}"
        elif gained == 0:
            play["result"] = "incomplete pass" if play_type == "pass" else "no gain"
        else:
            play["result"] = f"{abs(gained)}-yard loss"
        return

    if down is not None and next_down == down:
        play["result"] = "no gain / incomplete / penalty"
        return

    if gained:
        play["result"] = f"gain of {gained}"
        return

    play["result"] = "play run"


def describe_play(play: dict[str, Any]) -> str:
    parts: list[str] = []
    if play.get("quarter"):
        parts.append(f"Q{play['quarter']}")
    if play.get("game_clock"):
        parts.append(play["game_clock"])
    header = " ".join(parts)

    where = play.get("down_distance") or ""
    yard = play.get("yard_line")
    if yard is not None:
        where = f"{where} at the {yard}".strip()

    off = play.get("offensive_play_call")
    deff = play.get("defensive_play_call")
    call = f"{off or '?'} vs {deff or '?'}"

    result = play.get("result", "")
    end_dd = play.get("end_down_distance")
    end_yd = play.get("end_yard_line")
    end_bits = ", ".join(
        b for b in [end_dd, (f"at the {end_yd}" if end_yd is not None else "")] if b
    )
    tail = f"{result}" + (f" (→ {end_bits})" if end_bits else "")

    prefix = f"{header} — " if header else ""
    return f"{prefix}{where} — {call} — {tail}".strip()


def build_per_play_records(
    records: list[FrameRecord],
    boundaries: list[BoundaryRecord],
    scoreboard_region: Region,
    play_call_region: Region,
    state_cache_path: str | None = None,
    reuse_state_cache: bool = False,
    ocr_workers: int = 4,
    video_path: str | None = None,
    refine_calls: bool = True,
) -> list[dict[str, Any]]:
    if not records:
        return []
    # Plays are anchored to real football-situation changes read from the scoreboard,
    # not raw pixel-diff spikes. `boundaries` is retained for the boundaries artifact only.
    return build_situation_plays(
        records,
        scoreboard_region,
        play_call_region,
        state_cache_path,
        reuse_state_cache,
        ocr_workers,
        video_path,
        refine_calls,
    )


def write_analysis_outputs(
    output_dir: str,
    boundaries: list[BoundaryRecord],
    plays: list[dict[str, Any]],
) -> tuple[str, str]:
    boundaries_path = os.path.join(output_dir, "play_boundaries.json")
    per_play_path = os.path.join(output_dir, "per_play_records.json")

    with open(boundaries_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "frame_index": b.frame_index,
                    "timestamp_seconds": round(b.timestamp_seconds, 3),
                    "diff_score": round(b.diff_score, 3),
                }
                for b in boundaries
            ],
            f,
            indent=2,
        )

    with open(per_play_path, "w", encoding="utf-8") as f:
        json.dump(plays, f, indent=2)

    return boundaries_path, per_play_path


def has_game_state_anchor(play: dict[str, Any]) -> bool:
    if play.get("down") is not None and play.get("distance") is not None:
        return True
    return bool(play.get("offensive_play_call") or play.get("defensive_play_call"))


FOOTBALL_CSV_HEADERS = [
    "play_number",
    "start_timestamp_seconds",
    "end_timestamp_seconds",
    "quarter",
    "game_clock",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "down",
    "distance",
    "down_distance",
    "yard_line_observed",
    "yard_line",
    "yard_line_provenance",
    "stream_side",
    "possession_team_observed",
    "possession_team",
    "possession_team_provenance",
    "field_side_observed",
    "field_side",
    "field_side_provenance",
    "personnel",
    "formation_observed",
    "formation",
    "formation_provenance",
    "previous_offensive_play_call",
    "previous_defensive_play_call",
    "previous_play_banner_shown",
    "offensive_play_call_ocr",
    "defensive_play_call_ocr",
    "offensive_play_call_evidence_count",
    "defensive_play_call_evidence_count",
    "offensive_play_call_evidence",
    "defensive_play_call_evidence",
    "previous_play_heading_evidence",
    "offensive_play_call_validation",
    "defensive_play_call_validation",
    "offensive_play_call_catalog_match",
    "defensive_play_call_catalog_match",
    "offensive_play_call_catalog_url",
    "defensive_play_call_catalog_url",
    "offensive_play_call",
    "defensive_play_call",
    "play_type",
    "result",
    "yards_gained",
    "possession_change",
    "end_down_distance",
    "end_yard_line",
    "situation_confidence",
    "call_confidence",
    "result_confidence",
    "overall_confidence",
    "situation_provenance",
    "call_provenance",
    "result_provenance",
    "transition_evidence",
    "description",
]


def football_csv_row(play_number: int, play: dict[str, Any]) -> list[Any]:
    return [
        play_number,
        play.get("start_timestamp_seconds"),
        play.get("end_timestamp_seconds"),
        play.get("quarter"),
        play.get("game_clock"),
        play.get("away_team"),
        play.get("away_score"),
        play.get("home_team"),
        play.get("home_score"),
        play.get("down"),
        play.get("distance"),
        play.get("down_distance"),
        play.get("yard_line_observed"),
        play.get("yard_line"),
        play.get("yard_line_provenance"),
        play.get("stream_side"),
        play.get("possession_team_observed"),
        play.get("possession_team"),
        play.get("possession_team_provenance"),
        play.get("field_side_observed"),
        play.get("field_side"),
        play.get("field_side_provenance"),
        play.get("personnel"),
        play.get("formation_observed"),
        play.get("formation"),
        play.get("formation_provenance"),
        play.get("previous_offensive_play_call"),
        play.get("previous_defensive_play_call"),
        play.get("previous_play_banner_shown"),
        play.get("offensive_play_call_ocr"),
        play.get("defensive_play_call_ocr"),
        play.get("offensive_play_call_evidence_count"),
        play.get("defensive_play_call_evidence_count"),
        play.get("offensive_play_call_evidence"),
        play.get("defensive_play_call_evidence"),
        play.get("previous_play_heading_evidence"),
        play.get("offensive_play_call_validation"),
        play.get("defensive_play_call_validation"),
        play.get("offensive_play_call_catalog_match"),
        play.get("defensive_play_call_catalog_match"),
        play.get("offensive_play_call_catalog_url"),
        play.get("defensive_play_call_catalog_url"),
        play.get("offensive_play_call"),
        play.get("defensive_play_call"),
        play.get("play_type"),
        play.get("result"),
        play.get("yards_gained"),
        play.get("possession_change"),
        play.get("end_down_distance"),
        play.get("end_yard_line"),
        play.get("situation_confidence"),
        play.get("call_confidence"),
        play.get("result_confidence"),
        play.get("overall_confidence"),
        play.get("situation_provenance"),
        play.get("call_provenance"),
        play.get("result_provenance"),
        json.dumps(play.get("transition_evidence", [])),
        play.get("description"),
    ]


def final_football_csv_name(
    plays: list[dict[str, Any]],
    completed_date: date | None = None,
) -> str | None:
    if not plays:
        return None
    last_play = plays[-1]
    home_team = last_play.get("home_team")
    away_team = last_play.get("away_team")
    home_score = last_play.get("end_home_score")
    away_score = last_play.get("end_away_score")
    if home_score is None:
        home_score = last_play.get("home_score")
    if away_score is None:
        away_score = last_play.get("away_score")
    if None in (home_team, home_score, away_score, away_team):
        return None

    completion_text = (completed_date or date.today()).isoformat()
    filename = (
        f"{home_team} {home_score} - {away_score} {away_team} {completion_text}.csv"
    )
    return re.sub(r'[<>:"/\\|?*]', "-", filename)


def final_football_csv_path(
    output_dir: str,
    plays: list[dict[str, Any]],
    completed_date: date | None = None,
) -> str | None:
    final_name = final_football_csv_name(plays, completed_date)
    if final_name is None:
        return None
    return os.path.join(output_dir, GAME_DATA_DIRECTORY, final_name)


def team_football_csv_paths(
    output_dir: str,
    plays: list[dict[str, Any]],
    final_name: str,
) -> list[str]:
    if not plays:
        return []
    last_play = plays[-1]
    team_names = {last_play.get("away_team"), last_play.get("home_team")}
    paths: list[str] = []
    for team_name in sorted(team for team in team_names if team):
        safe_team_name = re.sub(r'[<>:"/\\|?*]', "-", str(team_name)).strip()
        if safe_team_name:
            paths.append(
                os.path.join(
                    output_dir,
                    TEAM_DATA_DIRECTORY,
                    safe_team_name,
                    final_name,
                )
            )
    return paths


def initialize_team_database(output_dir: str) -> list[str]:
    with open(TEAM_REFERENCE_PATH, encoding="utf-8") as reference_file:
        team_names = json.load(reference_file)
    if not isinstance(team_names, list) or not all(
        isinstance(team_name, str) and team_name for team_name in team_names
    ):
        raise RuntimeError(f"Invalid team reference: {TEAM_REFERENCE_PATH}")
    team_directories = [
        os.path.join(output_dir, TEAM_DATA_DIRECTORY, team_name)
        for team_name in team_names
    ]
    for team_directory in team_directories:
        os.makedirs(team_directory, exist_ok=True)
    return team_directories


def write_football_csv(
    plays: list[dict[str, Any]],
    output_dir: str,
    finalized: bool = False,
) -> str:
    assert_catalog_valid_calls(plays)
    write_catalog_validation_report(plays, output_dir)
    write_full_game_quality_report(plays, output_dir)
    csv_path = os.path.join(output_dir, "football_play_data.csv")
    rows: list[list[Any]] = [FOOTBALL_CSV_HEADERS]
    for play_number, play in enumerate(p for p in plays if has_game_state_anchor(p)):
        rows.append(football_csv_row(play_number, play))

    try:
        target = csv_path
        with open(target, "w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerows(rows)
    except PermissionError:
        target = os.path.join(output_dir, "football_play_data_latest.csv")
        with open(target, "w", newline="", encoding="utf-8") as csv_file:
            csv.writer(csv_file).writerows(rows)
        print(f"NOTE: {csv_path} was locked (open elsewhere); wrote {target} instead.")

    if finalized:
        final_target = final_football_csv_path(output_dir, plays)
        if final_target:
            initialize_team_database(output_dir)
            os.makedirs(os.path.dirname(final_target), exist_ok=True)
            with open(final_target, "w", newline="", encoding="utf-8") as csv_file:
                csv.writer(csv_file).writerows(rows)
            final_name = os.path.basename(final_target)
            for team_target in team_football_csv_paths(output_dir, plays, final_name):
                os.makedirs(os.path.dirname(team_target), exist_ok=True)
                with open(team_target, "w", newline="", encoding="utf-8") as csv_file:
                    csv.writer(csv_file).writerows(rows)
            return final_target

    return target


def write_full_game_quality_report(plays: list[dict[str, Any]], output_dir: str) -> str:
    intervals: dict[tuple[Any, Any], int] = {}
    possession_anomalies: list[int] = []
    low_confidence: list[dict[str, Any]] = []
    missing_fields: dict[str, int] = {}
    required_fields = (
        "quarter",
        "game_clock",
        "away_team",
        "home_team",
        "possession_team",
        "play_type",
        "result",
    )
    for index, play in enumerate(plays):
        interval = (
            play.get("start_timestamp_seconds"),
            play.get("end_timestamp_seconds"),
        )
        intervals[interval] = intervals.get(interval, 0) + 1
        teams = {play.get("away_team"), play.get("home_team")}
        if (
            play.get("possession_team") is not None
            and play.get("possession_team") not in teams
        ):
            possession_anomalies.append(index)
        if (play.get("overall_confidence") or 0) < 0.7:
            low_confidence.append(
                {
                    "play_number": index,
                    "overall_confidence": play.get("overall_confidence"),
                    "description": play.get("description"),
                }
            )
        for field in required_fields:
            if play.get(field) in (None, ""):
                missing_fields[field] = missing_fields.get(field, 0) + 1

    report = {
        "plays": len(plays),
        "duplicate_intervals": [
            {"interval": interval, "count": count}
            for interval, count in intervals.items()
            if count > 1
        ],
        "possession_anomaly_plays": possession_anomalies,
        "low_confidence_plays": low_confidence,
        "missing_required_fields": missing_fields,
        "play_type_counts": dict(
            Counter(play.get("play_type") or "unknown" for play in plays)
        ),
        "result_counts": dict(
            Counter(play.get("result") or "unknown" for play in plays)
        ),
        "transition_evidence_counts": dict(
            Counter(
                reason
                for play in plays
                for reason in play.get("transition_evidence", [])
            )
        ),
        "confidence_averages": {
            field: round(
                sum(float(play.get(field) or 0) for play in plays) / len(plays),
                3,
            )
            if plays
            else 0.0
            for field in (
                "situation_confidence",
                "call_confidence",
                "result_confidence",
                "overall_confidence",
            )
        },
    }
    report_path = os.path.join(output_dir, "full_game_quality_report.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    return report_path


def assert_catalog_valid_calls(plays: list[dict[str, Any]]) -> None:
    catalog = load_play_catalog()
    catalog_names = {
        side: {normalize_text(name) for name in catalog.get(side, [])}
        for side in ("offense", "defense")
    }
    violations: list[str] = []
    fields = (
        ("previous_offensive_play_call", "offense"),
        ("previous_defensive_play_call", "defense"),
        ("offensive_play_call", "offense"),
        ("defensive_play_call", "defense"),
    )
    for index, play in enumerate(plays):
        for field, side in fields:
            call = play.get(field)
            if call in (None, "", CALL_BANNER_NOT_SHOWN):
                continue
            if field == "offensive_play_call" and call in {"PAT Kick", "Kickoff"}:
                if play.get("play_type") in {"extra_point", "kickoff"}:
                    continue
            if normalize_text(call) not in catalog_names[side]:
                violations.append(f"play {index} {field}={call!r}")
    if violations:
        detail = "; ".join(violations[:10])
        raise RuntimeError(f"Non-catalog play calls cannot be written: {detail}")


def write_catalog_validation_report(
    plays: list[dict[str, Any]], output_dir: str
) -> str:
    status_fields = (
        "offensive_play_call_validation",
        "defensive_play_call_validation",
    )
    status_counts: dict[str, int] = {}
    for play in plays:
        for field in status_fields:
            status = str(play.get(field) or "missing")
            status_counts[status] = status_counts.get(status, 0) + 1

    claimed_fields = (
        "previous_offensive_play_call",
        "previous_defensive_play_call",
        "offensive_play_call",
        "defensive_play_call",
    )
    claimed_calls = sum(
        1
        for play in plays
        for field in claimed_fields
        if play.get(field) not in (None, "", CALL_BANNER_NOT_SHOWN)
    )
    unverified_ocr = sum(
        1
        for play in plays
        for side in ("offensive", "defensive")
        if play.get(f"{side}_play_call_validation")
        in {"unverified", "unreadable", "catalog_unavailable"}
    )
    report = {
        "catalog_source": load_play_catalog().get("source", PLAY_CATALOG_PATH),
        "plays": len(plays),
        "claimed_call_cells": claimed_calls,
        "unverified_ocr_cells": unverified_ocr,
        "status_counts": status_counts,
        "non_catalog_claimed_calls": 0,
        "all_claimed_calls_in_catalog": True,
    }
    report_path = os.path.join(output_dir, "play_call_validation_report.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    return report_path


def draw_live_overlay(
    frame: Any,
    regions: dict[str, Region],
    state: dict[str, Any],
    last_play: dict[str, Any] | None,
    play_count: int,
    csv_path: str,
    total_duration_seconds: float | None = None,
) -> None:
    height, width = frame.shape[:2]
    for region in regions.values():
        x1, y1, x2, y2 = region_to_rect(region, width, height)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    dd = (
        f"{state['down']} & {state['distance']}"
        if state.get("down") and state.get("distance") is not None
        else "?"
    )

    def val(key: str, default: str = "?") -> str:
        v = state.get(key)
        return str(v) if v is not None else default

    current_ts = state.get("timestamp")
    if current_ts is not None and total_duration_seconds and total_duration_seconds > 0:
        pct = min(100.0, (current_ts / total_duration_seconds) * 100.0)
        progress_text = f"progress {current_ts:05.1f}s / {total_duration_seconds:05.1f}s ({pct:0.0f}%)"
    else:
        progress_text = f"plays logged: {play_count}"

    lines = [
        progress_text,
        f"Q{val('quarter')} {val('game_clock', '--:--')}   {dd}   yd {val('yard_line')}",
        f"{val('away_team')} {val('away_score')} - {val('home_score')} {val('home_team')}",
        f"prev off: {val('prev_off_call', '-')} | def: {val('prev_def_call', '-')}",
        f"csv: {os.path.basename(csv_path)}",
        f"plays logged: {play_count}",
    ]
    if last_play:
        lines.append(f"last: {str(last_play.get('description', ''))[:88]}")

    y0 = 26
    for i, text in enumerate(lines):
        pos = (12, y0 + i * 22)
        cv2.putText(
            frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            frame,
            text,
            pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )


def run_live(
    video_path: str,
    output_dir: str,
    scoreboard_region: Region,
    banner_region: Region,
    sample_every_n_frames: int,
    max_frames: int | None,
    start_seconds: float,
    max_duration_seconds: float | None,
    calibration_path: str | None = None,
    preview_frame_path: str | None = None,
    show_window: bool = True,
) -> str:
    regions = build_element_regions(scoreboard_region, banner_region)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = write_football_csv([], output_dir)

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    if start_seconds > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000.0)

    states: list[dict[str, Any]] = []
    visible_plays: list[dict[str, Any]] = []
    play_count = 0
    last_play: dict[str, Any] | None = None

    def refresh_live_plays() -> None:
        nonlocal csv_path, visible_plays, play_count, last_play
        plays = plays_from_summaries(
            build_situation_summaries([dict(state) for state in states])
        )
        visible_plays = [play for play in plays if has_game_state_anchor(play)]
        previous_count = play_count
        play_count = len(visible_plays)
        last_play = visible_plays[-1] if visible_plays else None
        csv_path = write_football_csv(visible_plays, output_dir)
        for play in visible_plays[previous_count:]:
            print(f"[{play['play_number']}] {play['description']}")

    window = "Madden Scout - live scan (press q to quit)"
    display_enabled = show_window
    frame_index = 0
    saved = 0
    total_duration_seconds = None
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count and frame_count > 0:
        total_duration_seconds = frame_count / fps

    while True:
        success, frame = capture.read()
        if not success:
            break

        if frame_index % sample_every_n_frames == 0:
            timestamp = start_seconds + (frame_index / fps)
            if (
                max_duration_seconds is not None
                and timestamp - start_seconds > max_duration_seconds
            ):
                break

            state = extract_state_from_frame(frame, regions)
            state["timestamp"] = timestamp
            states.append(state)
            refresh_live_plays()

            if display_enabled or preview_frame_path:
                draw_live_overlay(
                    frame,
                    regions,
                    state,
                    last_play,
                    play_count,
                    csv_path,
                    total_duration_seconds=total_duration_seconds,
                )
            if preview_frame_path:
                preview_path = Path(preview_frame_path)
                preview_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_preview = preview_path.with_suffix(".tmp.png")
                cv2.imwrite(str(temporary_preview), frame)
                os.replace(temporary_preview, preview_path)
            if display_enabled:
                try:
                    cv2.imshow(window, frame)
                    if (cv2.waitKey(1) & 0xFF) == ord("q"):
                        break
                except cv2.error:
                    display_enabled = False
                    print(
                        "NOTE: no display available; continuing without a video window."
                    )

            saved += 1
            if max_frames is not None and saved >= max_frames:
                break

        frame_index += 1

    capture.release()
    if display_enabled:
        cv2.destroyAllWindows()
    final_summaries = build_situation_summaries([dict(state) for state in states])
    refine_summary_calls(video_path, final_summaries, regions)
    visible_plays = [
        play
        for play in plays_from_summaries(final_summaries)
        if has_game_state_anchor(play)
    ]
    csv_path = write_football_csv(visible_plays, output_dir, finalized=True)
    play_count = len(visible_plays)
    with open(
        os.path.join(output_dir, "frame_states.json"), "w", encoding="utf-8"
    ) as state_file:
        json.dump(
            {
                "version": FRAME_STATE_CACHE_VERSION,
                "sample_every_n_frames": sample_every_n_frames,
                "start_seconds": start_seconds,
                "states": states,
            },
            state_file,
            indent=2,
        )
    with open(
        os.path.join(output_dir, "per_play_records.json"), "w", encoding="utf-8"
    ) as output_file:
        json.dump(visible_plays, output_file, indent=2)
    if calibration_path:
        calibration_result = write_calibration_report(
            visible_plays,
            calibration_path,
            output_dir,
        )
        if calibration_result is not None:
            summary = calibration_result[2]["summary"]
            print(
                "Calibration: "
                f"{summary['matched_fields']} matched, "
                f"{summary['mismatched_fields']} mismatched, "
                f"{summary['missing_fields']} missing "
                f"(accuracy {summary['field_accuracy']:.1%})"
            )
    print(f"Live scan complete: {play_count} plays written to {csv_path}")
    return csv_path


def write_metadata_csv(records: list[FrameRecord], output_dir: str) -> str:
    metadata_path = os.path.join(output_dir, "frame_metadata.csv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "timestamp_seconds", "output_path"])
        for record in records:
            writer.writerow(
                [
                    record.frame_index,
                    f"{record.timestamp_seconds:.3f}",
                    record.output_path,
                ]
            )
    return metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest a video and extract sampled frames with timestamp metadata. "
            "This is the first step before deeper Madden play analysis."
        )
    )
    parser.add_argument(
        "video",
        nargs="?",
        default="output/downloads/source_video.mp4",
        help="Path to input video file (defaults to the local test video)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where frames and metadata are written",
    )
    parser.add_argument(
        "--sample-every-n-frames",
        type=int,
        default=30,
        help="Save one frame every N frames (default: 30)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on number of extracted frames",
    )
    parser.add_argument(
        "--url-clip-seconds",
        type=int,
        default=None,
        help=(
            "For URL inputs, optionally download only the first N seconds "
            "for faster testing (default: download the complete video)"
        ),
    )
    parser.add_argument(
        "--record-live",
        action="store_true",
        help="Record a live Twitch URL from the beginning of available live media until it ends",
    )
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
        help="Use authorized browser cookies for subscriber-only or private video access",
    )
    parser.add_argument(
        "--frame-diff-threshold",
        type=float,
        default=18.0,
        help="Threshold for frame-difference based play boundary detection",
    )
    parser.add_argument(
        "--min-boundary-gap-samples",
        type=int,
        default=2,
        help="Minimum sample gap between detected play boundaries",
    )
    parser.add_argument(
        "--layout-config",
        default="configs/layouts/source_video.json",
        help="Path to JSON layout config with visual artifact regions",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="Start sampling at this timestamp in seconds",
    )
    parser.add_argument(
        "--save-discovered-layout",
        action="store_true",
        help="Save auto-discovered layout when no config file is found",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=None,
        help="Optional full path to tesseract executable",
    )
    parser.add_argument(
        "--max-duration-seconds",
        type=float,
        default=None,
        help="Optional extraction duration in seconds counted from --start-seconds",
    )
    parser.add_argument(
        "--skip-no-scoreboard",
        action="store_true",
        help="Skip sampled frames where in-game scoreboard is not visible",
    )
    parser.add_argument(
        "--min-scoreboard-visibility",
        type=float,
        default=0.6,
        help="Visibility threshold for --skip-no-scoreboard, range 0.0-1.0",
    )
    parser.add_argument(
        "--stop-at-possession-change",
        action="store_true",
        help="Truncate output at the first detected offense-team possession change",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Play the video in a window while scanning and append plays to the CSV as they are detected",
    )
    parser.add_argument(
        "--live-preview-path",
        default=None,
        help="Write the current overlaid live-monitor frame to this image path",
    )
    parser.add_argument(
        "--no-live-window",
        action="store_true",
        help="Run live monitoring without opening a separate OpenCV window",
    )
    parser.add_argument(
        "--calibration-file",
        default="configs/calibration/source_video_first_14.json",
        help="Optional labeled truth set used to write accuracy reports without changing extracted data",
    )
    parser.add_argument(
        "--reuse-state-cache",
        action="store_true",
        help="Reuse output/frame_states.json when its sampled-frame signature matches",
    )
    parser.add_argument(
        "--ocr-workers",
        type=int,
        default=4,
        help="Concurrent frame OCR workers (default: 4)",
    )
    parser.add_argument(
        "--no-refine-calls",
        action="store_true",
        help="Disable targeted half-second rescans for partially readable previous-play banners",
    )

    args = parser.parse_args()

    if args.sample_every_n_frames <= 0:
        raise ValueError("--sample-every-n-frames must be greater than 0")
    if args.url_clip_seconds is not None and args.url_clip_seconds <= 0:
        raise ValueError("--url-clip-seconds must be greater than 0")
    if args.frame_diff_threshold <= 0:
        raise ValueError("--frame-diff-threshold must be greater than 0")
    if args.min_boundary_gap_samples < 1:
        raise ValueError("--min-boundary-gap-samples must be at least 1")
    if args.start_seconds < 0:
        raise ValueError("--start-seconds must be at least 0")
    if args.max_duration_seconds is not None and args.max_duration_seconds <= 0:
        raise ValueError("--max-duration-seconds must be greater than 0")
    if not 0.0 <= args.min_scoreboard_visibility <= 1.0:
        raise ValueError("--min-scoreboard-visibility must be between 0.0 and 1.0")
    if args.ocr_workers < 1:
        raise ValueError("--ocr-workers must be at least 1")

    return args


def main() -> None:
    args = parse_args()
    configure_tesseract(args.tesseract_cmd)

    video_input = args.video
    if is_url(video_input):
        print(f"Downloading video from URL: {video_input}")
        video_input = download_video(
            video_input,
            args.output_dir,
            clip_seconds=args.url_clip_seconds,
            record_live=args.record_live,
            cookies_from_browser=args.cookies_from_browser,
        )
        print(f"Downloaded video: {video_input}")

    if args.live:
        layout_regions = load_layout_config(args.layout_config)
        if layout_regions is None:
            raise RuntimeError(
                "Live mode needs a layout config. Run once without --live "
                "(or with --save-discovered-layout) to create one first."
            )
        run_live(
            video_path=video_input,
            output_dir=args.output_dir,
            scoreboard_region=layout_regions["scoreboard"],
            banner_region=layout_regions["play_call_banner"],
            sample_every_n_frames=args.sample_every_n_frames,
            max_frames=args.max_frames,
            start_seconds=args.start_seconds,
            max_duration_seconds=args.max_duration_seconds,
            calibration_path=args.calibration_file,
            preview_frame_path=args.live_preview_path,
            show_window=not args.no_live_window,
        )
        return

    records = extract_frames(
        video_path=video_input,
        output_dir=args.output_dir,
        sample_every_n_frames=args.sample_every_n_frames,
        max_frames=args.max_frames,
        start_seconds=args.start_seconds,
        max_duration_seconds=args.max_duration_seconds,
    )

    if not records:
        raise RuntimeError(
            "No frames were extracted. Check input video and sampling settings."
        )

    layout_regions = load_layout_config(args.layout_config)
    if args.save_discovered_layout or layout_regions is None:
        layout_regions = discover_layout_from_records(records)

        if args.save_discovered_layout:
            save_layout_config(args.layout_config, layout_regions)

    scoreboard_region = layout_regions["scoreboard"]
    play_call_region = layout_regions["play_call_banner"]

    skipped_frames = 0
    if args.skip_no_scoreboard:
        records, skipped_frames = filter_records_by_scoreboard_visibility(
            records,
            scoreboard_region,
            min_visibility_score=args.min_scoreboard_visibility,
        )
        if not records:
            raise RuntimeError(
                "All sampled frames were skipped because scoreboard was not visible."
            )

    metadata_path = write_metadata_csv(records, args.output_dir)

    boundaries = detect_play_boundaries(
        records,
        diff_threshold=args.frame_diff_threshold,
        min_boundary_gap_samples=args.min_boundary_gap_samples,
    )

    debug_index = len(records) // 2
    write_layout_debug_images(
        args.output_dir, records[debug_index].output_path, layout_regions
    )

    plays = build_per_play_records(
        records,
        boundaries,
        scoreboard_region,
        play_call_region,
        os.path.join(args.output_dir, "frame_states.json"),
        args.reuse_state_cache,
        args.ocr_workers,
        video_input,
        not args.no_refine_calls,
    )
    if args.stop_at_possession_change:
        plays = truncate_at_possession_change(plays)
    boundaries_path, per_play_path = write_analysis_outputs(
        args.output_dir, boundaries, plays
    )
    football_csv_path = write_football_csv(plays, args.output_dir, finalized=True)
    calibration_result = write_calibration_report(
        plays,
        args.calibration_file,
        args.output_dir,
    )

    print(f"Saved {len(records)} frames")
    if args.skip_no_scoreboard:
        print(f"Skipped frames (no scoreboard): {skipped_frames}")
    print(f"Metadata: {metadata_path}")
    print(f"Detected boundaries: {len(boundaries)}")
    print(f"Boundary data: {boundaries_path}")
    print(f"Per-play records: {per_play_path}")
    print(f"Football CSV: {football_csv_path}")
    if calibration_result is not None:
        calibration_json, calibration_csv, report = calibration_result
        summary = report["summary"]
        print(f"Calibration JSON: {calibration_json}")
        print(f"Calibration CSV: {calibration_csv}")
        print(
            "Calibration: "
            f"{summary['matched_fields']} matched, "
            f"{summary['mismatched_fields']} mismatched, "
            f"{summary['missing_fields']} missing "
            f"(accuracy {summary['field_accuracy']:.1%})"
        )
    print(f"Layout config: {args.layout_config}")


if __name__ == "__main__":
    main()
