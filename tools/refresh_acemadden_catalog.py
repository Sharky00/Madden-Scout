"""Refresh the offline Madden 27 play-name catalog from AceMadden."""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BASE_URL = "https://acemadden.com/playbooks/plays/madden27"
DEFAULT_OUTPUT = Path("configs/reference/acemadden_madden27_plays.json")
SCRIPT_PATTERN = re.compile(r"self\.__next_f\.push\((\[.*?\])\)</script>", re.DOTALL)
PLAY_PATTERN = re.compile(r"\{\"side\":\"(?:offense|defense)\".*?\"teamCount\":\d+\}")
TOTAL_PAGES_PATTERN = re.compile(r"\"totalPages\":(\d+)")
USER_AGENT = "Madden-Scout/1.0 (+offline play-name validation)"


def fetch_page(side: str, page: int) -> str:
    query = urlencode({"side": side, "page": page})
    request = Request(f"{BASE_URL}?{query}", headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def parse_page(payload: str) -> tuple[list[dict[str, object]], int]:
    decoded_payloads: list[str] = []
    for raw_value in SCRIPT_PATTERN.findall(payload):
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            continue
        if len(value) > 1 and isinstance(value[1], str):
            decoded_payloads.append(value[1])

    plays = [
        json.loads(raw_play)
        for decoded in decoded_payloads
        for raw_play in PLAY_PATTERN.findall(decoded)
    ]
    page_match = next(
        (
            TOTAL_PAGES_PATTERN.search(decoded)
            for decoded in decoded_payloads
            if TOTAL_PAGES_PATTERN.search(decoded)
        ),
        None,
    )
    if page_match is None or not plays:
        raise RuntimeError("AceMadden response did not contain totalPages metadata")
    return plays, int(page_match.group(1))


def fetch_side(side: str, delay_seconds: float) -> list[dict[str, object]]:
    first_payload = fetch_page(side, 1)
    plays, total_pages = parse_page(first_payload)
    print(f"{side}: page 1/{total_pages}, {len(plays)} plays")

    for page in range(2, total_pages + 1):
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        page_plays, observed_pages = parse_page(fetch_page(side, page))
        if observed_pages != total_pages:
            raise RuntimeError(
                f"AceMadden page count changed during refresh: {total_pages} -> {observed_pages}"
            )
        plays.extend(page_plays)
        if page % 25 == 0 or page == total_pages:
            print(f"{side}: page {page}/{total_pages}, {len(plays)} plays")

    unique = {
        (play["side"], play["formationSlug"], play["playSlug"]): play for play in plays
    }
    return list(unique.values())


def is_special_teams(play: dict[str, object]) -> bool:
    searchable = " ".join(
        str(play.get(field, ""))
        for field in ("formationFamily", "formation", "playName", "playType")
    ).upper()
    return any(
        phrase in searchable
        for phrase in (
            "SPECIAL TEAM",
            "KICKOFF",
            "FIELD GOAL",
            "PUNT",
            "FAKE FG",
            "FAKE PUNT",
        )
    )


def catalog_record(play: dict[str, object]) -> dict[str, object]:
    side = str(play["side"])
    formation_slug = str(play["formationSlug"])
    play_slug = str(play["playSlug"])
    return {
        "side": side,
        "unit": "special_teams" if is_special_teams(play) else side,
        "formation_family": play["formationFamily"],
        "formation": play["formation"],
        "formation_slug": formation_slug,
        "play_name": play["playName"],
        "play_slug": play_slug,
        "play_type": play["playType"],
        "team_count": play["teamCount"],
        "source_url": f"{BASE_URL}/{side}/{formation_slug}/{play_slug}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--delay-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_plays = fetch_side("offense", args.delay_seconds) + fetch_side(
        "defense", args.delay_seconds
    )
    plays = sorted(
        (catalog_record(play) for play in raw_plays),
        key=lambda play: (
            str(play["side"]),
            str(play["formation_family"]),
            str(play["formation"]),
            str(play["play_name"]),
        ),
    )
    names = {
        side: sorted(
            {str(play["play_name"]) for play in plays if play["side"] == side},
            key=str.casefold,
        )
        for side in ("offense", "defense")
    }
    catalog = {
        "source": BASE_URL,
        "game": "Madden NFL 27",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_key": ["side", "formation_slug", "play_slug"],
        "counts": {
            "total": len(plays),
            "offense": sum(play["unit"] == "offense" for play in plays),
            "defense": sum(play["unit"] == "defense" for play in plays),
            "special_teams": sum(play["unit"] == "special_teams" for play in plays),
        },
        "offense": names["offense"],
        "defense": names["defense"],
        "plays": plays,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {catalog['counts']}")


if __name__ == "__main__":
    main()
