# pip install requests python-dotenv
"""
Build a complete ESPN <-> Odds API team name map.

Samples odds snapshots from busy game days throughout the season,
collects every unique team name, and cross-references against ESPN's
full D1 directory. Outputs team_name_map.json.

Usage:
    python build_team_name_map.py
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

import requests
from dotenv import load_dotenv

from ncaab_config import (
    TEAMS, ODDS_NAME_MAP, ESPN_NAME_ALIASES, HARDCODED_ESPN_IDS,
    ESPN_BASE, ODDS_BASE, RATE_LIMIT_SEC, MAX_RETRIES, RETRY_BACKOFF,
)

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR.parent / ".env", override=True)
load_dotenv(SCRIPT_DIR / ".env", override=True)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
OUTPUT_FILE = SCRIPT_DIR / "team_name_map.json"

# Busy game dates spread across the season for maximum team coverage
SAMPLE_DATES = [
    "2025-11-04", "2025-11-08", "2025-11-15", "2025-11-22", "2025-11-29",
    "2025-12-06", "2025-12-13", "2025-12-20",
    "2026-01-04", "2026-01-11", "2026-01-18", "2026-01-25",
    "2026-02-01", "2026-02-08", "2026-02-15", "2026-02-22",
    "2026-03-01", "2026-03-08",
]


def _now():
    return datetime.now().strftime("%H:%M:%S")


def api_get(url, params=None, tag=""):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                time.sleep(RATE_LIMIT_SEC)
                return resp.json()
            if resp.status_code == 429:
                wait = RETRY_BACKOFF ** attempt * 5
                print(f"  [{_now()}] Rate-limited ({tag}), waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            print(f"  [{_now()}] HTTP {resp.status_code} for {tag}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
            return None
        except requests.RequestException as exc:
            print(f"  [{_now()}] Request error ({tag}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
            return None
    return None


# ── Step 1: ESPN team directory ──────────────────────────────────────────

def fetch_espn_teams() -> dict[str, str]:
    """Return {espn_display_name_lower: espn_display_name} for all D1 teams."""
    print(f"[{_now()}] Fetching ESPN team directory ...")
    teams: dict[str, str] = {}
    page = 1
    while True:
        data = api_get(
            f"{ESPN_BASE}/teams",
            params={"limit": 500, "page": page},
            tag=f"espn-teams-p{page}",
        )
        if not data:
            break
        block = (
            data.get("sports", [{}])[0]
            .get("leagues", [{}])[0]
            .get("teams", [])
        )
        if not block:
            break
        for entry in block:
            t = entry.get("team", entry)
            display = t.get("displayName", "")
            if display:
                teams[display.lower()] = display
        if len(block) < 500:
            break
        page += 1
    print(f"  Found {len(teams)} ESPN teams.")
    return teams


# ── Step 2: Odds API team names ──────────────────────────────────────────

def fetch_odds_team_names() -> set[str]:
    """Sample busy dates and collect every unique team name from Odds API."""
    print(f"\n[{_now()}] Sampling {len(SAMPLE_DATES)} dates from Odds API "
          f"({len(SAMPLE_DATES)} API calls) ...")
    names: set[str] = set()

    for i, date_str in enumerate(SAMPLE_DATES, 1):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        offset_h = 5 if (dt.month >= 11 or dt.month <= 2) else 4
        snap_iso = dt.replace(
            hour=17 + offset_h, minute=0, second=0,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"  [{_now()}] ({i}/{len(SAMPLE_DATES)}) {date_str}")
        data = api_get(
            f"{ODDS_BASE}/v4/historical/sports/basketball_ncaab/odds",
            params={
                "apiKey": THE_ODDS_API_KEY,
                "regions": "us",
                "markets": "h2h,spreads",
                "oddsFormat": "american",
                "date": snap_iso,
            },
            tag=f"odds-{date_str}",
        )
        if not data:
            continue
        for game in data.get("data", []):
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if home:
                names.add(home)
            if away:
                names.add(away)

    print(f"  Collected {len(names)} unique Odds API team names.")
    return names


# ── Step 3: Match names ─────────────────────────────────────────────────

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def build_map(
    espn_teams: dict[str, str],
    odds_names: set[str],
) -> dict[str, str]:
    """Build espn_display_name -> odds_api_name mapping."""
    # Index odds names by lowercase
    odds_lower: dict[str, str] = {n.lower(): n for n in odds_names}

    matched: dict[str, str] = {}
    unmatched_espn: list[str] = []

    espn_list = sorted(set(espn_teams.values()))
    print(f"\n[{_now()}] Matching {len(espn_list)} ESPN names "
          f"against {len(odds_names)} Odds API names ...")

    for espn_name in espn_list:
        key = espn_name.lower()

        # Exact match
        if key in odds_lower:
            matched[espn_name] = odds_lower[key]
            continue

        # Fuzzy match: find best candidate above threshold
        best_score = 0.0
        best_odds = ""
        for odds_name in odds_names:
            score = _similarity(espn_name, odds_name)
            if score > best_score:
                best_score = score
                best_odds = odds_name

        if best_score >= 0.75:
            matched[espn_name] = best_odds
        else:
            unmatched_espn.append(espn_name)

    # Also check for odds names with no ESPN match
    matched_odds = set(matched.values())
    unmatched_odds = [n for n in sorted(odds_names) if n not in matched_odds]

    print(f"  Matched: {len(matched)}")
    if unmatched_espn:
        print(f"  Unmatched ESPN ({len(unmatched_espn)}):")
        for n in sorted(unmatched_espn):
            print(f"    {n}")
    if unmatched_odds:
        print(f"  Unmatched Odds API ({len(unmatched_odds)}):")
        for n in sorted(unmatched_odds):
            print(f"    {n}")

    return matched


# ── Step 4: Add canonical name entries ───────────────────────────────────

def add_canonical_entries(
    name_map: dict[str, str],
    espn_teams: dict[str, str],
) -> dict[str, str]:
    """Add canonical_name -> odds_api_name entries for tracked teams."""
    # Build canonical -> ESPN display name using ESPN team directory
    espn_ids_file = SCRIPT_DIR / "espn_team_ids.json"
    if not espn_ids_file.exists():
        return name_map

    with open(espn_ids_file) as f:
        team_ids = json.load(f)

    # Reverse lookup: find ESPN display name for each canonical name
    # by searching the ESPN directory for the team ID
    espn_display_by_id: dict[str, str] = {}
    for entry_key, entry_val in espn_teams.items():
        # We need the ID too — re-fetch is expensive, so match by name
        pass

    # Simpler: for each canonical name, find its ESPN display name
    # by checking aliases and the team directory
    for canonical in TEAMS:
        if canonical in name_map:
            continue
        # Try existing ODDS_NAME_MAP first
        if canonical in ODDS_NAME_MAP:
            odds_name = ODDS_NAME_MAP[canonical]
            name_map[canonical] = odds_name
            continue
        # Try to find ESPN display name via aliases
        candidates = [canonical] + ESPN_NAME_ALIASES.get(canonical, [])
        for cand in candidates:
            if cand in name_map:
                name_map[canonical] = name_map[cand]
                break
            # Check with common suffixes
            for espn_display, odds_name in list(name_map.items()):
                if espn_display.lower().startswith(cand.lower()):
                    name_map[canonical] = odds_name
                    break
            if canonical in name_map:
                break

    return name_map


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    if not THE_ODDS_API_KEY:
        print("ERROR: THE_ODDS_API_KEY not set in .env")
        return

    print("=" * 60)
    print("  Build Team Name Map: ESPN <-> Odds API")
    print("=" * 60)
    print()

    espn_teams = fetch_espn_teams()
    odds_names = fetch_odds_team_names()
    name_map = build_map(espn_teams, odds_names)
    name_map = add_canonical_entries(name_map, espn_teams)

    # Sort for readability
    sorted_map = dict(sorted(name_map.items(), key=lambda x: x[0].lower()))

    with open(OUTPUT_FILE, "w") as f:
        json.dump(sorted_map, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Saved {len(sorted_map)} entries to {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
