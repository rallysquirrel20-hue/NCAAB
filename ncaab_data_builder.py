# pip install requests python-dotenv tqdm
"""
NCAAB Data Builder
==================
Pulls full-season game logs for 76 NCAAB teams from ESPN's API,
fetches historical opening/closing odds from The Odds API, computes
Point-In-Time (PIT) statistics, and outputs everything to CSV.

Season: 2025-26 (ESPN season=2026)

Usage:
    python ncaab_data_builder.py            # Full build (skips cached data)
    python ncaab_data_builder.py --update   # Incremental update (re-fetches
                                            # schedules to find new games,
                                            # only fetches odds for new dates)
"""

import csv
import gzip
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Load .env from the same directory as this script
load_dotenv(SCRIPT_DIR / ".env", override=True)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
if not THE_ODDS_API_KEY or THE_ODDS_API_KEY == "PASTE_YOUR_KEY_HERE":
    THE_ODDS_API_KEY = None
    print("WARNING: THE_ODDS_API_KEY not set in .env -- odds fetching will be skipped")

SEASON = 2026  # ESPN uses the ending year for the season label
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ODDS_BASE = "https://api.the-odds-api.com"

OUTPUT_DIR = SCRIPT_DIR
CACHE_FILE = OUTPUT_DIR / "ncaab_cache.json"
CACHE_FILE_GZ = OUTPUT_DIR / "ncaab_cache.json.gz"
CSV_FILE = OUTPUT_DIR / "ncaab_game_logs.csv"

RATE_LIMIT_SEC = 1.0          # seconds between API calls
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0           # exponential backoff multiplier

# ---------------------------------------------------------------------------
# The 76 teams we need
# ---------------------------------------------------------------------------

TEAMS = [
    # --- Tournament Field (68) ---
    # East
    "Duke", "Siena", "Georgia", "TCU", "Arkansas", "Yale", "Purdue",
    "Northern Iowa", "Louisville", "Missouri", "VCU", "Illinois",
    "North Dakota State", "Kentucky", "Santa Clara", "Iowa State",
    "Tennessee State",
    # South
    "Florida", "Southern", "Lehigh", "Ohio State", "Clemson",
    "St. John's", "McNeese", "Texas Tech", "Utah Valley",
    "North Carolina", "South Florida", "Nebraska", "Wright State",
    "Saint Mary's", "Saint Louis", "Houston", "Furman",
    # Midwest
    "Michigan", "Howard", "Idaho", "Utah State", "Iowa", "Tennessee",
    "Akron", "Kansas", "Sam Houston", "Wisconsin", "Miami (OH)",
    "Alabama", "Troy", "Miami", "UCF", "UConn", "UMBC",
    # West
    "Arizona", "Long Island University", "UCLA", "Texas A&M",
    "Vanderbilt", "High Point", "Virginia", "Hofstra", "BYU", "Texas",
    "SMU", "Gonzaga", "UC Irvine", "Villanova", "NC State",
    "Michigan State", "Queens",
    # --- Bubble (8) ---
    "Oklahoma", "Auburn", "Indiana", "New Mexico", "San Diego State",
    "Stanford", "Cincinnati", "Seton Hall",
]

# ---------------------------------------------------------------------------
# Conference map: canonical team name -> 2025-26 conference
# ---------------------------------------------------------------------------

CONFERENCE_MAP: dict[str, str] = {
    # SEC (12 of 16)
    "Florida": "SEC", "Alabama": "SEC", "Kentucky": "SEC",
    "Tennessee": "SEC", "Georgia": "SEC", "Vanderbilt": "SEC",
    "Missouri": "SEC", "Texas A&M": "SEC", "Auburn": "SEC",
    "Oklahoma": "SEC", "Texas": "SEC", "Arkansas": "SEC",
    # Big Ten
    "Michigan": "Big Ten", "Illinois": "Big Ten", "Purdue": "Big Ten",
    "Ohio State": "Big Ten", "Iowa": "Big Ten", "Wisconsin": "Big Ten",
    "Nebraska": "Big Ten", "UCLA": "Big Ten", "Michigan State": "Big Ten",
    "Indiana": "Big Ten",
    # Big 12
    "Arizona": "Big 12", "Houston": "Big 12", "Iowa State": "Big 12",
    "Kansas": "Big 12", "Texas Tech": "Big 12", "TCU": "Big 12",
    "BYU": "Big 12", "UCF": "Big 12", "Cincinnati": "Big 12",
    # ACC
    "Duke": "ACC", "North Carolina": "ACC", "Louisville": "ACC",
    "Clemson": "ACC", "Virginia": "ACC", "Miami": "ACC",
    "NC State": "ACC", "SMU": "ACC", "Stanford": "ACC",
    # Big East
    "St. John's": "Big East", "UConn": "Big East",
    "Villanova": "Big East", "Seton Hall": "Big East",
    # WCC
    "Gonzaga": "WCC", "Saint Mary's": "WCC", "Santa Clara": "WCC",
    # A-10
    "VCU": "A-10", "Saint Louis": "A-10",
    # Mountain West
    "San Diego State": "Mountain West", "New Mexico": "Mountain West",
    "Utah State": "Mountain West",
    # AAC
    "South Florida": "AAC",
    # MVC
    "Northern Iowa": "MVC",
    # MAC
    "Akron": "MAC", "Miami (OH)": "MAC",
    # WAC
    "Sam Houston": "WAC", "Utah Valley": "WAC",
    # Big South
    "High Point": "Big South",
    # CAA
    "Hofstra": "CAA",
    # Sun Belt
    "Troy": "Sun Belt",
    # Ivy
    "Yale": "Ivy",
    # SoCon
    "Furman": "SoCon",
    # America East
    "UMBC": "America East",
    # SWAC
    "Southern": "SWAC", "Howard": "SWAC",
    # OVC
    "Tennessee State": "OVC",
    # NEC
    "Long Island University": "NEC",
    # Patriot
    "Lehigh": "Patriot",
    # Summit
    "North Dakota State": "Summit",
    # Big Sky
    "Idaho": "Big Sky",
    # Big West
    "UC Irvine": "Big West",
    # ASUN
    "Queens": "ASUN",
    # Horizon
    "Wright State": "Horizon",
    # Southland
    "McNeese": "Southland",
    # MAAC
    "Siena": "MAAC",
}

# ---------------------------------------------------------------------------
# Aliases: map a canonical short name to known ESPN display variants.
# We'll also try these when matching ESPN team names during lookup.
# ---------------------------------------------------------------------------

ESPN_NAME_ALIASES: dict[str, list[str]] = {
    "St. John's":              ["St. John's", "St. John's Red Storm", "St. John's (NY)"],
    "Miami (OH)":              ["Miami (OH)", "Miami Ohio", "Miami RedHawks"],
    "Miami":                   ["Miami", "Miami Hurricanes", "Miami (FL)"],
    "UConn":                   ["UConn", "Connecticut", "Connecticut Huskies"],
    "UCF":                     ["UCF", "Central Florida"],
    "VCU":                     ["VCU", "Virginia Commonwealth"],
    "SMU":                     ["SMU", "Southern Methodist"],
    "BYU":                     ["BYU", "Brigham Young"],
    "UMBC":                    ["UMBC", "Maryland-Baltimore County"],
    "NC State":                ["NC State", "North Carolina State"],
    "Long Island University":  ["Long Island University", "LIU", "Long Island"],
    "McNeese":                 ["McNeese", "McNeese State"],
    "Sam Houston":             ["Sam Houston", "Sam Houston State"],
    "North Dakota State":      ["North Dakota State", "North Dakota St"],
    "Wright State":            ["Wright State", "Wright St"],
    "Tennessee State":         ["Tennessee State", "Tennessee St"],
    "Southern":                ["Southern", "Southern University", "Southern Jaguars"],
    "South Florida":           ["South Florida", "USF"],
    "Queens":                  ["Queens", "Queens University", "Queens (NC)", "Queens University Royals", "Queens NC"],
    "High Point":              ["High Point"],
    "TCU":                     ["TCU", "Texas Christian"],
    "Utah Valley":             ["Utah Valley", "Utah Valley State"],
    "Northern Iowa":           ["Northern Iowa", "UNI"],
    "Michigan State":          ["Michigan State"],
    "Iowa State":              ["Iowa State"],
    "Texas A&M":               ["Texas A&M"],
    "San Diego State":         ["San Diego State"],
    "Seton Hall":              ["Seton Hall"],
    "Saint Mary's":            ["Saint Mary's", "Saint Mary's (CA)"],
    "Saint Louis":             ["Saint Louis"],
    "Santa Clara":             ["Santa Clara"],
    "North Carolina":          ["North Carolina", "UNC"],
    "Ohio State":              ["Ohio State"],
    "Texas Tech":              ["Texas Tech"],
}

# Odds API uses long-form names. We map our canonical names to likely Odds API names.
ODDS_NAME_MAP: dict[str, str] = {
    "Duke":                    "Duke Blue Devils",
    "Siena":                   "Siena Saints",
    "Georgia":                 "Georgia Bulldogs",
    "TCU":                     "TCU Horned Frogs",
    "Arkansas":                "Arkansas Razorbacks",
    "Yale":                    "Yale Bulldogs",
    "Purdue":                  "Purdue Boilermakers",
    "Northern Iowa":           "Northern Iowa Panthers",
    "Louisville":              "Louisville Cardinals",
    "Missouri":                "Missouri Tigers",
    "VCU":                     "VCU Rams",
    "Illinois":                "Illinois Fighting Illini",
    "North Dakota State":      "North Dakota St Bison",
    "Kentucky":                "Kentucky Wildcats",
    "Santa Clara":             "Santa Clara Broncos",
    "Iowa State":              "Iowa State Cyclones",
    "Tennessee State":         "Tennessee St Tigers",
    "Florida":                 "Florida Gators",
    "Southern":                "Southern Jaguars",
    "Lehigh":                  "Lehigh Mountain Hawks",
    "Ohio State":              "Ohio State Buckeyes",
    "Clemson":                 "Clemson Tigers",
    "St. John's":              "St. John's Red Storm",
    "McNeese":                 "McNeese Cowboys",
    "Texas Tech":              "Texas Tech Red Raiders",
    "Utah Valley":             "Utah Valley Wolverines",
    "North Carolina":          "North Carolina Tar Heels",
    "South Florida":           "South Florida Bulls",
    "Nebraska":                "Nebraska Cornhuskers",
    "Wright State":            "Wright St Raiders",
    "Saint Mary's":            "Saint Mary's Gaels",
    "Saint Louis":             "Saint Louis Billikens",
    "Houston":                 "Houston Cougars",
    "Furman":                  "Furman Paladins",
    "Michigan":                "Michigan Wolverines",
    "Howard":                  "Howard Bison",
    "Idaho":                   "Idaho Vandals",
    "Utah State":              "Utah State Aggies",
    "Iowa":                    "Iowa Hawkeyes",
    "Tennessee":               "Tennessee Volunteers",
    "Akron":                   "Akron Zips",
    "Kansas":                  "Kansas Jayhawks",
    "Sam Houston":             "Sam Houston St Bearkats",
    "Wisconsin":               "Wisconsin Badgers",
    "Miami (OH)":              "Miami (OH) RedHawks",
    "Alabama":                 "Alabama Crimson Tide",
    "Troy":                    "Troy Trojans",
    "Miami":                   "Miami Hurricanes",
    "UCF":                     "UCF Knights",
    "UConn":                   "UConn Huskies",
    "UMBC":                    "UMBC Retrievers",
    "Arizona":                 "Arizona Wildcats",
    "Long Island University":  "LIU Sharks",
    "UCLA":                    "UCLA Bruins",
    "Texas A&M":               "Texas A&M Aggies",
    "Vanderbilt":              "Vanderbilt Commodores",
    "High Point":              "High Point Panthers",
    "Virginia":                "Virginia Cavaliers",
    "Hofstra":                 "Hofstra Pride",
    "BYU":                     "BYU Cougars",
    "Texas":                   "Texas Longhorns",
    "SMU":                     "SMU Mustangs",
    "Gonzaga":                 "Gonzaga Bulldogs",
    "UC Irvine":               "UC Irvine Anteaters",
    "Villanova":               "Villanova Wildcats",
    "NC State":                "NC State Wolfpack",
    "Michigan State":          "Michigan St Spartans",
    "Queens":                  "Queens University Royals",
    "Oklahoma":                "Oklahoma Sooners",
    "Auburn":                  "Auburn Tigers",
    "Indiana":                 "Indiana Hoosiers",
    "New Mexico":              "New Mexico Lobos",
    "San Diego State":         "San Diego St Aztecs",
    "Stanford":                "Stanford Cardinal",
    "Cincinnati":              "Cincinnati Bearcats",
    "Seton Hall":              "Seton Hall Pirates",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_str() -> str:
    """Compact timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _sleep():
    time.sleep(RATE_LIMIT_SEC)


def api_get(url: str, params: dict | None = None, tag: str = "") -> dict | None:
    """GET *url* with retries, backoff, and rate-limiting.  Returns parsed
    JSON or ``None`` on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                _sleep()
                return resp.json()
            elif resp.status_code == 429:
                wait = RETRY_BACKOFF ** attempt * 5
                print(f"  [{_now_str()}] Rate-limited ({tag}), waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            else:
                print(f"  [{_now_str()}] HTTP {resp.status_code} for {tag}: {resp.text[:200]}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF ** attempt)
                    continue
                return None
        except requests.RequestException as exc:
            print(f"  [{_now_str()}] Request error ({tag}): {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
            return None
    return None


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------

def load_cache() -> dict:
    # Try uncompressed first, then compressed
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not load cache ({exc}), trying compressed.")
    if CACHE_FILE_GZ.exists():
        try:
            with gzip.open(CACHE_FILE_GZ, "rt", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: could not load compressed cache ({exc}), starting fresh.")
    return {}


def save_cache(cache: dict):
    # Save uncompressed (for speed during runs) and compressed (for git)
    tmp = CACHE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    tmp.replace(CACHE_FILE)
    # Also save compressed copy for git
    with gzip.open(CACHE_FILE_GZ, "wt", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


# ---------------------------------------------------------------------------
# Step 1 -- Resolve ESPN team IDs
# ---------------------------------------------------------------------------

def fetch_espn_team_map() -> dict[str, dict]:
    """Return {display_name_lower: {id, displayName, shortName, ...}} for
    every D-I team from ESPN's bulk endpoint (paginated)."""
    all_teams: dict[str, dict] = {}
    page = 1
    while True:
        url = f"{ESPN_BASE}/teams"
        data = api_get(url, params={"limit": 500, "page": page}, tag="teams-list")
        if not data:
            break
        teams_block = data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])
        if not teams_block:
            break
        for entry in teams_block:
            t = entry.get("team", entry)
            tid = str(t.get("id", ""))
            info = {
                "id": tid,
                "displayName": t.get("displayName", ""),
                "shortDisplayName": t.get("shortDisplayName", ""),
                "abbreviation": t.get("abbreviation", ""),
                "nickname": t.get("nickname", ""),
                "location": t.get("location", ""),
                "name": t.get("name", ""),
            }
            # Index under several variations for easy lookup later
            for key_field in ("displayName", "shortDisplayName", "location", "abbreviation", "nickname"):
                val = info.get(key_field, "")
                if val:
                    all_teams[val.lower()] = info
            # Also index the combined "location nickname"
            combo = f"{info['location']} {info['name']}".strip()
            if combo:
                all_teams[combo.lower()] = info
        # ESPN returns all teams in one page typically, but handle paging
        if len(teams_block) < 500:
            break
        page += 1
    return all_teams


# Hardcoded ESPN IDs for teams that may not appear in the paginated
# /teams endpoint (e.g. newer D-I programs).
HARDCODED_ESPN_IDS: dict[str, str] = {
    "Queens": "2511",
}


def resolve_team_ids(team_map: dict[str, dict]) -> dict[str, str]:
    """Map each canonical team name in TEAMS to an ESPN team ID."""
    resolved: dict[str, str] = {}
    not_found: list[str] = []

    for name in TEAMS:
        # Check hardcoded first
        if name in HARDCODED_ESPN_IDS:
            resolved[name] = HARDCODED_ESPN_IDS[name]
            continue

        # Build a list of candidate search strings
        candidates = [name]
        candidates.extend(ESPN_NAME_ALIASES.get(name, []))
        found = False
        for cand in candidates:
            key = cand.lower()
            if key in team_map:
                resolved[name] = team_map[key]["id"]
                found = True
                break
        if not found:
            # Try substring matching as fallback -- but require the match
            # to be at least 4 chars to avoid false positives
            for map_key, info in team_map.items():
                for cand in candidates:
                    cl = cand.lower()
                    if len(cl) >= 4 and (cl == map_key or cl in map_key or map_key in cl):
                        resolved[name] = info["id"]
                        found = True
                        break
                if found:
                    break
        if not found:
            not_found.append(name)

    if not_found:
        print(f"\nWARNING: Could not resolve ESPN IDs for: {not_found}")
        print("These teams will be skipped.\n")

    return resolved


# ---------------------------------------------------------------------------
# Step 2 -- Fetch schedules from ESPN
# ---------------------------------------------------------------------------

def _extract_score(competitor: dict) -> int:
    """Robustly extract an integer score from an ESPN competitor dict.

    The ``score`` field can be:
      - a dict like ``{"value": 75.0, "displayValue": "75"}``
      - a plain string ``"75"``
      - a plain int / float
    """
    raw = competitor.get("score", 0)
    if isinstance(raw, dict):
        try:
            return int(float(raw.get("value", raw.get("displayValue", 0))))
        except (ValueError, TypeError):
            try:
                return int(raw.get("displayValue", 0))
            except (ValueError, TypeError):
                return 0
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return 0


def fetch_schedule(team_id: str, team_name: str) -> list[dict]:
    """Return list of completed-game dicts from a team's schedule."""
    url = f"{ESPN_BASE}/teams/{team_id}/schedule"
    data = api_get(url, params={"season": SEASON}, tag=f"schedule-{team_name}")
    if not data:
        return []

    games: list[dict] = []
    events = data.get("events", [])
    for ev in events:
        competitions = ev.get("competitions", [])
        if not competitions:
            continue
        comp = competitions[0]

        # Only completed games
        status = comp.get("status", {}).get("type", {}).get("name", "")
        if status != "STATUS_FINAL":
            continue

        event_id = ev.get("id", "")
        date_raw = comp.get("date", "")
        game_date = ""
        if date_raw:
            try:
                dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                game_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                game_date = date_raw[:10]

        # Neutral site flag
        neutral_site = comp.get("neutralSite", False)

        # Determine competitors
        competitors = comp.get("competitors", [])
        team_comp = None
        opp_comp = None
        for c in competitors:
            cid = str(c.get("id", c.get("team", {}).get("id", "")))
            if cid == team_id:
                team_comp = c
            else:
                opp_comp = c

        if not team_comp or not opp_comp:
            continue

        home_away = team_comp.get("homeAway", "")
        # Override home_away to "neutral" when the game is at a neutral site
        if neutral_site:
            home_away = "neutral"

        team_score = _extract_score(team_comp)
        opp_score = _extract_score(opp_comp)

        opp_name_full = opp_comp.get("team", {}).get("displayName", "")
        opp_name_short = opp_comp.get("team", {}).get("shortDisplayName", opp_name_full)
        # Capture opponent ESPN team ID for later cross-referencing
        opp_team_id = str(
            opp_comp.get("id", opp_comp.get("team", {}).get("id", ""))
        )
        win_loss = "W" if team_score > opp_score else ("L" if team_score < opp_score else "T")

        games.append({
            "event_id": event_id,
            "date": game_date,
            "opponent": opp_name_full,
            "opponent_short": opp_name_short,
            "opponent_id": opp_team_id,
            "home_away": home_away,
            "neutral_site": neutral_site,
            "team_score": team_score,
            "opp_score": opp_score,
            "win_loss": win_loss,
            # Half scores to be filled from summary endpoint
            "team_1h": None,
            "opp_1h": None,
        })

    return games


# ---------------------------------------------------------------------------
# Step 3 -- Fetch half-time scores from ESPN summary endpoint
# ---------------------------------------------------------------------------

def fetch_half_scores(event_id: str, team_id: str) -> tuple:
    """Return (team_1h_score, opp_1h_score) or (None, None)."""
    url = f"{ESPN_BASE}/summary"
    data = api_get(url, params={"event": event_id}, tag=f"summary-{event_id}")
    if not data:
        return (None, None)

    # The header.competitions[0].competitors[] has linescores
    header = data.get("header", {})
    competitions = header.get("competitions", [])
    if not competitions:
        # Fallback: try boxscore
        return _half_scores_from_boxscore(data, team_id)

    comp = competitions[0]
    competitors = comp.get("competitors", [])
    team_1h = None
    opp_1h = None

    for c in competitors:
        cid = str(c.get("id", ""))
        linescores = c.get("linescores", [])
        if linescores:
            try:
                first_half = int(linescores[0].get("displayValue", linescores[0].get("value", 0)))
            except (ValueError, TypeError, IndexError):
                first_half = None
        else:
            first_half = None

        if cid == team_id:
            team_1h = first_half
        else:
            opp_1h = first_half

    return (team_1h, opp_1h)


def _half_scores_from_boxscore(data: dict, team_id: str) -> tuple:
    """Fallback: try to extract 1H scores from boxscore data."""
    boxscore = data.get("boxscore", {})
    teams = boxscore.get("teams", [])
    team_1h = None
    opp_1h = None
    for t in teams:
        tid = str(t.get("team", {}).get("id", ""))
        stats = t.get("statistics", [])
        # Not reliable for half scores, return None
    return (team_1h, opp_1h)


# ---------------------------------------------------------------------------
# Step 4 -- Fetch historical odds from The Odds API
# ---------------------------------------------------------------------------

def fetch_historical_odds(date_str: str, iso_timestamp: str) -> dict | None:
    """Fetch historical odds for basketball_ncaab at *iso_timestamp*.

    Returns the raw JSON response or None.
    """
    url = f"{ODDS_BASE}/v4/historical/sports/basketball_ncaab/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "date": iso_timestamp,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }
    data = api_get(url, params=params, tag=f"odds-fg-{date_str}")
    return data


def fetch_historical_odds_1h(date_str: str, iso_timestamp: str) -> dict | None:
    """Fetch historical 1H odds."""
    url = f"{ODDS_BASE}/v4/historical/sports/basketball_ncaab/odds"
    params = {
        "apiKey": THE_ODDS_API_KEY,
        "date": iso_timestamp,
        "regions": "us",
        "markets": "h2h_h1,spreads_h1",
        "oddsFormat": "american",
    }
    data = api_get(url, params=params, tag=f"odds-1h-{date_str}")
    return data


def _build_odds_team_lookup() -> dict[str, str]:
    """Build a reverse lookup from lower-cased Odds API team name to our
    canonical name."""
    lookup: dict[str, str] = {}
    for canonical, odds_name in ODDS_NAME_MAP.items():
        lookup[odds_name.lower()] = canonical
        # Also add without mascot (just first word(s))
        parts = odds_name.rsplit(" ", 1)
        if len(parts) == 2:
            lookup[parts[0].lower()] = canonical
    return lookup


def _consensus_line(bookmaker_list: list[dict], market_key: str, team_name_lower: str,
                    field: str = "price") -> tuple:
    """Extract odds for *team_name_lower* from the first bookmaker that has
    the market.  Returns (team_value, opp_value) -- e.g. spread points or
    moneyline price.

    *field*: 'price' for moneyline, or we return both price and point for
    spreads.
    """
    for bm in bookmaker_list:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            outcomes = mkt.get("outcomes", [])
            team_val = None
            opp_val = None
            for oc in outcomes:
                if oc.get("name", "").lower() == team_name_lower:
                    if market_key in ("spreads", "spreads_h1"):
                        team_val = oc.get("point")
                    else:
                        team_val = oc.get("price")
                else:
                    if market_key in ("spreads", "spreads_h1"):
                        opp_val = oc.get("point")
                    else:
                        opp_val = oc.get("price")
            if team_val is not None:
                return (team_val, opp_val)
    return (None, None)


def _consensus_moneyline(bookmaker_list: list[dict], market_key: str,
                         team_name_lower: str) -> tuple:
    """Return (team_ml, opp_ml)."""
    for bm in bookmaker_list:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            outcomes = mkt.get("outcomes", [])
            team_ml = None
            opp_ml = None
            for oc in outcomes:
                if oc.get("name", "").lower() == team_name_lower:
                    team_ml = oc.get("price")
                else:
                    opp_ml = oc.get("price")
            if team_ml is not None:
                return (team_ml, opp_ml)
    return (None, None)


def parse_odds_snapshot(raw: dict | None, canonical_name: str,
                        opp_hint: str = "") -> dict:
    """Pull team-specific odds out of a raw historical-odds response.

    *opp_hint*: optional opponent display name (from ESPN) used to verify
    we're matching the correct game when searching adjacent dates.

    Returns dict with keys like 'ml', 'spread' (each a value or None).
    """
    result = {"ml": None, "spread": None, "ml_1h": None, "spread_1h": None}
    if not raw:
        return result

    odds_name = ODDS_NAME_MAP.get(canonical_name, "")
    if not odds_name:
        return result
    odds_lower = odds_name.lower()

    # Build a keyword from the opponent hint for verification
    # e.g. "Kansas State Wildcats" -> ["kansas", "state", "wildcats"]
    opp_keywords = [w.lower() for w in opp_hint.split() if len(w) >= 3] if opp_hint else []

    # The historical endpoint returns {"data": [...games...], ...}
    games_list = raw.get("data", [])
    if not games_list:
        return result

    # Find the game that contains our team (exact match first, then fuzzy)
    odds_parts = odds_lower.rsplit(" ", 1)
    odds_prefix = odds_parts[0] if len(odds_parts) == 2 else odds_lower

    matched_game = None
    matched_team_key = odds_lower

    for game in games_list:
        home = game.get("home_team", "").lower()
        away = game.get("away_team", "").lower()

        # Exact match only — no fuzzy prefix matching to avoid
        # "arizona wildcats" matching "arizona st sun devils" etc.
        if odds_lower == home:
            matched_team_key = odds_lower
        elif odds_lower == away:
            matched_team_key = odds_lower
        else:
            continue

        # Verify opponent matches (if hint provided)
        if opp_keywords:
            other_team = away if odds_lower == home else home
            # Check that at least 1 keyword from the opponent name appears
            if not any(kw in other_team for kw in opp_keywords):
                continue

        matched_game = game
        break

    if not matched_game:
        return result

    bookmakers = matched_game.get("bookmakers", [])
    if not bookmakers:
        return result

    ml_val, _ = _consensus_moneyline(bookmakers, "h2h", matched_team_key)
    spread_val, _ = _consensus_line(bookmakers, "spreads", matched_team_key)
    result["ml"] = ml_val
    result["spread"] = spread_val

    return result


def parse_odds_snapshot_1h(raw: dict | None, canonical_name: str) -> dict:
    """Same as parse_odds_snapshot but for 1H markets."""
    result = {"ml_1h": None, "spread_1h": None}
    if not raw:
        return result

    odds_name = ODDS_NAME_MAP.get(canonical_name, "")
    if not odds_name:
        return result
    odds_lower = odds_name.lower()

    games_list = raw.get("data", [])
    if not games_list:
        return result

    for game in games_list:
        home = game.get("home_team", "").lower()
        away = game.get("away_team", "").lower()
        if odds_lower != home and odds_lower != away:
            continue

        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            continue

        ml_val, _ = _consensus_moneyline(bookmakers, "h2h_h1", odds_lower)
        spread_val, _ = _consensus_line(bookmakers, "spreads_h1", odds_lower)
        result["ml_1h"] = ml_val
        result["spread_1h"] = spread_val
        break

    return result


# ---------------------------------------------------------------------------
# PIT (Point-In-Time) statistics computation
# ---------------------------------------------------------------------------

def _build_espn_id_to_canonical(team_ids: dict[str, str]) -> dict[str, str]:
    """Build a reverse mapping: ESPN team ID -> canonical team name.

    Only includes our 76 tracked teams.
    """
    return {tid: name for name, tid in team_ids.items()}


def _is_conference_game(team_name: str, opp_team_id: str,
                        espn_id_to_canonical: dict[str, str]) -> bool:
    """Return True if the opponent is in the same conference as team_name."""
    team_conf = CONFERENCE_MAP.get(team_name)
    if not team_conf:
        return False
    opp_canonical = espn_id_to_canonical.get(opp_team_id)
    if not opp_canonical:
        return False
    opp_conf = CONFERENCE_MAP.get(opp_canonical)
    return team_conf == opp_conf


def compute_pit_stats(
    schedules: dict[str, list[dict]],
    team_ids: dict[str, str],
    odds_cache: dict[str, dict],
) -> dict[str, dict[str, dict]]:
    """Compute Point-In-Time stats for every team and game.

    Returns: {canonical_team_name -> {event_id -> stats_dict}}

    Each stats_dict contains records, ATS records, and PPG -- all reflecting
    the team's performance PRIOR to the game identified by event_id.
    """
    espn_id_to_canonical = _build_espn_id_to_canonical(team_ids)

    # ------------------------------------------------------------------
    # Pre-compute closing spreads for every (team, game) so the PIT loop
    # can use them for ATS tracking.
    # ------------------------------------------------------------------
    closing_spreads: dict[str, dict[str, float | None]] = {}
    # closing_spreads[team_name][event_id] = closing spread value or None

    def _find_closing_spread(date_str: str, team_name: str,
                             opp_name: str) -> float | None:
        """Extract only the closing spread for a game."""
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        hits: list[tuple[str, dict]] = []
        for offset in range(-3, 4):
            d = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
            entry = odds_cache.get(d, {})
            for snap_key in ["fg_open", "fg_close"]:
                snap = entry.get(snap_key)
                if not snap:
                    continue
                ts = snap.get("timestamp", "")
                parsed = parse_odds_snapshot(snap, team_name, opp_hint=opp_name)
                if parsed.get("spread") is not None:
                    hits.append((ts, parsed))
        if not hits:
            return None
        hits.sort(key=lambda x: x[0])
        # Closing = last snapshot with a spread
        for _, parsed in reversed(hits):
            if parsed.get("spread") is not None:
                return parsed["spread"]
        return None

    print(f"  [{_now_str()}] Pre-computing closing spreads for ATS ...")
    for team_name in TEAMS:
        closing_spreads[team_name] = {}
        for g in schedules.get(team_name, []):
            cs = _find_closing_spread(
                g.get("date", ""), team_name, g.get("opponent", "")
            )
            closing_spreads[team_name][g["event_id"]] = cs

    # ------------------------------------------------------------------
    # Compute PIT stats per team
    # ------------------------------------------------------------------
    print(f"  [{_now_str()}] Computing PIT records, ATS, and PPG ...")
    pit_stats: dict[str, dict[str, dict]] = {}

    for team_name in TEAMS:
        games = schedules.get(team_name, [])
        sorted_games = sorted(games, key=lambda g: g["date"])

        # Running win/loss counters
        overall_w, overall_l = 0, 0
        home_w, home_l = 0, 0
        neutral_w, neutral_l = 0, 0
        away_w, away_l = 0, 0

        # Running ATS counters
        ats_overall_w, ats_overall_l, ats_overall_p = 0, 0, 0
        ats_home_w, ats_home_l, ats_home_p = 0, 0, 0
        ats_neutral_w, ats_neutral_l, ats_neutral_p = 0, 0, 0
        ats_away_w, ats_away_l, ats_away_p = 0, 0, 0

        # Running PPG counters
        total_pts, total_gp = 0, 0
        home_pts, home_gp = 0, 0
        neutral_pts, neutral_gp = 0, 0
        away_pts, away_gp = 0, 0

        team_pit: dict[str, dict] = {}

        for g in sorted_games:
            event_id = g["event_id"]

            # ---- SAVE current running stats (BEFORE this game) ----
            team_pit[event_id] = {
                "record": f"{overall_w}-{overall_l}",
                "home_record": f"{home_w}-{home_l}",
                "neutral_record": f"{neutral_w}-{neutral_l}",
                "away_record": f"{away_w}-{away_l}",
                "ats_record": f"{ats_overall_w}-{ats_overall_l}-{ats_overall_p}",
                "ats_home": f"{ats_home_w}-{ats_home_l}-{ats_home_p}",
                "ats_neutral": f"{ats_neutral_w}-{ats_neutral_l}-{ats_neutral_p}",
                "ats_away": f"{ats_away_w}-{ats_away_l}-{ats_away_p}",
                "ppg": round(total_pts / total_gp, 1) if total_gp > 0 else "",
                "home_ppg": round(home_pts / home_gp, 1) if home_gp > 0 else "",
                "neutral_ppg": round(neutral_pts / neutral_gp, 1) if neutral_gp > 0 else "",
                "away_ppg": round(away_pts / away_gp, 1) if away_gp > 0 else "",
            }

            # ---- UPDATE running stats with this game's results ----
            w = g["win_loss"] == "W"
            ha = g["home_away"]  # "home", "away", or "neutral"
            score = g["team_score"]

            overall_w += int(w)
            overall_l += int(not w)
            total_pts += score
            total_gp += 1

            if ha == "home":
                home_w += int(w)
                home_l += int(not w)
                home_pts += score
                home_gp += 1
            elif ha == "neutral":
                neutral_w += int(w)
                neutral_l += int(not w)
                neutral_pts += score
                neutral_gp += 1
            else:  # away
                away_w += int(w)
                away_l += int(not w)
                away_pts += score
                away_gp += 1

            # ATS update (only when closing spread exists)
            cs = closing_spreads.get(team_name, {}).get(event_id)
            if cs is not None:
                margin = g["team_score"] - g["opp_score"]
                ats_val = margin + cs  # spread is from team perspective
                if ats_val > 0:
                    ats_overall_w += 1
                    if ha == "home":
                        ats_home_w += 1
                    elif ha == "neutral":
                        ats_neutral_w += 1
                    else:
                        ats_away_w += 1
                elif ats_val < 0:
                    ats_overall_l += 1
                    if ha == "home":
                        ats_home_l += 1
                    elif ha == "neutral":
                        ats_neutral_l += 1
                    else:
                        ats_away_l += 1
                else:  # push
                    ats_overall_p += 1
                    if ha == "home":
                        ats_home_p += 1
                    elif ha == "neutral":
                        ats_neutral_p += 1
                    else:
                        ats_away_p += 1

        pit_stats[team_name] = team_pit

    return pit_stats


# ---------------------------------------------------------------------------
# Step 5 -- Main orchestration
# ---------------------------------------------------------------------------

def _merge_schedules(existing: list[dict], fresh: list[dict]) -> tuple[list[dict], int]:
    """Merge freshly fetched games into existing cached games.

    Returns (merged_list, count_of_new_games).
    Existing games are kept as-is (preserving cached half scores etc.).
    Only truly new event_ids are appended.
    """
    existing_ids = {g["event_id"] for g in existing}
    new_games = [g for g in fresh if g["event_id"] not in existing_ids]
    return existing + new_games, len(new_games)


def main():
    # Parse CLI flags
    update_mode = "--update" in sys.argv

    mode_label = "INCREMENTAL UPDATE" if update_mode else "FULL BUILD"
    print("=" * 70)
    print(f"  NCAAB Data Builder  --  Season 2025-26  [{mode_label}]")
    print("=" * 70)
    print()

    cache = load_cache()

    # ------------------------------------------------------------------
    # 1. Resolve ESPN team IDs
    # ------------------------------------------------------------------
    if "team_ids" in cache and len(cache["team_ids"]) >= len(TEAMS) - 5:
        print(f"[{_now_str()}] Using cached ESPN team IDs ({len(cache['team_ids'])} teams)")
        team_ids = cache["team_ids"]
    else:
        print(f"[{_now_str()}] Fetching ESPN team directory ...")
        team_map = fetch_espn_team_map()
        print(f"  Found {len(team_map)} index entries in ESPN directory")
        team_ids = resolve_team_ids(team_map)
        print(f"  Resolved {len(team_ids)}/{len(TEAMS)} team IDs")
        cache["team_ids"] = team_ids
        save_cache(cache)

    # ------------------------------------------------------------------
    # 2. Fetch schedules + half scores
    # ------------------------------------------------------------------
    if "schedules" not in cache:
        cache["schedules"] = {}

    schedules: dict[str, list[dict]] = cache["schedules"]

    # Check if schedules need re-fetch (missing neutral_site or opponent_id)
    if not update_mode and schedules and any(
        any("neutral_site" not in g or "opponent_id" not in g for g in games)
        for games in schedules.values() if games
    ):
        print(f"[{_now_str()}] Schedules missing neutral_site/opponent_id, re-fetching ...")
        schedules = {}
        cache["schedules"] = schedules

    if update_mode:
        # --- INCREMENTAL MODE ---
        # Re-fetch ALL schedules from ESPN (free API) and merge new games
        teams_with_ids = [t for t in TEAMS if t in team_ids]
        print(f"\n[{_now_str()}] Incremental update: re-fetching schedules for {len(teams_with_ids)} teams ...")
        total_new = 0
        for i, team_name in enumerate(teams_with_ids, 1):
            tid = team_ids[team_name]
            fresh_games = fetch_schedule(tid, team_name)
            existing_games = schedules.get(team_name, [])
            merged, new_count = _merge_schedules(existing_games, fresh_games)
            schedules[team_name] = merged
            if new_count > 0:
                print(f"  [{_now_str()}] ({i}/{len(teams_with_ids)}) {team_name}: +{new_count} new games")
                total_new += new_count

            if i % 10 == 0:
                cache["schedules"] = schedules
                save_cache(cache)

        print(f"  [{_now_str()}] Update complete: {total_new} new games found across all teams")
    else:
        # --- FULL BUILD MODE (original behavior) ---
        teams_to_fetch = [t for t in TEAMS if t in team_ids and t not in schedules]

        if teams_to_fetch:
            print(f"\n[{_now_str()}] Fetching schedules for {len(teams_to_fetch)} teams ...")
        else:
            print(f"\n[{_now_str()}] All {len(schedules)} team schedules already cached.")

        for i, team_name in enumerate(teams_to_fetch, 1):
            tid = team_ids[team_name]
            print(f"  [{_now_str()}] ({i}/{len(teams_to_fetch)}) {team_name} (ESPN id={tid}) ...")
            games = fetch_schedule(tid, team_name)
            schedules[team_name] = games
            print(f"    -> {len(games)} completed games")

            # Save every 5 teams
            if i % 5 == 0:
                cache["schedules"] = schedules
                save_cache(cache)

    cache["schedules"] = schedules
    save_cache(cache)

    # ------------------------------------------------------------------
    # 2b. Fetch half-time scores for games missing them
    # ------------------------------------------------------------------
    half_score_needed: list[tuple[str, int]] = []
    for team_name, games in schedules.items():
        if team_name not in team_ids:
            continue
        for idx, g in enumerate(games):
            if g.get("team_1h") is None and g.get("event_id"):
                half_score_needed.append((team_name, idx))

    if half_score_needed:
        # Deduplicate by event_id to avoid fetching the same summary twice
        event_cache: dict[str, tuple] = {}
        if "half_score_cache" in cache:
            event_cache = {k: tuple(v) for k, v in cache["half_score_cache"].items()}

        needed_events: dict[str, list[tuple[str, int]]] = {}
        for team_name, idx in half_score_needed:
            eid = schedules[team_name][idx]["event_id"]
            if eid in event_cache:
                continue
            needed_events.setdefault(eid, []).append((team_name, idx))

        total_events = len(needed_events)
        if total_events > 0:
            print(f"\n[{_now_str()}] Fetching half-time scores for {total_events} unique events ...")

        done = 0
        for eid, entries in needed_events.items():
            done += 1
            # Use the first team's ID for the summary call (doesn't matter which)
            first_team = entries[0][0]
            tid = team_ids[first_team]
            team_1h, opp_1h = fetch_half_scores(eid, tid)
            event_cache[eid] = (team_1h, opp_1h, tid)

            if done % 25 == 0 or done == total_events:
                print(f"    [{_now_str()}] Half scores: {done}/{total_events}")
                cache["half_score_cache"] = {k: list(v) for k, v in event_cache.items()}
                save_cache(cache)

        cache["half_score_cache"] = {k: list(v) for k, v in event_cache.items()}

        # Now apply the cached half scores to all games.
        # The summary endpoint returns both teams' linescores keyed by
        # competitor ID, so if the event was fetched for team A we can
        # simply swap team/opp for team B (the other competitor).
        for team_name, games in schedules.items():
            if team_name not in team_ids:
                continue
            tid = team_ids[team_name]
            for g in games:
                eid = g.get("event_id", "")
                if eid in event_cache and g.get("team_1h") is None:
                    cached = event_cache[eid]
                    cached_tid = str(cached[2]) if len(cached) > 2 else None
                    if cached_tid == tid:
                        # Same perspective -- use as-is
                        g["team_1h"] = cached[0]
                        g["opp_1h"] = cached[1]
                    else:
                        # Different perspective -- swap team and opp
                        g["team_1h"] = cached[1]
                        g["opp_1h"] = cached[0]

        cache["schedules"] = schedules
        save_cache(cache)
    else:
        print(f"[{_now_str()}] All half-time scores already cached.")

    # ------------------------------------------------------------------
    # 3. Collect all unique game dates
    # ------------------------------------------------------------------
    game_dates: set[str] = set()
    for team_name, games in schedules.items():
        for g in games:
            d = g.get("date", "")
            if d:
                game_dates.add(d)

    game_dates_sorted = sorted(game_dates)
    print(f"\n[{_now_str()}] Total unique game dates: {len(game_dates_sorted)}")
    if game_dates_sorted:
        print(f"    Range: {game_dates_sorted[0]} to {game_dates_sorted[-1]}")

    # ------------------------------------------------------------------
    # 4. Fetch historical odds grouped by date
    # ------------------------------------------------------------------
    if "odds_cache" not in cache:
        cache["odds_cache"] = {}
    odds_cache: dict[str, dict] = cache["odds_cache"]

    # The historical endpoint only supports featured markets (h2h, spreads, totals).
    # 1H markets (h2h_h1, spreads_h1) are NOT available historically.
    # So we fetch 2 calls per date: opening + closing for full-game markets.
    #
    # Opening: 10:00 AM ET on game day
    # Closing: 5:00 PM ET (most NCAAB games start evening; this
    #          approximates a late-afternoon closing line)

    dates_needing_odds = [d for d in game_dates_sorted if d not in odds_cache]

    if not THE_ODDS_API_KEY and dates_needing_odds:
        print(f"\n[{_now_str()}] Skipping odds fetch for {len(dates_needing_odds)} dates (no API key)")
        dates_needing_odds = []

    if dates_needing_odds:
        print(f"\n[{_now_str()}] Fetching odds for {len(dates_needing_odds)} dates ...")
        print(f"    (2 API calls per date: open FG, close FG)")
        print(f"    Estimated credit cost: {len(dates_needing_odds) * 2 * 20} credits")
        print(f"    Note: 1H odds not available via historical endpoint")
    else:
        print(f"\n[{_now_str()}] Odds already cached for all {len(game_dates_sorted)} dates.")

    for i, date_str in enumerate(dates_needing_odds, 1):
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        # Simple DST approximation: -5 for Nov-Feb (EST), -4 for Mar-Oct (EDT)
        if dt.month >= 11 or dt.month <= 2:
            offset_hours = 5
        else:
            offset_hours = 4

        open_utc = dt.replace(hour=10 + offset_hours, minute=0, second=0)
        close_utc = dt.replace(hour=17 + offset_hours, minute=0, second=0)

        open_iso = open_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
        close_iso = close_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

        print(f"  [{_now_str()}] ({i}/{len(dates_needing_odds)}) {date_str} ...")

        entry: dict = {}

        # Full game opening
        fg_open = fetch_historical_odds(date_str, open_iso)
        entry["fg_open"] = fg_open

        # Full game closing
        fg_close = fetch_historical_odds(date_str, close_iso)
        entry["fg_close"] = fg_close

        odds_cache[date_str] = entry

        # Save every 5 dates
        if i % 5 == 0 or i == len(dates_needing_odds):
            cache["odds_cache"] = odds_cache
            save_cache(cache)
            print(f"    [{_now_str()}] Cache saved ({i}/{len(dates_needing_odds)} dates done)")

    cache["odds_cache"] = odds_cache
    save_cache(cache)

    # ------------------------------------------------------------------
    # 5. Compute PIT stats
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Computing Point-In-Time statistics ...")
    pit_stats = compute_pit_stats(schedules, team_ids, odds_cache)

    # Build reverse mapping for opponent lookups
    espn_id_to_canonical = _build_espn_id_to_canonical(team_ids)

    # ------------------------------------------------------------------
    # 6. Build final rows + write CSV
    # ------------------------------------------------------------------
    print(f"\n[{_now_str()}] Building CSV rows ...")

    csv_headers = [
        "Team", "Date", "Opponent", "Home_Away", "Conference_Game",
        "Team_Final_Score", "Opp_Final_Score",
        "W_L",
        "Opening_FG_ML", "Closing_FG_ML",
        "Opening_FG_Spread", "Closing_FG_Spread",
        "Team_Record", "Team_Home_Record", "Team_Neutral_Record", "Team_Away_Record",
        "Opp_Record", "Opp_Home_Record", "Opp_Neutral_Record", "Opp_Away_Record",
        "Team_ATS", "Team_Home_ATS", "Team_Neutral_ATS", "Team_Away_ATS",
        "Opp_ATS", "Opp_Home_ATS", "Opp_Neutral_ATS", "Opp_Away_ATS",
        "Team_PPG", "Team_Home_PPG", "Team_Neutral_PPG", "Team_Away_PPG",
        "Opp_PPG", "Opp_Home_PPG", "Opp_Neutral_PPG", "Opp_Away_PPG",
    ]

    rows: list[list] = []
    odds_match_count = 0
    total_games = 0

    def _find_opening_closing(date_str: str, team_name: str,
                              opp_name: str) -> dict:
        """Find true opening and closing for each market independently.

        Opening ML    = first snapshot that has ML
        Closing ML    = last snapshot that has ML
        Opening Spread = first snapshot that has spread
        Closing Spread = last snapshot that has spread

        Returns dict with open_ml, close_ml, open_spread, close_spread.
        """
        dt = datetime.strptime(date_str, "%Y-%m-%d")

        # Collect all snapshots sorted by timestamp
        hits: list[tuple[str, dict]] = []

        for offset in range(-3, 4):
            d = (dt + timedelta(days=offset)).strftime("%Y-%m-%d")
            entry = odds_cache.get(d, {})
            for snap_key in ["fg_open", "fg_close"]:
                snap = entry.get(snap_key)
                if not snap:
                    continue
                ts = snap.get("timestamp", "")
                parsed = parse_odds_snapshot(snap, team_name, opp_hint=opp_name)
                if parsed.get("ml") is not None or parsed.get("spread") is not None:
                    hits.append((ts, parsed))

        result = {"open_ml": None, "close_ml": None,
                  "open_spread": None, "close_spread": None}

        if not hits:
            return result

        hits.sort(key=lambda x: x[0])

        # Find first/last for each market independently
        for _, parsed in hits:
            if parsed.get("ml") is not None and result["open_ml"] is None:
                result["open_ml"] = parsed["ml"]
            if parsed.get("spread") is not None and result["open_spread"] is None:
                result["open_spread"] = parsed["spread"]

        for _, parsed in reversed(hits):
            if parsed.get("ml") is not None and result["close_ml"] is None:
                result["close_ml"] = parsed["ml"]
            if parsed.get("spread") is not None and result["close_spread"] is None:
                result["close_spread"] = parsed["spread"]

        return result

    for team_name in TEAMS:
        games = schedules.get(team_name, [])
        team_pit = pit_stats.get(team_name, {})

        for g in games:
            total_games += 1
            date_str = g.get("date", "")
            opp_name = g.get("opponent", "")
            event_id = g.get("event_id", "")

            # --- Conference game determination ---
            opp_team_id = g.get("opponent_id", "")
            conf_game = _is_conference_game(team_name, opp_team_id,
                                            espn_id_to_canonical)
            conf_flag = "Y" if conf_game else "N"

            # --- Odds --- Find true opening/closing per market
            odds_result = _find_opening_closing(date_str, team_name, opp_name)

            open_fg_ml = odds_result["open_ml"]
            close_fg_ml = odds_result["close_ml"]
            open_fg_spread = odds_result["open_spread"]
            close_fg_spread = odds_result["close_spread"]

            if any(v is not None for v in [open_fg_ml, close_fg_ml, open_fg_spread, close_fg_spread]):
                odds_match_count += 1

            # --- PIT stats for this team ---
            t_pit = team_pit.get(event_id, {})
            team_record = t_pit.get("record", "0-0")
            team_home_record = t_pit.get("home_record", "0-0")
            team_neutral_record = t_pit.get("neutral_record", "0-0")
            team_away_record = t_pit.get("away_record", "0-0")
            team_ats = t_pit.get("ats_record", "0-0-0")
            team_ats_home = t_pit.get("ats_home", "0-0-0")
            team_ats_neutral = t_pit.get("ats_neutral", "0-0-0")
            team_ats_away = t_pit.get("ats_away", "0-0-0")
            team_ppg = t_pit.get("ppg", "")
            team_home_ppg = t_pit.get("home_ppg", "")
            team_neutral_ppg = t_pit.get("neutral_ppg", "")
            team_away_ppg = t_pit.get("away_ppg", "")

            # --- PIT stats for opponent (if in our 76-team list) ---
            opp_canonical = espn_id_to_canonical.get(opp_team_id)
            if opp_canonical and opp_canonical in pit_stats:
                o_pit = pit_stats[opp_canonical].get(event_id, {})
                opp_record = o_pit.get("record", "0-0")
                opp_home_record = o_pit.get("home_record", "0-0")
                opp_neutral_record = o_pit.get("neutral_record", "0-0")
                opp_away_record = o_pit.get("away_record", "0-0")
                opp_ats = o_pit.get("ats_record", "0-0-0")
                opp_ats_home = o_pit.get("ats_home", "0-0-0")
                opp_ats_neutral = o_pit.get("ats_neutral", "0-0-0")
                opp_ats_away = o_pit.get("ats_away", "0-0-0")
                opp_ppg = o_pit.get("ppg", "")
                opp_home_ppg = o_pit.get("home_ppg", "")
                opp_neutral_ppg = o_pit.get("neutral_ppg", "")
                opp_away_ppg = o_pit.get("away_ppg", "")
            else:
                # Opponent not in our 76-team list -- leave blank
                opp_record = ""
                opp_home_record = ""
                opp_neutral_record = ""
                opp_away_record = ""
                opp_ats = ""
                opp_ats_home = ""
                opp_ats_neutral = ""
                opp_ats_away = ""
                opp_ppg = ""
                opp_home_ppg = ""
                opp_neutral_ppg = ""
                opp_away_ppg = ""

            rows.append([
                team_name,
                date_str,
                g.get("opponent", ""),
                g.get("home_away", ""),
                conf_flag,
                g.get("team_score", ""),
                g.get("opp_score", ""),
                g.get("win_loss", ""),
                open_fg_ml if open_fg_ml is not None else "",
                close_fg_ml if close_fg_ml is not None else "",
                open_fg_spread if open_fg_spread is not None else "",
                close_fg_spread if close_fg_spread is not None else "",
                team_record,
                team_home_record,
                team_neutral_record,
                team_away_record,
                opp_record,
                opp_home_record,
                opp_neutral_record,
                opp_away_record,
                team_ats,
                team_ats_home,
                team_ats_neutral,
                team_ats_away,
                opp_ats,
                opp_ats_home,
                opp_ats_neutral,
                opp_ats_away,
                team_ppg,
                team_home_ppg,
                team_neutral_ppg,
                team_away_ppg,
                opp_ppg,
                opp_home_ppg,
                opp_neutral_ppg,
                opp_away_ppg,
            ])

    # Sort by date, then team
    rows.sort(key=lambda r: (r[1], r[0]))

    # Write CSV
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(rows)

    print(f"\n{'=' * 70}")
    print(f"  DONE!")
    print(f"  Total game rows:        {total_games}")
    print(f"  Games with FG odds:     {odds_match_count}")
    print(f"  CSV written to:         {CSV_FILE}")
    print(f"  Cache file:             {CACHE_FILE}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
