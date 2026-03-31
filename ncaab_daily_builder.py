# pip install requests python-dotenv pandas pyarrow
"""
NCAAB Daily Game Log Builder
=============================
Four-phase incremental pipeline:

  Phase 1 -- Fetch new games      (ESPN scoreboard, 1 API call / date)
             + boxscore stats     (ESPN summary, 1 API call / event)
  Phase 2 -- Fetch odds           (The Odds API, 1 call / NEW date only)
  Phase 3 -- Compute PIT stats    (pure math, zero API calls)
  Phase 4 -- Export CSV           (write final output)

Safe to re-run daily.  Only dates with new games trigger API calls.
Tournament games are picked up automatically.

Season: 2025-26

Usage:
    python ncaab_daily_builder.py
    python ncaab_daily_builder.py --backfill-odds
    python ncaab_daily_builder.py --backfill-boxscore
"""

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

from ncaab_config import (
    TEAMS, CONFERENCE_MAP, ESPN_NAME_ALIASES, ODDS_NAME_MAP,
    HARDCODED_ESPN_IDS, SEASON, ESPN_BASE, ODDS_BASE,
    RATE_LIMIT_SEC, MAX_RETRIES, RETRY_BACKOFF,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

load_dotenv(PROJECT_DIR / ".env", override=True)
load_dotenv(SCRIPT_DIR / ".env", override=True)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")

PARQUET_FILE = SCRIPT_DIR / "ncaab_game_logs.parquet"
TEAM_IDS_CACHE = SCRIPT_DIR / "espn_team_ids.json"
TEAM_NAME_MAP_FILE = SCRIPT_DIR / "team_name_map.json"
CSV_FILE = SCRIPT_DIR / "ncaab_game_logs.csv"
ODDS_CACHE_DIR = SCRIPT_DIR / "odds_cache"
ODDS_HISTORY_FILE = SCRIPT_DIR / "ncaab_odds_history.parquet"

SEASON_START = datetime(2025, 11, 3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _sleep():
    time.sleep(RATE_LIMIT_SEC)


def _safe_int(val) -> int | None:
    """Parse a value as int, returning None on failure."""
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


def _safe_acc(val) -> int:
    """Safe accumulate: return int value or 0 if null/NaN."""
    if pd.notna(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0
    return 0


# Track Odds API credit usage across the run
_odds_api_calls = 0
_odds_api_credits_used = 0
_odds_api_credits_remaining = None


def api_get(url: str, params: dict | None = None, tag: str = "") -> dict | None:
    """GET with retries, backoff, and rate-limiting."""
    global _odds_api_calls, _odds_api_credits_used, _odds_api_credits_remaining

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                # Track Odds API credit usage
                if ODDS_BASE in url:
                    _odds_api_calls += 1
                    last_cost = resp.headers.get("x-requests-last")
                    remaining = resp.headers.get("x-requests-remaining")
                    if last_cost is not None:
                        _odds_api_credits_used += int(last_cost)
                    if remaining is not None:
                        _odds_api_credits_remaining = int(remaining)
                    _sleep()
                return resp.json()
            if resp.status_code == 429:
                wait = RETRY_BACKOFF ** attempt * 5
                print(f"  [{_now_str()}] Rate-limited ({tag}), waiting {wait:.0f}s ...")
                time.sleep(wait)
                continue
            print(f"  [{_now_str()}] HTTP {resp.status_code} for {tag}")
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


def print_odds_api_usage():
    """Print a summary of Odds API credit usage for this run."""
    if _odds_api_calls > 0:
        print(f"  Odds API: {_odds_api_calls} calls, "
              f"{_odds_api_credits_used} credits used, "
              f"{_odds_api_credits_remaining:,} remaining")


# ---------------------------------------------------------------------------
# ESPN Team ID Resolution
# ---------------------------------------------------------------------------

def fetch_espn_team_map() -> dict[str, dict]:
    all_teams: dict[str, dict] = {}
    page = 1
    while True:
        url = f"{ESPN_BASE}/teams"
        data = api_get(url, params={"limit": 500, "page": page}, tag="teams-list")
        if not data:
            break
        teams_block = (
            data.get("sports", [{}])[0]
            .get("leagues", [{}])[0]
            .get("teams", [])
        )
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
            for field in ("displayName", "shortDisplayName", "location",
                          "abbreviation", "nickname"):
                val = info.get(field, "")
                if val:
                    all_teams[val.lower()] = info
            combo = f"{info['location']} {info['name']}".strip()
            if combo:
                all_teams[combo.lower()] = info
        if len(teams_block) < 500:
            break
        page += 1
    return all_teams


def resolve_team_ids(team_map: dict[str, dict]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    not_found: list[str] = []
    for name in TEAMS:
        if name in HARDCODED_ESPN_IDS:
            resolved[name] = HARDCODED_ESPN_IDS[name]
            continue
        candidates = [name] + ESPN_NAME_ALIASES.get(name, [])
        found = False
        for cand in candidates:
            key = cand.lower()
            if key in team_map:
                resolved[name] = team_map[key]["id"]
                found = True
                break
        if not found:
            for map_key, info in team_map.items():
                for cand in candidates:
                    cl = cand.lower()
                    if len(cl) >= 4 and (cl == map_key or cl in map_key
                                         or map_key in cl):
                        resolved[name] = info["id"]
                        found = True
                        break
                if found:
                    break
        if not found:
            not_found.append(name)
    if not_found:
        print(f"  WARNING: Could not resolve ESPN IDs for: {not_found}")
    return resolved


def get_team_ids() -> dict[str, str]:
    if TEAM_IDS_CACHE.exists():
        with open(TEAM_IDS_CACHE, "r") as f:
            cached = json.load(f)
        if len(cached) >= len(TEAMS) - 5:
            print(f"[{_now_str()}] Using cached ESPN team IDs "
                  f"({len(cached)} teams)")
            return cached
    print(f"[{_now_str()}] Fetching ESPN team directory ...")
    team_map = fetch_espn_team_map()
    team_ids = resolve_team_ids(team_map)
    print(f"  Resolved {len(team_ids)}/{len(TEAMS)} team IDs")
    with open(TEAM_IDS_CACHE, "w") as f:
        json.dump(team_ids, f, indent=2)
    return team_ids


# ---------------------------------------------------------------------------
# Full D1 Team + Conference Resolution
# ---------------------------------------------------------------------------

D1_TEAM_IDS_CACHE = SCRIPT_DIR / "d1_team_ids.json"
D1_CONF_CACHE = SCRIPT_DIR / "d1_conferences.json"


def get_all_d1_ids() -> dict[str, str]:
    """Return {canonical_name: espn_id} for ALL D1 teams (~362)."""
    if D1_TEAM_IDS_CACHE.exists():
        with open(D1_TEAM_IDS_CACHE, "r") as f:
            cached = json.load(f)
        if len(cached) >= 350:
            print(f"[{_now_str()}] Using cached D1 team IDs ({len(cached)} teams)")
            return cached

    print(f"[{_now_str()}] Fetching full D1 team directory ...")
    team_map = fetch_espn_team_map()

    # Deduplicate by ESPN ID → use displayName as canonical
    seen_ids: set[str] = set()
    d1_ids: dict[str, str] = {}
    for _key, info in team_map.items():
        tid = info["id"]
        display = info["displayName"]
        if tid not in seen_ids and display:
            d1_ids[display] = tid
            seen_ids.add(tid)

    # Layer on hardcoded overrides and tournament aliases
    for name, tid in HARDCODED_ESPN_IDS.items():
        d1_ids[name] = tid

    # Also register tournament canonical names (short names like "Duke")
    # so they map to the same IDs as their displayName counterparts
    tournament_ids = resolve_team_ids(team_map)
    for name, tid in tournament_ids.items():
        d1_ids[name] = tid

    print(f"  Resolved {len(d1_ids)} D1 team IDs")
    with open(D1_TEAM_IDS_CACHE, "w") as f:
        json.dump(d1_ids, f, indent=2)
    return d1_ids


def fetch_d1_conference_map() -> dict[str, str]:
    """Return {espn_id: conference_name} for all D1 teams.

    Uses ESPN /groups endpoint (covers ~299 teams) + individual team
    lookups for the remainder (~63 teams in missing conferences).
    """
    if D1_CONF_CACHE.exists():
        with open(D1_CONF_CACHE, "r") as f:
            cache = json.load(f)
        cached_map = cache.get("conferences", {})
        cached_date = cache.get("date", "")
        today = datetime.now().strftime("%Y-%m-%d")
        if cached_date == today and len(cached_map) >= 350:
            print(f"[{_now_str()}] Using cached D1 conference map "
                  f"({len(cached_map)} teams)")
            return cached_map

    print(f"[{_now_str()}] Fetching D1 conference map ...")
    conf_map: dict[str, str] = {}

    # Step 1: /groups endpoint (covers most conferences)
    data = api_get(f"{ESPN_BASE}/groups", tag="groups")
    if data:
        groups = data.get("groups", data if isinstance(data, list) else [])
        for top_group in groups:
            if top_group.get("name") == "Non-NCAA Division I":
                continue
            for conf in top_group.get("children", []):
                conf_name = conf.get("name", "")
                # Shorten standard suffixes
                conf_name = conf_name.replace(" Conference", "")
                for t in conf.get("teams", []):
                    conf_map[str(t["id"])] = conf_name

    # Step 2: Fill in missing teams from individual /teams/{id} endpoints
    all_d1_ids = get_all_d1_ids()
    all_espn_ids = set(all_d1_ids.values())
    missing_ids = all_espn_ids - set(conf_map.keys())

    if missing_ids:
        print(f"  Fetching conferences for {len(missing_ids)} remaining teams ...")
        for i, tid in enumerate(sorted(missing_ids), 1):
            data = api_get(f"{ESPN_BASE}/teams/{tid}", tag=f"team-{tid}")
            if data:
                team = data.get("team", {})
                standing = team.get("standingSummary", "")
                # Parse "8th in Summit" → "Summit"
                if " in " in standing:
                    conf_name = standing.split(" in ", 1)[1].strip()
                    conf_map[tid] = conf_name
            if i % 25 == 0:
                print(f"    {i}/{len(missing_ids)} ...")

    print(f"  Conference map: {len(conf_map)} teams across "
          f"{len(set(conf_map.values()))} conferences")

    with open(D1_CONF_CACHE, "w") as f:
        json.dump({
            "date": datetime.now().strftime("%Y-%m-%d"),
            "conferences": conf_map,
        }, f, indent=2)
    return conf_map


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "team", "date", "event_id", "commence_time_utc",
    "opponent", "opponent_short", "opponent_id",
    "home_away", "neutral_site", "conference_game",
    "team_score", "opp_score", "win_loss",
    "team_1h", "opp_1h",
    # Boxscore raw stats (per game, from summary)
    "team_fgm", "team_fga", "team_3pm", "team_3pa", "team_ftm", "team_fta",
    "team_oreb", "team_dreb", "team_to",
    "opp_fgm", "opp_fga", "opp_3pm", "opp_3pa", "opp_ftm", "opp_fta",
    "opp_oreb", "opp_dreb", "opp_to",
    # Opponent AP rank at game time
    "opp_ap_rank",
    # Tracking flags
    "is_tracked",
    "ap_rank_checked",
    "odds_checked",
]

ODDS_COLUMNS = [
    "opening_fg_ml", "closing_fg_ml",
    "opening_fg_spread", "closing_fg_spread",
]

PIT_COLUMNS = [
    # Record (games + win pct)
    "team_games", "team_win_pct",
    "team_home_games", "team_home_win_pct",
    "team_neutral_games", "team_neutral_win_pct",
    "team_away_games", "team_away_win_pct",
    "opp_games", "opp_win_pct",
    "opp_home_games", "opp_home_win_pct",
    "opp_neutral_games", "opp_neutral_win_pct",
    "opp_away_games", "opp_away_win_pct",
    # ATS (games + win pct)
    "team_ats_games", "team_ats_win_pct",
    "team_home_ats_games", "team_home_ats_win_pct",
    "team_neutral_ats_games", "team_neutral_ats_win_pct",
    "team_away_ats_games", "team_away_ats_win_pct",
    "opp_ats_games", "opp_ats_win_pct",
    "opp_home_ats_games", "opp_home_ats_win_pct",
    "opp_neutral_ats_games", "opp_neutral_ats_win_pct",
    "opp_away_ats_games", "opp_away_ats_win_pct",
    # PPG
    "team_ppg", "team_home_ppg", "team_neutral_ppg", "team_away_ppg",
    "opp_ppg", "opp_home_ppg", "opp_neutral_ppg", "opp_away_ppg",
    # Team offensive shooting
    "team_ftm_pg", "team_fta_pg", "team_ft_pct",
    "team_3pm_pg", "team_3pa_pg", "team_3pt_pct",
    "team_2pm_pg", "team_2pa_pg", "team_2pt_pct",
    # Team defensive shooting allowed
    "team_def_ftm_pg", "team_def_fta_pg", "team_def_ft_pct",
    "team_def_3pm_pg", "team_def_3pa_pg", "team_def_3pt_pct",
    "team_def_2pm_pg", "team_def_2pa_pg", "team_def_2pt_pct",
    # Team rebounding & turnovers
    "team_oreb_pg", "team_dreb_pg",
    "team_to_pg", "team_forced_to_pg",
    # Team pace & SOS
    "team_pace", "team_sos",
    # Team derived per-possession & efficiency
    "team_ppp", "team_def_ppg", "team_def_ppp",
    "team_3pa_per_poss", "team_2pa_per_poss", "team_fta_per_poss",
    "team_oreb_per_poss", "team_to_per_poss",
    "team_def_3pa_per_poss", "team_def_2pa_per_poss", "team_def_fta_per_poss",
    "team_dreb_per_poss", "team_forced_to_per_poss",
    # Team ATS margin running averages
    "team_ats_margin_wins", "team_ats_margin_losses",
    # Opponent offensive shooting
    "opp_ftm_pg", "opp_fta_pg", "opp_ft_pct",
    "opp_3pm_pg", "opp_3pa_pg", "opp_3pt_pct",
    "opp_2pm_pg", "opp_2pa_pg", "opp_2pt_pct",
    # Opponent defensive shooting allowed
    "opp_def_ftm_pg", "opp_def_fta_pg", "opp_def_ft_pct",
    "opp_def_3pm_pg", "opp_def_3pa_pg", "opp_def_3pt_pct",
    "opp_def_2pm_pg", "opp_def_2pa_pg", "opp_def_2pt_pct",
    # Opponent rebounding & turnovers
    "opp_oreb_pg", "opp_dreb_pg",
    "opp_to_pg", "opp_forced_to_pg",
    # Opponent pace & SOS
    "opp_pace", "opp_sos",
    # Opponent derived per-possession & efficiency
    "opp_ppp", "opp_def_ppg", "opp_def_ppp",
    "opp_3pa_per_poss", "opp_2pa_per_poss", "opp_fta_per_poss",
    "opp_oreb_per_poss", "opp_to_per_poss",
    "opp_def_3pa_per_poss", "opp_def_2pa_per_poss", "opp_def_fta_per_poss",
    "opp_dreb_per_poss", "opp_forced_to_per_poss",
    # Opponent ATS margin running averages
    "opp_ats_margin_wins", "opp_ats_margin_losses",
]

ALL_COLUMNS = RAW_COLUMNS + ODDS_COLUMNS + PIT_COLUMNS


def load_game_logs() -> pd.DataFrame:
    if PARQUET_FILE.exists():
        return pd.read_parquet(PARQUET_FILE)
    return pd.DataFrame(columns=ALL_COLUMNS)


def save_game_logs(df: pd.DataFrame) -> None:
    # Nullable integer columns
    int_cols = [
        "team_1h", "opp_1h",
        "team_fgm", "team_fga", "team_3pm", "team_3pa",
        "team_ftm", "team_fta", "team_oreb", "team_dreb", "team_to",
        "opp_fgm", "opp_fga", "opp_3pm", "opp_3pa",
        "opp_ftm", "opp_fta", "opp_oreb", "opp_dreb", "opp_to",
        "opp_ap_rank",
        # PIT game counts
        "team_games", "team_home_games", "team_neutral_games", "team_away_games",
        "opp_games", "opp_home_games", "opp_neutral_games", "opp_away_games",
        "team_ats_games", "team_home_ats_games",
        "team_neutral_ats_games", "team_away_ats_games",
        "opp_ats_games", "opp_home_ats_games",
        "opp_neutral_ats_games", "opp_away_ats_games",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Ensure all expected columns exist
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Drop orphaned columns not in ALL_COLUMNS
    orphans = [c for c in df.columns if c not in ALL_COLUMNS]
    if orphans:
        df.drop(columns=orphans, inplace=True)

    # Numeric PIT columns: coerce to float for pyarrow
    numeric_pit = [c for c in PIT_COLUMNS
                   if c.endswith(("_ppg", "_pct", "_pg", "_pace", "_sos"))]
    for col in numeric_pit:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.to_parquet(PARQUET_FILE, index=False, engine="pyarrow")


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Fetch new games (ESPN scoreboard + summary boxscore)
# ═══════════════════════════════════════════════════════════════════════════

def _extract_score(competitor: dict) -> int:
    raw = competitor.get("score", 0)
    if isinstance(raw, dict):
        try:
            return int(float(raw.get("value", raw.get("displayValue", 0))))
        except (ValueError, TypeError):
            return 0
    try:
        return int(float(raw))
    except (ValueError, TypeError):
        return 0


def _parse_first_half(linescores: list) -> int | None:
    if not linescores:
        return None
    try:
        first = linescores[0]
        if isinstance(first, dict):
            val = first.get("value", first.get("displayValue"))
            if val is not None:
                return int(float(val))
        else:
            return int(float(first))
    except (ValueError, TypeError, IndexError):
        return None
    return None


def _extract_ap_rank(competitor: dict) -> int | None:
    """Extract AP rank from a scoreboard competitor. None = unranked."""
    rank_info = competitor.get("curatedRank", {})
    if not rank_info:
        return None
    rank = rank_info.get("current")
    if rank is not None and rank < 99:
        return int(rank)
    return None


def fetch_scoreboard(date: datetime) -> list[dict]:
    date_param = date.strftime("%Y%m%d")
    data = api_get(
        f"{ESPN_BASE}/scoreboard",
        params={"dates": date_param, "limit": 400, "groups": 50},
        tag=f"scoreboard-{date_param}",
    )
    if not data:
        return []
    return data.get("events", [])


def parse_scoreboard(
    events: list[dict],
    tournament_ids: set[str],
    d1_id_to_name: dict[str, str],
    d1_conf_by_id: dict[str, str],
    scoreboard_date: str = "",
) -> list[dict]:
    rows: list[dict] = []
    for ev in events:
        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]
        if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
            continue

        event_id = ev.get("id", "")
        date_raw = comp.get("date", "")
        # Use the ESPN scoreboard date (the date we queried) as the
        # authoritative game date.  This avoids UTC-to-ET drift for
        # late-night games.
        if scoreboard_date:
            game_date = scoreboard_date
        elif date_raw:
            try:
                dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                from zoneinfo import ZoneInfo
                dt_et = dt.astimezone(ZoneInfo("America/New_York"))
                game_date = dt_et.strftime("%Y-%m-%d")
            except (ValueError, KeyError):
                game_date = date_raw[:10]
        else:
            game_date = ""

        neutral_site = comp.get("neutralSite", False)
        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        comp_data = []
        for c in competitors:
            cid = str(c.get("id", c.get("team", {}).get("id", "")))
            team_info = c.get("team", {})
            comp_data.append({
                "id": cid,
                "score": _extract_score(c),
                "home_away": c.get("homeAway", ""),
                "display_name": team_info.get("displayName", ""),
                "short_name": team_info.get("shortDisplayName",
                                            team_info.get("displayName", "")),
                "first_half": _parse_first_half(c.get("linescores", [])),
                "ap_rank": _extract_ap_rank(c),
            })

        # Create a row for BOTH teams in every game
        for i, team_c in enumerate(comp_data):
            opp_c = comp_data[1 - i]
            canonical = d1_id_to_name.get(team_c["id"],
                                          team_c["display_name"])
            if not canonical:
                continue

            ha = "neutral" if neutral_site else team_c["home_away"]
            win_loss = ("W" if team_c["score"] > opp_c["score"]
                        else ("L" if team_c["score"] < opp_c["score"]
                              else "T"))

            # Conference game detection using full D1 conference map
            tc = d1_conf_by_id.get(team_c["id"])
            oc = d1_conf_by_id.get(opp_c["id"])
            conf_game = bool(tc and tc == oc)

            rows.append({
                "team": canonical,
                "date": game_date,
                "event_id": event_id,
                "commence_time_utc": date_raw,
                "opponent": opp_c["display_name"],
                "opponent_short": opp_c["short_name"],
                "opponent_id": opp_c["id"],
                "home_away": ha,
                "neutral_site": neutral_site,
                "conference_game": conf_game,
                "team_score": team_c["score"],
                "opp_score": opp_c["score"],
                "win_loss": win_loss,
                "team_1h": team_c["first_half"],
                "opp_1h": opp_c["first_half"],
                "opp_ap_rank": opp_c["ap_rank"],
                "is_tracked": team_c["id"] in tournament_ids,
            })

    return rows


# ---------------------------------------------------------------------------
# Game summary: half scores + boxscore stats
# ---------------------------------------------------------------------------

def fetch_game_summary(event_id: str) -> dict:
    """Fetch summary and return {team_id: {stat_dict}} for both teams."""
    data = api_get(
        f"{ESPN_BASE}/summary",
        params={"event": event_id},
        tag=f"summary-{event_id}",
    )
    if not data:
        return {}

    result: dict[str, dict] = {}

    # Half scores + AP rank from header
    header = data.get("header", {})
    comps = header.get("competitions", [])
    if comps:
        for c in comps[0].get("competitors", []):
            cid = str(c.get("id", ""))
            result.setdefault(cid, {})
            result[cid]["1h"] = _parse_first_half(c.get("linescores", []))
            result[cid]["ap_rank"] = _extract_ap_rank(c)

    # Boxscore stats
    for team_box in data.get("boxscore", {}).get("teams", []):
        tid = str(team_box.get("team", {}).get("id", ""))
        if not tid:
            continue
        result.setdefault(tid, {})

        stats: dict[str, str] = {}
        for s in team_box.get("statistics", []):
            stats[s.get("name", "")] = s.get("displayValue", "")

        fg = stats.get("fieldGoalsMade-fieldGoalsAttempted", "").split("-")
        result[tid]["fgm"] = _safe_int(fg[0]) if len(fg) == 2 else None
        result[tid]["fga"] = _safe_int(fg[1]) if len(fg) == 2 else None

        tp = stats.get(
            "threePointFieldGoalsMade-threePointFieldGoalsAttempted", ""
        ).split("-")
        result[tid]["3pm"] = _safe_int(tp[0]) if len(tp) == 2 else None
        result[tid]["3pa"] = _safe_int(tp[1]) if len(tp) == 2 else None

        ft = stats.get("freeThrowsMade-freeThrowsAttempted", "").split("-")
        result[tid]["ftm"] = _safe_int(ft[0]) if len(ft) == 2 else None
        result[tid]["fta"] = _safe_int(ft[1]) if len(ft) == 2 else None

        result[tid]["oreb"] = _safe_int(stats.get("offensiveRebounds"))
        result[tid]["dreb"] = _safe_int(stats.get("defensiveRebounds"))
        result[tid]["to"] = _safe_int(
            stats.get("totalTurnovers", stats.get("turnovers"))
        )

    return result


def backfill_game_details(
    day_games: list[dict], team_ids: dict[str, str],
) -> None:
    """Fetch summary for games missing half scores or boxscore stats (deduped)."""
    needed_eids: set[str] = set()
    for g in day_games:
        if g.get("team_1h") is None or g.get("team_fgm") is None:
            needed_eids.add(g["event_id"])
    if not needed_eids:
        return

    # Fetch once per unique event
    cache: dict[str, dict] = {}
    for eid in needed_eids:
        cache[eid] = fetch_game_summary(eid)

    # Distribute to game rows
    for g in day_games:
        eid = g["event_id"]
        if eid not in cache:
            continue
        summary = cache[eid]
        g_tid = team_ids.get(g["team"], "")

        # Find opponent's team ID in the summary
        opp_tid = None
        for tid in summary:
            if tid != g_tid:
                opp_tid = tid
                break

        team_data = summary.get(g_tid, {})
        opp_data = summary.get(opp_tid, {}) if opp_tid else {}

        # Half scores
        if g.get("team_1h") is None:
            g["team_1h"] = team_data.get("1h")
            g["opp_1h"] = opp_data.get("1h")

        # AP rank (fill from summary if scoreboard didn't have it)
        if g.get("opp_ap_rank") is None and opp_data.get("ap_rank") is not None:
            g["opp_ap_rank"] = opp_data["ap_rank"]

        # Boxscore stats
        if g.get("team_fgm") is None:
            for stat in ("fgm", "fga", "3pm", "3pa", "ftm", "fta",
                         "oreb", "dreb", "to"):
                g[f"team_{stat}"] = team_data.get(stat)
                g[f"opp_{stat}"] = opp_data.get(stat)


def backfill_boxscore_data(
    df: pd.DataFrame, team_ids: dict[str, str],
) -> pd.DataFrame:
    """Backfill boxscore stats + AP rank for existing games missing them."""
    missing_mask = df["team_fgm"].isna()
    if not missing_mask.any():
        print(f"  [{_now_str()}] All rows already have boxscore data.")
        return df

    missing_events = df[missing_mask]["event_id"].unique()
    est_minutes = len(missing_events) * (RATE_LIMIT_SEC + 0.5) / 60
    print(f"  [{_now_str()}] Backfilling boxscore for "
          f"{len(missing_events)} events (~{est_minutes:.1f} min) ...")

    for i, eid in enumerate(missing_events, 1):
        if i % 25 == 0 or i == 1:
            pct = i / len(missing_events) * 100
            print(f"  [{_now_str()}] Boxscore progress: "
                  f"{i}/{len(missing_events)} ({pct:.0f}%)")

        summary = fetch_game_summary(eid)
        if not summary:
            continue

        event_mask = df["event_id"] == eid
        for idx in df[event_mask].index:
            row = df.loc[idx]
            g_tid = team_ids.get(row["team"], "")

            opp_tid = None
            for tid in summary:
                if tid != g_tid:
                    opp_tid = tid
                    break

            team_data = summary.get(g_tid, {})
            opp_data = summary.get(opp_tid, {}) if opp_tid else {}

            # Half scores
            if pd.isna(row.get("team_1h")):
                df.at[idx, "team_1h"] = team_data.get("1h")
                df.at[idx, "opp_1h"] = opp_data.get("1h")

            # AP rank
            if pd.isna(row.get("opp_ap_rank")) and opp_data.get("ap_rank") is not None:
                df.at[idx, "opp_ap_rank"] = opp_data["ap_rank"]

            # Boxscore stats
            if pd.isna(row.get("team_fgm")):
                for stat in ("fgm", "fga", "3pm", "3pa", "ftm", "fta",
                             "oreb", "dreb", "to"):
                    df.at[idx, f"team_{stat}"] = team_data.get(stat)
                    df.at[idx, f"opp_{stat}"] = opp_data.get(stat)

    filled = df["team_fgm"].notna().sum()
    print(f"  [{_now_str()}] Backfill done: {filled}/{len(df)} rows "
          f"have boxscore data.")
    return df


def backfill_ap_ranks(
    df: pd.DataFrame,
    tracked_ids: set[str],
    id_to_name: dict[str, str],
) -> pd.DataFrame:
    """Re-scan historical scoreboards to fill in opp_ap_rank for games that
    haven't been checked yet.

    Uses a sentinel column 'ap_rank_checked' to distinguish 'unranked' (NULL
    but already checked) from 'never checked'.  Only re-fetches dates with
    unchecked rows.
    """
    if "ap_rank_checked" not in df.columns:
        df["ap_rank_checked"] = False

    unchecked = df[~df["ap_rank_checked"].fillna(False).astype(bool)]
    if unchecked.empty:
        print(f"  [{_now_str()}] All rows already checked for AP ranks.")
        return df

    missing_dates = sorted(unchecked["date"].unique())
    print(f"  [{_now_str()}] Backfilling AP ranks for "
          f"{len(missing_dates)} dates ({len(unchecked)} unchecked rows) ...")

    # Build a lookup: (event_id, competitor_id) -> ap_rank
    rank_lookup: dict[tuple[str, str], int | None] = {}

    for i, date_str in enumerate(missing_dates, 1):
        if i % 20 == 0 or i == 1:
            print(f"  [{_now_str()}] Progress: {i}/{len(missing_dates)} dates")
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        events = fetch_scoreboard(dt)
        for ev in events:
            event_id = ev.get("id", "")
            comps = ev.get("competitions", [])
            if not comps:
                continue
            for c in comps[0].get("competitors", []):
                cid = str(c.get("id", c.get("team", {}).get("id", "")))
                rank = _extract_ap_rank(c)
                rank_lookup[(event_id, cid)] = rank

    # Apply ranks to unchecked rows
    filled = 0
    for idx in unchecked.index:
        row = df.loc[idx]
        opp_id = str(row.get("opponent_id", ""))
        eid = row["event_id"]
        rank = rank_lookup.get((eid, opp_id))
        if rank is not None:
            df.at[idx, "opp_ap_rank"] = rank
            filled += 1
        df.at[idx, "ap_rank_checked"] = True

    total_ranked = df["opp_ap_rank"].notna().sum()
    print(f"  [{_now_str()}] AP rank backfill done: "
          f"{filled} new ranks filled, {total_ranked}/{len(df)} total.")
    return df


def backfill_commence_times(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill commence_time_utc from ESPN scoreboard for existing rows."""
    if "commence_time_utc" not in df.columns:
        df["commence_time_utc"] = None

    missing = df[df["commence_time_utc"].isna() | (df["commence_time_utc"] == "")]
    if missing.empty:
        print(f"  [{_now_str()}] All rows already have commence times.")
        return df

    # Group by date to minimize API calls (one scoreboard call per date)
    missing_dates = sorted(missing["date"].unique())
    print(f"  [{_now_str()}] Backfilling commence times for "
          f"{len(missing_dates)} dates ({len(missing)} rows) ...")

    # Build lookup: event_id -> commence_time_utc
    time_lookup: dict[str, str] = {}
    for i, date_str in enumerate(missing_dates, 1):
        if i % 20 == 0:
            print(f"  [{_now_str()}] Progress: {i}/{len(missing_dates)} dates")
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        events = fetch_scoreboard(dt)
        for ev in events:
            event_id = ev.get("id", "")
            comps = ev.get("competitions", [])
            if not comps:
                continue
            date_raw = comps[0].get("date", "")
            if event_id and date_raw:
                time_lookup[event_id] = date_raw

    filled = 0
    for idx in missing.index:
        eid = df.at[idx, "event_id"]
        ct = time_lookup.get(str(eid))
        if ct:
            df.at[idx, "commence_time_utc"] = ct
            filled += 1

    print(f"  [{_now_str()}] Commence time backfill done: "
          f"{filled}/{len(missing)} rows filled.")
    return df


def create_mirror_rows(
    df: pd.DataFrame,
    tracked_ids: set[str],
    team_ids: dict[str, str],
) -> pd.DataFrame:
    """Create mirror rows for games where a tournament team played a non-tournament opponent.

    For each such game, creates a row from the opponent's perspective by swapping
    team/opp columns.  Mirror rows have is_tracked = False.
    """
    mirror_rows: list[dict] = []
    existing_keys: set[tuple[str, str]] = set(zip(df["team"], df["event_id"]))

    for _, row in df.iterrows():
        opp_id = str(row.get("opponent_id", ""))
        if opp_id in tracked_ids:
            continue  # opponent is tracked; they already have their own row

        opp_display = row["opponent"]
        eid = row["event_id"]
        if (opp_display, eid) in existing_keys:
            continue  # mirror already exists

        # Determine home_away for mirror
        orig_ha = row["home_away"]
        if orig_ha == "home":
            mirror_ha = "away"
        elif orig_ha == "away":
            mirror_ha = "home"
        else:
            mirror_ha = "neutral"

        mirror_wl = "L" if row["win_loss"] == "W" else (
            "W" if row["win_loss"] == "L" else "T")

        # Look up the tournament team's ESPN ID for opponent_id in mirror
        team_espn_id = team_ids.get(row["team"], "")

        mirror = {
            "team": opp_display,
            "date": row["date"],
            "event_id": eid,
            "commence_time_utc": row.get("commence_time_utc"),
            "opponent": row["team"],
            "opponent_short": row["team"],
            "opponent_id": team_espn_id,
            "home_away": mirror_ha,
            "neutral_site": row.get("neutral_site", False),
            "conference_game": False,
            "team_score": row["opp_score"],
            "opp_score": row["team_score"],
            "win_loss": mirror_wl,
            "team_1h": row.get("opp_1h"),
            "opp_1h": row.get("team_1h"),
            "opp_ap_rank": None,  # tournament team's rank not tracked here
            "is_tracked": False,
            # Swap boxscore stats
            "team_fgm": row.get("opp_fgm"), "team_fga": row.get("opp_fga"),
            "team_3pm": row.get("opp_3pm"), "team_3pa": row.get("opp_3pa"),
            "team_ftm": row.get("opp_ftm"), "team_fta": row.get("opp_fta"),
            "team_oreb": row.get("opp_oreb"), "team_dreb": row.get("opp_dreb"),
            "team_to": row.get("opp_to"),
            "opp_fgm": row.get("team_fgm"), "opp_fga": row.get("team_fga"),
            "opp_3pm": row.get("team_3pm"), "opp_3pa": row.get("team_3pa"),
            "opp_ftm": row.get("team_ftm"), "opp_fta": row.get("team_fta"),
            "opp_oreb": row.get("team_oreb"), "opp_dreb": row.get("team_dreb"),
            "opp_to": row.get("team_to"),
        }
        mirror_rows.append(mirror)
        existing_keys.add((opp_display, eid))

    if not mirror_rows:
        return df

    mirror_df = pd.DataFrame(mirror_rows)
    for col in ALL_COLUMNS:
        if col not in mirror_df.columns:
            mirror_df[col] = None

    combined = pd.concat([df, mirror_df[ALL_COLUMNS]], ignore_index=True)
    combined.sort_values(["date", "team"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"  [{_now_str()}] Created {len(mirror_rows)} mirror rows "
          f"(total: {len(combined)})")
    return combined


def phase1_fetch_opponent_games(
    df: pd.DataFrame,
    team_ids: dict[str, str],
    tracked_ids: set[str],
) -> pd.DataFrame:
    """Fetch full schedules for non-tournament opponents to fill in their game logs.

    Only includes games where at least one team is a tournament team or a
    known opponent (i.e., has played a tournament team).
    """
    # Collect all unique non-tournament opponent IDs
    all_opp_ids: set[str] = set()
    opp_id_to_name: dict[str, str] = {}
    for _, row in df.iterrows():
        opp_id = str(row.get("opponent_id", ""))
        if opp_id and opp_id not in tracked_ids:
            all_opp_ids.add(opp_id)
            if opp_id not in opp_id_to_name:
                opp_id_to_name[opp_id] = row["opponent"]

    # Also include non-tracked teams that appear as "team" (from mirrors)
    known_team_names: set[str] = set()
    for _, row in df.iterrows():
        if row.get("is_tracked") is False:
            known_team_names.add(row["team"])

    # Build set of known IDs (tracked + known opponents)
    known_ids = tracked_ids | all_opp_ids

    existing_keys: set[tuple[str, str]] = set(zip(df["team"], df["event_id"]))

    # Skip opponents that already have game rows in the dataset
    # (they were fetched in a prior run)
    teams_with_rows = set(df["team"].unique())
    already_fetched = {oid for oid, name in opp_id_to_name.items()
                       if name in teams_with_rows}
    new_opp_ids = all_opp_ids - already_fetched
    if already_fetched:
        print(f"  [{_now_str()}] Skipping {len(already_fetched)} opponents "
              f"already in dataset, {len(new_opp_ids)} new to fetch")
    all_opp_ids = new_opp_ids

    if not all_opp_ids:
        print(f"  [{_now_str()}] No non-tournament opponents to fetch.")
        return df

    est_minutes = len(all_opp_ids) * (RATE_LIMIT_SEC + 0.5) / 60
    print(f"[{_now_str()}] Fetching schedules for {len(all_opp_ids)} "
          f"non-tournament opponents (~{est_minutes:.1f} min) ...")

    all_new_rows: list[dict] = []
    events_fetched: set[str] = set()

    for i, opp_id in enumerate(sorted(all_opp_ids), 1):
        opp_name = opp_id_to_name.get(opp_id, f"Team-{opp_id}")
        if i % 25 == 0 or i == 1:
            pct = i / len(all_opp_ids) * 100
            print(f"  [{_now_str()}] Opponent schedule {i}/{len(all_opp_ids)} "
                  f"({pct:.0f}%): {opp_name} — "
                  f"{len(all_new_rows)} new rows so far")

        url = f"{ESPN_BASE}/teams/{opp_id}/schedule"
        data = api_get(url, params={"season": SEASON}, tag=f"schedule-{opp_id}")
        if not data:
            continue

        for ev in data.get("events", []):
            event_id = ev.get("id", "")
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                continue

            # Skip if we already have this row
            if (opp_name, event_id) in existing_keys:
                continue

            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            # Find this opponent and the other team
            opp_c = None
            other_c = None
            for c in competitors:
                cid = str(c.get("id", c.get("team", {}).get("id", "")))
                if cid == opp_id:
                    opp_c = c
                else:
                    other_c = c
            if not opp_c or not other_c:
                continue

            other_id = str(other_c.get("id", other_c.get("team", {}).get("id", "")))

            # Skip if neither team is known
            if other_id not in known_ids:
                continue

            neutral_site = comp.get("neutralSite", False)
            date_raw = comp.get("date", "")
            game_date = ""
            if date_raw:
                try:
                    dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                    from zoneinfo import ZoneInfo
                    dt_et = dt.astimezone(ZoneInfo("America/New_York"))
                    game_date = dt_et.strftime("%Y-%m-%d")
                except (ValueError, KeyError):
                    game_date = date_raw[:10]

            opp_score_val = _extract_score(opp_c)
            other_score_val = _extract_score(other_c)
            opp_ha_raw = opp_c.get("homeAway", "")
            ha = "neutral" if neutral_site else opp_ha_raw
            wl = ("W" if opp_score_val > other_score_val
                  else ("L" if opp_score_val < other_score_val else "T"))

            other_team_info = other_c.get("team", {})
            other_display = other_team_info.get("displayName", "")
            other_short = other_team_info.get(
                "shortDisplayName", other_team_info.get("displayName", ""))

            row_dict = {
                "team": opp_name,
                "date": game_date,
                "event_id": event_id,
                "commence_time_utc": date_raw,
                "opponent": other_display,
                "opponent_short": other_short,
                "opponent_id": other_id,
                "home_away": ha,
                "neutral_site": neutral_site,
                "conference_game": False,
                "team_score": opp_score_val,
                "opp_score": other_score_val,
                "win_loss": wl,
                "team_1h": None,
                "opp_1h": None,
                "opp_ap_rank": None,
                "is_tracked": False,
            }

            # Fetch boxscore if not already fetched for this event
            if event_id not in events_fetched:
                summary = fetch_game_summary(event_id)
                events_fetched.add(event_id)
                if summary:
                    team_data = summary.get(opp_id, {})
                    other_data = summary.get(other_id, {})
                    row_dict["team_1h"] = team_data.get("1h")
                    row_dict["opp_1h"] = other_data.get("1h")
                    for stat in ("fgm", "fga", "3pm", "3pa", "ftm", "fta",
                                 "oreb", "dreb", "to"):
                        row_dict[f"team_{stat}"] = team_data.get(stat)
                        row_dict[f"opp_{stat}"] = other_data.get(stat)

            all_new_rows.append(row_dict)
            existing_keys.add((opp_name, event_id))

    if not all_new_rows:
        print(f"  [{_now_str()}] No new opponent games found.")
        return df

    new_df = pd.DataFrame(all_new_rows)
    for col in ALL_COLUMNS:
        if col not in new_df.columns:
            new_df[col] = None

    combined = pd.concat([df, new_df[ALL_COLUMNS]], ignore_index=True)
    combined.sort_values(["date", "team"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    print(f"  [{_now_str()}] Opponent games: +{len(all_new_rows)} rows "
          f"(total: {len(combined)})")
    return combined


def phase1_fetch_games(
    df: pd.DataFrame,
    d1_team_ids: dict[str, str],
    tournament_ids: set[str],
    d1_id_to_name: dict[str, str],
    d1_conf_by_id: dict[str, str],
    force_start: datetime | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    """Return (updated_df, set_of_new_date_strings)."""
    existing_keys: set[tuple[str, str]] = set()
    if not df.empty:
        existing_keys = set(zip(df["team"], df["event_id"]))

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if force_start:
        start_date = force_start
    elif df.empty:
        start_date = SEASON_START
    else:
        start_date = datetime.strptime(df["date"].max(), "%Y-%m-%d")

    if start_date > today:
        print(f"[{_now_str()}] Phase 1: already up to date.")
        return df, set()

    dates_to_fetch: list[datetime] = []
    d = start_date
    while d <= today:
        dates_to_fetch.append(d)
        d += timedelta(days=1)

    print(f"[{_now_str()}] Phase 1: scanning {len(dates_to_fetch)} date(s) "
          f"({dates_to_fetch[0]:%Y-%m-%d} to {dates_to_fetch[-1]:%Y-%m-%d})")

    all_new_rows: list[dict] = []
    new_dates: set[str] = set()

    for i, date in enumerate(dates_to_fetch, 1):
        date_str = date.strftime("%Y-%m-%d")
        if i % 10 == 0 or i == 1 or len(dates_to_fetch) <= 5:
            print(f"  [{_now_str()}] Scanning date {i}/{len(dates_to_fetch)}: "
                  f"{date_str}")

        events = fetch_scoreboard(date)
        if not events:
            continue

        day_games = parse_scoreboard(
            events, tournament_ids, d1_id_to_name, d1_conf_by_id, date_str,
        )
        day_games = [
            g for g in day_games
            if (g["team"], g["event_id"]) not in existing_keys
        ]
        if not day_games:
            continue

        print(f"  [{_now_str()}] ({i}/{len(dates_to_fetch)}) "
              f"{date_str}: {len(day_games)} new rows")

        backfill_game_details(day_games, d1_team_ids)

        all_new_rows.extend(day_games)
        new_dates.add(date_str)

        for g in day_games:
            existing_keys.add((g["team"], g["event_id"]))

    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        # Ensure all columns exist (odds + pit will be null for new rows)
        for col in ALL_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = None
        combined = pd.concat([df, new_df[ALL_COLUMNS]], ignore_index=True)
        combined.sort_values(["date", "team"], inplace=True)
        combined.reset_index(drop=True, inplace=True)
        print(f"  [{_now_str()}] Phase 1 done: +{len(all_new_rows)} rows "
              f"(total: {len(combined)})")
        return combined, new_dates

    print(f"  [{_now_str()}] Phase 1 done: no new games.")
    return df, new_dates


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Fetch odds for new dates only (The Odds API)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_opening_snap_time(date_str: str) -> str:
    """Compute an 'opening' snapshot timestamp for a game date.

    Uses 9 AM ET on the game day — lines are typically posted by then
    and should differ from the closing (game-time) snapshot.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    offset_h = 5 if (dt.month >= 11 or dt.month <= 2) else 4
    snap_dt = dt + timedelta(hours=9 + offset_h)  # 9 AM ET in UTC
    return snap_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_team_name_map() -> dict[str, str]:
    """Load the ESPN -> Odds API team name map."""
    if TEAM_NAME_MAP_FILE.exists():
        with open(TEAM_NAME_MAP_FILE) as f:
            return json.load(f)
    return {}


_TEAM_NAME_MAP: dict[str, str] = _load_team_name_map()


def _find_team_in_odds(
    snap: dict | None, team_odds_lower: str,
) -> dict | None:
    """Find a game by team name. One game per team per day."""
    if not snap:
        return None
    for game in snap.get("data", []):
        home = game.get("home_team", "").lower()
        away = game.get("away_team", "").lower()
        if team_odds_lower == home or team_odds_lower == away:
            return game
    return None


def _first_bookmaker_value(
    bookmakers: list[dict], market_key: str, team_lower: str,
    field: str,
) -> float | int | None:
    for bm in bookmakers:
        for mkt in bm.get("markets", []):
            if mkt.get("key") != market_key:
                continue
            for oc in mkt.get("outcomes", []):
                if oc.get("name", "").lower() == team_lower:
                    return oc.get(field)
    return None


def _odds_cache_path(timestamp_iso: str) -> Path:
    """Return the cache file path for an odds snapshot timestamp."""
    safe_name = timestamp_iso.replace(":", "-").replace("Z", "")
    return ODDS_CACHE_DIR / f"{safe_name}.json"


def _load_cached_odds(timestamp_iso: str) -> dict | None:
    """Load a cached odds snapshot if it exists."""
    path = _odds_cache_path(timestamp_iso)
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


def _save_odds_cache(timestamp_iso: str, data: dict) -> None:
    """Save an odds snapshot to the local cache."""
    ODDS_CACHE_DIR.mkdir(exist_ok=True)
    path = _odds_cache_path(timestamp_iso)
    with open(path, "w") as f:
        json.dump(data, f)


def fetch_odds_snapshot(timestamp_iso: str) -> dict | None:
    """Fetch a historical odds snapshot, using local cache when available."""
    cached = _load_cached_odds(timestamp_iso)
    if cached is not None:
        return cached

    url = f"{ODDS_BASE}/v4/historical/sports/basketball_ncaab/odds"
    data = api_get(url, params={
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
        "date": timestamp_iso,
    }, tag=f"odds-snap-{timestamp_iso[:16]}")

    if data is not None:
        _save_odds_cache(timestamp_iso, data)
    return data


def phase2_backfill_odds(
    df: pd.DataFrame, date_filter: str | None = None,
) -> pd.DataFrame:
    """Backfill true opening and closing lines using per-game snapshots.

    For each unique commence_time, fetches the snapshot at floor(commence_time)
    for closing lines.  Opening lines come from the earliest snapshot where
    each game first appeared.

    Args:
        date_filter: Optional YYYY-MM-DD string to scope backfill to a single
                     date. If None, backfills all missing.
    """
    if not THE_ODDS_API_KEY:
        print(f"[{_now_str()}] Backfill odds: skipped (no Odds API key).")
        return df

    # Ensure odds_checked column exists
    if "odds_checked" not in df.columns:
        df["odds_checked"] = False

    # Only tracked rows that haven't been checked yet and have commence times
    needs_odds = (
        (~df["odds_checked"].fillna(False).astype(bool)) &
        (df["commence_time_utc"].notna()) &
        (df["commence_time_utc"] != "") &
        (df["is_tracked"] == True)  # noqa: E712
    )

    if date_filter:
        needs_odds = needs_odds & (df["date"] == date_filter)

    if not needs_odds.any():
        scope = f"on {date_filter}" if date_filter else "overall"
        print(f"[{_now_str()}] Backfill odds: no tracked rows missing odds "
              f"{scope}.")
        return df

    scope_msg = f" (date: {date_filter})" if date_filter else ""
    print(f"[{_now_str()}] Backfill odds: {needs_odds.sum()} tracked rows "
          f"missing odds{scope_msg}")

    # Compute floor(commence_time) ONLY for rows needing odds
    ct_series = pd.to_datetime(df.loc[needs_odds, "commence_time_utc"], utc=True,
                               errors="coerce")
    floor_times = ct_series.dt.floor("h")

    # Deduplicate snapshot times (only hours where tracked rows need odds)
    unique_snaps = sorted(floor_times.dropna().unique())

    # Check how many are already cached
    snap_isos = [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ")
                 for t in unique_snaps]
    cached_count = sum(1 for s in snap_isos if _load_cached_odds(s) is not None)
    api_count = len(unique_snaps) - cached_count
    est_minutes = api_count * (RATE_LIMIT_SEC + 0.5) / 60
    print(f"[{_now_str()}] Backfill odds: {len(unique_snaps)} snapshots needed "
          f"({cached_count} cached, {api_count} to fetch"
          f"{f', ~{est_minutes:.1f} min, ~{api_count * 20} credits' if api_count else ''})")

    # Fetch all snapshots chronologically (cache hits are instant)
    snapshots: dict[str, dict] = {}
    failed = 0
    fetched = 0
    for i, snap_iso in enumerate(snap_isos, 1):
        is_cached = _load_cached_odds(snap_iso) is not None
        if not is_cached and (fetched % 10 == 0 or fetched == 0):
            pct = i / len(unique_snaps) * 100
            print(f"  [{_now_str()}] Fetching snapshot {i}/{len(unique_snaps)} "
                  f"({pct:.0f}%) — {snap_iso}")
        snap_data = fetch_odds_snapshot(snap_iso)
        if snap_data:
            games_in_snap = len(snap_data.get("data", []))
            snapshots[snap_iso] = snap_data
            if not is_cached:
                fetched += 1
                if fetched <= 3 or fetched % 25 == 0:
                    print(f"    -> {games_in_snap} games in snapshot")
        else:
            failed += 1

    if not snapshots:
        print(f"  [{_now_str()}] Backfill odds: no snapshots returned "
              f"({failed} failed).")
        print_odds_api_usage()
        return df
    print(f"  [{_now_str()}] Loaded {len(snapshots)} snapshots "
          f"({cached_count} cached, {fetched} fetched, {failed} failed)")
    print_odds_api_usage()

    # Build opening index: (team_lower, commence_time) -> earliest_snapshot_iso
    opening_index: dict[tuple[str, str], str] = {}
    for snap_iso in sorted(snapshots.keys()):
        snap_data = snapshots[snap_iso]
        for game in snap_data.get("data", []):
            ct = game.get("commence_time", "")
            home = game.get("home_team", "").lower()
            away = game.get("away_team", "").lower()
            for team_lower in (home, away):
                key = (team_lower, ct)
                if key not in opening_index:
                    opening_index[key] = snap_iso

    # Assign lines to rows that need odds
    target_indices = df[needs_odds].index
    total_rows = len(target_indices)
    filled = 0
    no_odds_name = 0
    no_closing_snap = 0
    no_closing_match = 0
    no_opening_snap = 0
    print(f"  [{_now_str()}] Matching odds to {total_rows} unchecked rows ...")

    for row_num, idx in enumerate(target_indices, 1):
        row = df.loc[idx]
        canonical = row["team"]
        odds_name = _TEAM_NAME_MAP.get(canonical, "") or \
                    ODDS_NAME_MAP.get(canonical, "")
        if not odds_name:
            no_odds_name += 1
            df.at[idx, "odds_checked"] = True
            continue
        team_odds_lower = odds_name.lower()

        ct_val = pd.to_datetime(row["commence_time_utc"], utc=True,
                                errors="coerce")
        if pd.isna(ct_val):
            df.at[idx, "odds_checked"] = True
            continue

        floor_iso = ct_val.floor("h").strftime("%Y-%m-%dT%H:%M:%SZ")

        # Closing line: snapshot at floor(commence_time)
        closing_snap = snapshots.get(floor_iso)
        if closing_snap:
            game = _find_team_in_odds(closing_snap, team_odds_lower)
            if game:
                bm = game.get("bookmakers", [])
                df.at[idx, "closing_fg_ml"] = _first_bookmaker_value(
                    bm, "h2h", team_odds_lower, "price")
                df.at[idx, "closing_fg_spread"] = _first_bookmaker_value(
                    bm, "spreads", team_odds_lower, "point")
            else:
                no_closing_match += 1
        else:
            no_closing_snap += 1

        # Opening line: earliest snapshot where this game appeared
        ct_str = row["commence_time_utc"]
        # Try to match against opening_index using the raw commence_time
        open_snap_iso = opening_index.get((team_odds_lower, ct_str))
        if not open_snap_iso:
            # Try ISO-formatted version
            ct_iso = ct_val.strftime("%Y-%m-%dT%H:%M:%SZ")
            open_snap_iso = opening_index.get((team_odds_lower, ct_iso))
        if open_snap_iso and open_snap_iso in snapshots:
            open_snap = snapshots[open_snap_iso]
            game = _find_team_in_odds(open_snap, team_odds_lower)
            if game:
                bm = game.get("bookmakers", [])
                df.at[idx, "opening_fg_ml"] = _first_bookmaker_value(
                    bm, "h2h", team_odds_lower, "price")
                df.at[idx, "opening_fg_spread"] = _first_bookmaker_value(
                    bm, "spreads", team_odds_lower, "point")
                filled += 1
        else:
            no_opening_snap += 1

        # Mark as checked regardless of whether we found odds
        df.at[idx, "odds_checked"] = True

        if row_num % 500 == 0:
            print(f"    [{_now_str()}] Processed {row_num}/{total_rows} rows "
                  f"({filled} matched so far)")

    matched = df["closing_fg_spread"].notna().sum()
    print(f"  [{_now_str()}] Backfill odds done: {matched}/{len(df)} rows "
          f"have closing spreads, {filled} rows got opening lines.")
    if no_odds_name or no_closing_snap or no_closing_match:
        print(f"    Skipped: {no_odds_name} no odds name, "
              f"{no_closing_snap} no closing snapshot, "
              f"{no_closing_match} no closing match, "
              f"{no_opening_snap} no opening snapshot")
    return df


def phase2_fetch_odds(df: pd.DataFrame, new_dates: set[str]) -> pd.DataFrame:
    """Fetch odds for dates that had new games in Phase 1.

    Uses game-time snapshots for closing lines and an early-day snapshot
    for opening lines — same approach as backfill, scoped to new dates.
    """
    if not THE_ODDS_API_KEY:
        print(f"[{_now_str()}] Phase 2: skipped (no Odds API key).")
        return df

    if not new_dates:
        print(f"[{_now_str()}] Phase 2: no new dates to fetch odds for.")
        return df

    # Filter to tracked rows on new dates that need odds
    mask = (
        df["date"].isin(new_dates) &
        (df["is_tracked"] == True) &  # noqa: E712
        df["closing_fg_spread"].isna() &
        df["commence_time_utc"].notna() &
        (df["commence_time_utc"] != "")
    )
    if not mask.any():
        print(f"[{_now_str()}] Phase 2: no tracked rows need odds on new dates.")
        return df

    # Compute unique closing snapshot times (floor of commence_time)
    ct_series = pd.to_datetime(df.loc[mask, "commence_time_utc"], utc=True,
                               errors="coerce")
    closing_hours = sorted(ct_series.dt.floor("h").dropna().unique())

    # Compute unique opening snapshot times (9 AM ET on each game date)
    opening_times = sorted({_compute_opening_snap_time(d) for d in new_dates})

    # Deduplicate all snapshot times
    all_snap_times = sorted(set(
        [pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M:%SZ") for t in closing_hours]
        + opening_times
    ))

    cached_count = sum(1 for s in all_snap_times
                       if _load_cached_odds(s) is not None)
    api_count = len(all_snap_times) - cached_count
    print(f"[{_now_str()}] Phase 2: {mask.sum()} tracked rows need odds "
          f"across {len(new_dates)} date(s)")
    print(f"  {len(all_snap_times)} snapshots "
          f"({len(closing_hours)} closing + {len(opening_times)} opening, "
          f"deduped to {len(all_snap_times)}) — "
          f"{cached_count} cached, {api_count} to fetch")

    # Fetch all snapshots (cache hits are instant)
    snapshots: dict[str, dict] = {}
    for i, snap_iso in enumerate(all_snap_times, 1):
        is_cached = _load_cached_odds(snap_iso) is not None
        snap_data = fetch_odds_snapshot(snap_iso)
        if snap_data:
            games_in_snap = len(snap_data.get("data", []))
            snapshots[snap_iso] = snap_data
            label = "cached" if is_cached else "fetched"
            print(f"  [{_now_str()}] ({i}/{len(all_snap_times)}) {snap_iso} "
                  f"— {games_in_snap} games ({label})")

    print_odds_api_usage()

    if not snapshots:
        print(f"  [{_now_str()}] Phase 2: no snapshots returned.")
        return df

    # Build opening index: (team_lower, commence_time) -> earliest snapshot
    opening_index: dict[tuple[str, str], str] = {}
    for snap_iso in sorted(snapshots.keys()):
        snap_data = snapshots[snap_iso]
        for game in snap_data.get("data", []):
            ct = game.get("commence_time", "")
            home = game.get("home_team", "").lower()
            away = game.get("away_team", "").lower()
            for team_lower in (home, away):
                key = (team_lower, ct)
                if key not in opening_index:
                    opening_index[key] = snap_iso

    # Ensure odds_checked column exists
    if "odds_checked" not in df.columns:
        df["odds_checked"] = False

    # Assign lines to rows
    filled_closing = 0
    filled_opening = 0
    no_odds_name = 0

    for idx in df[mask].index:
        row = df.loc[idx]
        canonical = row["team"]
        odds_name = _TEAM_NAME_MAP.get(canonical, "") or \
                    ODDS_NAME_MAP.get(canonical, "")
        if not odds_name:
            no_odds_name += 1
            df.at[idx, "odds_checked"] = True
            continue
        team_odds_lower = odds_name.lower()

        ct_val = pd.to_datetime(row["commence_time_utc"], utc=True,
                                errors="coerce")
        if pd.isna(ct_val):
            df.at[idx, "odds_checked"] = True
            continue

        # Closing line: snapshot at floor(commence_time)
        floor_iso = ct_val.floor("h").strftime("%Y-%m-%dT%H:%M:%SZ")
        closing_snap = snapshots.get(floor_iso)
        if closing_snap:
            game = _find_team_in_odds(closing_snap, team_odds_lower)
            if game:
                bm = game.get("bookmakers", [])
                df.at[idx, "closing_fg_ml"] = _first_bookmaker_value(
                    bm, "h2h", team_odds_lower, "price")
                df.at[idx, "closing_fg_spread"] = _first_bookmaker_value(
                    bm, "spreads", team_odds_lower, "point")
                filled_closing += 1

        # Opening line: earliest snapshot where this game appeared
        ct_str = row["commence_time_utc"]
        open_snap_iso = opening_index.get((team_odds_lower, ct_str))
        if not open_snap_iso:
            ct_iso = ct_val.strftime("%Y-%m-%dT%H:%M:%SZ")
            open_snap_iso = opening_index.get((team_odds_lower, ct_iso))
        if open_snap_iso and open_snap_iso in snapshots:
            open_snap = snapshots[open_snap_iso]
            game = _find_team_in_odds(open_snap, team_odds_lower)
            if game:
                bm = game.get("bookmakers", [])
                df.at[idx, "opening_fg_ml"] = _first_bookmaker_value(
                    bm, "h2h", team_odds_lower, "price")
                df.at[idx, "opening_fg_spread"] = _first_bookmaker_value(
                    bm, "spreads", team_odds_lower, "point")
                filled_opening += 1

        # Mark as checked regardless of whether we found odds
        df.at[idx, "odds_checked"] = True

    matched = df["closing_fg_spread"].notna().sum()
    print(f"  [{_now_str()}] Phase 2 done: {filled_closing} closing, "
          f"{filled_opening} opening lines matched "
          f"({no_odds_name} skipped — no odds name)")
    print(f"  [{_now_str()}] Total rows with spreads: {matched}/{len(df)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Compute PIT stats (pure math, no API calls)
# ═══════════════════════════════════════════════════════════════════════════

# Keys used to write new PIT stats into team_records / opp_records dicts
_NEW_PIT_KEYS = [
    "ftm_pg", "fta_pg", "ft_pct",
    "3pm_pg", "3pa_pg", "3pt_pct",
    "2pm_pg", "2pa_pg", "2pt_pct",
    "def_ftm_pg", "def_fta_pg", "def_ft_pct",
    "def_3pm_pg", "def_3pa_pg", "def_3pt_pct",
    "def_2pm_pg", "def_2pa_pg", "def_2pt_pct",
    "oreb_pg", "dreb_pg",
    "to_pg", "forced_to_pg",
    "pace", "sos",
    # Derived per-possession & efficiency
    "ppp", "def_ppg", "def_ppp",
    "3pa_per_poss", "2pa_per_poss", "fta_per_poss",
    "oreb_per_poss", "to_per_poss",
    "def_3pa_per_poss", "def_2pa_per_poss", "def_fta_per_poss",
    "dreb_per_poss", "forced_to_per_poss",
    # ATS margin running averages
    "ats_margin_wins", "ats_margin_losses",
]


def phase3_compute_pit(
    df: pd.DataFrame, team_ids: dict[str, str],
) -> pd.DataFrame:
    """Compute Point-In-Time stats for every row.

    For each team + game, PIT stats reflect performance PRIOR to that game.
    Always recomputed from scratch (fast, no API).
    """
    print(f"[{_now_str()}] Phase 3: computing PIT stats ...")

    id_to_name = {tid: name for name, tid in team_ids.items()}

    # Extend id_to_name with non-tournament opponents
    for _, row in df.iterrows():
        opp_id = str(row.get("opponent_id", ""))
        if opp_id and opp_id not in id_to_name:
            id_to_name[opp_id] = row["opponent"]

    # -- Pre-compute per-team PIT: {team -> {event_id -> stats_dict}} --
    pit: dict[str, dict[str, dict]] = {}

    all_teams = list(df["team"].unique())
    print(f"  [{_now_str()}] Computing PIT for {len(all_teams)} teams, "
          f"{len(df)} rows ...")
    for t_idx, team_name in enumerate(all_teams, 1):
        if t_idx % 100 == 0 or t_idx == 1:
            print(f"  [{_now_str()}] PIT progress: {t_idx}/{len(all_teams)} teams")
        team_rows = df[df["team"] == team_name].sort_values("date")
        if team_rows.empty:
            pit[team_name] = {}
            continue

        # Running counters — existing
        ow, ol = 0, 0
        hw, hl = 0, 0
        nw, nl = 0, 0
        aw, al = 0, 0

        ats_ow, ats_ol, ats_op = 0, 0, 0
        ats_hw, ats_hl, ats_hp = 0, 0, 0
        ats_nw, ats_nl, ats_np = 0, 0, 0
        ats_aw, ats_al, ats_ap = 0, 0, 0

        tp, tg = 0, 0      # total pts, games
        hp, hg = 0, 0      # home
        np_, ng = 0, 0      # neutral
        ap, ag = 0, 0      # away

        # Running counters — new (boxscore)
        s_fgm, s_fga = 0, 0   # team shooting
        s_3pm, s_3pa = 0, 0
        s_ftm, s_fta = 0, 0
        s_oreb, s_dreb = 0, 0
        s_to = 0

        d_fgm, d_fga = 0, 0   # defensive (opponent's numbers in game)
        d_3pm, d_3pa = 0, 0
        d_ftm, d_fta = 0, 0
        d_to = 0               # opponent TOs = forced TOs

        sos_sum, sos_count = 0, 0
        bg = 0  # games with boxscore data

        # ATS margin running averages
        ats_margin_win_sum, ats_margin_win_count = 0.0, 0
        ats_margin_loss_sum, ats_margin_loss_count = 0.0, 0

        team_pit: dict[str, dict] = {}

        for _, row in team_rows.iterrows():
            eid = row["event_id"]

            # ── SAVE stats as-of BEFORE this game ──
            entry: dict = {
                "games": ow + ol,
                "win_pct": round(ow / (ow + ol), 3) if (ow + ol) else None,
                "home_games": hw + hl,
                "home_win_pct": round(hw / (hw + hl), 3) if (hw + hl) else None,
                "neutral_games": nw + nl,
                "neutral_win_pct": round(nw / (nw + nl), 3) if (nw + nl) else None,
                "away_games": aw + al,
                "away_win_pct": round(aw / (aw + al), 3) if (aw + al) else None,
                "ats_games": ats_ow + ats_ol + ats_op,
                "ats_win_pct": round(ats_ow / (ats_ow + ats_ol + ats_op), 3) if (ats_ow + ats_ol + ats_op) else None,
                "ats_home_games": ats_hw + ats_hl + ats_hp,
                "ats_home_win_pct": round(ats_hw / (ats_hw + ats_hl + ats_hp), 3) if (ats_hw + ats_hl + ats_hp) else None,
                "ats_neutral_games": ats_nw + ats_nl + ats_np,
                "ats_neutral_win_pct": round(ats_nw / (ats_nw + ats_nl + ats_np), 3) if (ats_nw + ats_nl + ats_np) else None,
                "ats_away_games": ats_aw + ats_al + ats_ap,
                "ats_away_win_pct": round(ats_aw / (ats_aw + ats_al + ats_ap), 3) if (ats_aw + ats_al + ats_ap) else None,
                "ppg": round(tp / tg, 1) if tg else "",
                "home_ppg": round(hp / hg, 1) if hg else "",
                "neutral_ppg": round(np_ / ng, 1) if ng else "",
                "away_ppg": round(ap / ag, 1) if ag else "",
            }

            # New PIT stats (need at least 1 prior game with boxscore)
            if bg > 0:
                # Offensive shooting per game
                entry["ftm_pg"] = round(s_ftm / bg, 1)
                entry["fta_pg"] = round(s_fta / bg, 1)
                entry["ft_pct"] = round(s_ftm / s_fta * 100, 1) if s_fta else ""
                entry["3pm_pg"] = round(s_3pm / bg, 1)
                entry["3pa_pg"] = round(s_3pa / bg, 1)
                entry["3pt_pct"] = round(s_3pm / s_3pa * 100, 1) if s_3pa else ""
                s_2pm = s_fgm - s_3pm
                s_2pa = s_fga - s_3pa
                entry["2pm_pg"] = round(s_2pm / bg, 1)
                entry["2pa_pg"] = round(s_2pa / bg, 1)
                entry["2pt_pct"] = round(s_2pm / s_2pa * 100, 1) if s_2pa else ""

                # Defensive shooting allowed per game
                entry["def_ftm_pg"] = round(d_ftm / bg, 1)
                entry["def_fta_pg"] = round(d_fta / bg, 1)
                entry["def_ft_pct"] = round(d_ftm / d_fta * 100, 1) if d_fta else ""
                entry["def_3pm_pg"] = round(d_3pm / bg, 1)
                entry["def_3pa_pg"] = round(d_3pa / bg, 1)
                entry["def_3pt_pct"] = round(d_3pm / d_3pa * 100, 1) if d_3pa else ""
                d_2pm = d_fgm - d_3pm
                d_2pa = d_fga - d_3pa
                entry["def_2pm_pg"] = round(d_2pm / bg, 1)
                entry["def_2pa_pg"] = round(d_2pa / bg, 1)
                entry["def_2pt_pct"] = round(d_2pm / d_2pa * 100, 1) if d_2pa else ""

                # Rebounding
                entry["oreb_pg"] = round(s_oreb / bg, 1)
                entry["dreb_pg"] = round(s_dreb / bg, 1)

                # Turnovers
                entry["to_pg"] = round(s_to / bg, 1)
                entry["forced_to_pg"] = round(d_to / bg, 1)

                # Pace (possessions per game)
                poss = s_fga - s_oreb + s_to + 0.475 * s_fta
                pace_pg = round(poss / bg, 1)
                entry["pace"] = pace_pg

                # Derived per-possession stats
                if pace_pg > 0:
                    ppg_val = tp / tg if tg else 0
                    entry["ppp"] = round(ppg_val / pace_pg, 2)

                    # Defensive PPG & PPP
                    def_ppg_val = (d_2pm * 2 + d_3pm * 3 + d_ftm) / bg
                    entry["def_ppg"] = round(def_ppg_val, 1)
                    entry["def_ppp"] = round(def_ppg_val / pace_pg, 2)

                    # Offense per-possession
                    entry["3pa_per_poss"] = round((s_3pa / bg) / pace_pg, 2)
                    entry["2pa_per_poss"] = round((s_2pa / bg) / pace_pg, 2)
                    entry["fta_per_poss"] = round((s_fta / bg) / pace_pg, 2)
                    entry["oreb_per_poss"] = round((s_oreb / bg) / pace_pg, 2)
                    entry["to_per_poss"] = round((s_to / bg) / pace_pg, 2)

                    # Defense per-possession
                    entry["def_3pa_per_poss"] = round((d_3pa / bg) / pace_pg, 2)
                    entry["def_2pa_per_poss"] = round((d_2pa / bg) / pace_pg, 2)
                    entry["def_fta_per_poss"] = round((d_fta / bg) / pace_pg, 2)
                    entry["dreb_per_poss"] = round((s_dreb / bg) / pace_pg, 2)
                    entry["forced_to_per_poss"] = round((d_to / bg) / pace_pg, 2)
                else:
                    for k in ["ppp", "def_ppg", "def_ppp",
                              "3pa_per_poss", "2pa_per_poss", "fta_per_poss",
                              "oreb_per_poss", "to_per_poss",
                              "def_3pa_per_poss", "def_2pa_per_poss", "def_fta_per_poss",
                              "dreb_per_poss", "forced_to_per_poss"]:
                        entry[k] = None
            else:
                for k in _NEW_PIT_KEYS:
                    if k not in ("sos", "ats_margin_wins", "ats_margin_losses"):
                        entry[k] = None

            # SOS (uses all games, not just boxscore games)
            entry["sos"] = round(sos_sum / sos_count, 1) if sos_count else None

            # ATS margin running averages (PIT — before this game)
            entry["ats_margin_wins"] = round(ats_margin_win_sum / ats_margin_win_count, 1) if ats_margin_win_count else None
            entry["ats_margin_losses"] = round(ats_margin_loss_sum / ats_margin_loss_count, 1) if ats_margin_loss_count else None

            team_pit[eid] = entry

            # ── UPDATE counters with this game ──
            w = row["win_loss"] == "W"
            ha = row["home_away"]
            score = int(row["team_score"])

            ow += int(w); ol += int(not w)
            tp += score; tg += 1

            if ha == "home":
                hw += int(w); hl += int(not w)
                hp += score; hg += 1
            elif ha == "neutral":
                nw += int(w); nl += int(not w)
                np_ += score; ng += 1
            else:
                aw += int(w); al += int(not w)
                ap += score; ag += 1

            # ATS (only when closing spread exists)
            cs = row.get("closing_fg_spread")
            if pd.notna(cs):
                margin = int(row["team_score"]) - int(row["opp_score"])
                ats_val = margin + float(cs)
                if ats_val > 0:
                    ats_ow += 1
                    if ha == "home": ats_hw += 1
                    elif ha == "neutral": ats_nw += 1
                    else: ats_aw += 1
                elif ats_val < 0:
                    ats_ol += 1
                    if ha == "home": ats_hl += 1
                    elif ha == "neutral": ats_nl += 1
                    else: ats_al += 1
                else:
                    ats_op += 1
                    if ha == "home": ats_hp += 1
                    elif ha == "neutral": ats_np += 1
                    else: ats_ap += 1

                # ATS margin running averages
                if ats_val > 0:
                    ats_margin_win_sum += ats_val
                    ats_margin_win_count += 1
                elif ats_val < 0:
                    ats_margin_loss_sum += ats_val
                    ats_margin_loss_count += 1

            # Boxscore counters (only if data exists)
            if pd.notna(row.get("team_fgm")):
                bg += 1
                s_fgm += _safe_acc(row["team_fgm"])
                s_fga += _safe_acc(row["team_fga"])
                s_3pm += _safe_acc(row["team_3pm"])
                s_3pa += _safe_acc(row["team_3pa"])
                s_ftm += _safe_acc(row["team_ftm"])
                s_fta += _safe_acc(row["team_fta"])
                s_oreb += _safe_acc(row["team_oreb"])
                s_dreb += _safe_acc(row["team_dreb"])
                s_to += _safe_acc(row["team_to"])
                d_fgm += _safe_acc(row["opp_fgm"])
                d_fga += _safe_acc(row["opp_fga"])
                d_3pm += _safe_acc(row["opp_3pm"])
                d_3pa += _safe_acc(row["opp_3pa"])
                d_ftm += _safe_acc(row["opp_ftm"])
                d_fta += _safe_acc(row["opp_fta"])
                d_to += _safe_acc(row["opp_to"])

            # SOS counter (every game counts; unranked = 100)
            opp_rank = row.get("opp_ap_rank")
            if pd.notna(opp_rank):
                sos_sum += int(opp_rank)
            else:
                sos_sum += 100
            sos_count += 1

        pit[team_name] = team_pit

    # -- Write PIT columns back to df --
    team_records = []
    opp_records = []

    for _, row in df.iterrows():
        team = row["team"]
        eid = row["event_id"]

        t = pit.get(team, {}).get(eid, {})
        team_dict = {
            "team_games": t.get("games", 0),
            "team_win_pct": t.get("win_pct"),
            "team_home_games": t.get("home_games", 0),
            "team_home_win_pct": t.get("home_win_pct"),
            "team_neutral_games": t.get("neutral_games", 0),
            "team_neutral_win_pct": t.get("neutral_win_pct"),
            "team_away_games": t.get("away_games", 0),
            "team_away_win_pct": t.get("away_win_pct"),
            "team_ats_games": t.get("ats_games", 0),
            "team_ats_win_pct": t.get("ats_win_pct"),
            "team_home_ats_games": t.get("ats_home_games", 0),
            "team_home_ats_win_pct": t.get("ats_home_win_pct"),
            "team_neutral_ats_games": t.get("ats_neutral_games", 0),
            "team_neutral_ats_win_pct": t.get("ats_neutral_win_pct"),
            "team_away_ats_games": t.get("ats_away_games", 0),
            "team_away_ats_win_pct": t.get("ats_away_win_pct"),
            "team_ppg": t.get("ppg", ""),
            "team_home_ppg": t.get("home_ppg", ""),
            "team_neutral_ppg": t.get("neutral_ppg", ""),
            "team_away_ppg": t.get("away_ppg", ""),
        }
        # New team PIT columns
        for k in _NEW_PIT_KEYS:
            team_dict[f"team_{k}"] = t.get(k, "")
        team_records.append(team_dict)

        # Opponent PIT (only if opponent is one of our 68 teams)
        opp_id = str(row.get("opponent_id", ""))
        opp_canonical = id_to_name.get(opp_id)
        if opp_canonical and opp_canonical in pit:
            o = pit[opp_canonical].get(eid, {})
            opp_dict = {
                "opp_games": o.get("games", 0),
                "opp_win_pct": o.get("win_pct"),
                "opp_home_games": o.get("home_games", 0),
                "opp_home_win_pct": o.get("home_win_pct"),
                "opp_neutral_games": o.get("neutral_games", 0),
                "opp_neutral_win_pct": o.get("neutral_win_pct"),
                "opp_away_games": o.get("away_games", 0),
                "opp_away_win_pct": o.get("away_win_pct"),
                "opp_ats_games": o.get("ats_games", 0),
                "opp_ats_win_pct": o.get("ats_win_pct"),
                "opp_home_ats_games": o.get("ats_home_games", 0),
                "opp_home_ats_win_pct": o.get("ats_home_win_pct"),
                "opp_neutral_ats_games": o.get("ats_neutral_games", 0),
                "opp_neutral_ats_win_pct": o.get("ats_neutral_win_pct"),
                "opp_away_ats_games": o.get("ats_away_games", 0),
                "opp_away_ats_win_pct": o.get("ats_away_win_pct"),
                "opp_ppg": o.get("ppg", ""),
                "opp_home_ppg": o.get("home_ppg", ""),
                "opp_neutral_ppg": o.get("neutral_ppg", ""),
                "opp_away_ppg": o.get("away_ppg", ""),
            }
            for k in _NEW_PIT_KEYS:
                opp_dict[f"opp_{k}"] = o.get(k, "")
            opp_records.append(opp_dict)
        else:
            opp_records.append(
                {col: None for col in PIT_COLUMNS if col.startswith("opp_")}
            )

    team_pit_df = pd.DataFrame(team_records, index=df.index)
    opp_pit_df = pd.DataFrame(opp_records, index=df.index)

    for col in team_pit_df.columns:
        df[col] = team_pit_df[col]
    for col in opp_pit_df.columns:
        df[col] = opp_pit_df[col]

    print(f"  [{_now_str()}] Phase 3 done.")
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4 — Export CSV
# ═══════════════════════════════════════════════════════════════════════════

CSV_COLUMNS = {
    "team": "Team",
    "date": "Date",
    "commence_time_utc": "Commence_Time_UTC",
    "opponent": "Opponent",
    "home_away": "Home_Away",
    "conference_game": "Conference_Game",
    "team_score": "Team_Final_Score",
    "opp_score": "Opp_Final_Score",
    "win_loss": "W_L",
    "opening_fg_ml": "Opening_FG_ML",
    "closing_fg_ml": "Closing_FG_ML",
    "opening_fg_spread": "Opening_FG_Spread",
    "closing_fg_spread": "Closing_FG_Spread",
    # Record (games + win pct)
    "team_games": "Team_Games",
    "team_win_pct": "Team_Win_Pct",
    "team_home_games": "Team_Home_Games",
    "team_home_win_pct": "Team_Home_Win_Pct",
    "team_neutral_games": "Team_Neutral_Games",
    "team_neutral_win_pct": "Team_Neutral_Win_Pct",
    "team_away_games": "Team_Away_Games",
    "team_away_win_pct": "Team_Away_Win_Pct",
    "opp_games": "Opp_Games",
    "opp_win_pct": "Opp_Win_Pct",
    "opp_home_games": "Opp_Home_Games",
    "opp_home_win_pct": "Opp_Home_Win_Pct",
    "opp_neutral_games": "Opp_Neutral_Games",
    "opp_neutral_win_pct": "Opp_Neutral_Win_Pct",
    "opp_away_games": "Opp_Away_Games",
    "opp_away_win_pct": "Opp_Away_Win_Pct",
    # ATS (games + win pct)
    "team_ats_games": "Team_ATS_Games",
    "team_ats_win_pct": "Team_ATS_Win_Pct",
    "team_home_ats_games": "Team_Home_ATS_Games",
    "team_home_ats_win_pct": "Team_Home_ATS_Win_Pct",
    "team_neutral_ats_games": "Team_Neutral_ATS_Games",
    "team_neutral_ats_win_pct": "Team_Neutral_ATS_Win_Pct",
    "team_away_ats_games": "Team_Away_ATS_Games",
    "team_away_ats_win_pct": "Team_Away_ATS_Win_Pct",
    "opp_ats_games": "Opp_ATS_Games",
    "opp_ats_win_pct": "Opp_ATS_Win_Pct",
    "opp_home_ats_games": "Opp_Home_ATS_Games",
    "opp_home_ats_win_pct": "Opp_Home_ATS_Win_Pct",
    "opp_neutral_ats_games": "Opp_Neutral_ATS_Games",
    "opp_neutral_ats_win_pct": "Opp_Neutral_ATS_Win_Pct",
    "opp_away_ats_games": "Opp_Away_ATS_Games",
    "opp_away_ats_win_pct": "Opp_Away_ATS_Win_Pct",
    # PPG
    "team_ppg": "Team_PPG",
    "team_home_ppg": "Team_Home_PPG",
    "team_neutral_ppg": "Team_Neutral_PPG",
    "team_away_ppg": "Team_Away_PPG",
    "opp_ppg": "Opp_PPG",
    "opp_home_ppg": "Opp_Home_PPG",
    "opp_neutral_ppg": "Opp_Neutral_PPG",
    "opp_away_ppg": "Opp_Away_PPG",
    # Team offensive shooting
    "team_ftm_pg": "Team_FTM_PG",
    "team_fta_pg": "Team_FTA_PG",
    "team_ft_pct": "Team_FT_Pct",
    "team_3pm_pg": "Team_3PM_PG",
    "team_3pa_pg": "Team_3PA_PG",
    "team_3pt_pct": "Team_3PT_Pct",
    "team_2pm_pg": "Team_2PM_PG",
    "team_2pa_pg": "Team_2PA_PG",
    "team_2pt_pct": "Team_2PT_Pct",
    # Team defensive shooting allowed
    "team_def_ftm_pg": "Team_Def_FTM_PG",
    "team_def_fta_pg": "Team_Def_FTA_PG",
    "team_def_ft_pct": "Team_Def_FT_Pct",
    "team_def_3pm_pg": "Team_Def_3PM_PG",
    "team_def_3pa_pg": "Team_Def_3PA_PG",
    "team_def_3pt_pct": "Team_Def_3PT_Pct",
    "team_def_2pm_pg": "Team_Def_2PM_PG",
    "team_def_2pa_pg": "Team_Def_2PA_PG",
    "team_def_2pt_pct": "Team_Def_2PT_Pct",
    # Team rebounding & turnovers
    "team_oreb_pg": "Team_OREB_PG",
    "team_dreb_pg": "Team_DREB_PG",
    "team_to_pg": "Team_TO_PG",
    "team_forced_to_pg": "Team_Forced_TO_PG",
    # Team pace & SOS
    "team_pace": "Team_Pace",
    "team_sos": "Team_SOS",
    # Opponent offensive shooting
    "opp_ftm_pg": "Opp_FTM_PG",
    "opp_fta_pg": "Opp_FTA_PG",
    "opp_ft_pct": "Opp_FT_Pct",
    "opp_3pm_pg": "Opp_3PM_PG",
    "opp_3pa_pg": "Opp_3PA_PG",
    "opp_3pt_pct": "Opp_3PT_Pct",
    "opp_2pm_pg": "Opp_2PM_PG",
    "opp_2pa_pg": "Opp_2PA_PG",
    "opp_2pt_pct": "Opp_2PT_Pct",
    # Opponent defensive shooting allowed
    "opp_def_ftm_pg": "Opp_Def_FTM_PG",
    "opp_def_fta_pg": "Opp_Def_FTA_PG",
    "opp_def_ft_pct": "Opp_Def_FT_Pct",
    "opp_def_3pm_pg": "Opp_Def_3PM_PG",
    "opp_def_3pa_pg": "Opp_Def_3PA_PG",
    "opp_def_3pt_pct": "Opp_Def_3PT_Pct",
    "opp_def_2pm_pg": "Opp_Def_2PM_PG",
    "opp_def_2pa_pg": "Opp_Def_2PA_PG",
    "opp_def_2pt_pct": "Opp_Def_2PT_Pct",
    # Opponent rebounding & turnovers
    "opp_oreb_pg": "Opp_OREB_PG",
    "opp_dreb_pg": "Opp_DREB_PG",
    "opp_to_pg": "Opp_TO_PG",
    "opp_forced_to_pg": "Opp_Forced_TO_PG",
    # Opponent pace & SOS
    "opp_pace": "Opp_Pace",
    "opp_sos": "Opp_SOS",
    # Opponent AP rank (raw per-game)
    "opp_ap_rank": "Opp_AP_Rank",
}


def rebuild_odds_history() -> None:
    """Append new cached snapshots to the odds history parquet.

    Reads existing parquet (if any), flattens any new cache JSON files
    that haven't been incorporated yet, and appends them.  Never drops
    existing rows.
    """
    if not ODDS_CACHE_DIR.exists():
        return

    cache_files = sorted(ODDS_CACHE_DIR.glob("*.json"))
    if not cache_files:
        return

    # Load existing history
    existing = pd.DataFrame()
    existing_snapshots: set[str] = set()
    if ODDS_HISTORY_FILE.exists():
        existing = pd.read_parquet(ODDS_HISTORY_FILE)
        # Track which snapshot_times we already have to avoid duplicates
        if "snapshot_time" in existing.columns:
            existing_snapshots = set(existing["snapshot_time"].unique())

    # Check if any cache file is newer than the parquet
    if ODDS_HISTORY_FILE.exists() and len(existing) > 0:
        history_mtime = ODDS_HISTORY_FILE.stat().st_mtime
        newest_cache = max(f.stat().st_mtime for f in cache_files)
        if newest_cache <= history_mtime:
            print(f"[{_now_str()}] Odds history: up to date "
                  f"({len(existing):,} rows)")
            return

    new_rows: list[dict] = []
    new_snap_count = 0
    for f in cache_files:
        with open(f, "r") as fh:
            data = json.load(fh)
        snapshot_time = data.get("timestamp", "")
        # Skip snapshots we already have
        if snapshot_time in existing_snapshots:
            continue
        new_snap_count += 1
        for game in data.get("data", []):
            game_id = game.get("id", "")
            commence_time = game.get("commence_time", "")
            home_team = game.get("home_team", "")
            away_team = game.get("away_team", "")
            for bm in game.get("bookmakers", []):
                bookmaker = bm.get("key", "")
                last_update = bm.get("last_update", "")
                for mkt in bm.get("markets", []):
                    market = mkt.get("key", "")
                    for oc in mkt.get("outcomes", []):
                        new_rows.append({
                            "snapshot_time": snapshot_time,
                            "game_id": game_id,
                            "commence_time": commence_time,
                            "home_team": home_team,
                            "away_team": away_team,
                            "bookmaker": bookmaker,
                            "last_update": last_update,
                            "market": market,
                            "outcome": oc.get("name", ""),
                            "price": oc.get("price"),
                            "point": oc.get("point"),
                        })

    if not new_rows:
        print(f"[{_now_str()}] Odds history: up to date "
              f"({len(existing):,} rows)")
        return

    new_df = pd.DataFrame(new_rows)
    if len(existing) > 0:
        merged = pd.concat([existing, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_parquet(ODDS_HISTORY_FILE, index=False, engine="pyarrow")
    print(f"[{_now_str()}] Odds history: appended {new_snap_count} new snapshots "
          f"({len(new_rows):,} new rows, {len(merged):,} total, "
          f"{ODDS_HISTORY_FILE.stat().st_size // 1024}KB)")


def phase4_export_csv(df: pd.DataFrame) -> None:
    # Only export tournament team rows
    tracked = df[df["is_tracked"] == True]  # noqa: E712
    export = tracked[list(CSV_COLUMNS.keys())].copy()
    # Map conference_game bool -> Y/N
    export["conference_game"] = export["conference_game"].map(
        {True: "Y", False: "N", "Y": "Y", "N": "N"}).fillna("N")
    export.rename(columns=CSV_COLUMNS, inplace=True)
    export.to_csv(CSV_FILE, index=False)
    print(f"[{_now_str()}] Phase 4: CSV written to {CSV_FILE}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill-odds", nargs="?", const="all", default=None,
                        metavar="DATE",
                        help="Fetch odds for missing data. Optional DATE "
                             "(YYYY-MM-DD) to scope to a single date, or "
                             "omit for all missing dates.")
    parser.add_argument("--backfill-boxscore", action="store_true",
                        help="Fetch boxscore stats for existing games missing them")
    parser.add_argument("--backfill-ranks", action="store_true",
                        help="Re-scan scoreboards to fill AP ranks for all games")
    parser.add_argument("--backfill-times", action="store_true",
                        help="Backfill commence_time_utc for all games")
    parser.add_argument("--rebuild-all", action="store_true",
                        help="Re-scan all dates to capture all D1 teams")
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously on an interval")
    parser.add_argument("--interval", type=int, default=21600,
                        help="Seconds between runs in loop mode (default: 21600 = 6 hours)")
    args = parser.parse_args()

    def run_once():
        print("=" * 60)
        print("  NCAAB Daily Builder -- Season 2025-26 (Full D1)")
        print("=" * 60)
        print()

        # Full D1 team directory
        d1_team_ids = get_all_d1_ids()
        d1_id_to_name = {tid: name for name, tid in d1_team_ids.items()}
        d1_conf_by_id = fetch_d1_conference_map()

        # Tournament subset (for odds, CSV export, is_tracked flag)
        tournament_team_ids = get_team_ids()
        tournament_ids = set(tournament_team_ids.values())

        df = load_game_logs()

        # Ensure is_tracked is set for existing rows
        if "is_tracked" not in df.columns:
            df["is_tracked"] = None
        if not df.empty:
            tracked_names = set(TEAMS)
            df.loc[df["is_tracked"].isna(), "is_tracked"] = (
                df.loc[df["is_tracked"].isna(), "team"].isin(tracked_names)
            )

        # --rebuild-all: drop old non-tournament rows (partial mirror/opponent data)
        # and re-scan from season start to capture all D1 teams
        if args.rebuild_all:
            if not df.empty:
                old_len = len(df)
                df = df[df["is_tracked"] == True].copy()  # noqa: E712
                print(f"[{_now_str()}] --rebuild-all: kept {len(df)} tournament rows, "
                      f"dropped {old_len - len(df)} non-tournament rows")

        # Show existing data summary
        existing_spreads = df["closing_fg_spread"].notna().sum() if not df.empty else 0
        existing_boxscore = df["team_fgm"].notna().sum() if not df.empty else 0
        print(f"[{_now_str()}] Loaded {len(df)} existing rows "
              f"({existing_spreads} with spreads, {existing_boxscore} with boxscore)")

        # Show what we're about to do
        flags = []
        if args.rebuild_all:
            flags.append("rebuild-all (full D1 re-scan)")
        if args.backfill_odds:
            if args.backfill_odds == "all":
                flags.append("backfill-odds (ALL missing dates)")
            else:
                flags.append(f"backfill-odds (date: {args.backfill_odds})")
        if args.backfill_boxscore:
            flags.append("backfill-boxscore")
        if args.backfill_ranks:
            flags.append("backfill-ranks")
        if args.backfill_times:
            flags.append("backfill-times")
        if flags:
            print(f"[{_now_str()}] Flags: {', '.join(flags)}")
        else:
            print(f"[{_now_str()}] Mode: daily incremental (new dates only)")
        print()

        # Phase 1: new games (all D1 teams)
        force_start = SEASON_START if args.rebuild_all else None
        df, new_dates = phase1_fetch_games(
            df, d1_team_ids, tournament_ids, d1_id_to_name, d1_conf_by_id,
            force_start=force_start,
        )

        # Phase 2: odds (tournament teams only)
        if args.backfill_odds:
            date_filter = None if args.backfill_odds == "all" else args.backfill_odds
            if date_filter:
                print(f"[{_now_str()}] Backfill mode: odds for {date_filter}")
            else:
                print(f"[{_now_str()}] Backfill mode: true opening/closing lines (all)")
            df = phase2_backfill_odds(df, date_filter=date_filter)
        else:
            df = phase2_fetch_odds(df, new_dates)

        # Backfill boxscore stats for existing games
        if args.backfill_boxscore:
            print(f"[{_now_str()}] Backfilling boxscore data ...")
            df = backfill_boxscore_data(df, d1_team_ids)

        # Backfill AP ranks from scoreboards
        if args.backfill_ranks:
            print(f"[{_now_str()}] Backfilling AP ranks ...")
            df = backfill_ap_ranks(df, tournament_ids, d1_id_to_name)

        # Backfill commence times
        if args.backfill_times:
            print(f"[{_now_str()}] Backfilling commence times ...")
            df = backfill_commence_times(df)

        # Save after API phases (in case Phase 3/4 crash, data is safe)
        save_game_logs(df)

        # Phase 3: PIT stats (always recomputed, no API)
        df = phase3_compute_pit(df, d1_team_ids)

        # Phase 4: export CSV
        phase4_export_csv(df)

        # Save final with PIT columns
        save_game_logs(df)

        # Rebuild odds history parquet from cache (no API calls)
        rebuild_odds_history()

        tracked_df = df[df["is_tracked"] == True]  # noqa: E712
        spread_count = tracked_df["closing_fg_spread"].notna().sum()
        boxscore_count = df["team_fgm"].notna().sum()
        opp_pit_filled = tracked_df["opp_games"].notna().sum()
        print(f"\n{'=' * 60}")
        print(f"  DONE! {len(df)} total rows ({len(tracked_df)} tracked), "
              f"{df['team'].nunique()} teams")
        print_odds_api_usage()
        print(f"  Rows with spreads:  {spread_count}/{len(tracked_df)}")
        print(f"  Rows with boxscore: {boxscore_count}/{len(df)}")
        print(f"  Opp PIT filled:     {opp_pit_filled}/{len(tracked_df)}")
        print(f"  Parquet: {PARQUET_FILE}")
        print(f"  CSV:     {CSV_FILE}")
        print(f"  Odds DB: {ODDS_HISTORY_FILE}")
        print(f"{'=' * 60}")

    # Run once immediately
    run_once()

    # Loop mode: re-run on interval
    if args.loop:
        print(f"\n[{_now_str()}] Looping every {args.interval}s "
              f"({args.interval // 3600}h). Ctrl+C to stop.")
        while True:
            time.sleep(args.interval)
            try:
                run_once()
            except Exception as e:
                print(f"  [{_now_str()}] Error during run: {e}")


if __name__ == "__main__":
    main()
