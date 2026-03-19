# pip install requests python-dotenv pandas pyarrow
"""
NCAAB Daily Game Log Builder
=============================
Four-phase incremental pipeline:

  Phase 1 -- Fetch new games      (ESPN scoreboard, 1 API call / date)
             + boxscore stats     (ESPN summary, 1 API call / event)
  Phase 2 -- Fetch odds           (The Odds API, 2 calls / NEW date only)
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
CSV_FILE = SCRIPT_DIR / "ncaab_game_logs.csv"

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


def api_get(url: str, params: dict | None = None, tag: str = "") -> dict | None:
    """GET with retries, backoff, and rate-limiting."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
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
# Parquet I/O
# ---------------------------------------------------------------------------

RAW_COLUMNS = [
    "team", "date", "event_id",
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
]

ODDS_COLUMNS = [
    "opening_fg_ml", "closing_fg_ml",
    "opening_fg_spread", "closing_fg_spread",
]

PIT_COLUMNS = [
    # Record
    "team_record", "team_home_record", "team_neutral_record", "team_away_record",
    "opp_record", "opp_home_record", "opp_neutral_record", "opp_away_record",
    # ATS
    "team_ats", "team_home_ats", "team_neutral_ats", "team_away_ats",
    "opp_ats", "opp_home_ats", "opp_neutral_ats", "opp_away_ats",
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
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Ensure all expected columns exist
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = None

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
    tracked_ids: set[str],
    id_to_name: dict[str, str],
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
        game_date = ""
        if date_raw:
            try:
                dt = datetime.fromisoformat(date_raw.replace("Z", "+00:00"))
                game_date = dt.strftime("%Y-%m-%d")
            except ValueError:
                game_date = date_raw[:10]

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
                "is_tracked": cid in tracked_ids,
            })

        for i, team_c in enumerate(comp_data):
            if not team_c["is_tracked"]:
                continue
            opp_c = comp_data[1 - i]
            canonical = id_to_name.get(team_c["id"], "")
            if not canonical:
                continue

            ha = "neutral" if neutral_site else team_c["home_away"]
            win_loss = ("W" if team_c["score"] > opp_c["score"]
                        else ("L" if team_c["score"] < opp_c["score"]
                              else "T"))

            opp_canonical = id_to_name.get(opp_c["id"])
            conf_game = False
            if opp_canonical:
                tc = CONFERENCE_MAP.get(canonical)
                oc = CONFERENCE_MAP.get(opp_canonical)
                conf_game = bool(tc and tc == oc)

            rows.append({
                "team": canonical,
                "date": game_date,
                "event_id": event_id,
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
    print(f"  [{_now_str()}] Backfilling boxscore for "
          f"{len(missing_events)} events ...")

    for i, eid in enumerate(missing_events, 1):
        if i % 50 == 0:
            print(f"  [{_now_str()}] Progress: {i}/{len(missing_events)}")

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
    """Re-scan historical scoreboards to fill in opp_ap_rank for all games."""
    missing_dates = sorted(df[df["opp_ap_rank"].isna()]["date"].unique())
    if not missing_dates:
        print(f"  [{_now_str()}] All rows already have AP rank data.")
        return df

    print(f"  [{_now_str()}] Backfilling AP ranks for "
          f"{len(missing_dates)} dates ...")

    # Build a lookup: (event_id, competitor_id) -> ap_rank
    rank_lookup: dict[tuple[str, str], int | None] = {}

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
            for c in comps[0].get("competitors", []):
                cid = str(c.get("id", c.get("team", {}).get("id", "")))
                rank = _extract_ap_rank(c)
                rank_lookup[(event_id, cid)] = rank

    # Apply ranks to dataframe
    filled = 0
    for idx in df[df["opp_ap_rank"].isna()].index:
        row = df.loc[idx]
        opp_id = str(row.get("opponent_id", ""))
        eid = row["event_id"]
        rank = rank_lookup.get((eid, opp_id))
        if rank is not None:
            df.at[idx, "opp_ap_rank"] = rank
            filled += 1

    total_ranked = df["opp_ap_rank"].notna().sum()
    print(f"  [{_now_str()}] AP rank backfill done: "
          f"{filled} new ranks filled, {total_ranked}/{len(df)} total.")
    return df


def phase1_fetch_games(
    df: pd.DataFrame,
    team_ids: dict[str, str],
    tracked_ids: set[str],
    id_to_name: dict[str, str],
) -> tuple[pd.DataFrame, set[str]]:
    """Return (updated_df, set_of_new_date_strings)."""
    existing_keys: set[tuple[str, str]] = set()
    if not df.empty:
        existing_keys = set(zip(df["team"], df["event_id"]))

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    if df.empty:
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

        events = fetch_scoreboard(date)
        if not events:
            continue

        day_games = parse_scoreboard(events, tracked_ids, id_to_name)
        day_games = [
            g for g in day_games
            if (g["team"], g["event_id"]) not in existing_keys
        ]
        if not day_games:
            continue

        print(f"  [{_now_str()}] ({i}/{len(dates_to_fetch)}) "
              f"{date_str}: {len(day_games)} new rows")

        backfill_game_details(day_games, team_ids)

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

def fetch_daily_odds(date_str: str) -> dict:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    offset_h = 5 if (dt.month >= 11 or dt.month <= 2) else 4

    open_iso = dt.replace(
        hour=10 + offset_h, minute=0, second=0,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    close_iso = dt.replace(
        hour=17 + offset_h, minute=0, second=0,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    base_params = {
        "apiKey": THE_ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h,spreads",
        "oddsFormat": "american",
    }
    url = f"{ODDS_BASE}/v4/historical/sports/basketball_ncaab/odds"

    fg_open = api_get(url, params={**base_params, "date": open_iso},
                      tag=f"odds-open-{date_str}")
    fg_close = api_get(url, params={**base_params, "date": close_iso},
                       tag=f"odds-close-{date_str}")
    return {"fg_open": fg_open, "fg_close": fg_close}


def _find_team_in_odds(
    snap: dict | None, odds_lower: str, opp_keywords: list[str],
) -> dict | None:
    if not snap:
        return None
    for game in snap.get("data", []):
        home = game.get("home_team", "").lower()
        away = game.get("away_team", "").lower()
        if odds_lower != home and odds_lower != away:
            continue
        if opp_keywords:
            other = away if odds_lower == home else home
            if not any(kw in other for kw in opp_keywords):
                continue
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


def extract_odds_for_game(
    odds_data: dict, canonical_name: str, opp_display_name: str,
) -> dict:
    result = {
        "opening_fg_ml": None, "closing_fg_ml": None,
        "opening_fg_spread": None, "closing_fg_spread": None,
    }
    odds_name = ODDS_NAME_MAP.get(canonical_name, "")
    if not odds_name:
        return result
    odds_lower = odds_name.lower()

    opp_keywords = (
        [w.lower() for w in opp_display_name.split() if len(w) >= 3]
        if opp_display_name else []
    )

    for snap_key, ml_col, spread_col in [
        ("fg_open",  "opening_fg_ml",  "opening_fg_spread"),
        ("fg_close", "closing_fg_ml",  "closing_fg_spread"),
    ]:
        game = _find_team_in_odds(
            odds_data.get(snap_key), odds_lower, opp_keywords,
        )
        if not game:
            continue
        bookmakers = game.get("bookmakers", [])
        result[ml_col] = _first_bookmaker_value(
            bookmakers, "h2h", odds_lower, "price")
        result[spread_col] = _first_bookmaker_value(
            bookmakers, "spreads", odds_lower, "point")

    return result


def phase2_fetch_odds(df: pd.DataFrame, new_dates: set[str]) -> pd.DataFrame:
    """Fetch odds ONLY for dates that had new games in Phase 1."""
    if not THE_ODDS_API_KEY:
        print(f"[{_now_str()}] Phase 2: skipped (no Odds API key).")
        return df

    if not new_dates:
        print(f"[{_now_str()}] Phase 2: no new dates to fetch odds for.")
        return df

    sorted_dates = sorted(new_dates)
    print(f"[{_now_str()}] Phase 2: fetching odds for {len(sorted_dates)} "
          f"date(s) ({len(sorted_dates) * 2} API calls) ...")

    for i, date_str in enumerate(sorted_dates, 1):
        print(f"  [{_now_str()}] ({i}/{len(sorted_dates)}) {date_str}")
        odds_data = fetch_daily_odds(date_str)

        mask = df["date"] == date_str
        for idx in df[mask].index:
            row = df.loc[idx]
            # Only fill in odds if not already present
            if pd.notna(row.get("closing_fg_spread")):
                continue
            odds = extract_odds_for_game(
                odds_data, row["team"], row["opponent"])
            for col, val in odds.items():
                if val is not None:
                    df.at[idx, col] = val

    matched = df["closing_fg_spread"].notna().sum()
    print(f"  [{_now_str()}] Phase 2 done: {matched}/{len(df)} rows have spreads.")
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

    # -- Pre-compute per-team PIT: {team -> {event_id -> stats_dict}} --
    pit: dict[str, dict[str, dict]] = {}

    for team_name in TEAMS:
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

        team_pit: dict[str, dict] = {}

        for _, row in team_rows.iterrows():
            eid = row["event_id"]

            # ── SAVE stats as-of BEFORE this game ──
            entry: dict = {
                "record": f"{ow}-{ol}",
                "home_record": f"{hw}-{hl}",
                "neutral_record": f"{nw}-{nl}",
                "away_record": f"{aw}-{al}",
                "ats": f"{ats_ow}-{ats_ol}-{ats_op}",
                "ats_home": f"{ats_hw}-{ats_hl}-{ats_hp}",
                "ats_neutral": f"{ats_nw}-{ats_nl}-{ats_np}",
                "ats_away": f"{ats_aw}-{ats_al}-{ats_ap}",
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
                entry["pace"] = round(poss / bg, 1)
            else:
                for k in _NEW_PIT_KEYS:
                    if k != "sos":
                        entry[k] = ""

            # SOS (uses all games, not just boxscore games)
            entry["sos"] = round(sos_sum / sos_count, 1) if sos_count else ""

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
            "team_record": t.get("record", "0-0"),
            "team_home_record": t.get("home_record", "0-0"),
            "team_neutral_record": t.get("neutral_record", "0-0"),
            "team_away_record": t.get("away_record", "0-0"),
            "team_ats": t.get("ats", "0-0-0"),
            "team_home_ats": t.get("ats_home", "0-0-0"),
            "team_neutral_ats": t.get("ats_neutral", "0-0-0"),
            "team_away_ats": t.get("ats_away", "0-0-0"),
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
                "opp_record": o.get("record", "0-0"),
                "opp_home_record": o.get("home_record", "0-0"),
                "opp_neutral_record": o.get("neutral_record", "0-0"),
                "opp_away_record": o.get("away_record", "0-0"),
                "opp_ats": o.get("ats", "0-0-0"),
                "opp_home_ats": o.get("ats_home", "0-0-0"),
                "opp_neutral_ats": o.get("ats_neutral", "0-0-0"),
                "opp_away_ats": o.get("ats_away", "0-0-0"),
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
                {col: "" for col in PIT_COLUMNS if col.startswith("opp_")}
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
    # Record
    "team_record": "Team_Record",
    "team_home_record": "Team_Home_Record",
    "team_neutral_record": "Team_Neutral_Record",
    "team_away_record": "Team_Away_Record",
    "opp_record": "Opp_Record",
    "opp_home_record": "Opp_Home_Record",
    "opp_neutral_record": "Opp_Neutral_Record",
    "opp_away_record": "Opp_Away_Record",
    # ATS
    "team_ats": "Team_ATS",
    "team_home_ats": "Team_Home_ATS",
    "team_neutral_ats": "Team_Neutral_ATS",
    "team_away_ats": "Team_Away_ATS",
    "opp_ats": "Opp_ATS",
    "opp_home_ats": "Opp_Home_ATS",
    "opp_neutral_ats": "Opp_Neutral_ATS",
    "opp_away_ats": "Opp_Away_ATS",
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


def phase4_export_csv(df: pd.DataFrame) -> None:
    export = df[list(CSV_COLUMNS.keys())].copy()
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
    parser.add_argument("--backfill-odds", action="store_true",
                        help="Fetch odds for ALL dates with missing odds data")
    parser.add_argument("--backfill-boxscore", action="store_true",
                        help="Fetch boxscore stats for existing games missing them")
    parser.add_argument("--backfill-ranks", action="store_true",
                        help="Re-scan scoreboards to fill AP ranks for all games")
    args = parser.parse_args()

    print("=" * 60)
    print("  NCAAB Daily Builder -- Season 2025-26")
    print("=" * 60)
    print()

    team_ids = get_team_ids()
    tracked_ids = set(team_ids.values())
    id_to_name = {tid: name for name, tid in team_ids.items()}

    df = load_game_logs()
    print(f"[{_now_str()}] Loaded {len(df)} existing rows.\n")

    # Phase 1: new games
    df, new_dates = phase1_fetch_games(df, team_ids, tracked_ids, id_to_name)

    # Phase 2: odds
    if args.backfill_odds:
        # Find all dates where at least one row has no closing spread
        missing = df[df["closing_fg_spread"].isna()]["date"].unique()
        odds_dates = set(missing)
        print(f"[{_now_str()}] Backfill mode: {len(odds_dates)} dates "
              f"with missing odds ({len(odds_dates) * 2} API calls)")
        df = phase2_fetch_odds(df, odds_dates)
    else:
        df = phase2_fetch_odds(df, new_dates)

    # Backfill boxscore stats for existing games
    if args.backfill_boxscore:
        print(f"[{_now_str()}] Backfilling boxscore data ...")
        df = backfill_boxscore_data(df, team_ids)

    # Backfill AP ranks from scoreboards
    if args.backfill_ranks:
        print(f"[{_now_str()}] Backfilling AP ranks ...")
        df = backfill_ap_ranks(df, tracked_ids, id_to_name)

    # Save after API phases (in case Phase 3/4 crash, data is safe)
    save_game_logs(df)

    # Phase 3: PIT stats (always recomputed, no API)
    df = phase3_compute_pit(df, team_ids)

    # Phase 4: export CSV
    phase4_export_csv(df)

    # Save final with PIT columns
    save_game_logs(df)

    spread_count = df["closing_fg_spread"].notna().sum()
    boxscore_count = df["team_fgm"].notna().sum()
    print(f"\n{'=' * 60}")
    print(f"  DONE! {len(df)} total rows, {df['team'].nunique()} teams")
    print(f"  Rows with spreads:  {spread_count}/{len(df)}")
    print(f"  Rows with boxscore: {boxscore_count}/{len(df)}")
    print(f"  Parquet: {PARQUET_FILE}")
    print(f"  CSV:     {CSV_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
