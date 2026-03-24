"""
NCAAB Schedule Refresher
=========================
Fetches today's games from ESPN + current odds from The Odds API.
Writes ncaab_today.json for the web app backend to serve.

Usage:
    python ncaab_schedule_refresher.py              # Run once
    python ncaab_schedule_refresher.py --loop        # Run every 5 minutes
    python ncaab_schedule_refresher.py --loop --interval 120  # Custom interval (seconds)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env", override=True)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
ODDS_BASE = "https://api.the-odds-api.com"
OUTPUT_FILE = SCRIPT_DIR / "ncaab_today.json"
GAME_LOGS_FILE = SCRIPT_DIR / "ncaab_game_logs.parquet"

# Import config
sys.path.insert(0, str(SCRIPT_DIR))
from ncaab_config import TEAMS, CONFERENCE_MAP, ESPN_NAME_ALIASES, ODDS_NAME_MAP, HARDCODED_ESPN_IDS


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def load_team_ids() -> dict[str, str]:
    cache = SCRIPT_DIR / "espn_team_ids.json"
    if cache.exists():
        with open(cache) as f:
            return json.load(f)
    return {}


def get_latest_pit_stats() -> dict[str, dict]:
    """Load latest PIT stats for each tracked team from game_logs."""
    if not GAME_LOGS_FILE.exists():
        return {}
    df = pd.read_parquet(GAME_LOGS_FILE)
    tracked = df[df["is_tracked"] == True].copy()
    if tracked.empty:
        return {}
    tracked.sort_values("date", inplace=True)
    latest = tracked.groupby("team").last()

    pit_stats = {}
    for team, row in latest.iterrows():
        stats = {
            "record": f"{int(row.get('team_games', 0) * row.get('team_win_pct', 0)) if pd.notna(row.get('team_win_pct')) else 0}-{int(row.get('team_games', 0) * (1 - row.get('team_win_pct', 0))) if pd.notna(row.get('team_win_pct')) else 0}" if pd.notna(row.get('team_games')) and row.get('team_games', 0) > 0 else "0-0",
            "conference": CONFERENCE_MAP.get(team, ""),
        }
        # Compute W-L properly from the running totals
        games = row.get("team_games", 0)
        win_pct = row.get("team_win_pct", 0)
        if pd.notna(games) and pd.notna(win_pct) and games > 0:
            # These are pre-game stats for the LAST game, so add 1 game
            total_games = int(games) + 1
            last_wl = row.get("win_loss", "")
            wins = round(games * win_pct)
            losses = int(games) - wins
            if last_wl == "W":
                wins += 1
            else:
                losses += 1
            stats["record"] = f"{wins}-{losses}"
            stats["games"] = total_games
        else:
            stats["record"] = "0-0"
            stats["games"] = 0

        # ATS
        ats_games = row.get("team_ats_games", 0)
        ats_pct = row.get("team_ats_win_pct", 0)
        if pd.notna(ats_games) and pd.notna(ats_pct) and ats_games > 0:
            ats_w = round(ats_games * ats_pct)
            ats_l = int(ats_games) - ats_w
            stats["ats"] = f"{ats_w}-{ats_l}"
        else:
            stats["ats"] = "0-0"

        # Key PIT stats
        for key in ["team_ppg", "team_win_pct", "team_ats_win_pct",
                     "team_ft_pct", "team_3pt_pct", "team_2pt_pct",
                     "team_def_3pt_pct", "team_def_2pt_pct",
                     "team_oreb_pg", "team_dreb_pg",
                     "team_to_pg", "team_forced_to_pg",
                     "team_pace", "team_sos",
                     "opp_ppg"]:
            val = row.get(key)
            if pd.notna(val) and val != "":
                stats[key] = round(float(val), 1) if isinstance(val, (int, float)) else val
        pit_stats[team] = stats
    return pit_stats


LOOKAHEAD_DAYS = 7


def fetch_schedule(date_param: str) -> list[dict]:
    """Fetch games for a single date (YYYYMMDD) from ESPN scoreboard."""
    try:
        resp = requests.get(
            f"{ESPN_BASE}/scoreboard",
            params={"dates": date_param, "limit": 400, "groups": 50},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("events", [])
    except Exception as e:
        print(f"  [{_now_str()}] ESPN scoreboard error for {date_param}: {e}")
        return []


def fetch_todays_schedule() -> list[dict]:
    """Fetch games from today through LOOKAHEAD_DAYS days ahead."""
    today = datetime.now()
    all_events = []
    dates_checked = []
    event_espn_date: dict[str, str] = {}
    for offset in range(LOOKAHEAD_DAYS + 1):
        d = today + timedelta(days=offset)
        date_param = d.strftime("%Y%m%d")
        date_str = d.strftime("%Y-%m-%d")
        dates_checked.append(date_str)
        events = fetch_schedule(date_param)
        for ev in events:
            eid = ev.get("id", "")
            if eid not in event_espn_date:
                event_espn_date[eid] = date_str
        all_events.extend(events)

    print(f"  [{_now_str()}] Checked {len(dates_checked)} dates: {dates_checked[0]} to {dates_checked[-1]}")

    team_ids = load_team_ids()
    id_to_name = {tid: name for name, tid in team_ids.items()}
    tracked_ids = set(team_ids.values())

    games = []
    seen_events = set()

    for ev in all_events:
        event_id = ev.get("id", "")
        if event_id in seen_events:
            continue

        comps = ev.get("competitions", [])
        if not comps:
            continue
        comp = comps[0]

        status_type = comp.get("status", {}).get("type", {})
        status_name = status_type.get("name", "")
        status_detail = comp.get("status", {}).get("type", {}).get("shortDetail", "")

        competitors = comp.get("competitors", [])
        if len(competitors) != 2:
            continue

        # Check if at least one team is tracked
        home_c = away_c = None
        for c in competitors:
            if c.get("homeAway") == "home":
                home_c = c
            else:
                away_c = c
        if not home_c or not away_c:
            continue

        home_id = str(home_c.get("id", home_c.get("team", {}).get("id", "")))
        away_id = str(away_c.get("id", away_c.get("team", {}).get("id", "")))
        home_tracked = home_id in tracked_ids
        away_tracked = away_id in tracked_ids

        if not home_tracked and not away_tracked:
            continue

        seen_events.add(event_id)

        neutral_site = comp.get("neutralSite", False)
        commence_time = comp.get("date", "")

        home_team_info = home_c.get("team", {})
        away_team_info = away_c.get("team", {})

        home_name = id_to_name.get(home_id, home_team_info.get("displayName", "Unknown"))
        away_name = id_to_name.get(away_id, away_team_info.get("displayName", "Unknown"))

        game = {
            "game_id": event_id,
            "commence_time": commence_time,
            "game_date": event_espn_date.get(event_id, commence_time[:10]),
            "status": status_name,
            "status_detail": status_detail,
            "neutral_site": neutral_site,
            "home": {
                "name": home_name,
                "display_name": home_team_info.get("displayName", home_name),
                "abbreviation": home_team_info.get("abbreviation", ""),
                "logo": home_team_info.get("logo", ""),
                "id": home_id,
                "score": None,
                "is_tracked": home_tracked,
            },
            "away": {
                "name": away_name,
                "display_name": away_team_info.get("displayName", away_name),
                "abbreviation": away_team_info.get("abbreviation", ""),
                "logo": away_team_info.get("logo", ""),
                "id": away_id,
                "score": None,
                "is_tracked": away_tracked,
            },
            "odds": None,
        }

        # Add scores if game is in progress or final
        if status_name in ("STATUS_IN_PROGRESS", "STATUS_FINAL", "STATUS_HALFTIME",
                           "STATUS_END_PERIOD"):
            try:
                game["home"]["score"] = int(float(home_c.get("score", 0)))
                game["away"]["score"] = int(float(away_c.get("score", 0)))
            except (ValueError, TypeError):
                pass

        games.append(game)

    return games


def fetch_current_odds() -> dict:
    """Fetch current live odds from The Odds API. Returns {game_key: odds_data}."""
    if not THE_ODDS_API_KEY:
        return {}
    try:
        resp = requests.get(
            f"{ODDS_BASE}/v4/sports/basketball_ncaab/odds",
            params={
                "apiKey": THE_ODDS_API_KEY,
                "regions": "us",
                "markets": "h2h,spreads,totals",
                "oddsFormat": "american",
            },
            timeout=30,
        )
        resp.raise_for_status()
        remaining = resp.headers.get("x-requests-remaining", "?")
        used = resp.headers.get("x-requests-last", "?")
        print(f"  [{_now_str()}] Odds API: {used} credits used, {remaining} remaining")
        data = resp.json()
    except Exception as e:
        print(f"  [{_now_str()}] Odds API error: {e}")
        return {}

    # Index by (home_team_lower, away_team_lower) for matching
    odds_by_teams = {}
    for game in data:
        home = game.get("home_team", "").lower()
        away = game.get("away_team", "").lower()
        odds_by_teams[(home, away)] = game
        # Also index by odds API game id
        odds_by_teams[game.get("id", "")] = game
    return odds_by_teams


def _find_odds_name(team_name: str) -> str:
    """Get the Odds API name for a team."""
    return ODDS_NAME_MAP.get(team_name, team_name)


def _extract_bookmaker_odds(bookmakers: list[dict], team_name_lower: str) -> dict:
    """Extract structured odds from bookmaker data."""
    result = {"spread": None, "ml": None, "total": None, "bookmakers": []}

    for bm in bookmakers:
        bm_data = {"name": bm.get("title", bm.get("key", "")), "markets": {}}
        for mkt in bm.get("markets", []):
            key = mkt.get("key", "")
            for oc in mkt.get("outcomes", []):
                if key == "spreads" and oc.get("name", "").lower() == team_name_lower:
                    bm_data["markets"]["spread"] = {
                        "point": oc.get("point"),
                        "price": oc.get("price"),
                    }
                    if result["spread"] is None:
                        result["spread"] = oc.get("point")
                elif key == "h2h" and oc.get("name", "").lower() == team_name_lower:
                    bm_data["markets"]["ml"] = oc.get("price")
                    if result["ml"] is None:
                        result["ml"] = oc.get("price")
                elif key == "totals" and oc.get("name") == "Over":
                    bm_data["markets"]["total"] = oc.get("point")
                    if result["total"] is None:
                        result["total"] = oc.get("point")
        if bm_data["markets"]:
            result["bookmakers"].append(bm_data)
    return result


def match_odds_to_games(games: list[dict], odds_data: dict) -> None:
    """Match odds data to games in place."""
    for game in games:
        home_odds_name = _find_odds_name(game["home"]["name"]).lower()
        away_odds_name = _find_odds_name(game["away"]["name"]).lower()

        # Try to find by team names
        matched = odds_data.get((home_odds_name, away_odds_name))
        if not matched:
            # Try reverse lookup through all odds
            for key, od in odds_data.items():
                if not isinstance(key, tuple):
                    continue
                if home_odds_name in key[0] or key[0] in home_odds_name:
                    if away_odds_name in key[1] or key[1] in away_odds_name:
                        matched = od
                        break

        if matched:
            bm = matched.get("bookmakers", [])
            game["odds"] = _extract_bookmaker_odds(bm, home_odds_name)
            # Also add away odds
            away_odds = _extract_bookmaker_odds(bm, away_odds_name)
            game["odds"]["away_spread"] = away_odds["spread"]
            game["odds"]["away_ml"] = away_odds["ml"]


def run_refresh():
    """Run a single refresh cycle."""
    print(f"[{_now_str()}] Refreshing schedule...")

    games = fetch_todays_schedule()
    print(f"  [{_now_str()}] Found {len(games)} games (today + {LOOKAHEAD_DAYS} days)")

    if games:
        odds_data = fetch_current_odds()
        if odds_data:
            match_odds_to_games(games, odds_data)
            matched = sum(1 for g in games if g["odds"] is not None)
            print(f"  [{_now_str()}] Matched odds for {matched}/{len(games)} games")

    # Load PIT stats and attach
    pit_stats = get_latest_pit_stats()
    for game in games:
        for side in ("home", "away"):
            name = game[side]["name"]
            if name in pit_stats:
                game[side]["stats"] = pit_stats[name]

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "game_count": len(games),
        "games": games,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"  [{_now_str()}] Wrote {OUTPUT_FILE.name} ({len(games)} games)")
    return output


def main():
    parser = argparse.ArgumentParser(description="NCAAB Schedule Refresher")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=300,
                        help="Seconds between refreshes (default: 300 = 5 min)")
    args = parser.parse_args()

    run_refresh()

    if args.loop:
        print(f"\n[{_now_str()}] Looping every {args.interval}s. Ctrl+C to stop.")
        while True:
            time.sleep(args.interval)
            try:
                run_refresh()
            except Exception as e:
                print(f"  [{_now_str()}] Error: {e}")


if __name__ == "__main__":
    main()
