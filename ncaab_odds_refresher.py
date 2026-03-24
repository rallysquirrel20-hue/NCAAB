"""
NCAAB Odds History Refresher
==============================
Snapshots current live odds and appends to ncaab_odds_history.parquet.

Usage:
    python ncaab_odds_refresher.py              # Run once
    python ncaab_odds_refresher.py --loop        # Run every 15 minutes
    python ncaab_odds_refresher.py --loop --interval 600  # Custom interval
"""

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env", override=True)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
ODDS_BASE = "https://api.the-odds-api.com"
HISTORY_FILE = SCRIPT_DIR / "ncaab_odds_history.parquet"

HISTORY_COLUMNS = [
    "snapshot_time", "game_id", "commence_time",
    "home_team", "away_team", "bookmaker",
    "last_update", "market", "outcome", "price", "point",
]


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def fetch_live_odds() -> list[dict] | None:
    """Fetch current odds from The Odds API and flatten to rows."""
    if not THE_ODDS_API_KEY:
        print(f"  [{_now_str()}] No THE_ODDS_API_KEY set. Skipping.")
        return None

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
        return None

    snapshot_time = datetime.now(timezone.utc).isoformat()
    rows = []

    for game in data:
        game_id = game.get("id", "")
        commence_time = game.get("commence_time", "")
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        for bm in game.get("bookmakers", []):
            bm_key = bm.get("key", "")
            last_update = bm.get("last_update", "")

            for mkt in bm.get("markets", []):
                market_key = mkt.get("key", "")
                for oc in mkt.get("outcomes", []):
                    rows.append({
                        "snapshot_time": snapshot_time,
                        "game_id": game_id,
                        "commence_time": commence_time,
                        "home_team": home_team,
                        "away_team": away_team,
                        "bookmaker": bm_key,
                        "last_update": last_update,
                        "market": market_key,
                        "outcome": oc.get("name", ""),
                        "price": oc.get("price"),
                        "point": oc.get("point"),
                    })

    print(f"  [{_now_str()}] Fetched {len(data)} games, {len(rows)} odds rows")
    return rows


def append_to_history(rows: list[dict]) -> None:
    """Append new odds rows to the history parquet."""
    new_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)

    if HISTORY_FILE.exists():
        try:
            existing = pd.read_parquet(HISTORY_FILE)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_parquet(HISTORY_FILE, index=False, engine="pyarrow")
            print(f"  [{_now_str()}] History: {len(combined)} total rows (+{len(rows)} new)")
        except Exception as e:
            print(f"  [{_now_str()}] History write error: {e}. Starting fresh.")
            new_df.to_parquet(HISTORY_FILE, index=False, engine="pyarrow")
    else:
        new_df.to_parquet(HISTORY_FILE, index=False, engine="pyarrow")
        print(f"  [{_now_str()}] History: Created new file with {len(rows)} rows")


def run_refresh():
    """Run a single odds snapshot."""
    print(f"[{_now_str()}] Snapshotting odds...")
    rows = fetch_live_odds()
    if rows:
        append_to_history(rows)
    elif rows is not None:
        print(f"  [{_now_str()}] No odds data returned")


def main():
    parser = argparse.ArgumentParser(description="NCAAB Odds History Refresher")
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=900,
                        help="Seconds between snapshots (default: 900 = 15 min)")
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
