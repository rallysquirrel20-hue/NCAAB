"""
NCAAB Odds History Refresher
==============================
Snapshots current live odds and appends to per-day parquet files in odds_history/.

Per-day files prevent git merge conflicts when syncing between multiple PCs.
Before each API call, checks if a recent snapshot already exists to avoid
duplicate calls when multiple PCs are running.

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
HISTORY_DIR = SCRIPT_DIR / "odds_history"
# Legacy single-file path (still read by backend for migration)
LEGACY_HISTORY_FILE = SCRIPT_DIR / "ncaab_odds_history.parquet"

# Minimum minutes between snapshots — skip API call if a snapshot exists
# within this window (prevents duplicate calls across PCs)
MIN_SNAPSHOT_GAP_MINUTES = 10

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


def _today_file() -> Path:
    """Return the per-day parquet path for today."""
    return HISTORY_DIR / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.parquet"


def _has_recent_snapshot(gap_minutes: int = MIN_SNAPSHOT_GAP_MINUTES) -> bool:
    """Check if today's file already has a snapshot within the last gap_minutes."""
    day_file = _today_file()
    if not day_file.exists():
        return False
    try:
        df = pd.read_parquet(day_file)
        if df.empty:
            return False
        latest = pd.to_datetime(df["snapshot_time"]).max()
        now = pd.Timestamp.now(tz="UTC")
        if latest.tzinfo is None:
            latest = latest.tz_localize("UTC")
        age = (now - latest).total_seconds() / 60
        if age < gap_minutes:
            print(f"  [{_now_str()}] Recent snapshot exists ({age:.0f}m ago, "
                  f"threshold {gap_minutes}m). Skipping API call.")
            return True
    except Exception:
        pass
    return False


def append_to_history(rows: list[dict]) -> None:
    """Append new odds rows to today's per-day parquet file."""
    HISTORY_DIR.mkdir(exist_ok=True)
    new_df = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    day_file = _today_file()

    if day_file.exists():
        try:
            existing = pd.read_parquet(day_file)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined.to_parquet(day_file, index=False, engine="pyarrow")
            print(f"  [{_now_str()}] {day_file.name}: {len(combined)} rows (+{len(rows)} new)")
        except Exception as e:
            print(f"  [{_now_str()}] Write error: {e}. Starting fresh for today.")
            new_df.to_parquet(day_file, index=False, engine="pyarrow")
    else:
        new_df.to_parquet(day_file, index=False, engine="pyarrow")
        print(f"  [{_now_str()}] {day_file.name}: Created with {len(rows)} rows")


def run_refresh():
    """Run a single odds snapshot (skips if a recent snapshot exists)."""
    print(f"[{_now_str()}] Snapshotting odds...")
    if _has_recent_snapshot():
        return
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
