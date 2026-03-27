"""
NCAAB Historical Odds Backfill
================================
Fetches historical odds snapshots at 30-minute intervals (:00 and :30)
for the full NCAAB season and appends to ncaab_odds_history.parquet.

Markets: h2h, spreads, totals
Resolution: 30 minutes

Usage:
    python ncaab_odds_backfill.py              # Run full backfill
    python ncaab_odds_backfill.py --dry-run    # Show what would be fetched
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
load_dotenv(SCRIPT_DIR.parent / ".env", override=True)
load_dotenv(SCRIPT_DIR / ".env", override=True)

THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")
ODDS_BASE = "https://api.the-odds-api.com"
HISTORY_FILE = SCRIPT_DIR / "ncaab_odds_history.parquet"

HISTORY_COLUMNS = [
    "snapshot_time", "game_id", "commence_time",
    "home_team", "away_team", "bookmaker",
    "last_update", "market", "outcome", "price", "point",
]

SEASON_START = "2025-11-03T13:00:00+00:00"
SEASON_END = "2026-04-03T03:00:00+00:00"

RATE_LIMIT_SEC = 1.0
BATCH_SIZE = 100  # Save to parquet every N snapshots


def _now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def get_existing_snapshot_slots() -> set:
    """Load existing parquet and return set of 30-min slots already covered."""
    slots: set = set()
    for f in [HISTORY_FILE, STAGING_FILE]:
        if not f.exists():
            continue
        df = pd.read_parquet(f, columns=["snapshot_time"])
        snap_times = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")
        slots.update(snap_times.dt.floor("30min").dropna().unique())
    return slots


def get_all_target_slots() -> list:
    """Generate all :00 and :30 slots for the full season."""
    slots = pd.date_range(SEASON_START, SEASON_END, freq="30min", tz="UTC")
    return sorted(slots)


def fetch_historical_snapshot(timestamp_iso: str) -> tuple[list[dict], dict]:
    """Fetch a historical odds snapshot. Returns (rows, headers_info)."""
    resp = requests.get(
        f"{ODDS_BASE}/v4/historical/sports/basketball_ncaab/odds",
        params={
            "apiKey": THE_ODDS_API_KEY,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "american",
            "date": timestamp_iso,
        },
        timeout=30,
    )
    resp.raise_for_status()

    credits_used = resp.headers.get("x-requests-last", "?")
    credits_remaining = resp.headers.get("x-requests-remaining", "?")
    headers_info = {"used": credits_used, "remaining": credits_remaining}

    data = resp.json()

    # Historical endpoint wraps in {"timestamp": ..., "data": [...]}
    if isinstance(data, dict):
        actual_timestamp = data.get("timestamp", timestamp_iso)
        games = data.get("data", [])
    else:
        actual_timestamp = timestamp_iso
        games = data

    rows = []
    for game in games:
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
                        "snapshot_time": actual_timestamp,
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

    return rows, headers_info


STAGING_FILE = SCRIPT_DIR / "ncaab_odds_backfill_staging.parquet"


def append_to_staging(all_rows: list[dict]) -> int:
    """Append rows to a staging parquet (avoids loading the full history)."""
    if not all_rows:
        return 0
    new_df = pd.DataFrame(all_rows, columns=HISTORY_COLUMNS)
    if STAGING_FILE.exists():
        existing = pd.read_parquet(STAGING_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_parquet(STAGING_FILE, index=False, engine="pyarrow")
    return len(combined)


def merge_staging_to_history() -> None:
    """Merge staging file into the main history parquet."""
    if not STAGING_FILE.exists():
        return
    staging = pd.read_parquet(STAGING_FILE)
    if staging.empty:
        STAGING_FILE.unlink()
        return
    print(f"[{_now_str()}] Merging {len(staging):,} staging rows into history...")
    if HISTORY_FILE.exists():
        existing = pd.read_parquet(HISTORY_FILE)
        combined = pd.concat([existing, staging], ignore_index=True)
    else:
        combined = staging
    combined.to_parquet(HISTORY_FILE, index=False, engine="pyarrow")
    STAGING_FILE.unlink()
    print(f"[{_now_str()}] History: {len(combined):,} total rows "
          f"({HISTORY_FILE.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="NCAAB Historical Odds Backfill")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without calling API")
    args = parser.parse_args()

    if not THE_ODDS_API_KEY:
        print("ERROR: THE_ODDS_API_KEY not set")
        return

    print(f"[{_now_str()}] Loading existing odds history...")
    existing_slots = get_existing_snapshot_slots()
    all_slots = get_all_target_slots()
    missing_slots = [s for s in all_slots if s not in existing_slots]

    print(f"[{_now_str()}] Season: {SEASON_START} to {SEASON_END}")
    print(f"  Total 30-min slots:    {len(all_slots):,}")
    print(f"  Already covered:       {len(existing_slots):,}")
    print(f"  Missing:               {len(missing_slots):,}")
    print(f"  Credits (@ 30/call):   {len(missing_slots) * 30:,}")
    print(f"  Est. runtime:          {len(missing_slots) * RATE_LIMIT_SEC / 60:.0f} min")
    print()

    if args.dry_run:
        print("[DRY RUN] No API calls made.")
        print(f"First 10 missing slots:")
        for s in missing_slots[:10]:
            print(f"  {s}")
        if len(missing_slots) > 10:
            print(f"  ... and {len(missing_slots) - 10} more")
        return

    if not missing_slots:
        print(f"[{_now_str()}] All slots covered. Nothing to do.")
        return

    total = len(missing_slots)
    total_rows = 0
    total_credits = 0
    batch_rows = []
    errors = 0

    print(f"[{_now_str()}] Starting backfill of {total:,} snapshots...")
    print()

    for i, slot in enumerate(missing_slots):
        ts_iso = slot.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            rows, info = fetch_historical_snapshot(ts_iso)
            total_rows += len(rows)
            batch_rows.extend(rows)

            credits = int(info["used"]) if info["used"] != "?" else 30
            total_credits += credits
            remaining = info["remaining"]

            pct = (i + 1) / total * 100
            print(f"  [{_now_str()}] {i+1:,}/{total:,} ({pct:.1f}%) "
                  f"{ts_iso} — {len(rows):,} rows, "
                  f"{credits} credits (remaining: {remaining})")

        except requests.exceptions.HTTPError as e:
            errors += 1
            print(f"  [{_now_str()}] {i+1:,}/{total:,} ERROR {ts_iso}: {e}")
            if e.response is not None and e.response.status_code == 429:
                print(f"  [{_now_str()}] Rate limited. Waiting 60s...")
                time.sleep(60)
            elif e.response is not None and e.response.status_code == 422:
                # No data for this timestamp, skip
                pass
            else:
                time.sleep(5)
            continue
        except Exception as e:
            errors += 1
            print(f"  [{_now_str()}] {i+1:,}/{total:,} ERROR {ts_iso}: {e}")
            time.sleep(5)
            continue

        # Save batch periodically
        if len(batch_rows) > 0 and (i + 1) % BATCH_SIZE == 0:
            print(f"  [{_now_str()}] Saving batch ({len(batch_rows):,} rows)...")
            total_in_file = append_to_staging(batch_rows)
            print(f"  [{_now_str()}] Parquet now: {total_in_file:,} total rows")
            batch_rows = []

        time.sleep(RATE_LIMIT_SEC)

    # Save remaining rows
    if batch_rows:
        print(f"  [{_now_str()}] Saving final batch ({len(batch_rows):,} rows)...")
        total_in_file = append_to_history(batch_rows)
        print(f"  [{_now_str()}] Parquet now: {total_in_file:,} total rows")

    print()
    print(f"[{_now_str()}] Backfill fetching complete.")
    print(f"  Snapshots fetched:  {total - errors:,}")
    print(f"  Errors:             {errors}")
    print(f"  Total odds rows:    {total_rows:,}")
    print(f"  Credits used:       {total_credits:,}")

    # Merge staging into main history
    merge_staging_to_history()

    print(f"  File: {HISTORY_FILE}")
    print(f"  Size: {HISTORY_FILE.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
