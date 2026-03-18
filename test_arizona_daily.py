"""
Test: Simulate running the daily builder throughout the season for Arizona.

For each date from season start to today, this:
  1. Fetches that day's scoreboard (one API call)
  2. Checks if Arizona played
  3. If so, appends to the game log ONLY if not already stored
  4. Prints the incremental state after each game-day

This mirrors exactly how the parquet file would grow over a real season
of daily runs.
"""

import time
from datetime import datetime, timedelta

import requests

from ncaab_config import ESPN_BASE

RATE_LIMIT = 0.35


def api_get(url, params=None, tag=""):
    for attempt in range(1, 4):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                time.sleep(RATE_LIMIT)
                return resp.json()
            if resp.status_code == 429:
                time.sleep(2 ** attempt * 5)
                continue
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            if attempt < 3:
                time.sleep(2 ** attempt)
                continue
            return None
    return None


def extract_score(competitor):
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


def resolve_arizona_id():
    data = api_get(f"{ESPN_BASE}/teams", params={"limit": 500, "page": 1},
                   tag="teams-list")
    if not data:
        raise SystemExit("Could not fetch ESPN team directory")
    for entry in (data.get("sports", [{}])[0]
                      .get("leagues", [{}])[0]
                      .get("teams", [])):
        t = entry.get("team", entry)
        if t.get("location", "").lower() == "arizona":
            return str(t.get("id", ""))
    raise SystemExit("Could not find Arizona")


def main():
    arizona_id = resolve_arizona_id()
    print(f"Arizona Wildcats (ESPN id={arizona_id})\n")

    season_start = datetime(2025, 11, 3)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # This simulates the parquet file — starts empty
    game_log = []                       # accumulated rows
    stored_event_ids = set()            # tracks (event_id) already stored

    dates = []
    d = season_start
    while d <= today:
        dates.append(d)
        d += timedelta(days=1)

    print(f"Simulating daily runs across {len(dates)} dates ...\n")
    print("=" * 90)

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        date_param = date.strftime("%Y%m%d")

        # -- This is the ONE scoreboard call the daily builder makes --
        data = api_get(f"{ESPN_BASE}/scoreboard",
                       params={"dates": date_param, "limit": 400, "groups": 50},
                       tag=date_param)
        if not data:
            continue

        for ev in data.get("events", []):
            comps = ev.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            if comp.get("status", {}).get("type", {}).get("name") != "STATUS_FINAL":
                continue
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            az_comp = opp_comp = None
            for c in competitors:
                cid = str(c.get("id", c.get("team", {}).get("id", "")))
                if cid == arizona_id:
                    az_comp = c
                else:
                    opp_comp = c

            if not az_comp or not opp_comp:
                continue

            event_id = ev.get("id", "")

            # -- KEY CHECK: skip if already in our "parquet" --
            if event_id in stored_event_ids:
                print(f"  {date_str}  SKIP (event {event_id} already stored)")
                continue

            # -- New game: append to log --
            neutral = comp.get("neutralSite", False)
            ha = "neutral" if neutral else az_comp.get("homeAway", "")
            az_score = extract_score(az_comp)
            opp_score = extract_score(opp_comp)
            wl = "W" if az_score > opp_score else ("L" if az_score < opp_score else "T")
            opp_name = opp_comp.get("team", {}).get("displayName", "?")

            game_log.append({
                "date": date_str,
                "event_id": event_id,
                "opponent": opp_name,
                "home_away": ha,
                "az_score": az_score,
                "opp_score": opp_score,
                "wl": wl,
            })
            stored_event_ids.add(event_id)

            wins = sum(1 for g in game_log if g["wl"] == "W")
            losses = sum(1 for g in game_log if g["wl"] == "L")

            print(f"  {date_str}  +APPEND  {ha:<8} {opp_name:<30} "
                  f"{az_score}-{opp_score}  {wl}   "
                  f"| log: {len(game_log)} game(s), record: {wins}-{losses}")

    # ── Final state ────────────────────────────────────────────────────
    print("=" * 90)
    wins = sum(1 for g in game_log if g["wl"] == "W")
    losses = sum(1 for g in game_log if g["wl"] == "L")
    print(f"\nFinal game log ({len(game_log)} games, {wins}-{losses}):\n")

    print(f"{'#':>3}  {'Date':<12} {'Loc':<8} {'Opponent':<30} {'Score':<10} {'W/L'}")
    print("-" * 75)
    for i, g in enumerate(game_log, 1):
        print(f"{i:>3}  {g['date']:<12} {g['home_away']:<8} {g['opponent']:<30} "
              f"{g['az_score']}-{g['opp_score']:<7} {g['wl']}")


if __name__ == "__main__":
    main()
