"""
NCAAB Dashboard — FastAPI Backend
==================================
Serves game data, strategy analytics, and backtesting for the NCAAB web app.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aiohttp
import uvicorn

# ── paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent
PARQUET_FILE = REPO_DIR / "ncaab_game_logs.parquet"
ODDS_HISTORY_FILE = REPO_DIR / "ncaab_odds_history.parquet"

# Add repo dir so we can import config
sys.path.insert(0, str(REPO_DIR))
from ncaab_config import TEAMS, CONFERENCE_MAP, ODDS_NAME_MAP  # noqa: E402

# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(title="NCAAB Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── data globals ───────────────────────────────────────────────────────────
DF: pd.DataFrame = pd.DataFrame()
GAMES_DF: pd.DataFrame = pd.DataFrame()  # one row per game, from favorite's perspective
ODDS_DF: pd.DataFrame = pd.DataFrame()
TEAM_STATS_TIMELINE: dict = {}  # team -> sorted list of (date, {stat: value})

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

# Approximate 2025-26 date boundaries
SEASON_START = "2025-11-03"
CONF_TOURNEY_START = "2026-03-08"
NCAA_START = "2026-03-17"

JUICE = -110

# PIT stat columns available for backtesting
PIT_STATS = [
    "win_pct", "home_win_pct", "neutral_win_pct", "away_win_pct",
    "ats_win_pct", "home_ats_win_pct", "neutral_ats_win_pct", "away_ats_win_pct",
    "ppg", "home_ppg", "neutral_ppg", "away_ppg",
    "ft_pct", "ftm_pg", "fta_pg",
    "3pt_pct", "3pm_pg", "3pa_pg",
    "2pt_pct", "2pm_pg", "2pa_pg",
    "def_ft_pct", "def_ftm_pg", "def_fta_pg",
    "def_3pt_pct", "def_3pm_pg", "def_3pa_pg",
    "def_2pt_pct", "def_2pm_pg", "def_2pa_pg",
    "oreb_pg", "dreb_pg", "to_pg", "forced_to_pg",
    "pace", "sos",
]


# ── data loading ───────────────────────────────────────────────────────────

def load_data():
    global DF, GAMES_DF, ODDS_DF, TEAM_STATS_TIMELINE

    print("Loading game logs...")
    DF = pd.read_parquet(PARQUET_FILE)
    DF["date"] = DF["date"].astype(str)
    # Ensure numeric types
    for col in ["team_score", "opp_score", "closing_fg_spread", "opening_fg_spread",
                 "closing_fg_ml", "opening_fg_ml"]:
        if col in DF.columns:
            DF[col] = pd.to_numeric(DF[col], errors="coerce")

    # Compute ATS fields
    mask = DF["closing_fg_spread"].notna()
    DF.loc[mask, "margin"] = DF.loc[mask, "team_score"] - DF.loc[mask, "opp_score"]
    DF.loc[mask, "ats_margin"] = DF.loc[mask, "margin"] + DF.loc[mask, "closing_fg_spread"]
    DF["covered"] = False
    DF.loc[mask, "covered"] = DF.loc[mask, "ats_margin"] > 0
    DF["push"] = False
    DF.loc[mask, "push"] = DF.loc[mask, "ats_margin"] == 0
    print(f"  Loaded {len(DF)} game rows ({DF['date'].nunique()} dates)")

    # Build team stats timeline for rankings
    print("Building team stats timeline...")
    tracked = DF[DF["is_tracked"] == True].sort_values("date")
    TEAM_STATS_TIMELINE = {}
    for team in TEAMS:
        team_rows = tracked[tracked["team"] == team]
        timeline = []
        for _, row in team_rows.iterrows():
            stats = {}
            for s in PIT_STATS:
                col = f"team_{s}"
                if col in row.index and pd.notna(row[col]):
                    stats[s] = float(row[col])
            if stats:
                timeline.append((row["date"], stats))
        TEAM_STATS_TIMELINE[team] = timeline

    # Load odds history
    if ODDS_HISTORY_FILE.exists():
        print("Loading odds history...")
        ODDS_DF = pd.read_parquet(ODDS_HISTORY_FILE)
        ODDS_DF["snapshot_time"] = pd.to_datetime(ODDS_DF["snapshot_time"])
        print(f"  Loaded {len(ODDS_DF)} odds snapshots")

        # Precompute closing spread prices (actual juice per game)
        print("Computing closing spread prices...")
        _attach_closing_spread_prices(DF, ODDS_DF)
    else:
        print("  No odds history file found")
        DF["closing_spread_price"] = np.nan

    # Build GAMES_DF: one row per game, deduplicated
    # For each event_id with a spread, keep the favorite's row (spread < 0).
    # For pick'em (spread == 0), keep the home team's row.
    # Tag tournament rounds before building GAMES_DF
    print("Tagging tournament rounds...")
    _tag_tournament_rounds(DF)

    # This gives us exactly 1 row per game for strategy/equity calculations.
    print("Building deduplicated games table...")
    has_spread = DF[
        (DF["is_tracked"] == True) &
        DF["closing_fg_spread"].notna() &
        DF["event_id"].notna()
    ].copy()

    # For events with 2 tracked rows, keep the favorite
    # For events with 1 tracked row, keep it (it may be fav or dog)
    rows_to_keep = []
    for eid, group in has_spread.groupby("event_id"):
        if len(group) == 1:
            rows_to_keep.append(group.index[0])
        else:
            # Pick the favorite (most negative spread)
            fav_idx = group["closing_fg_spread"].idxmin()
            rows_to_keep.append(fav_idx)

    GAMES_DF = has_spread.loc[rows_to_keep].copy()

    # Add underdog spread price: for games between 2 tracked teams, look up
    # the other row's price. For single-row games, we need the opponent price
    # from odds history (already done if both sides were matched).
    GAMES_DF["dog_spread_price"] = np.nan
    for idx in GAMES_DF.index:
        eid = GAMES_DF.at[idx, "event_id"]
        team = GAMES_DF.at[idx, "team"]
        other = has_spread[(has_spread["event_id"] == eid) & (has_spread["team"] != team)]
        if len(other) > 0:
            opp_price = other.iloc[0].get("closing_spread_price")
            if pd.notna(opp_price):
                GAMES_DF.at[idx, "dog_spread_price"] = float(opp_price)

    # For single-row games (opponent not tracked), try to get dog price
    # from odds history via the opponent's side of the same odds game
    _fill_opponent_spread_prices(GAMES_DF, ODDS_DF)

    # Normalize: ensure every row is from the FAVORITE's perspective.
    # For rows where tracked team is the underdog (spread > 0), flip:
    #   - fav_spread = -closing_fg_spread (make it negative)
    #   - fav_covered = ~covered (favorite covered when underdog didn't)
    #   - swap fav/dog prices
    dog_rows = GAMES_DF["closing_fg_spread"] > 0
    n_flipped = dog_rows.sum()
    if n_flipped > 0:
        # Save original values for the flip
        orig_spread = GAMES_DF.loc[dog_rows, "closing_fg_spread"].copy()
        orig_covered = GAMES_DF.loc[dog_rows, "covered"].copy()
        orig_price = GAMES_DF.loc[dog_rows, "closing_spread_price"].copy()
        orig_dog_price = GAMES_DF.loc[dog_rows, "dog_spread_price"].copy()
        orig_ats = GAMES_DF.loc[dog_rows, "ats_margin"].copy()

        # Flip spread to negative (favorite's perspective)
        GAMES_DF.loc[dog_rows, "closing_fg_spread"] = -orig_spread
        # Favorite covered = underdog didn't cover
        GAMES_DF.loc[dog_rows, "covered"] = ~orig_covered
        # Swap prices: tracked team's price becomes dog price, opponent's becomes fav price
        GAMES_DF.loc[dog_rows, "closing_spread_price"] = orig_dog_price
        GAMES_DF.loc[dog_rows, "dog_spread_price"] = orig_price
        # Flip ATS margin sign (favorite's margin is opposite of underdog's)
        GAMES_DF.loc[dog_rows, "ats_margin"] = -orig_ats
        # Swap team/opponent names so "team" is always the favorite
        orig_team = GAMES_DF.loc[dog_rows, "team"].copy()
        orig_opp = GAMES_DF.loc[dog_rows, "opponent"].copy()
        orig_tscore = GAMES_DF.loc[dog_rows, "team_score"].copy()
        orig_oscore = GAMES_DF.loc[dog_rows, "opp_score"].copy()
        GAMES_DF.loc[dog_rows, "team"] = orig_opp
        GAMES_DF.loc[dog_rows, "opponent"] = orig_team
        GAMES_DF.loc[dog_rows, "team_score"] = orig_oscore
        GAMES_DF.loc[dog_rows, "opp_score"] = orig_tscore
        # Flip home/away (if tracked team was away underdog, favorite is home)
        ha = GAMES_DF.loc[dog_rows, "home_away"].copy()
        GAMES_DF.loc[dog_rows, "home_away"] = ha.map(
            {"home": "away", "away": "home", "neutral": "neutral"})

    # Precompute underdog covered (opposite of favorite covered, excluding pushes)
    GAMES_DF["dog_covered"] = False
    no_push = ~GAMES_DF["push"]
    GAMES_DF.loc[no_push, "dog_covered"] = ~GAMES_DF.loc[no_push, "covered"]

    n_games = len(GAMES_DF)
    n_fav = (GAMES_DF["closing_fg_spread"] < 0).sum()
    print(f"  Flipped {n_flipped} rows to favorite perspective")
    n_dog_prices = GAMES_DF["dog_spread_price"].notna().sum()
    print(f"  {n_games} unique games, {n_fav} with favorite row, {n_dog_prices} with dog price")

    print("Data loaded.")


def _attach_closing_spread_prices(gl: pd.DataFrame, odds_df: pd.DataFrame):
    """Join closing spread prices from odds history onto game logs.

    For each tracked game with a closing spread, find the matching odds-API
    game via (team, commence_time) and pull the last-snapshot spread price
    whose point matches closing_fg_spread.  Falls back to the last snapshot
    for that team regardless of point.
    """
    reverse_name = {v: k for k, v in ODDS_NAME_MAP.items()}

    spreads = odds_df[odds_df["market"] == "spreads"].copy()
    spreads["commence_time"] = pd.to_datetime(spreads["commence_time"])
    spreads["snapshot_time"] = pd.to_datetime(spreads["snapshot_time"])
    spreads["canonical"] = spreads["outcome"].map(reverse_name)

    # Build odds game directory: game_id -> (commence_time, home_canonical, away_canonical)
    games_dir = (
        spreads.groupby("game_id")
        .first()[["commence_time", "home_team", "away_team"]]
        .reset_index()
    )
    games_dir["home_canonical"] = games_dir["home_team"].map(reverse_name)
    games_dir["away_canonical"] = games_dir["away_team"].map(reverse_name)

    # Index for fast lookup: canonical team -> list of (game_id, commence_time)
    team_games: dict[str, list[tuple[str, pd.Timestamp]]] = {}
    for _, g in games_dir.iterrows():
        for canon in (g["home_canonical"], g["away_canonical"]):
            if pd.notna(canon):
                team_games.setdefault(canon, []).append(
                    (g["game_id"], g["commence_time"])
                )

    # Pre-group spreads by (game_id, canonical) for fast lookup
    spread_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    for (gid, canon), grp in spreads[spreads["canonical"].notna()].groupby(
        ["game_id", "canonical"]
    ):
        spread_lookup[(gid, canon)] = grp.sort_values("snapshot_time")

    # Walk tracked rows with a spread
    gl["closing_spread_price"] = np.nan
    gl["commence_time_utc"] = pd.to_datetime(gl["commence_time_utc"], errors="coerce")

    need = gl.index[
        (gl["is_tracked"] == True)
        & gl["closing_fg_spread"].notna()
        & gl["commence_time_utc"].notna()
    ]

    matched = 0
    for idx in need:
        team = gl.at[idx, "team"]
        ct = gl.at[idx, "commence_time_utc"]
        target_spread = gl.at[idx, "closing_fg_spread"]

        candidates = team_games.get(team, [])
        gid = None
        for cand_gid, cand_ct in candidates:
            if abs((cand_ct - ct).total_seconds()) < 7200:  # within 2 hours
                gid = cand_gid
                break
        if gid is None:
            continue

        team_sp = spread_lookup.get((gid, team))
        if team_sp is None or len(team_sp) == 0:
            continue

        # Prefer rows matching the exact spread point
        exact = team_sp[team_sp["point"] == target_spread]
        if len(exact) > 0:
            gl.at[idx, "closing_spread_price"] = float(exact.iloc[-1]["price"])
        else:
            gl.at[idx, "closing_spread_price"] = float(team_sp.iloc[-1]["price"])
        matched += 1

    print(f"  Matched closing spread prices for {matched}/{len(need)} games")


def _fill_opponent_spread_prices(games_df: pd.DataFrame, odds_df: pd.DataFrame):
    """For games without a dog_spread_price, try to find the opponent's
    closing spread price from the odds history."""
    if odds_df.empty:
        return

    reverse_name = {v: k for k, v in ODDS_NAME_MAP.items()}
    spreads = odds_df[odds_df["market"] == "spreads"].copy()
    spreads["commence_time"] = pd.to_datetime(spreads["commence_time"])
    spreads["snapshot_time"] = pd.to_datetime(spreads["snapshot_time"])

    # Build game directory
    games_dir = (
        spreads.groupby("game_id")
        .first()[["commence_time", "home_team", "away_team"]]
        .reset_index()
    )
    games_dir["home_canonical"] = games_dir["home_team"].map(reverse_name)
    games_dir["away_canonical"] = games_dir["away_team"].map(reverse_name)

    # Index: canonical team -> list of (game_id, commence_time)
    team_games: dict[str, list[tuple[str, pd.Timestamp]]] = {}
    for _, g in games_dir.iterrows():
        for canon in (g["home_canonical"], g["away_canonical"]):
            if pd.notna(canon):
                team_games.setdefault(canon, []).append(
                    (g["game_id"], g["commence_time"])
                )

    # Pre-group spreads by game_id for lookup
    spread_by_game: dict[str, pd.DataFrame] = {}
    for gid, grp in spreads.groupby("game_id"):
        spread_by_game[gid] = grp.sort_values("snapshot_time")

    need = games_df.index[
        games_df["dog_spread_price"].isna() &
        games_df["commence_time_utc"].notna()
    ]

    filled = 0
    for idx in need:
        team = games_df.at[idx, "team"]
        ct = games_df.at[idx, "commence_time_utc"]
        fav_spread = games_df.at[idx, "closing_fg_spread"]
        dog_spread = -fav_spread  # opponent has opposite spread

        # Find the odds game
        candidates = team_games.get(team, [])
        gid = None
        for cand_gid, cand_ct in candidates:
            if abs((cand_ct - ct).total_seconds()) < 7200:
                gid = cand_gid
                break
        if gid is None:
            continue

        game_spreads = spread_by_game.get(gid)
        if game_spreads is None:
            continue

        # Find opponent's outcome rows (not the team's canonical name)
        opp_rows = game_spreads[game_spreads["outcome"].map(
            lambda x: reverse_name.get(x, x)
        ) != team]

        if len(opp_rows) == 0:
            continue

        # Prefer exact match on dog spread point
        exact = opp_rows[opp_rows["point"] == dog_spread]
        if len(exact) > 0:
            games_df.at[idx, "dog_spread_price"] = float(exact.iloc[-1]["price"])
        else:
            games_df.at[idx, "dog_spread_price"] = float(opp_rows.iloc[-1]["price"])
        filled += 1

    if filled > 0:
        print(f"  Filled {filled} additional opponent spread prices")


def get_team_rankings(as_of_date: str) -> dict[str, dict[str, float]]:
    """Compute percentile rankings for all teams as of a given date.
    Returns {team: {stat: percentile_rank (0-100)}}."""
    latest_stats: dict[str, dict] = {}
    for team, timeline in TEAM_STATS_TIMELINE.items():
        for date, stats in timeline:
            if date <= as_of_date:
                latest_stats[team] = stats
            else:
                break

    if len(latest_stats) < 10:
        return {}

    rankings: dict[str, dict[str, float]] = {}
    stat_names = set()
    for stats in latest_stats.values():
        stat_names.update(stats.keys())

    for stat in stat_names:
        values = [(t, s[stat]) for t, s in latest_stats.items() if stat in s]
        if len(values) < 10:
            continue
        values.sort(key=lambda x: x[1])
        n = len(values)
        for i, (team, _) in enumerate(values):
            pct = (i / (n - 1)) * 100 if n > 1 else 50.0
            if team not in rankings:
                rankings[team] = {}
            rankings[team][stat] = pct

    return rankings


# ── timeframe helpers ──────────────────────────────────────────────────────

def _tag_tournament_rounds(df: pd.DataFrame) -> None:
    """Tag each game with its tournament round based on game sequence.

    For each team in the 68-team field, their tournament games (date >= NCAA_START)
    are numbered in chronological order:
      Game 1 for First Four teams = "first_four"
      Game 1 for non-First-Four teams (or game 2 for FF winners) = "round_of_64"
      Next game = "round_of_32", then "sweet_16", "elite_8", "final_four", "championship"
    """
    df["tournament_round"] = ""

    tourney = df[
        (df["date"] >= NCAA_START) &
        (df["is_tracked"] == True)
    ].copy()

    if tourney.empty:
        return

    # Identify First Four events: earliest 2 tournament dates
    first_dates = sorted(tourney["date"].unique())[:2]
    ff_events = set(tourney[tourney["date"].isin(first_dates)]["event_id"].unique())
    ff_teams = set(tourney[tourney["event_id"].isin(ff_events)]["team"].unique())

    # Tag First Four
    for idx in tourney.index:
        if tourney.at[idx, "event_id"] in ff_events:
            df.at[idx, "tournament_round"] = "first_four"

    # For all other tournament games, determine round by the event, not
    # per-team sequence.  Use a non-First-Four team in each game to
    # determine the round (their game number = the true round number).
    round_sequence = ["round_of_64", "round_of_32", "sweet_16",
                      "elite_8", "final_four", "championship"]

    non_ff = tourney[~tourney["event_id"].isin(ff_events)]

    # For each event, find a team that did NOT play in the First Four.
    # Their tournament game number (1st, 2nd, 3rd...) maps to the round.
    event_round: dict[str, str] = {}
    # First, build each non-FF team's game sequence
    team_game_num: dict[str, dict[str, int]] = {}  # team -> {event_id: game_number}
    for team in non_ff["team"].unique():
        if team in ff_teams:
            # FF team: their non-FF games are numbered starting from 1
            team_tourney = non_ff[non_ff["team"] == team].sort_values("date")
        else:
            team_tourney = non_ff[non_ff["team"] == team].sort_values("date")
        for i, (idx, row) in enumerate(team_tourney.iterrows()):
            team_game_num.setdefault(team, {})[row["event_id"]] = i

    # Assign round to each event
    for eid in non_ff["event_id"].unique():
        ev_rows = non_ff[non_ff["event_id"] == eid]
        # Prefer a non-FF team's game number
        game_num = None
        for _, row in ev_rows.iterrows():
            t = row["team"]
            if t not in ff_teams and t in team_game_num:
                game_num = team_game_num[t].get(eid, 0)
                break
        if game_num is None:
            # All teams are FF teams — use any team's number
            for _, row in ev_rows.iterrows():
                t = row["team"]
                if t in team_game_num:
                    game_num = team_game_num[t].get(eid, 0)
                    break
        if game_num is not None and game_num < len(round_sequence):
            event_round[eid] = round_sequence[game_num]

    # Apply round tags to DF
    for idx in non_ff.index:
        eid = non_ff.at[idx, "event_id"]
        if eid in event_round:
            df.at[idx, "tournament_round"] = event_round[eid]


def filter_by_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "full_season":
        return df
    elif timeframe == "regular_season":
        return df[df["date"] < CONF_TOURNEY_START]
    elif timeframe == "conf_tournaments":
        return df[(df["date"] >= CONF_TOURNEY_START) & (df["date"] < NCAA_START)]
    elif timeframe == "ncaa_tournament":
        return df[df["date"] >= NCAA_START]
    elif timeframe in ("first_four", "round_of_64", "round_of_32", "sweet_16",
                       "elite_8", "final_four", "championship"):
        return df[df["tournament_round"] == timeframe]
    return df


# ── strategy computations ─────────────────────────────────────────────────

def compute_strategy_stats(games: pd.DataFrame, name: str,
                           side: str = "favorite") -> dict:
    """Compute ATS stats for a set of games using actual closing spread prices.

    side: "favorite" uses the favorite's covered/price,
          "underdog" uses the underdog's covered/price,
          "home"/"away"/"neutral" use home_away + favorite's perspective.
    """
    testable = games[~games["push"]].copy()
    n = len(testable)
    if n == 0:
        return {"name": name, "n": 0, "wins": 0, "losses": 0,
                "win_pct": 0, "roi": 0, "profitable": False}

    if side == "underdog":
        wins = int(testable["dog_covered"].sum())
        price_col = "dog_spread_price"
    else:
        wins = int(testable["covered"].sum())
        price_col = "closing_spread_price"

    losses = n - wins
    win_pct = wins / n

    # ROI using actual closing spread prices
    covered_col = "dog_covered" if side == "underdog" else "covered"
    total_profit = 0.0
    for _, row in testable.iterrows():
        price = row.get(price_col)
        if pd.isna(price):
            price = JUICE
        if row[covered_col]:
            if price < 0:
                total_profit += 100 / abs(price)
            else:
                total_profit += price / 100
        else:
            total_profit -= 1.0
    roi = total_profit / n * 100

    return {
        "name": name, "n": n, "wins": wins, "losses": losses,
        "win_pct": round(win_pct, 4), "roi": round(roi, 2),
        "units": round(total_profit, 2),
        "profitable": win_pct > 0.524,
    }


def compute_equity_curve(games: pd.DataFrame,
                         side: str = "favorite") -> list[dict]:
    """Compute cumulative profit curve using actual closing spread prices.

    side: "favorite" bets the favorite in every game,
          "underdog" bets the underdog in every game.

    Bet sizing: risk 1 unit on every game.
      - If price is negative (e.g. -115): risk 1 unit, win 100/115 units
      - If price is positive (e.g. +105): risk 1 unit, win 105/100 units
    """
    testable = games[~games["push"]].sort_values("date")

    if len(testable) == 0:
        return []

    covered_col = "dog_covered" if side == "underdog" else "covered"
    price_col = "dog_spread_price" if side == "underdog" else "closing_spread_price"

    curve = []
    cum_profit = 0.0
    for _, row in testable.iterrows():
        price = row.get(price_col)
        if pd.isna(price):
            price = JUICE

        if row[covered_col]:
            if price < 0:
                cum_profit += 100 / abs(price)
            else:
                cum_profit += price / 100
        else:
            cum_profit -= 1.0

        curve.append({"date": row["date"], "profit": round(cum_profit, 3),
                       "team": row["team"], "opponent": row.get("opponent", ""),
                       "price": int(price)})
    return curve


# ── pydantic models ────────────────────────────────────────────────────────

class BacktestFilter(BaseModel):
    stat: str
    compare: str  # "gt", "lt", "gt_opp", "lt_opp", "gt_opp_margin", "lt_opp_margin", "rank_gt", "rank_lt"
    value: Optional[float] = None
    opp_stat: Optional[str] = None  # for cross-stat comparison


class BacktestRequest(BaseModel):
    filters: list[BacktestFilter]
    side: str = "all"  # "all", "favorites", "underdogs", "home", "away"
    min_games: int = 0


# ── endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/dates")
def get_dates():
    """Return all dates that have game data."""
    dates = sorted(DF["date"].unique().tolist())
    return {"dates": dates}


@app.get("/api/teams")
def get_teams():
    """Return the 68 tournament teams with conference info."""
    return [{"name": t, "conference": CONFERENCE_MAP.get(t, "")} for t in TEAMS]


@app.get("/api/games")
def get_games(date: str = Query(..., description="YYYY-MM-DD")):
    """Return all games for a specific date with stats and spread info."""
    day = DF[DF["date"] == date]
    tracked = day[day["is_tracked"] == True]

    # Deduplicate: for games between two tracked teams, pick each once
    seen_events = set()
    games = []
    for _, row in tracked.iterrows():
        eid = row.get("event_id", "")
        team = row["team"]
        key = f"{eid}_{team}" if eid else f"{date}_{team}"
        if key in seen_events:
            continue
        seen_events.add(key)

        game = {
            "event_id": str(eid) if pd.notna(eid) else "",
            "team": team,
            "opponent": row.get("opponent", ""),
            "home_away": row.get("home_away", ""),
            "neutral_site": bool(row.get("neutral_site", False)),
            "conference_game": bool(row.get("conference_game", False)),
            "team_score": _safe_int(row.get("team_score")),
            "opp_score": _safe_int(row.get("opp_score")),
            "win_loss": row.get("win_loss", ""),
            "opening_spread": _safe_float(row.get("opening_fg_spread")),
            "closing_spread": _safe_float(row.get("closing_fg_spread")),
            "opening_ml": _safe_int(row.get("opening_fg_ml")),
            "closing_ml": _safe_int(row.get("closing_fg_ml")),
            "covered": bool(row.get("covered", False)) if pd.notna(row.get("ats_margin")) else None,
            "ats_margin": _safe_float(row.get("ats_margin")),
            "team_conference": CONFERENCE_MAP.get(team, ""),
            "opp_conference": CONFERENCE_MAP.get(row.get("opponent", ""), ""),
            # PIT stats
            "team_stats": _extract_pit_stats(row, "team"),
            "opp_stats": _extract_pit_stats(row, "opp"),
        }
        games.append(game)

    games.sort(key=lambda g: (g["team_score"] is None, g["team"]))
    return {"date": date, "games": games}


@app.get("/api/games/{event_id}/line-movement")
def get_line_movement(event_id: str):
    """Return line movement data for a specific game."""
    if ODDS_DF.empty:
        return {"event_id": event_id, "movements": []}

    # Match by game_id in odds history
    game_odds = ODDS_DF[ODDS_DF["game_id"] == event_id]

    if game_odds.empty:
        # Try matching by commence_time or other means
        return {"event_id": event_id, "movements": []}

    # Filter to spreads market
    spreads = game_odds[game_odds["market"] == "spreads"].copy()
    if spreads.empty:
        return {"event_id": event_id, "movements": []}

    spreads = spreads.sort_values("snapshot_time")

    movements = []
    for _, row in spreads.iterrows():
        movements.append({
            "time": row["snapshot_time"].isoformat(),
            "bookmaker": row.get("bookmaker", ""),
            "outcome": row.get("outcome", ""),
            "point": _safe_float(row.get("point")),
            "price": _safe_int(row.get("price")),
        })

    return {"event_id": event_id, "movements": movements}


@app.get("/api/espn/scoreboard")
async def get_espn_scoreboard(date: str = Query("", description="YYYYMMDD")):
    """Proxy ESPN scoreboard for live scores."""
    if not date:
        date = datetime.now().strftime("%Y%m%d")

    url = f"{ESPN_BASE}/scoreboard?dates={date}&limit=200"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return _parse_espn_scoreboard(data)
                return {"error": f"ESPN returned {resp.status}", "games": []}
    except Exception as e:
        return {"error": str(e), "games": []}


SPREAD_BUCKETS = [
    ("0.5 to 2.5", -2.5, -0.5),
    ("3 to 5.5", -5.5, -3.0),
    ("6 to 9.5", -9.5, -6.0),
    ("10+", -999, -10.0),
]


def _spread_bucket(games: pd.DataFrame, lo: float, hi: float) -> pd.DataFrame:
    """Filter games where the favorite's spread falls in [lo, hi].
    lo/hi are negative (favorite perspective). lo < hi, e.g. lo=-5.5, hi=-3.0."""
    return games[(games["closing_fg_spread"] >= lo) & (games["closing_fg_spread"] <= hi)]


def _build_section(games: pd.DataFrame, label_prefix: str,
                   side: str, venue: str = "all") -> list[dict]:
    """Build a list of strategy rows: total + 4 spread buckets."""
    total = compute_strategy_stats(games, label_prefix, side=side)
    total["filter"] = {"side": side, "venue": venue, "spread_lo": -999, "spread_hi": 999}
    rows = [total]
    for bucket_label, lo, hi in SPREAD_BUCKETS:
        bucket = _spread_bucket(games, lo, hi)
        row = compute_strategy_stats(
            bucket, f"{label_prefix} ({bucket_label})", side=side)
        row["filter"] = {"side": side, "venue": venue, "spread_lo": lo, "spread_hi": hi}
        rows.append(row)
    return rows


@app.get("/api/strategy/summary")
def get_strategy_summary():
    """Return organized strategy stats across all timeframes.

    Structure per timeframe:
      all_games: [All Favs, buckets..., All Dogs, buckets...]
      home_away: [Home Favs, buckets..., Home Dogs, buckets...,
                  Away Favs, ..., Away Dogs, ...,
                  Neutral Favs, ..., Neutral Dogs, ...]
    """
    results = {}
    TIMEFRAME_KEYS = [
        "full_season", "regular_season", "conf_tournaments", "ncaa_tournament",
        "first_four", "round_of_64", "round_of_32", "sweet_16",
        "elite_8", "final_four", "championship",
    ]
    for tf_name in TIMEFRAME_KEYS:
        tf = filter_by_timeframe(GAMES_DF, tf_name)

        home = tf[tf["home_away"] == "home"]
        away = tf[tf["home_away"] == "away"]
        neutral = tf[tf["home_away"] == "neutral"]

        results[tf_name] = {
            "all_games": (
                _build_section(tf, "All Favorites", "favorite", "all") +
                _build_section(tf, "All Underdogs", "underdog", "all")
            ),
            "home_away": (
                _build_section(home, "Home Favorites", "favorite", "home") +
                _build_section(home, "Home Underdogs", "underdog", "home") +
                _build_section(away, "Away Favorites", "favorite", "away") +
                _build_section(away, "Away Underdogs", "underdog", "away") +
                _build_section(neutral, "Neutral Favorites", "favorite", "neutral") +
                _build_section(neutral, "Neutral Underdogs", "underdog", "neutral")
            ),
        }

    return results


@app.get("/api/strategy/games")
def get_strategy_games(
    side: str = Query("favorite"),
    venue: str = Query("all"),       # all, home, away, neutral
    spread_lo: float = Query(-999),
    spread_hi: float = Query(999),
    timeframe: str = Query("full_season"),
):
    """Return the list of games matching a strategy filter."""
    tf = filter_by_timeframe(GAMES_DF, timeframe)

    if venue != "all":
        tf = tf[tf["home_away"] == venue]

    # Spread bucket (favorite perspective, so values are negative)
    if spread_lo != -999 or spread_hi != 999:
        tf = tf[
            (tf["closing_fg_spread"] >= spread_lo) &
            (tf["closing_fg_spread"] <= spread_hi)
        ]

    testable = tf[~tf["push"]].sort_values("date", ascending=False)

    covered_col = "dog_covered" if side == "underdog" else "covered"
    price_col = "dog_spread_price" if side == "underdog" else "closing_spread_price"

    games = []
    for _, row in testable.iterrows():
        price = row.get(price_col)
        if pd.isna(price):
            price = JUICE
        won = bool(row[covered_col])
        if won:
            pnl = 100 / abs(price) if price < 0 else price / 100
        else:
            pnl = -1.0

        games.append({
            "date": row["date"],
            "team": row["team"],
            "opponent": row.get("opponent", ""),
            "home_away": row.get("home_away", ""),
            "team_score": _safe_int(row.get("team_score")),
            "opp_score": _safe_int(row.get("opp_score")),
            "spread": _safe_float(row.get("closing_fg_spread")),
            "price": int(price),
            "covered": won,
            "ats_margin": _safe_float(row.get("ats_margin")),
            "pnl": round(pnl, 3),
            "team_win_pct": _safe_float(row.get("team_win_pct")),
            "opp_win_pct": _safe_float(row.get("opp_win_pct")),
        })

    return {"games": games, "count": len(games)}


@app.get("/api/strategy/equity-curve")
def get_equity_curve(
    side: str = Query("favorite"),
    venue: str = Query("all"),
    spread_lo: float = Query(-999),
    spread_hi: float = Query(999),
    timeframe: str = Query("full_season"),
):
    """Return equity curve for a strategy filter using GAMES_DF."""
    tf = filter_by_timeframe(GAMES_DF, timeframe)

    if venue != "all":
        tf = tf[tf["home_away"] == venue]

    if spread_lo != -999 or spread_hi != 999:
        tf = tf[
            (tf["closing_fg_spread"] >= spread_lo) &
            (tf["closing_fg_spread"] <= spread_hi)
        ]

    curve = compute_equity_curve(tf, side=side)
    return {"curve": curve}


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """Run a backtest with user-defined filters."""
    testable = DF[
        (DF["is_tracked"] == True) &
        DF["closing_fg_spread"].notna() &
        DF["team_games"].notna() &
        (DF["team_games"] > 0)
    ].copy()

    # Apply side filter
    if req.side == "favorites":
        testable = testable[testable["closing_fg_spread"] < 0]
    elif req.side == "underdogs":
        testable = testable[testable["closing_fg_spread"] > 0]
    elif req.side == "home":
        testable = testable[testable["home_away"] == "home"]
    elif req.side == "away":
        testable = testable[testable["home_away"] == "away"]

    # Apply each filter
    mask = pd.Series(True, index=testable.index)
    filter_descriptions = []

    for f in req.filters:
        team_col = f"team_{f.stat}"
        opp_col = f"opp_{f.stat}"

        if f.compare == "gt" and f.value is not None:
            # Team stat > value
            if team_col in testable.columns:
                mask &= testable[team_col].notna() & (testable[team_col] > f.value)
                filter_descriptions.append(f"team_{f.stat} > {f.value}")

        elif f.compare == "lt" and f.value is not None:
            if team_col in testable.columns:
                mask &= testable[team_col].notna() & (testable[team_col] < f.value)
                filter_descriptions.append(f"team_{f.stat} < {f.value}")

        elif f.compare == "gt_opp":
            # Team stat > opponent stat
            actual_opp = f"opp_{f.opp_stat}" if f.opp_stat else opp_col
            if team_col in testable.columns and actual_opp in testable.columns:
                mask &= (
                    testable[team_col].notna() &
                    testable[actual_opp].notna() &
                    (testable[team_col] > testable[actual_opp])
                )
                filter_descriptions.append(f"team_{f.stat} > {actual_opp}")

        elif f.compare == "lt_opp":
            actual_opp = f"opp_{f.opp_stat}" if f.opp_stat else opp_col
            if team_col in testable.columns and actual_opp in testable.columns:
                mask &= (
                    testable[team_col].notna() &
                    testable[actual_opp].notna() &
                    (testable[team_col] < testable[actual_opp])
                )
                filter_descriptions.append(f"team_{f.stat} < {actual_opp}")

        elif f.compare == "gt_opp_margin" and f.value is not None:
            actual_opp = f"opp_{f.opp_stat}" if f.opp_stat else opp_col
            if team_col in testable.columns and actual_opp in testable.columns:
                mask &= (
                    testable[team_col].notna() &
                    testable[actual_opp].notna() &
                    ((testable[team_col] - testable[actual_opp]) > f.value)
                )
                filter_descriptions.append(f"team_{f.stat} - {actual_opp} > {f.value}")

        elif f.compare == "lt_opp_margin" and f.value is not None:
            actual_opp = f"opp_{f.opp_stat}" if f.opp_stat else opp_col
            if team_col in testable.columns and actual_opp in testable.columns:
                mask &= (
                    testable[team_col].notna() &
                    testable[actual_opp].notna() &
                    ((testable[team_col] - testable[actual_opp]) < f.value)
                )
                filter_descriptions.append(f"team_{f.stat} - {actual_opp} < {f.value}")

        elif f.compare == "rank_gt" and f.value is not None:
            # Team stat ranking > value (e.g., top 25 = rank > 75 percentile)
            rank_col = f"_rank_{f.stat}"
            if rank_col not in testable.columns:
                _add_rank_columns(testable, f.stat)
            if rank_col in testable.columns:
                mask &= testable[rank_col].notna() & (testable[rank_col] > f.value)
                filter_descriptions.append(f"team_{f.stat} rank > {f.value}th pctile")

        elif f.compare == "rank_lt" and f.value is not None:
            rank_col = f"_rank_{f.stat}"
            if rank_col not in testable.columns:
                _add_rank_columns(testable, f.stat)
            if rank_col in testable.columns:
                mask &= testable[rank_col].notna() & (testable[rank_col] < f.value)
                filter_descriptions.append(f"team_{f.stat} rank < {f.value}th pctile")

        elif f.compare == "opp_rank_gt" and f.value is not None:
            rank_col = f"_opp_rank_{f.stat}"
            if rank_col not in testable.columns:
                _add_opp_rank_columns(testable, f.stat)
            if rank_col in testable.columns:
                mask &= testable[rank_col].notna() & (testable[rank_col] > f.value)
                filter_descriptions.append(f"opp_{f.stat} rank > {f.value}th pctile")

        elif f.compare == "opp_rank_lt" and f.value is not None:
            rank_col = f"_opp_rank_{f.stat}"
            if rank_col not in testable.columns:
                _add_opp_rank_columns(testable, f.stat)
            if rank_col in testable.columns:
                mask &= testable[rank_col].notna() & (testable[rank_col] < f.value)
                filter_descriptions.append(f"opp_{f.stat} rank < {f.value}th pctile")

    if req.min_games > 0:
        mask &= testable["team_games"] >= req.min_games
        filter_descriptions.append(f"min {req.min_games} games played")

    filtered = testable[mask]

    # Compute stats
    no_push = filtered[~filtered["push"]]
    n = len(no_push)
    if n == 0:
        return {
            "filters": filter_descriptions,
            "n": 0, "wins": 0, "losses": 0,
            "win_pct": 0, "roi": 0, "profitable": False,
            "p_value": 1.0, "ci_low": 0, "ci_high": 0,
            "curve": [],
        }

    wins = int(no_push["covered"].sum())
    losses = n - wins
    win_pct = wins / n
    profit = wins * (100 / abs(JUICE)) - losses
    roi = profit / n * 100

    # Binomial test
    from scipy import stats as sp_stats
    p_value = sp_stats.binomtest(wins, n, 0.5, alternative="greater").pvalue

    # Wilson CI
    z = 1.96
    denom = 1 + z**2 / n
    center = (win_pct + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((win_pct * (1 - win_pct) + z**2 / (4 * n)) / n) / denom
    ci_low = center - margin
    ci_high = center + margin

    # Equity curve
    curve = compute_equity_curve(filtered)

    return {
        "filters": filter_descriptions,
        "side": req.side,
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_pct": round(win_pct, 4),
        "roi": round(roi, 2),
        "profitable": win_pct > 0.524,
        "p_value": round(p_value, 6),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "curve": curve,
    }


@app.get("/api/stats")
def get_available_stats():
    """Return the list of available PIT stats for backtesting."""
    return {"stats": PIT_STATS}


# ── helpers ────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return round(float(v), 2)
    except (TypeError, ValueError):
        return None


def _safe_int(v) -> Optional[int]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _extract_pit_stats(row: pd.Series, prefix: str) -> dict:
    """Extract PIT stats from a game row for a given prefix (team/opp)."""
    stats = {}
    for s in PIT_STATS:
        col = f"{prefix}_{s}"
        if col in row.index:
            stats[s] = _safe_float(row[col])
    return stats


def _add_rank_columns(df: pd.DataFrame, stat: str):
    """Add percentile rank column for a team stat, computed per date."""
    col = f"team_{stat}"
    rank_col = f"_rank_{stat}"
    if col not in df.columns:
        return

    # Group by date and compute percentile rank
    df[rank_col] = np.nan
    for date, group in df.groupby("date"):
        valid = group[col].dropna()
        if len(valid) < 3:
            continue
        ranks = valid.rank(pct=True) * 100
        df.loc[ranks.index, rank_col] = ranks


def _add_opp_rank_columns(df: pd.DataFrame, stat: str):
    """Add percentile rank column for an opponent stat, computed per date."""
    col = f"opp_{stat}"
    rank_col = f"_opp_rank_{stat}"
    if col not in df.columns:
        return

    df[rank_col] = np.nan
    for date, group in df.groupby("date"):
        valid = group[col].dropna()
        if len(valid) < 3:
            continue
        ranks = valid.rank(pct=True) * 100
        df.loc[ranks.index, rank_col] = ranks


def _parse_espn_scoreboard(data: dict) -> dict:
    """Parse ESPN scoreboard response into our format."""
    games = []
    for event in data.get("events", []):
        competition = event.get("competitions", [{}])[0]
        competitors = competition.get("competitors", [])
        if len(competitors) < 2:
            continue

        status = event.get("status", {})
        status_type = status.get("type", {})
        state = status_type.get("state", "pre")  # pre, in, post
        detail = status_type.get("detail", "")
        clock = status.get("displayClock", "")
        period = status.get("period", 0)

        home = away = None
        for c in competitors:
            info = {
                "id": c.get("id", ""),
                "name": c.get("team", {}).get("displayName", ""),
                "short": c.get("team", {}).get("shortDisplayName", ""),
                "abbreviation": c.get("team", {}).get("abbreviation", ""),
                "score": int(c.get("score", 0)) if c.get("score") else 0,
                "seed": c.get("curatedRank", {}).get("current", 0),
                "logo": c.get("team", {}).get("logo", ""),
            }
            if c.get("homeAway") == "home":
                home = info
            else:
                away = info

        if not home or not away:
            continue

        games.append({
            "event_id": event.get("id", ""),
            "state": state,
            "detail": detail,
            "clock": clock,
            "period": period,
            "home": home,
            "away": away,
            "neutral_site": competition.get("neutralSite", False),
            "start_time": event.get("date", ""),
        })

    return {"games": games}


# ── startup ────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_data()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
