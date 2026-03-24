"""
NCAAB Web App Backend
======================
FastAPI server serving today's games, odds, matchups, and backtesting.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy import stats

from engine import BacktestEngine
from models import SizedBacktestRequest, SizedBacktestResult

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
NCAAB_DIR = Path(os.getenv("NCAAB_DIR", SCRIPT_DIR.parent.parent))

load_dotenv(SCRIPT_DIR / ".env", override=True)
load_dotenv(NCAAB_DIR / ".env", override=True)

# Re-resolve after dotenv
NCAAB_DIR = Path(os.getenv("NCAAB_DIR", SCRIPT_DIR.parent.parent))

sys.path.insert(0, str(NCAAB_DIR))
from ncaab_config import TEAMS, CONFERENCE_MAP, ODDS_NAME_MAP

# Data files
GAME_LOGS_FILE = NCAAB_DIR / "ncaab_game_logs.parquet"
ODDS_HISTORY_FILE = NCAAB_DIR / "ncaab_odds_history.parquet"
TODAY_FILE = NCAAB_DIR / "ncaab_today.json"

# ---------------------------------------------------------------------------
# Data cache with mtime-based refresh
# ---------------------------------------------------------------------------

_cache: dict[str, Any] = {
    "game_logs": None,
    "game_logs_mtime": 0,
    "today": None,
    "today_mtime": 0,
    "odds_history": None,
    "odds_history_mtime": 0,
}


def _get_game_logs() -> pd.DataFrame:
    if GAME_LOGS_FILE.exists():
        mtime = GAME_LOGS_FILE.stat().st_mtime
        if _cache["game_logs"] is None or mtime > _cache["game_logs_mtime"]:
            _cache["game_logs"] = pd.read_parquet(GAME_LOGS_FILE)
            _cache["game_logs_mtime"] = mtime
    if _cache["game_logs"] is None:
        return pd.DataFrame()
    return _cache["game_logs"]


def _get_today() -> dict:
    if TODAY_FILE.exists():
        mtime = TODAY_FILE.stat().st_mtime
        if _cache["today"] is None or mtime > _cache["today_mtime"]:
            with open(TODAY_FILE) as f:
                _cache["today"] = json.load(f)
            _cache["today_mtime"] = mtime
    if _cache["today"] is None:
        return {"games": [], "updated_at": None, "game_count": 0}
    return _cache["today"]


def _get_odds_history() -> pd.DataFrame:
    if ODDS_HISTORY_FILE.exists():
        mtime = ODDS_HISTORY_FILE.stat().st_mtime
        if _cache["odds_history"] is None or mtime > _cache["odds_history_mtime"]:
            _cache["odds_history"] = pd.read_parquet(ODDS_HISTORY_FILE)
            _cache["odds_history_mtime"] = mtime
    if _cache["odds_history"] is None:
        return pd.DataFrame()
    return _cache["odds_history"]


# Build reverse lookup: canonical name -> odds API name
_CANONICAL_TO_ODDS = {k: v for k, v in ODDS_NAME_MAP.items()}

# Preferred bookmaker order
_BOOK_PREF = ["bovada", "draftkings", "fanduel", "betmgm", "betrivers",
              "williamhill_us", "fanatics", "betonlineag", "lowvig"]


def _lookup_odds_for_game(home_canonical: str, away_canonical: str,
                          commence_date: str) -> dict | None:
    """Look up full odds (both sides) from the odds history for a game.

    Returns a dict with spread/ml/total for both teams, or None if not found.
    Uses the last pre-game snapshot (or latest available).
    """
    odf = _get_odds_history()
    if odf.empty:
        return None

    home_odds_name = _CANONICAL_TO_ODDS.get(home_canonical, home_canonical)
    away_odds_name = _CANONICAL_TO_ODDS.get(away_canonical, away_canonical)

    mask = (
        (odf["home_team"] == home_odds_name) & (odf["away_team"] == away_odds_name)
    ) | (
        (odf["home_team"] == away_odds_name) & (odf["away_team"] == home_odds_name)
    )
    game_df = odf[mask]
    if game_df.empty:
        return None

    # Get the latest snapshot that's on or before the game date + 1 day
    # (pre-game odds, not live in-game odds)
    cutoff = f"{commence_date}T23:59:59Z"
    pre_game = game_df[game_df["snapshot_time"] <= cutoff]
    if pre_game.empty:
        # Fall back to earliest snapshot if all are after game date
        snap_time = game_df["snapshot_time"].min()
    else:
        snap_time = pre_game["snapshot_time"].max()

    snap = game_df[game_df["snapshot_time"] == snap_time]

    # Determine actual home/away from odds data
    actual_home = snap["home_team"].iloc[0]
    actual_away = snap["away_team"].iloc[0]

    # Pick best available bookmaker
    available_books = snap["bookmaker"].unique()
    chosen_book = None
    for b in _BOOK_PREF:
        if b in available_books:
            chosen_book = b
            break
    if chosen_book is None:
        chosen_book = available_books[0]

    book_rows = snap[snap["bookmaker"] == chosen_book]

    result = {
        "bookmaker": chosen_book,
        "spread": None, "spread_price": None,
        "away_spread": None, "away_spread_price": None,
        "ml": None, "away_ml": None,
        "total": None, "over_price": None, "under_price": None,
    }

    for _, row in book_rows.iterrows():
        mkt = row["market"]
        outcome = row["outcome"]
        price = row["price"] if pd.notna(row["price"]) else None
        point = row["point"] if pd.notna(row["point"]) else None

        if mkt == "spreads":
            if outcome == actual_home:
                result["spread"] = point
                result["spread_price"] = price
            elif outcome == actual_away:
                result["away_spread"] = point
                result["away_spread_price"] = price
        elif mkt == "h2h":
            if outcome == actual_home:
                result["ml"] = price
            elif outcome == actual_away:
                result["away_ml"] = price
        elif mkt == "totals":
            result["total"] = point
            if outcome == "Over":
                result["over_price"] = price
            elif outcome == "Under":
                result["under_price"] = price

    return result


# ---------------------------------------------------------------------------
# Backtest engine (adapted from ncaab_backtest.py)
# ---------------------------------------------------------------------------

JUICE = -110


def _enrich_with_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank columns for each ranked stat, computed per-date among tracked teams.

    Adds columns like rank_team_win_pct, rank_opp_win_pct for use in filters.
    """
    tracked = df[df["is_tracked"] == True].copy()
    if tracked.empty:
        return df

    dates = sorted(tracked["date"].unique())

    # Build a lookup: for each (date, team) -> ranks
    # We compute ranks cumulatively: for each date, rank all teams by their
    # latest PIT stats up to and including that date.
    all_rank_rows = []

    for date in dates:
        rankings = _compute_rankings(date)
        day_rows = tracked[tracked["date"] == date]
        for idx, row in day_rows.iterrows():
            team = row["team"]
            opponent = row.get("opponent", "")
            team_ranks = rankings.get(team, {})
            # Find opponent's canonical name — check if they're tracked
            opp_ranks = {}
            for opp_name, opp_r in rankings.items():
                if opp_name == opponent or opponent.startswith(opp_name):
                    opp_ranks = opp_r
                    break
            # Also try matching via the tracked rows for this event
            if not opp_ranks:
                opp_tracked = day_rows[
                    (day_rows["event_id"] == row.get("event_id"))
                    & (day_rows["team"] != team)
                ]
                if not opp_tracked.empty:
                    opp_name = opp_tracked.iloc[0]["team"]
                    opp_ranks = rankings.get(opp_name, {})

            rank_data = {"_idx": idx}
            for stat in _RANKED_STATS:
                if stat in team_ranks:
                    rank_data[f"rank_{stat}"] = team_ranks[stat]
                opp_stat = stat.replace("team_", "opp_", 1)
                if stat in opp_ranks:
                    rank_data[f"rank_{opp_stat}"] = opp_ranks[stat]
            all_rank_rows.append(rank_data)

    if not all_rank_rows:
        return df

    rank_df = pd.DataFrame(all_rank_rows).set_index("_idx")
    df = df.join(rank_df, how="left")
    return df


def _load_testable_games(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (df["is_tracked"] == True)
        & df["closing_fg_spread"].notna()
        & df["team_games"].notna()
        & (df["team_games"] > 0)
    )
    df = df[mask].copy()
    df["margin"] = df["team_score"].astype(float) - df["opp_score"].astype(float)
    df["ats_margin"] = df["margin"] + df["closing_fg_spread"].astype(float)
    df["covered"] = df["ats_margin"] > 0
    df["push"] = df["ats_margin"] == 0
    df = _enrich_with_ranks(df)
    return df


def _run_backtest(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    filtered = df[mask & ~df["push"]]
    n = len(filtered)
    if n == 0:
        return {"name": name, "n": 0}

    wins = int(filtered["covered"].sum())
    losses = n - wins
    win_pct = wins / n
    profit = wins * (100 / abs(JUICE)) - losses
    roi = profit / n * 100

    p_value = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue

    z = 1.96
    denom = 1 + z ** 2 / n
    center = (win_pct + z ** 2 / (2 * n)) / denom
    margin_val = z * np.sqrt((win_pct * (1 - win_pct) + z ** 2 / (4 * n)) / n) / denom

    return {
        "name": name,
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_pct": round(win_pct, 4),
        "roi": round(roi, 2),
        "p_value": round(p_value, 6),
        "ci_low": round(center - margin_val, 4),
        "ci_high": round(center + margin_val, 4),
        "profitable": win_pct > 0.524,
    }


def _build_mask_from_filters(
    df: pd.DataFrame, filters: list[dict], team: str | None
) -> pd.Series:
    mask = pd.Series(True, index=df.index)

    if team:
        mask = mask & (df["team"] == team)

    for f in filters:
        col = f.get("stat", "")
        op = f.get("op", "")
        val = f.get("value")
        compare_col = f.get("compare_col", "")

        if col not in df.columns:
            continue

        series = pd.to_numeric(df[col], errors="coerce")

        # Column-vs-column comparison (e.g., team_3pt_pct vs opp_def_3pt_pct)
        if compare_col and compare_col in df.columns:
            opp_series = pd.to_numeric(df[compare_col], errors="coerce")
            diff = series - opp_series
            try:
                threshold = float(val) if val else 0.0
            except (ValueError, TypeError):
                threshold = 0.0
            valid = series.notna() & opp_series.notna()
            if op == ">":
                mask = mask & valid & (diff > threshold)
            elif op == ">=":
                mask = mask & valid & (diff >= threshold)
            elif op == "<":
                mask = mask & valid & (diff < threshold)
            elif op == "<=":
                mask = mask & valid & (diff <= threshold)
            continue

        # Fixed-value comparison
        try:
            num_val = float(val)
            if op == ">":
                mask = mask & (series > num_val)
            elif op == ">=":
                mask = mask & (series >= num_val)
            elif op == "<":
                mask = mask & (series < num_val)
            elif op == "<=":
                mask = mask & (series <= num_val)
            elif op == "==":
                mask = mask & (series == num_val)
            elif op == "!=":
                mask = mask & (series != num_val)
        except (ValueError, TypeError):
            # String comparison
            raw = df[col]
            if op == "==":
                mask = mask & (raw == val)
            elif op == "!=":
                mask = mask & (raw != val)

    return mask


def _run_built_in_strategies(df: pd.DataFrame) -> list[dict]:
    results = []

    results.append(_run_backtest(df, pd.Series(True, index=df.index), "Baseline (all games)"))
    results.append(_run_backtest(df, df["closing_fg_spread"] < 0, "All favorites"))
    results.append(_run_backtest(df, df["closing_fg_spread"] > 0, "All underdogs"))
    results.append(_run_backtest(df, df["closing_fg_spread"] < -10, "Big favorites (spread < -10)"))
    results.append(_run_backtest(df, df["closing_fg_spread"] > 10, "Big underdogs (spread > +10)"))

    for thresh in [5, 10, 15]:
        m = (
            df["team_def_2pt_pct"].notna()
            & df["opp_2pt_pct"].notna()
            & (df["opp_2pt_pct"] - df["team_def_2pt_pct"] > thresh)
        )
        results.append(_run_backtest(df, m, f"Team def 2PT% allows {thresh}+ less than opp shoots"))

    for thresh in [3, 5]:
        m = (
            df["team_def_3pt_pct"].notna()
            & df["opp_3pt_pct"].notna()
            & (df["opp_3pt_pct"] - df["team_def_3pt_pct"] > thresh)
        )
        results.append(_run_backtest(df, m, f"Team def 3PT% allows {thresh}+ less than opp shoots"))

    for thresh in [3, 5, 8]:
        m = (
            df["team_ft_pct"].notna()
            & df["opp_ft_pct"].notna()
            & (df["team_ft_pct"] - df["opp_ft_pct"] > thresh)
        )
        results.append(_run_backtest(df, m, f"Team FT% {thresh}+ higher than opponent FT%"))

    for thresh in [3, 5]:
        m = (
            df["team_fta_pg"].notna()
            & df["opp_fta_pg"].notna()
            & (df["team_fta_pg"] - df["opp_fta_pg"] > thresh)
        )
        results.append(_run_backtest(df, m, f"Team FTA/game {thresh}+ more than opponent"))

    for thresh in [2, 4]:
        m = (
            df["team_forced_to_pg"].notna()
            & df["opp_forced_to_pg"].notna()
            & df["team_to_pg"].notna()
            & df["opp_to_pg"].notna()
            & (
                (df["team_forced_to_pg"] - df["team_to_pg"])
                - (df["opp_forced_to_pg"] - df["opp_to_pg"])
                > thresh
            )
        )
        results.append(_run_backtest(df, m, f"Turnover margin advantage {thresh}+/game"))

    for thresh in [5, 8]:
        m = df["team_pace"].notna() & df["opp_pace"].notna()
        results.append(
            _run_backtest(df, m & (df["team_pace"] - df["opp_pace"] > thresh),
                          f"Team pace {thresh}+ faster than opponent")
        )
        results.append(
            _run_backtest(df, m & (df["opp_pace"] - df["team_pace"] > thresh),
                          f"Team pace {thresh}+ slower than opponent")
        )

    for thresh in [3, 5]:
        m = (
            df["team_oreb_pg"].notna()
            & df["opp_oreb_pg"].notna()
            & (df["team_oreb_pg"] - df["opp_oreb_pg"] > thresh)
        )
        results.append(_run_backtest(df, m, f"Team OREB/game {thresh}+ more than opponent"))

    for thresh in [0.15, 0.25]:
        m = (
            df["team_win_pct"].notna()
            & df["opp_win_pct"].notna()
            & (df["team_win_pct"] - df["opp_win_pct"] > thresh)
        )
        results.append(_run_backtest(df, m, f"Team win% {thresh:.0%}+ higher than opponent"))

    for thresh in [0.55, 0.60]:
        m = (
            df["team_ats_win_pct"].notna()
            & (df["team_ats_win_pct"] > thresh)
            & (df["team_ats_games"] >= 5)
        )
        results.append(_run_backtest(df, m, f"Team ATS win% > {thresh:.0%} (min 5 games)"))

    for thresh in [10, 20]:
        m = (
            df["team_sos"].notna()
            & df["opp_sos"].notna()
            & (df["opp_sos"] - df["team_sos"] > thresh)
        )
        results.append(_run_backtest(df, m, f"Team SOS {thresh}+ tougher than opponent"))

    # Line movement
    has_both = df["opening_fg_spread"].notna() & df["closing_fg_spread"].notna()
    move = df["closing_fg_spread"] - df["opening_fg_spread"]
    for thresh in [1, 2, 3]:
        results.append(
            _run_backtest(df, has_both & (move > thresh),
                          f"Line moved {thresh}+ pts in team's favor")
        )
        results.append(
            _run_backtest(df, has_both & (move < -thresh),
                          f"Line moved {thresh}+ pts against team")
        )

    results.append(_run_backtest(df, df["home_away"] == "home", "Home teams"))
    results.append(_run_backtest(df, df["home_away"] == "away", "Away teams"))
    results.append(_run_backtest(df, df["home_away"] == "neutral", "Neutral site"))
    results.append(_run_backtest(df, df["conference_game"].isin([True, "Y"]), "Conference games"))
    results.append(
        _run_backtest(df, ~df["conference_game"].isin([True, "Y"]), "Non-conference games")
    )

    return [r for r in results if r["n"] > 0]


def _scan_differentials(df: pd.DataFrame, min_games: int = 20) -> list[dict]:
    results = []
    team_stats = [
        c for c in df.columns
        if c.startswith("team_") and c.endswith(("_pct", "_pg", "_pace", "_sos"))
    ]

    for team_col in team_stats:
        opp_col = "opp_" + team_col[5:]
        if opp_col not in df.columns:
            continue

        valid = df[team_col].notna() & df[opp_col].notna()
        if valid.sum() < 100:
            continue

        t_series = pd.to_numeric(df[team_col], errors="coerce")
        o_series = pd.to_numeric(df[opp_col], errors="coerce")
        diff = t_series - o_series
        stat_name = team_col[5:]

        for pct, label in [(75, "top 25%"), (90, "top 10%")]:
            thresh = np.nanpercentile(diff[valid], pct)
            if thresh == 0:
                continue
            mask = valid & (diff > thresh)
            r = _run_backtest(
                df, mask,
                f"team {stat_name} >> opp ({label}, diff > {thresh:.1f})",
            )
            if r["n"] >= min_games:
                results.append(r)

        for pct, label in [(25, "bottom 25%"), (10, "bottom 10%")]:
            thresh = np.nanpercentile(diff[valid], pct)
            if thresh == 0:
                continue
            mask = valid & (diff < thresh)
            r = _run_backtest(
                df, mask,
                f"team {stat_name} << opp ({label}, diff < {thresh:.1f})",
            )
            if r["n"] >= min_games:
                results.append(r)

    return results


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="NCAAB Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "ncaab_dir": str(NCAAB_DIR),
        "game_logs_exists": GAME_LOGS_FILE.exists(),
        "today_exists": TODAY_FILE.exists(),
    }


# -- Today's Games --


@app.get("/api/today")
def get_today():
    """Return today's games with odds and stats."""
    return _get_today()


def _safe_float(val):
    if pd.notna(val):
        try:
            return round(float(val), 2)
        except (ValueError, TypeError):
            pass
    return None


def _safe_int(val):
    if pd.notna(val):
        try:
            return int(float(val))
        except (ValueError, TypeError):
            pass
    return None


# Stats where higher = better, and stats where lower = better
_RANKED_STATS = {
    "team_win_pct": True,
    "team_ppg": True,
    "opp_ppg": False,       # points allowed — lower is better
    "team_ats_win_pct": True,
    "team_3pt_pct": True,
    "team_ft_pct": True,
    "team_2pt_pct": True,
    "team_pace": True,
    "team_sos": True,
    "team_oreb_pg": True,
    "team_dreb_pg": True,
    "team_to_pg": False,     # turnovers — lower is better
    "team_forced_to_pg": True,
    "team_def_3pt_pct": False,  # opponent shooting allowed — lower is better
    "team_def_2pt_pct": False,
}

_rankings_cache: dict[str, dict] = {}


def _compute_rankings(date: str) -> dict[str, dict[str, int]]:
    """Compute per-stat rank (1=best) for all tracked teams as of a date.

    Returns {team_name: {stat: rank}}.
    """
    if date in _rankings_cache:
        return _rankings_cache[date]

    df = _get_game_logs()
    if df.empty:
        return {}

    tracked = df[(df["is_tracked"] == True) & (df["date"] <= date)].copy()
    if tracked.empty:
        return {}

    # Get each team's latest row on or before this date
    tracked = tracked.sort_values("date")
    latest = tracked.groupby("team").last()

    rankings: dict[str, dict[str, int]] = {team: {} for team in latest.index}

    for stat, higher_is_better in _RANKED_STATS.items():
        if stat not in latest.columns:
            continue
        vals = pd.to_numeric(latest[stat], errors="coerce")
        ranked = vals.rank(ascending=not higher_is_better, method="min", na_option="bottom")
        for team in latest.index:
            r = ranked.get(team)
            if pd.notna(r) and pd.notna(vals.get(team)):
                rankings[team][stat] = int(r)

    _rankings_cache[date] = rankings
    return rankings


def _build_record(r, prefix="team"):
    """Build 'W-L' string from games + win_pct."""
    games = _safe_int(r.get(f"{prefix}_games"))
    wp = _safe_float(r.get(f"{prefix}_win_pct"))
    if games and wp is not None:
        wins = round(games * wp)
        return f"{wins}-{games - wins}"
    return None


def _build_ats(r, prefix="team"):
    """Build 'W-L' ATS string from ats_games + ats_win_pct."""
    ag = _safe_int(r.get(f"{prefix}_ats_games"))
    ap = _safe_float(r.get(f"{prefix}_ats_win_pct"))
    if ag and ap is not None:
        aw = round(ag * ap)
        return f"{aw}-{ag - aw}"
    return None


def _build_side(r, rankings: dict[str, dict[str, int]] | None = None):
    """Build a full side dict from a game log row."""
    team_name = r["team"]
    team_ranks = rankings.get(team_name, {}) if rankings else {}

    stat_keys = [
        "team_win_pct", "team_ppg", "opp_ppg", "team_ats_win_pct",
        "team_3pt_pct", "team_ft_pct", "team_2pt_pct",
        "team_pace", "team_sos", "team_oreb_pg", "team_dreb_pg",
        "team_to_pg", "team_forced_to_pg", "team_def_3pt_pct", "team_def_2pt_pct",
    ]
    stats: dict[str, Any] = {
        "record": _build_record(r),
        "ats": _build_ats(r),
    }
    ranks: dict[str, int] = {}
    for k in stat_keys:
        stats[k] = _safe_float(r.get(k))
        if k in team_ranks:
            ranks[k] = team_ranks[k]

    return {
        "name": team_name,
        "display_name": team_name,
        "abbreviation": "",
        "id": str(r.get("event_id", "")),
        "score": _safe_int(r.get("team_score")),
        "is_tracked": True,
        "stats": stats,
        "ranks": ranks,
    }


def _build_untracked_side(r):
    """Build a side dict for an untracked opponent."""
    return {
        "name": r["opponent"],
        "display_name": r.get("opponent_short", r["opponent"]),
        "abbreviation": "",
        "id": "",
        "score": _safe_int(r.get("opp_score")),
        "is_tracked": False,
        "stats": {},
    }


@app.get("/api/games/{date}")
def get_games_by_date(date: str):
    """Return all tracked games for a specific date (YYYY-MM-DD)."""
    df = _get_game_logs()
    if df.empty:
        return {"date": date, "games": [], "game_count": 0}

    day_df = df[(df["date"] == date) & (df["is_tracked"] == True)].copy()
    if day_df.empty:
        return {"date": date, "games": [], "game_count": 0}

    rankings = _compute_rankings(date)

    seen_events = set()
    games = []
    for _, row in day_df.iterrows():
        eid = row.get("event_id", "")
        if eid in seen_events:
            continue
        seen_events.add(eid)

        # Find opponent row if they're also tracked
        opp_rows = day_df[
            (day_df["event_id"] == eid) & (day_df["team"] != row["team"])
        ]
        opp_row = opp_rows.iloc[0] if not opp_rows.empty else None

        # Determine home/away from the row's perspective
        ha = row.get("home_away", "home")
        is_neutral = bool(row.get("neutral_site", False))

        if ha == "home" or ha == "neutral":
            home_side = _build_side(row, rankings)
            away_side = _build_side(opp_row, rankings) if opp_row is not None else _build_untracked_side(row)
        else:
            away_side = _build_side(row, rankings)
            home_side = _build_side(opp_row, rankings) if opp_row is not None else _build_untracked_side(row)

        # Build odds: prefer odds history (has both sides + totals + juice),
        # fall back to game logs (closing spread/ML only)
        home_name = home_side["name"]
        away_name = away_side["name"]
        live_odds = _lookup_odds_for_game(home_name, away_name, date)

        if live_odds:
            odds_obj = {
                "spread": live_odds["spread"],
                "spread_price": live_odds["spread_price"],
                "away_spread": live_odds["away_spread"],
                "away_spread_price": live_odds["away_spread_price"],
                "ml": live_odds["ml"],
                "away_ml": live_odds["away_ml"],
                "total": live_odds["total"],
                "over_price": live_odds["over_price"],
                "under_price": live_odds["under_price"],
                "bookmaker": live_odds["bookmaker"],
            }
        else:
            # Fallback to game logs
            home_row = row if ha in ("home", "neutral") else (opp_row if opp_row is not None else row)
            away_row = opp_row if ha in ("home", "neutral") else row
            home_closing_spread = _safe_float(home_row.get("closing_fg_spread"))
            away_closing_spread = -home_closing_spread if home_closing_spread is not None else None
            odds_obj = {
                "spread": home_closing_spread,
                "spread_price": None,
                "away_spread": away_closing_spread,
                "away_spread_price": None,
                "ml": _safe_float(home_row.get("closing_fg_ml")),
                "away_ml": _safe_float(away_row.get("closing_fg_ml")) if away_row is not None else None,
                "total": None,
                "over_price": None,
                "under_price": None,
                "bookmaker": None,
            }

        game = {
            "game_id": eid,
            "commence_time": f"{date}T00:00:00Z",
            "date": date,
            "status": "STATUS_FINAL",
            "status_detail": "Final",
            "neutral_site": is_neutral,
            "home": home_side,
            "away": away_side,
            "odds": odds_obj,
        }
        games.append(game)

    return {"date": date, "games": games, "game_count": len(games)}


@app.get("/api/dates")
def get_game_dates():
    """Return all dates that have tracked games, sorted."""
    df = _get_game_logs()
    if df.empty:
        return []
    tracked = df[df["is_tracked"] == True]
    dates = sorted(tracked["date"].unique().tolist())
    return dates


def _find_game_teams(game_id: str) -> tuple[str, str] | None:
    """Find home/away canonical names for a game_id from game logs."""
    df = _get_game_logs()
    if df.empty:
        return None
    rows = df[df["event_id"] == game_id]
    if rows.empty:
        return None
    # Get one row and determine home/away
    for _, r in rows.iterrows():
        ha = r.get("home_away", "home")
        if ha in ("home", "neutral"):
            home = r["team"]
            away = r["opponent"] if r.get("opponent_short") is None else r["opponent"]
            # Try to find tracked opponent name
            opp_rows = rows[rows["team"] != r["team"]]
            if not opp_rows.empty:
                away = opp_rows.iloc[0]["team"]
            return (home, away)
        else:
            away = r["team"]
            opp_rows = rows[rows["team"] != r["team"]]
            if not opp_rows.empty:
                home = opp_rows.iloc[0]["team"]
            else:
                home = r["opponent"]
            return (home, away)
    return None


@app.get("/api/today/{game_id}/odds")
@app.get("/api/game/{game_id}/odds")
def get_game_odds(game_id: str):
    """Return odds movement timeline for a game (flat format for charting)."""
    teams = _find_game_teams(game_id)
    if not teams:
        return {"game_id": game_id, "home": "", "away": "", "timeline": []}

    home_canonical, away_canonical = teams
    home_odds_name = _CANONICAL_TO_ODDS.get(home_canonical, home_canonical)
    away_odds_name = _CANONICAL_TO_ODDS.get(away_canonical, away_canonical)

    odf = _get_odds_history()
    if odf.empty:
        return {"game_id": game_id, "home": home_canonical, "away": away_canonical, "timeline": []}

    mask = (
        (odf["home_team"] == home_odds_name) & (odf["away_team"] == away_odds_name)
    ) | (
        (odf["home_team"] == away_odds_name) & (odf["away_team"] == home_odds_name)
    )
    matched = odf[mask]
    if matched.empty:
        return {"game_id": game_id, "home": home_canonical, "away": away_canonical, "timeline": []}

    actual_home = matched["home_team"].iloc[0]
    actual_away = matched["away_team"].iloc[0]

    # Build a flat timeline: one entry per snapshot with best-book odds
    timeline = []
    for snap_time, group in matched.groupby("snapshot_time"):
        # Pick preferred bookmaker
        available = group["bookmaker"].unique()
        chosen = None
        for b in _BOOK_PREF:
            if b in available:
                chosen = b
                break
        if not chosen:
            chosen = available[0]

        book_rows = group[group["bookmaker"] == chosen]
        entry = {"time": snap_time, "bookmaker": chosen}

        for _, row in book_rows.iterrows():
            mkt = row["market"]
            outcome = row["outcome"]
            price = float(row["price"]) if pd.notna(row["price"]) else None
            point = float(row["point"]) if pd.notna(row["point"]) else None

            if mkt == "spreads":
                if outcome == actual_home:
                    entry["home_spread"] = point
                    entry["home_spread_price"] = price
                elif outcome == actual_away:
                    entry["away_spread"] = point
                    entry["away_spread_price"] = price
            elif mkt == "h2h":
                if outcome == actual_home:
                    entry["home_ml"] = price
                elif outcome == actual_away:
                    entry["away_ml"] = price
            elif mkt == "totals":
                entry["total"] = point
                if outcome == "Over":
                    entry["over_price"] = price
                elif outcome == "Under":
                    entry["under_price"] = price

        timeline.append(entry)

    timeline.sort(key=lambda x: x["time"])

    return {
        "game_id": game_id,
        "home": home_canonical,
        "away": away_canonical,
        "snapshot_count": len(timeline),
        "timeline": timeline,
    }


@app.get("/api/today/{game_id}/matchup")
@app.get("/api/game/{game_id}/matchup")
def get_game_matchup(game_id: str):
    """Return PIT stats comparison for both teams in a game."""
    teams = _find_game_teams(game_id)
    if not teams:
        raise HTTPException(404, f"Game {game_id} not found")

    home_canonical, away_canonical = teams

    df = _get_game_logs()
    if df.empty:
        return {"game_id": game_id, "home": {}, "away": {}}

    pit_keys = [
        "team_games", "team_win_pct", "team_ppg", "opp_ppg",
        "team_ats_games", "team_ats_win_pct",
        "team_ft_pct", "team_3pt_pct", "team_2pt_pct",
        "team_def_ft_pct", "team_def_3pt_pct", "team_def_2pt_pct",
        "team_oreb_pg", "team_dreb_pg",
        "team_to_pg", "team_forced_to_pg",
        "team_pace", "team_sos",
        "team_home_win_pct", "team_away_win_pct",
        "team_home_ppg", "team_away_ppg",
    ]

    # Determine game date for rankings
    game_date = None
    tracked = df[df["event_id"] == game_id]
    if not tracked.empty:
        game_date = tracked.iloc[0]["date"]

    rankings = _compute_rankings(game_date) if game_date else {}

    result = {"game_id": game_id}
    for side, name in [("home", home_canonical), ("away", away_canonical)]:
        team_rows = df[(df["team"] == name) & (df["is_tracked"] == True)]
        if team_rows.empty:
            result[side] = {"name": name, "stats": {}, "ranks": {}}
            continue

        # Get the row for this specific game if possible, otherwise latest
        game_rows = team_rows[team_rows["event_id"] == game_id]
        if not game_rows.empty:
            row = game_rows.iloc[0]
        else:
            row = team_rows.sort_values("date").iloc[-1]

        team_ranks = rankings.get(name, {})
        stats_dict = {
            "name": name,
            "conference": CONFERENCE_MAP.get(name, ""),
            "record": _build_record(row),
            "ats": _build_ats(row),
        }
        ranks_dict = {}
        for key in pit_keys:
            val = row.get(key)
            if pd.notna(val) and val != "":
                try:
                    stats_dict[key] = round(float(val), 2)
                except (ValueError, TypeError):
                    stats_dict[key] = val
            if key in team_ranks:
                ranks_dict[key] = team_ranks[key]

        result[side] = {**stats_dict, "ranks": ranks_dict}

    return result


# -- Teams --


@app.get("/api/teams")
def get_teams():
    """Return all 68 teams with current stats."""
    df = _get_game_logs()
    if df.empty:
        return []

    tracked = df[df["is_tracked"] == True].copy()
    tracked.sort_values("date", inplace=True)

    teams = []
    for team_name in TEAMS:
        team_rows = tracked[tracked["team"] == team_name]
        if team_rows.empty:
            teams.append({
                "name": team_name,
                "conference": CONFERENCE_MAP.get(team_name, ""),
                "games": 0,
            })
            continue

        latest = team_rows.iloc[-1]
        total_games = len(team_rows)
        wins = (team_rows["win_loss"] == "W").sum()
        losses = total_games - wins

        ats_games = 0
        ats_wins = 0
        for _, r in team_rows.iterrows():
            cs = r.get("closing_fg_spread")
            if pd.notna(cs):
                m = float(r["team_score"]) - float(r["opp_score"])
                ats_m = m + float(cs)
                if ats_m != 0:
                    ats_games += 1
                    if ats_m > 0:
                        ats_wins += 1

        info = {
            "name": team_name,
            "conference": CONFERENCE_MAP.get(team_name, ""),
            "games": total_games,
            "record": f"{wins}-{losses}",
            "win_pct": round(wins / total_games, 3) if total_games else 0,
            "ats_record": f"{ats_wins}-{ats_games - ats_wins}" if ats_games else "0-0",
            "ats_pct": round(ats_wins / ats_games, 3) if ats_games else 0,
        }

        for key in ["team_ppg", "team_sos", "team_pace",
                     "team_3pt_pct", "team_ft_pct", "team_2pt_pct",
                     "team_def_3pt_pct"]:
            val = latest.get(key)
            if pd.notna(val) and val != "":
                try:
                    info[key] = round(float(val), 1)
                except (ValueError, TypeError):
                    pass

        teams.append(info)

    return teams


@app.get("/api/teams/{team_name}")
def get_team_detail(team_name: str):
    """Return full game log for a team."""
    df = _get_game_logs()
    if df.empty:
        raise HTTPException(404, "No game data")

    team_rows = df[
        (df["team"] == team_name) & (df["is_tracked"] == True)
    ].sort_values("date")

    if team_rows.empty:
        raise HTTPException(404, f"Team '{team_name}' not found")

    cols = [
        "date", "opponent", "home_away", "conference_game",
        "team_score", "opp_score", "win_loss",
        "opening_fg_spread", "closing_fg_spread",
        "opening_fg_ml", "closing_fg_ml",
        "team_ppg", "opp_ppg", "team_win_pct", "team_ats_win_pct",
        "team_3pt_pct", "team_ft_pct", "team_pace", "team_sos",
    ]
    available = [c for c in cols if c in team_rows.columns]
    records = team_rows[available].to_dict(orient="records")

    # Clean NaN values for JSON
    for r in records:
        for k, v in r.items():
            if pd.isna(v):
                r[k] = None

    return {
        "team": team_name,
        "conference": CONFERENCE_MAP.get(team_name, ""),
        "games": records,
    }


# -- Backtest --


class BacktestRequest(BaseModel):
    team: str | None = None
    filters: list[dict] = []
    min_games: int = 10
    name: str = "Custom backtest"


class ScanRequest(BaseModel):
    min_games: int = 20


@app.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """Run a custom backtest with filters."""
    df = _get_game_logs()
    if df.empty:
        raise HTTPException(500, "No game data loaded")

    testable = _load_testable_games(df)
    if testable.empty:
        return {"result": {"name": req.name, "n": 0}}

    mask = _build_mask_from_filters(testable, req.filters, req.team)
    result = _run_backtest(testable, mask, req.name)

    # Also return matched games for the table
    filtered = testable[mask & ~testable["push"]]
    game_list = []
    for _, row in filtered.iterrows():
        game_list.append({
            "date": row["date"],
            "team": row["team"],
            "opponent": row["opponent"],
            "home_away": row["home_away"],
            "spread": row.get("closing_fg_spread"),
            "team_score": int(row["team_score"]),
            "opp_score": int(row["opp_score"]),
            "margin": float(row["margin"]),
            "ats_margin": float(row["ats_margin"]),
            "covered": bool(row["covered"]),
        })

    # Cumulative P&L
    pnl = []
    cumulative = 0.0
    for g in game_list:
        if g["covered"]:
            cumulative += 100 / abs(JUICE)
        else:
            cumulative -= 1.0
        pnl.append({"date": g["date"], "pnl": round(cumulative, 2)})

    return {"result": result, "games": game_list, "pnl": pnl}


@app.get("/api/backtest/strategies")
def get_strategies():
    """Run all built-in strategies."""
    df = _get_game_logs()
    if df.empty:
        return []
    testable = _load_testable_games(df)
    if testable.empty:
        return []
    results = _run_built_in_strategies(testable)
    return sorted(results, key=lambda r: r.get("win_pct", 0), reverse=True)


@app.post("/api/backtest/scan")
def run_scan(req: ScanRequest):
    """Scan all stat differentials for edges."""
    df = _get_game_logs()
    if df.empty:
        return []
    testable = _load_testable_games(df)
    if testable.empty:
        return []
    results = _scan_differentials(testable, req.min_games)
    return sorted(results, key=lambda r: r.get("win_pct", 0), reverse=True)


# -- Stat columns metadata for frontend filter builder --


@app.get("/api/columns")
def get_columns():
    """Return available stat columns grouped by category for the filter builder.

    Each entry is {col, label} so the frontend can show human-readable names.
    """
    return {
        "record": [
            {"col": "team_games", "label": "Team Games"},
            {"col": "team_win_pct", "label": "Team Win %"},
            {"col": "team_home_win_pct", "label": "Team Home Win %"},
            {"col": "team_away_win_pct", "label": "Team Away Win %"},
            {"col": "team_neutral_win_pct", "label": "Team Neutral Win %"},
            {"col": "opp_games", "label": "Opp Games"},
            {"col": "opp_win_pct", "label": "Opp Win %"},
        ],
        "ats": [
            {"col": "team_ats_games", "label": "Team ATS Games"},
            {"col": "team_ats_win_pct", "label": "Team ATS %"},
            {"col": "team_home_ats_win_pct", "label": "Team Home ATS %"},
            {"col": "team_away_ats_win_pct", "label": "Team Away ATS %"},
            {"col": "opp_ats_games", "label": "Opp ATS Games"},
            {"col": "opp_ats_win_pct", "label": "Opp ATS %"},
        ],
        "scoring": [
            {"col": "team_ppg", "label": "Team Off PPG"},
            {"col": "opp_ppg", "label": "Team Def PPG (pts allowed)"},
            {"col": "team_home_ppg", "label": "Team Home PPG"},
            {"col": "team_away_ppg", "label": "Team Away PPG"},
            {"col": "opp_home_ppg", "label": "Opp Home PPG"},
            {"col": "opp_away_ppg", "label": "Opp Away PPG"},
        ],
        "shooting": [
            {"col": "team_ft_pct", "label": "Team FT %"},
            {"col": "team_3pt_pct", "label": "Team 3PT %"},
            {"col": "team_2pt_pct", "label": "Team 2PT %"},
            {"col": "opp_ft_pct", "label": "Opp FT %"},
            {"col": "opp_3pt_pct", "label": "Opp 3PT %"},
            {"col": "opp_2pt_pct", "label": "Opp 2PT %"},
        ],
        "defense": [
            {"col": "team_def_ft_pct", "label": "Team Def FT % (allowed)"},
            {"col": "team_def_3pt_pct", "label": "Team Def 3PT % (allowed)"},
            {"col": "team_def_2pt_pct", "label": "Team Def 2PT % (allowed)"},
            {"col": "opp_def_ft_pct", "label": "Opp Def FT % (allowed)"},
            {"col": "opp_def_3pt_pct", "label": "Opp Def 3PT % (allowed)"},
            {"col": "opp_def_2pt_pct", "label": "Opp Def 2PT % (allowed)"},
        ],
        "rebounding": [
            {"col": "team_oreb_pg", "label": "Team OREB/G"},
            {"col": "team_dreb_pg", "label": "Team DREB/G"},
            {"col": "opp_oreb_pg", "label": "Opp OREB/G"},
            {"col": "opp_dreb_pg", "label": "Opp DREB/G"},
        ],
        "turnovers": [
            {"col": "team_to_pg", "label": "Team TO/G"},
            {"col": "team_forced_to_pg", "label": "Team Forced TO/G"},
            {"col": "opp_to_pg", "label": "Opp TO/G"},
            {"col": "opp_forced_to_pg", "label": "Opp Forced TO/G"},
        ],
        "pace_sos": [
            {"col": "team_pace", "label": "Team Pace"},
            {"col": "team_sos", "label": "Team SOS"},
            {"col": "opp_pace", "label": "Opp Pace"},
            {"col": "opp_sos", "label": "Opp SOS"},
        ],
        "odds": [
            {"col": "opening_fg_spread", "label": "Opening Spread"},
            {"col": "closing_fg_spread", "label": "Closing Spread"},
            {"col": "opening_fg_ml", "label": "Opening ML"},
            {"col": "closing_fg_ml", "label": "Closing ML"},
        ],
        "location": [
            {"col": "home_away", "label": "Home/Away"},
            {"col": "conference_game", "label": "Conference Game"},
        ],
        "ranks": [
            {"col": "rank_team_win_pct", "label": "Team Win % Rank"},
            {"col": "rank_team_ppg", "label": "Team PPG Rank"},
            {"col": "rank_team_ats_win_pct", "label": "Team ATS % Rank"},
            {"col": "rank_team_3pt_pct", "label": "Team 3PT % Rank"},
            {"col": "rank_team_ft_pct", "label": "Team FT % Rank"},
            {"col": "rank_team_2pt_pct", "label": "Team 2PT % Rank"},
            {"col": "rank_team_pace", "label": "Team Pace Rank"},
            {"col": "rank_team_sos", "label": "Team SOS Rank"},
            {"col": "rank_team_oreb_pg", "label": "Team OREB/G Rank"},
            {"col": "rank_team_dreb_pg", "label": "Team DREB/G Rank"},
            {"col": "rank_team_to_pg", "label": "Team TO/G Rank"},
            {"col": "rank_team_forced_to_pg", "label": "Team Forced TO/G Rank"},
            {"col": "rank_team_def_3pt_pct", "label": "Team Def 3PT % Rank"},
            {"col": "rank_team_def_2pt_pct", "label": "Team Def 2PT % Rank"},
            {"col": "rank_opp_win_pct", "label": "Opp Win % Rank"},
            {"col": "rank_opp_ppg", "label": "Opp PPG Rank"},
            {"col": "rank_opp_ats_win_pct", "label": "Opp ATS % Rank"},
            {"col": "rank_opp_3pt_pct", "label": "Opp 3PT % Rank"},
            {"col": "rank_opp_ft_pct", "label": "Opp FT % Rank"},
            {"col": "rank_opp_2pt_pct", "label": "Opp 2PT % Rank"},
            {"col": "rank_opp_pace", "label": "Opp Pace Rank"},
            {"col": "rank_opp_sos", "label": "Opp SOS Rank"},
            {"col": "rank_opp_oreb_pg", "label": "Opp OREB/G Rank"},
            {"col": "rank_opp_dreb_pg", "label": "Opp DREB/G Rank"},
            {"col": "rank_opp_to_pg", "label": "Opp TO/G Rank"},
            {"col": "rank_opp_forced_to_pg", "label": "Opp Forced TO/G Rank"},
            {"col": "rank_opp_def_3pt_pct", "label": "Opp Def 3PT % Rank"},
            {"col": "rank_opp_def_2pt_pct", "label": "Opp Def 2PT % Rank"},
        ],
    }


# ---------------------------------------------------------------------------
# Sized Backtest (from NCAAB_Bets engine)
# ---------------------------------------------------------------------------


def _get_engine() -> BacktestEngine:
    """Get a BacktestEngine instance using cached game logs."""
    df = _get_game_logs()
    if df.empty:
        raise HTTPException(500, "No game data loaded")
    return BacktestEngine(df, NCAAB_DIR)


@app.post("/api/backtest/sized", response_model=SizedBacktestResult)
def run_sized_backtest(req: SizedBacktestRequest):
    """Run a backtest with bet sizing strategies (flat, martingale, d'alembert, kelly, units)."""
    engine = _get_engine()
    result = engine.run(req)
    if result is None:
        raise HTTPException(400, "No games matched the selected filters.")
    return result


@app.get("/api/stats")
def get_available_stats():
    """Return all numeric columns available for filtering."""
    df = _get_game_logs()
    if df.empty:
        return []
    testable = _load_testable_games(df)
    numeric_cols = testable.select_dtypes(include=["number"]).columns.tolist()
    return sorted(numeric_cols)


@app.post("/api/strategies/save")
def save_strategy(req: SizedBacktestRequest):
    """Save a strategy configuration for later use."""
    engine = _get_engine()
    engine.save_strategy(req.model_dump())
    return {"status": "saved"}


@app.get("/api/strategies/saved")
def list_saved_strategies():
    """List all saved strategy configurations."""
    engine = _get_engine()
    return engine.load_strategies()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
