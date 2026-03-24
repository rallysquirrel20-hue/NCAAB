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


# ---------------------------------------------------------------------------
# Backtest engine (adapted from ncaab_backtest.py)
# ---------------------------------------------------------------------------

JUICE = -110


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


@app.get("/api/today/{game_id}/odds")
def get_game_odds(game_id: str):
    """Return odds movement history for a specific game."""
    if not ODDS_HISTORY_FILE.exists():
        return {"game_id": game_id, "snapshots": []}

    # Find the game in today's data to get commence_time and team names
    today = _get_today()
    game = None
    for g in today.get("games", []):
        if g["game_id"] == game_id:
            game = g
            break

    if not game:
        raise HTTPException(404, f"Game {game_id} not found in today's data")

    home_name = game["home"]["display_name"]
    away_name = game["away"]["display_name"]

    # Also try odds API names
    home_odds = ODDS_NAME_MAP.get(game["home"]["name"], game["home"]["name"])
    away_odds = ODDS_NAME_MAP.get(game["away"]["name"], game["away"]["name"])

    # Read odds history — filter for this game's teams
    try:
        df = pd.read_parquet(ODDS_HISTORY_FILE)
    except Exception:
        return {"game_id": game_id, "snapshots": []}

    # Match by home_team/away_team columns
    mask = (
        (df["home_team"].str.lower().isin([
            home_name.lower(), home_odds.lower(),
            game["home"]["name"].lower()
        ]))
        & (df["away_team"].str.lower().isin([
            away_name.lower(), away_odds.lower(),
            game["away"]["name"].lower()
        ]))
    )
    matched = df[mask]

    if matched.empty:
        return {"game_id": game_id, "snapshots": [], "home": home_name, "away": away_name}

    # Group by snapshot_time for line movement
    snapshots = []
    for snap_time, group in matched.groupby("snapshot_time"):
        snap = {"time": snap_time, "bookmakers": {}}
        for _, row in group.iterrows():
            bm = row["bookmaker"]
            if bm not in snap["bookmakers"]:
                snap["bookmakers"][bm] = {}
            mkt = row["market"]
            if mkt not in snap["bookmakers"][bm]:
                snap["bookmakers"][bm][mkt] = []
            snap["bookmakers"][bm][mkt].append({
                "outcome": row["outcome"],
                "price": row.get("price"),
                "point": row.get("point"),
            })
        snapshots.append(snap)

    return {
        "game_id": game_id,
        "home": home_name,
        "away": away_name,
        "snapshot_count": len(snapshots),
        "snapshots": snapshots,
    }


@app.get("/api/today/{game_id}/matchup")
def get_game_matchup(game_id: str):
    """Return PIT stats comparison for both teams in a game."""
    today = _get_today()
    game = None
    for g in today.get("games", []):
        if g["game_id"] == game_id:
            game = g
            break

    if not game:
        raise HTTPException(404, f"Game {game_id} not found")

    df = _get_game_logs()
    if df.empty:
        return {"game_id": game_id, "home": {}, "away": {}}

    result = {"game_id": game_id}
    for side in ("home", "away"):
        name = game[side]["name"]
        team_rows = df[(df["team"] == name) & (df["is_tracked"] == True)]
        if team_rows.empty:
            result[side] = {"name": name, "stats": {}}
            continue

        latest = team_rows.sort_values("date").iloc[-1]
        stats_dict = {"name": name, "conference": CONFERENCE_MAP.get(name, "")}

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
        for key in pit_keys:
            val = latest.get(key)
            if pd.notna(val) and val != "":
                try:
                    stats_dict[key] = round(float(val), 2)
                except (ValueError, TypeError):
                    stats_dict[key] = val

        result[side] = stats_dict

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
    }


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
