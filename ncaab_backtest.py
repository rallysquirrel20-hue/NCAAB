"""
NCAAB ATS Backtesting Framework
================================
Test betting hypotheses against historical spread data.

Usage:
    python ncaab_backtest.py              # Run all built-in strategies
    python ncaab_backtest.py --scan       # Scan all stat differentials for edges

Profitability threshold: >52.4% ATS at -110 juice
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PARQUET_FILE = SCRIPT_DIR / "ncaab_game_logs.parquet"
JUICE = -110  # standard spread juice


def load_testable_games() -> pd.DataFrame:
    """Load games that can be backtested (tracked + has spread + has PIT stats)."""
    df = pd.read_parquet(PARQUET_FILE)
    mask = (
        (df["is_tracked"] == True) &
        df["closing_fg_spread"].notna() &
        df["team_games"].notna() &
        (df["team_games"] > 0)
    )
    df = df[mask].copy()

    # Compute ATS result
    df["margin"] = df["team_score"].astype(float) - df["opp_score"].astype(float)
    df["ats_margin"] = df["margin"] + df["closing_fg_spread"].astype(float)
    df["covered"] = df["ats_margin"] > 0
    df["push"] = df["ats_margin"] == 0

    return df


def backtest(df: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    """Run a backtest on filtered games. Returns stats dict."""
    filtered = df[mask & ~df["push"]]  # exclude pushes
    n = len(filtered)
    if n == 0:
        return {"name": name, "n": 0}

    wins = filtered["covered"].sum()
    losses = n - wins
    win_pct = wins / n

    # "To win" sizing: risk abs(JUICE)/100 per bet to win 1 unit
    risk_per_unit = abs(JUICE) / 100  # 1.1 for -110
    profit = wins - losses * risk_per_unit
    roi = profit / n * 100

    # Binomial test: is win rate significantly different from 50%?
    p_value = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue

    # Confidence interval (Wilson)
    z = 1.96
    denom = 1 + z**2 / n
    center = (win_pct + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((win_pct * (1 - win_pct) + z**2 / (4 * n)) / n) / denom
    ci_low = center - margin
    ci_high = center + margin

    return {
        "name": name,
        "n": n,
        "wins": wins,
        "losses": losses,
        "win_pct": win_pct,
        "roi": roi,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "profitable": win_pct > 0.524,
    }


def print_result(r: dict):
    """Print a single backtest result."""
    if r["n"] == 0:
        print(f"  {r['name']}: no games matched")
        return

    sig = ""
    if r["p_value"] < 0.01:
        sig = " ***"
    elif r["p_value"] < 0.05:
        sig = " **"
    elif r["p_value"] < 0.10:
        sig = " *"

    prof = "+" if r["profitable"] else " "

    print(f"  {prof} {r['name']}")
    print(f"    Record: {r['wins']}-{r['losses']} ({r['n']} games)")
    print(f"    ATS:    {r['win_pct']:.1%}  (CI: {r['ci_low']:.1%} - {r['ci_high']:.1%})")
    print(f"    ROI:    {r['roi']:+.1f}%  |  p-value: {r['p_value']:.4f}{sig}")
    print()


def print_results_table(results: list[dict]):
    """Print results as a compact sorted table."""
    # Filter to those with enough games
    valid = [r for r in results if r["n"] >= 20]
    valid.sort(key=lambda r: r.get("win_pct", 0), reverse=True)

    print(f"{'Strategy':<55} {'N':>5} {'W-L':>9} {'ATS%':>6} {'ROI':>7} {'p-val':>7}")
    print("-" * 95)
    for r in valid:
        sig = "***" if r["p_value"] < 0.01 else "**" if r["p_value"] < 0.05 else "*" if r["p_value"] < 0.10 else ""
        prof = "+" if r["profitable"] else " "
        wl = f"{r['wins']}-{r['losses']}"
        print(f"{prof} {r['name']:<53} {r['n']:>5} {wl:>9} {r['win_pct']:>5.1%} {r['roi']:>+6.1f}% {r['p_value']:>7.4f} {sig}")


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

def run_strategies(df: pd.DataFrame) -> list[dict]:
    """Run all defined strategies and return results."""
    results = []

    # --- Baseline ---
    results.append(backtest(df, pd.Series(True, index=df.index), "Baseline (all games)"))

    # --- Favorites vs Dogs ---
    results.append(backtest(df, df["closing_fg_spread"] < 0, "All favorites"))
    results.append(backtest(df, df["closing_fg_spread"] > 0, "All underdogs"))
    results.append(backtest(df, df["closing_fg_spread"] < -10, "Big favorites (spread < -10)"))
    results.append(backtest(df, df["closing_fg_spread"] > 10, "Big underdogs (spread > +10)"))

    # --- Your hypothesis: strong defense vs weak offense ---
    # Lower def_2pt_pct = better defense (allows fewer made shots)
    # Lower opp_2pt_pct = weaker offense
    for thresh in [5, 10, 15]:
        mask = (
            df["team_def_2pt_pct"].notna() &
            df["opp_2pt_pct"].notna() &
            (df["opp_2pt_pct"] - df["team_def_2pt_pct"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team def 2PT% allows {thresh}+ less than opp shoots"))

    # Team holds opponents to low 3PT% vs opponent that relies on 3s
    for thresh in [3, 5]:
        mask = (
            df["team_def_3pt_pct"].notna() &
            df["opp_3pt_pct"].notna() &
            (df["opp_3pt_pct"] - df["team_def_3pt_pct"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team def 3PT% allows {thresh}+ less than opp shoots"))

    # --- Your hypothesis: free throw edge ---
    for thresh in [3, 5, 8]:
        mask = (
            df["team_ft_pct"].notna() &
            df["opp_ft_pct"].notna() &
            (df["team_ft_pct"] - df["opp_ft_pct"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team FT% {thresh}+ higher than opponent FT%"))

    # Team shoots more FTs per game
    for thresh in [3, 5]:
        mask = (
            df["team_fta_pg"].notna() &
            df["opp_fta_pg"].notna() &
            (df["team_fta_pg"] - df["opp_fta_pg"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team FTA/game {thresh}+ more than opponent"))

    # --- Turnover differential ---
    for thresh in [2, 4]:
        mask = (
            df["team_forced_to_pg"].notna() &
            df["opp_forced_to_pg"].notna() &
            df["team_to_pg"].notna() &
            df["opp_to_pg"].notna() &
            ((df["team_forced_to_pg"] - df["team_to_pg"]) -
             (df["opp_forced_to_pg"] - df["opp_to_pg"]) > thresh)
        )
        results.append(backtest(df, mask,
            f"Turnover margin advantage {thresh}+/game"))

    # --- Pace mismatch ---
    for thresh in [5, 8]:
        mask = (
            df["team_pace"].notna() &
            df["opp_pace"].notna() &
            (df["team_pace"] - df["opp_pace"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team pace {thresh}+ faster than opponent"))
        results.append(backtest(df,
            df["team_pace"].notna() &
            df["opp_pace"].notna() &
            (df["opp_pace"] - df["team_pace"] > thresh),
            f"Team pace {thresh}+ slower than opponent"))

    # --- Rebounding edge ---
    for thresh in [3, 5]:
        mask = (
            df["team_oreb_pg"].notna() &
            df["opp_oreb_pg"].notna() &
            (df["team_oreb_pg"] - df["opp_oreb_pg"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team OREB/game {thresh}+ more than opponent"))

    # --- Win % vs spread ---
    for thresh in [0.15, 0.25]:
        mask = (
            df["team_win_pct"].notna() &
            df["opp_win_pct"].notna() &
            (df["team_win_pct"] - df["opp_win_pct"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team win% {thresh:.0%}+ higher than opponent"))

    # --- ATS momentum ---
    for thresh in [0.55, 0.60]:
        mask = (
            df["team_ats_win_pct"].notna() &
            (df["team_ats_win_pct"] > thresh) &
            (df["team_ats_games"] >= 5)
        )
        results.append(backtest(df, mask,
            f"Team ATS win% > {thresh:.0%} (min 5 games)"))

    # --- SOS edge (lower = tougher schedule) ---
    for thresh in [10, 20]:
        mask = (
            df["team_sos"].notna() &
            df["opp_sos"].notna() &
            (df["opp_sos"] - df["team_sos"] > thresh)
        )
        results.append(backtest(df, mask,
            f"Team SOS {thresh}+ tougher than opponent"))

    # --- Line movement (opening vs closing) ---
    mask = (
        df["opening_fg_spread"].notna() &
        df["closing_fg_spread"].notna()
    )
    move = df["closing_fg_spread"] - df["opening_fg_spread"]
    # Line moved in team's favor (got more points or smaller favorite)
    for thresh in [1, 2, 3]:
        results.append(backtest(df, mask & (move > thresh),
            f"Line moved {thresh}+ pts in team's favor"))
        results.append(backtest(df, mask & (move < -thresh),
            f"Line moved {thresh}+ pts against team"))

    # --- Home/Away splits ---
    results.append(backtest(df, df["home_away"] == "home", "Home teams"))
    results.append(backtest(df, df["home_away"] == "away", "Away teams"))
    results.append(backtest(df, df["home_away"] == "neutral", "Neutral site"))

    # --- Conference games ---
    results.append(backtest(df,
        df["conference_game"].isin([True, "Y"]), "Conference games"))
    results.append(backtest(df,
        ~df["conference_game"].isin([True, "Y"]), "Non-conference games"))

    return results


def scan_differentials(df: pd.DataFrame) -> list[dict]:
    """Automatically scan all stat pairs for edges.

    For each team_X vs opp_X stat, tests multiple thresholds
    to find where team having a higher value predicts ATS covers.
    """
    results = []

    # Find matching team/opp stat pairs
    team_stats = [c for c in df.columns
                  if c.startswith("team_") and c.endswith(("_pct", "_pg", "_pace", "_sos"))]

    for team_col in team_stats:
        opp_col = "opp_" + team_col[5:]  # team_X -> opp_X
        if opp_col not in df.columns:
            continue

        valid = df[team_col].notna() & df[opp_col].notna()
        if valid.sum() < 100:
            continue

        diff = df[team_col].astype(float) - df[opp_col].astype(float)
        stat_name = team_col[5:]  # strip "team_"

        # Test top quartile and bottom quartile of differential
        for pct, label in [(75, "top 25%"), (90, "top 10%")]:
            thresh = np.nanpercentile(diff[valid], pct)
            if thresh == 0:
                continue
            mask = valid & (diff > thresh)
            r = backtest(df, mask, f"team {stat_name} >> opp ({label}, diff > {thresh:.1f})")
            if r["n"] >= 20:
                results.append(r)

        for pct, label in [(25, "bottom 25%"), (10, "bottom 10%")]:
            thresh = np.nanpercentile(diff[valid], pct)
            if thresh == 0:
                continue
            mask = valid & (diff < thresh)
            r = backtest(df, mask, f"team {stat_name} << opp ({label}, diff < {thresh:.1f})")
            if r["n"] >= 20:
                results.append(r)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="NCAAB ATS Backtester")
    parser.add_argument("--scan", action="store_true",
                        help="Scan all stat differentials for edges")
    args = parser.parse_args()

    df = load_testable_games()
    print(f"Loaded {len(df)} testable games "
          f"(need >52.4% to profit at -110)\n")

    # Always run defined strategies
    print("=" * 95)
    print("  DEFINED STRATEGIES")
    print("=" * 95)
    results = run_strategies(df)
    print_results_table(results)

    # Highlight profitable strategies
    profitable = [r for r in results if r.get("profitable") and r["n"] >= 30]
    if profitable:
        print(f"\n{'=' * 95}")
        print(f"  POTENTIALLY PROFITABLE (>52.4% ATS, n >= 30)")
        print(f"{'=' * 95}")
        for r in sorted(profitable, key=lambda x: x["roi"], reverse=True):
            print_result(r)

    # Optional: scan all differentials
    if args.scan:
        print(f"\n{'=' * 95}")
        print(f"  STAT DIFFERENTIAL SCAN")
        print(f"{'=' * 95}")
        scan_results = scan_differentials(df)
        print_results_table(scan_results)

        scan_profitable = [r for r in scan_results
                          if r.get("profitable") and r["n"] >= 30 and r["p_value"] < 0.10]
        if scan_profitable:
            print(f"\n{'=' * 95}")
            print(f"  SCAN: SIGNIFICANT EDGES (p < 0.10, n >= 30, ATS > 52.4%)")
            print(f"{'=' * 95}")
            for r in sorted(scan_profitable, key=lambda x: x["roi"], reverse=True):
                print_result(r)


if __name__ == "__main__":
    main()
