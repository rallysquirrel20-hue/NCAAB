"""
BacktestEngine with bet sizing strategies.

Supports: flat, martingale, d'alembert, kelly, and confidence-based unit sizing.
Tracks pushes, equity curve, and max drawdown.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy import stats

from models import StatFilter, SizingConfig, SizedBacktestRequest, SizedBacktestResult


class BacktestEngine:
    """Runs backtests with configurable bet sizing on pre-loaded game data."""

    def __init__(self, df: pd.DataFrame, ncaab_dir: Path):
        self.ncaab_dir = ncaab_dir
        self.strategies_file = ncaab_dir / "strategies.json"
        self._prepare(df)

    def _prepare(self, df: pd.DataFrame):
        self.df = df[
            (df["is_tracked"] == True)
            & df["closing_fg_spread"].notna()
        ].copy()

        if "date" in self.df.columns:
            self.df["date"] = pd.to_datetime(self.df["date"])
            self.df = self.df.sort_values("date")

        self.df["margin"] = self.df["team_score"].astype(float) - self.df["opp_score"].astype(float)
        self.df["ats_margin"] = self.df["margin"] + self.df["closing_fg_spread"].astype(float)
        self.df["covered"] = self.df["ats_margin"] > 0
        self.df["push"] = self.df["ats_margin"] == 0

    def apply_filters(self, filters: List[StatFilter]) -> pd.Series:
        mask = pd.Series(True, index=self.df.index)
        for f in filters:
            if f.stat not in self.df.columns:
                continue
            val = self.df[f.stat].astype(float)
            if f.operator == ">":
                mask &= val > f.value
            elif f.operator == "<":
                mask &= val < f.value
            elif f.operator == "==":
                mask &= val == f.value
            elif f.operator == ">=":
                mask &= val >= f.value
            elif f.operator == "<=":
                mask &= val <= f.value
        return mask

    def run(self, req: SizedBacktestRequest) -> SizedBacktestResult | None:
        mask = self.apply_filters(req.filters)
        test_df = self.df[mask].copy()

        if len(test_df) == 0:
            return None

        bankroll = req.sizing.bankroll
        current_bankroll = bankroll
        equity_curve = [current_bankroll]

        wins, losses, pushes = 0, 0, 0
        total_profit = 0.0
        sequence_multiplier = 1.0

        for _, row in test_df.iterrows():
            stake = self._calc_stake(req.sizing, req.juice, current_bankroll, sequence_multiplier, row)
            payout_mult = 100.0 / abs(req.juice)

            if row["push"]:
                pushes += 1
            elif row["covered"]:
                wins += 1
                profit = stake * payout_mult
                current_bankroll += profit
                total_profit += profit
                sequence_multiplier = 1.0
            else:
                losses += 1
                current_bankroll -= stake
                total_profit -= stake
                if req.sizing.method == "martingale":
                    sequence_multiplier *= 2.0
                elif req.sizing.method == "dalembert":
                    sequence_multiplier += 1.0

            equity_curve.append(current_bankroll)

        n = wins + losses
        win_pct = wins / n if n > 0 else 0
        roi = (total_profit / (n * req.sizing.base_unit)) * 100 if n > 0 else 0
        p_val = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue if n > 0 else 1.0

        peaks = np.maximum.accumulate(equity_curve)
        drawdowns = (np.array(peaks) - np.array(equity_curve)) / np.array(peaks)
        max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

        return SizedBacktestResult(
            n_games=len(test_df),
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_pct=round(win_pct, 4),
            roi=round(roi, 2),
            profit=round(total_profit, 2),
            equity_curve=[round(v, 2) for v in equity_curve],
            drawdown=round(max_drawdown, 4),
            p_value=round(p_val, 6),
        )

    def _calc_stake(self, sizing: SizingConfig, juice: float,
                    current_bankroll: float, seq_mult: float, row) -> float:
        if sizing.method == "flat":
            return sizing.base_unit
        elif sizing.method in ("martingale", "dalembert"):
            return sizing.base_unit * seq_mult
        elif sizing.method == "kelly":
            win_prob = 0.55
            odds = 1.0 / (abs(juice) / 100.0)
            k_pct = (win_prob * (odds + 1) - 1) / odds
            return current_bankroll * k_pct * (sizing.fraction or 1.0)
        elif sizing.method == "units":
            edge = row.get("team_ats_win_pct", 0.5) - 0.5
            units = 1
            if edge > 0.10:
                units = 5
            elif edge > 0.05:
                units = 3
            return sizing.base_unit * units
        return sizing.base_unit

    def save_strategy(self, strategy: dict):
        strategies = []
        if self.strategies_file.exists():
            with open(self.strategies_file) as f:
                strategies = json.load(f)
        strategies.append(strategy)
        with open(self.strategies_file, "w") as f:
            json.dump(strategies, f, indent=2)

    def load_strategies(self) -> list:
        if not self.strategies_file.exists():
            return []
        with open(self.strategies_file) as f:
            return json.load(f)
