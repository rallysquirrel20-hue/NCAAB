"""
NCAAB Score Prediction Model
====================================
Reads ncaab_game_logs.csv, engineers point-in-time features from
cumulative stats, and trains a model to predict final scores for each team.

ATS picks are derived directly from the score model's predicted margin
vs the spread — no separate ATS model needed.

Usage:
    python ncaab_predict.py
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_FILE = SCRIPT_DIR / "ncaab_game_logs.csv"

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_record(rec_str):
    """Parse 'W-L' record string into (wins, losses, total, win_pct)."""
    if pd.isna(rec_str) or rec_str == "":
        return np.nan, np.nan, np.nan, np.nan
    parts = rec_str.strip().split("-")
    if len(parts) != 2:
        return np.nan, np.nan, np.nan, np.nan
    try:
        w, l = int(parts[0]), int(parts[1])
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan
    total = w + l
    pct = w / total if total > 0 else 0.5
    return w, l, total, pct


def parse_ats(ats_str):
    """Parse 'W-L-P' ATS record into (wins, losses, pushes, total, cover_pct)."""
    if pd.isna(ats_str) or ats_str == "":
        return np.nan, np.nan, np.nan, np.nan, np.nan
    parts = ats_str.strip().split("-")
    if len(parts) != 3:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        w, l, p = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    total = w + l + p
    pct = w / total if total > 0 else 0.5
    return w, l, p, total, pct


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _pit_expanding(group_series):
    """Point-in-time expanding mean (excludes current game)."""
    return group_series.shift(1).expanding().mean()


def _pit_roll(group_series, window):
    """Point-in-time rolling mean over last `window` non-NaN values."""
    return group_series.shift(1).rolling(window, min_periods=1).mean()


def build_features(df):
    """Build numeric features from the raw game log DataFrame.

    All stats are point-in-time: only use data available BEFORE each game.
    Features are organized into systematic categories:
      - Venue (home/away/neutral)
      - Offense (overall, conference, home/away splits, form)
      - Defense (overall, conference, home/away splits, form)
      - Win % (overall, conference, home/away)
      - ATS % (overall, conference, home/away)
      - Form (rolling 3 and 5 game windows)
      - Market (spread, moneyline)
      - Matchup (offense vs defense interactions)
    """
    df = df.copy()

    # Sort chronologically per team
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Team", "Date"]).reset_index(drop=True)

    # =====================================================================
    # PARSE RAW COLUMNS
    # =====================================================================

    # --- Parse record columns ---
    for prefix in ["Team", "Opp"]:
        for suffix in ["Record", "Home_Record", "Away_Record"]:
            col = f"{prefix}_{suffix}"
            parsed = df[col].apply(parse_record)
            df[f"{col}_wins"] = parsed.apply(lambda x: x[0])
            df[f"{col}_losses"] = parsed.apply(lambda x: x[1])
            df[f"{col}_games"] = parsed.apply(lambda x: x[2])
            df[f"{col}_pct"] = parsed.apply(lambda x: x[3])

    # --- Parse ATS columns ---
    for prefix in ["Team", "Opp"]:
        for suffix in ["ATS", "Home_ATS", "Away_ATS"]:
            col = f"{prefix}_{suffix}"
            parsed = df[col].apply(parse_ats)
            df[f"{col}_cover_pct"] = parsed.apply(lambda x: x[4])

    # --- PPG features (from CSV — point-in-time cumulative averages) ---
    for col in ["Team_PPG", "Team_Home_PPG", "Team_Away_PPG",
                "Opp_PPG", "Opp_Home_PPG", "Opp_Away_PPG"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # =====================================================================
    # 1. VENUE
    # =====================================================================
    df["is_home"] = (df["Home_Away"] == "home").astype(int)
    df["is_away"] = (df["Home_Away"] == "away").astype(int)
    df["is_neutral"] = (df["Home_Away"] == "neutral").astype(int)
    df["is_conf"] = (df["Conference_Game"] == "Y").astype(int)

    # =====================================================================
    # 2. MARKET (spread, moneyline)
    # =====================================================================
    df["spread"] = pd.to_numeric(df["Closing_FG_Spread"], errors="coerce")
    df["spread_open"] = pd.to_numeric(df["Opening_FG_Spread"], errors="coerce")
    df["spread_move"] = df["spread"] - df["spread_open"]
    df["ml"] = pd.to_numeric(df["Closing_FG_ML"], errors="coerce")

    def ml_to_prob(ml_val):
        if pd.isna(ml_val):
            return np.nan
        if ml_val < 0:
            return abs(ml_val) / (abs(ml_val) + 100)
        else:
            return 100 / (ml_val + 100)

    df["ml_implied_prob"] = df["ml"].apply(ml_to_prob)

    # =====================================================================
    # 3. CORE STATS — base columns for splits
    # =====================================================================
    df["margin"] = df["Team_Final_Score"] - df["Opp_Final_Score"]
    conf_mask = df["Conference_Game"] == "Y"
    home_mask = df["Home_Away"] == "home"
    away_mask = df["Home_Away"] == "away"

    # ATS covered flag (for computing ATS splits)
    _ats_margin = df["Team_Final_Score"] + df["spread"] - df["Opp_Final_Score"]
    df["_covered"] = (_ats_margin > 0).astype(float)
    df.loc[df["spread"].isna(), "_covered"] = np.nan

    # Win flag
    df["_won"] = (df["margin"] > 0).astype(float)

    # =====================================================================
    # 4. OFFENSE — PPG (overall, conference, home, away)
    # =====================================================================
    # Overall (season-long)
    df["off_ppg"] = df.groupby("Team")["Team_Final_Score"].transform(_pit_expanding)
    # Conference only
    df["_conf_pts"] = df["Team_Final_Score"].where(conf_mask)
    df["off_ppg_conf"] = df.groupby("Team")["_conf_pts"].transform(_pit_expanding)
    # Home only
    df["_home_pts"] = df["Team_Final_Score"].where(home_mask)
    df["off_ppg_home"] = df.groupby("Team")["_home_pts"].transform(_pit_expanding)
    # Away only
    df["_away_pts"] = df["Team_Final_Score"].where(away_mask)
    df["off_ppg_away"] = df.groupby("Team")["_away_pts"].transform(_pit_expanding)

    # =====================================================================
    # 5. DEFENSE — Opp PPG allowed (overall, conference, home, away)
    # =====================================================================
    df["def_ppg"] = df.groupby("Team")["Opp_Final_Score"].transform(_pit_expanding)
    df["_conf_opp_pts"] = df["Opp_Final_Score"].where(conf_mask)
    df["def_ppg_conf"] = df.groupby("Team")["_conf_opp_pts"].transform(_pit_expanding)
    df["_home_opp_pts"] = df["Opp_Final_Score"].where(home_mask)
    df["def_ppg_home"] = df.groupby("Team")["_home_opp_pts"].transform(_pit_expanding)
    df["_away_opp_pts"] = df["Opp_Final_Score"].where(away_mask)
    df["def_ppg_away"] = df.groupby("Team")["_away_opp_pts"].transform(_pit_expanding)

    # =====================================================================
    # 6. WIN % (overall, conference, home, away)
    # =====================================================================
    df["win_pct"] = df.groupby("Team")["_won"].transform(_pit_expanding)
    df["_won_conf"] = df["_won"].where(conf_mask)
    df["win_pct_conf"] = df.groupby("Team")["_won_conf"].transform(_pit_expanding)
    df["_won_home"] = df["_won"].where(home_mask)
    df["win_pct_home"] = df.groupby("Team")["_won_home"].transform(_pit_expanding)
    df["_won_away"] = df["_won"].where(away_mask)
    df["win_pct_away"] = df.groupby("Team")["_won_away"].transform(_pit_expanding)

    # =====================================================================
    # 7. ATS % (overall, conference, home, away)
    # =====================================================================
    df["ats_pct"] = df.groupby("Team")["_covered"].transform(_pit_expanding)
    df["_cov_conf"] = df["_covered"].where(conf_mask)
    df["ats_pct_conf"] = df.groupby("Team")["_cov_conf"].transform(_pit_expanding)
    df["_cov_home"] = df["_covered"].where(home_mask)
    df["ats_pct_home"] = df.groupby("Team")["_cov_home"].transform(_pit_expanding)
    df["_cov_away"] = df["_covered"].where(away_mask)
    df["ats_pct_away"] = df.groupby("Team")["_cov_away"].transform(_pit_expanding)

    # =====================================================================
    # 8. FORM — rolling 3 and 5 game windows
    # =====================================================================
    for window in [3, 5]:
        w = str(window)
        df[f"form{w}_off"] = df.groupby("Team")["Team_Final_Score"].transform(
            lambda x, win=window: _pit_roll(x, win))
        df[f"form{w}_def"] = df.groupby("Team")["Opp_Final_Score"].transform(
            lambda x, win=window: _pit_roll(x, win))
        df[f"form{w}_margin"] = df.groupby("Team")["margin"].transform(
            lambda x, win=window: _pit_roll(x, win))
        df[f"form{w}_ats"] = df.groupby("Team")["_covered"].transform(
            lambda x, win=window: _pit_roll(x, win))
        df[f"form{w}_win"] = df.groupby("Team")["_won"].transform(
            lambda x, win=window: _pit_roll(x, win))

    # Conference-specific rolling form
    for window in [3, 5]:
        w = str(window)
        df[f"form{w}_off_conf"] = df.groupby("Team")["_conf_pts"].transform(
            lambda x, win=window: _pit_roll(x, win))
        df[f"form{w}_def_conf"] = df.groupby("Team")["_conf_opp_pts"].transform(
            lambda x, win=window: _pit_roll(x, win))
        df[f"form{w}_ats_conf"] = df.groupby("Team")["_cov_conf"].transform(
            lambda x, win=window: _pit_roll(x, win))

    # =====================================================================
    # 9. MARGIN (overall, conference)
    # =====================================================================
    df["avg_margin"] = df.groupby("Team")["margin"].transform(_pit_expanding)
    df["_conf_margin"] = df["margin"].where(conf_mask)
    df["avg_margin_conf"] = df.groupby("Team")["_conf_margin"].transform(_pit_expanding)

    # =====================================================================
    # 10. OPPONENT STATS via self-join (mirror rows)
    # =====================================================================
    # Columns to look up from the opponent's side
    opp_cols_to_join = [
        "off_ppg", "off_ppg_conf", "off_ppg_home", "off_ppg_away",
        "def_ppg", "def_ppg_conf", "def_ppg_home", "def_ppg_away",
        "win_pct", "win_pct_conf", "win_pct_home", "win_pct_away",
        "ats_pct", "ats_pct_conf", "ats_pct_home", "ats_pct_away",
        "avg_margin", "avg_margin_conf",
        "form5_off", "form5_def", "form5_margin", "form5_ats",
        "form3_off", "form3_def", "form3_margin", "form3_ats",
        "form5_off_conf", "form5_def_conf", "form5_ats_conf",
    ]

    opp_lookup = df[["Date", "Team_Final_Score", "Opp_Final_Score"]
                     + opp_cols_to_join].copy()
    opp_lookup = opp_lookup.rename(columns={c: f"opp_{c}" for c in opp_cols_to_join})
    opp_lookup = opp_lookup.drop_duplicates(
        subset=["Date", "Team_Final_Score", "Opp_Final_Score"]
    )

    df = df.merge(
        opp_lookup,
        left_on=["Date", "Team_Final_Score", "Opp_Final_Score"],
        right_on=["Date", "Opp_Final_Score", "Team_Final_Score"],
        how="left",
        suffixes=("", "_merge"),
    )
    for c in list(df.columns):
        if c.endswith("_merge"):
            df = df.drop(columns=[c])

    # =====================================================================
    # 11. MATCHUP FEATURES — offense vs defense interactions
    # =====================================================================
    # Team offense vs opponent defense
    df["off_vs_def"] = df["off_ppg"] - df["opp_def_ppg"]
    df["off_vs_def_conf"] = df["off_ppg_conf"] - df["opp_def_ppg_conf"]
    df["off_vs_def_form5"] = df["form5_off"] - df["opp_form5_def"]
    df["off_vs_def_form3"] = df["form3_off"] - df["opp_form3_def"]
    # Opponent offense vs team defense
    df["opp_off_vs_def"] = df["opp_off_ppg"] - df["def_ppg"]
    df["opp_off_vs_def_conf"] = df["opp_off_ppg_conf"] - df["def_ppg_conf"]
    # PPG differentials
    df["ppg_diff"] = df["off_ppg"] - df["opp_off_ppg"]
    df["ppg_diff_conf"] = df["off_ppg_conf"] - df["opp_off_ppg_conf"]

    # =====================================================================
    # 12. CONFERENCE-WEIGHTED BLENDS
    # =====================================================================
    # For conference/tournament games, conference stats are more predictive.
    # Create blended features: 70% conference + 30% overall when conf data
    # exists, otherwise fall back to overall.
    CONF_WEIGHT = 0.7

    def _blend(overall, conf):
        """Blend conference and overall stats, favoring conference."""
        return conf.fillna(overall) * CONF_WEIGHT + overall * (1 - CONF_WEIGHT)

    df["blend_off"] = _blend(df["off_ppg"], df["off_ppg_conf"])
    df["blend_def"] = _blend(df["def_ppg"], df["def_ppg_conf"])
    df["blend_margin"] = _blend(df["avg_margin"], df["avg_margin_conf"])
    df["blend_ats"] = _blend(df["ats_pct"], df["ats_pct_conf"])
    df["blend_win"] = _blend(df["win_pct"], df["win_pct_conf"])
    # Opponent blends
    df["opp_blend_off"] = _blend(df["opp_off_ppg"], df["opp_off_ppg_conf"])
    df["opp_blend_def"] = _blend(df["opp_def_ppg"], df["opp_def_ppg_conf"])
    df["opp_blend_margin"] = _blend(df["opp_avg_margin"], df["opp_avg_margin_conf"])
    df["opp_blend_ats"] = _blend(df["opp_ats_pct"], df["opp_ats_pct_conf"])
    # Blended matchup
    df["blend_off_vs_def"] = df["blend_off"] - df["opp_blend_def"]
    df["blend_opp_off_vs_def"] = df["opp_blend_off"] - df["blend_def"]
    df["blend_ppg_diff"] = df["blend_off"] - df["opp_blend_off"]
    df["blend_margin_diff"] = df["blend_margin"] - df["opp_blend_margin"]

    # =====================================================================
    # CLEANUP temp columns
    # =====================================================================
    temp_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=temp_cols)

    # =====================================================================
    # TARGET VARIABLES
    # =====================================================================
    df["total_points"] = df["Team_Final_Score"] + df["Opp_Final_Score"]
    df["ats_margin"] = df["Team_Final_Score"] + df["spread"] - df["Opp_Final_Score"]
    df["covered"] = (df["ats_margin"] > 0).astype(int)
    df["ats_push"] = (df["ats_margin"] == 0).astype(int)

    return df


# ---------------------------------------------------------------------------
# Model feature selection
# ---------------------------------------------------------------------------

SCORE_FEATURES = [
    # --- Venue ---
    "is_home", "is_away", "is_neutral", "is_conf",
    # --- Market ---
    "spread", "spread_open", "ml_implied_prob",
    # --- Conference-weighted blends (70% conf / 30% overall — single number per stat) ---
    "blend_off", "blend_def", "blend_margin", "blend_ats", "blend_win",
    "opp_blend_off", "opp_blend_def", "opp_blend_margin", "opp_blend_ats",
    "blend_off_vs_def", "blend_opp_off_vs_def", "blend_ppg_diff", "blend_margin_diff",
    # --- Form: rolling 5 ---
    "form5_off", "form5_def", "form5_margin", "form5_ats", "form5_win",
    # --- Form: rolling 3 ---
    "form3_off", "form3_def", "form3_margin", "form3_ats", "form3_win",
    # --- Opponent form ---
    "opp_form5_off", "opp_form5_def", "opp_form5_margin", "opp_form5_ats",
    "opp_form3_off", "opp_form3_def", "opp_form3_margin", "opp_form3_ats",
]


# ATS confidence thresholds (points of predicted margin vs spread)
# These define how far the predicted margin must exceed the spread
# to trigger a pick at each confidence level.
ATS_STRONG_THRESHOLD = 4.0   # 4+ points beyond spread = STRONG pick
ATS_LEAN_THRESHOLD = 1.5     # 1.5-4 points beyond spread = LEAN pick


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(df):
    """Train score models and evaluate. ATS is derived from margin vs spread."""
    df = df.copy()

    # Filter to rows that have spread info and enough history
    mask = (
        df["spread"].notna()
        & df["off_ppg"].notna()
        & df["avg_margin"].notna()
        & (df["Team_Record_games"] >= 3)
    )
    model_df = df[mask].copy().reset_index(drop=True)
    print(f"Games with usable features: {len(model_df)}")

    # Fill remaining NaNs in features
    for col in SCORE_FEATURES:
        if col in model_df.columns:
            model_df[col] = model_df[col].fillna(model_df[col].median())

    # --- Time-based split: use last 20% of games as test ---
    model_df = model_df.sort_values("Date").reset_index(drop=True)
    split_idx = int(len(model_df) * 0.8)
    train = model_df.iloc[:split_idx]
    test = model_df.iloc[split_idx:]
    print(f"Train: {len(train)} games | Test: {len(test)} games")
    print(f"Train dates: {train['Date'].min().date()} to {train['Date'].max().date()}")
    print(f"Test dates:  {test['Date'].min().date()} to {test['Date'].max().date()}")

    # =====================================================================
    # SCORE PREDICTION (Team Score & Opponent Score)
    # =====================================================================
    print("\n" + "=" * 70)
    print("SCORE PREDICTION MODEL")
    print("=" * 70)

    X_train_s = train[SCORE_FEATURES].values
    X_test_s = test[SCORE_FEATURES].values

    scaler_s = StandardScaler()
    X_train_s = scaler_s.fit_transform(X_train_s)
    X_test_s = scaler_s.transform(X_test_s)

    # --- Team Score ---
    y_train_team = train["Team_Final_Score"].values
    y_test_team = test["Team_Final_Score"].values

    model_team = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model_team.fit(X_train_s, y_train_team)
    pred_team = model_team.predict(X_test_s)

    mae_team = mean_absolute_error(y_test_team, pred_team)
    rmse_team = np.sqrt(mean_squared_error(y_test_team, pred_team))
    print(f"\nTeam Score Prediction:")
    print(f"  MAE:  {mae_team:.2f} points")
    print(f"  RMSE: {rmse_team:.2f} points")

    # --- Opponent Score ---
    y_train_opp = train["Opp_Final_Score"].values
    y_test_opp = test["Opp_Final_Score"].values

    model_opp = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    model_opp.fit(X_train_s, y_train_opp)
    pred_opp = model_opp.predict(X_test_s)

    mae_opp = mean_absolute_error(y_test_opp, pred_opp)
    rmse_opp = np.sqrt(mean_squared_error(y_test_opp, pred_opp))
    print(f"\nOpponent Score Prediction:")
    print(f"  MAE:  {mae_opp:.2f} points")
    print(f"  RMSE: {rmse_opp:.2f} points")

    # --- Combined margin prediction ---
    pred_margin = pred_team - pred_opp
    actual_margin = y_test_team - y_test_opp
    mae_margin = mean_absolute_error(actual_margin, pred_margin)
    print(f"\nMargin Prediction:")
    print(f"  MAE:  {mae_margin:.2f} points")

    # Straight-up winner accuracy from score prediction
    su_correct = np.sum((pred_margin > 0) == (actual_margin > 0))
    su_acc = su_correct / len(actual_margin)
    print(f"  Straight-Up Winner Accuracy: {su_acc:.1%}")

    # =====================================================================
    # ATS EVALUATION (margin-vs-spread rule, no separate model)
    # =====================================================================
    print("\n" + "=" * 70)
    print("ATS EVALUATION (Margin vs Spread — No Separate Model)")
    print("=" * 70)

    test_spreads = test["spread"].values
    test_covered = test["covered"].values
    test_push = test["ats_push"].values
    pred_margin_vs_spread = pred_margin - test_spreads

    # Remove pushes
    non_push = test_push == 0
    pmvs = pred_margin_vs_spread[non_push]
    actual_cov = test_covered[non_push]

    # Simple rule: if pred_margin > spread (i.e. pmvs > 0), predict cover
    ats_pred = (pmvs > 0).astype(int)
    ats_acc = accuracy_score(actual_cov, ats_pred)
    print(f"\nOverall ATS Accuracy (margin > spread): {ats_acc:.1%}")

    # Baseline
    base_rate = actual_cov.mean()
    print(f"Test Set Cover Rate (baseline): {base_rate:.1%}")

    # Accuracy by edge size (how far predicted margin is from the spread)
    print("\nATS Accuracy by Edge Size:")
    print(f"  {'Edge Bucket':<25} {'Games':>6} {'Accuracy':>10} {'Cover Rate':>12}")
    print(f"  {'-'*55}")
    for lo, hi, label in [
        (ATS_STRONG_THRESHOLD, 99, f"STRONG Cover (>{ATS_STRONG_THRESHOLD:.1f} pts)"),
        (ATS_LEAN_THRESHOLD, ATS_STRONG_THRESHOLD, f"LEAN Cover ({ATS_LEAN_THRESHOLD:.1f}-{ATS_STRONG_THRESHOLD:.1f} pts)"),
        (-ATS_LEAN_THRESHOLD, ATS_LEAN_THRESHOLD, f"Toss-Up (<{ATS_LEAN_THRESHOLD:.1f} pts)"),
        (-ATS_STRONG_THRESHOLD, -ATS_LEAN_THRESHOLD, f"LEAN No Cover"),
        (-99, -ATS_STRONG_THRESHOLD, f"STRONG No Cover"),
    ]:
        bucket_mask = (pmvs >= lo) & (pmvs < hi)
        n = bucket_mask.sum()
        if n > 0:
            bucket_acc = accuracy_score(actual_cov[bucket_mask], ats_pred[bucket_mask])
            actual_cover = actual_cov[bucket_mask].mean()
            print(f"  {label:<25} {n:>6} {bucket_acc:>10.1%} {actual_cover:>12.1%}")

    # =====================================================================
    # FEATURE IMPORTANCE
    # =====================================================================
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Score Prediction - Team Score)")
    print("=" * 70)
    feat_imp = sorted(zip(SCORE_FEATURES, model_team.feature_importances_),
                      key=lambda x: -x[1])
    for fname, imp in feat_imp[:15]:
        bar = "█" * int(imp * 200)
        print(f"  {fname:<30} {imp:.4f}  {bar}")

    # =====================================================================
    # SAMPLE PREDICTIONS ON TEST SET
    # =====================================================================
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS vs ACTUALS (last 20 test games)")
    print("=" * 70)
    sample = test.tail(20).copy()
    sample_X = scaler_s.transform(sample[SCORE_FEATURES].values)
    sample_pred_team = model_team.predict(sample_X)
    sample_pred_opp = model_opp.predict(sample_X)
    sample_pred_margin = sample_pred_team - sample_pred_opp
    sample_pmvs = sample_pred_margin - sample["spread"].values

    print(f"\n  {'Team':<18} {'Opp':<18} {'Pred':>10} {'Actual':>10} {'Spread':>7} {'Edge':>6} {'ATS':>5} {'Result':>7}")
    print(f"  {'-'*82}")
    for i, (_, row) in enumerate(sample.iterrows()):
        pred_sc = f"{sample_pred_team[i]:.0f}-{sample_pred_opp[i]:.0f}"
        actual_sc = f"{row['Team_Final_Score']:.0f}-{row['Opp_Final_Score']:.0f}"
        spread = row["spread"]
        edge = sample_pmvs[i]
        ats_call = "COV" if edge > 0 else "NO"
        actual_cov_str = "COV" if row["covered"] == 1 else "NO"
        right = "✓" if ats_call == actual_cov_str else "✗"
        team_short = row["Team"][:16]
        opp_short = str(row["Opponent"])[:16]
        print(f"  {team_short:<18} {opp_short:<18} {pred_sc:>10} {actual_sc:>10} {spread:>+7.1f} {edge:>+6.1f} {ats_call:>5} {right:>7}")

    # =====================================================================
    # CROSS-VALIDATION
    # =====================================================================
    print("\n" + "=" * 70)
    print("TIME-SERIES CROSS-VALIDATION (5 folds)")
    print("=" * 70)

    all_X_score = model_df[SCORE_FEATURES].values
    all_y_team = model_df["Team_Final_Score"].values
    all_y_opp = model_df["Opp_Final_Score"].values
    all_y_cov = model_df["covered"].values
    all_spreads = model_df["spread"].values
    all_push = model_df["ats_push"].values

    tscv = TimeSeriesSplit(n_splits=5)
    cv_mae = []
    cv_ats_acc = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(all_X_score)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(all_X_score[tr_idx])
        Xte = sc.transform(all_X_score[te_idx])

        m_team = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=42)
        m_team.fit(Xtr, all_y_team[tr_idx])
        p_team = m_team.predict(Xte)
        cv_mae.append(mean_absolute_error(all_y_team[te_idx], p_team))

        m_opp = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, subsample=0.8,
                                           random_state=42)
        m_opp.fit(Xtr, all_y_opp[tr_idx])
        p_opp = m_opp.predict(Xte)

        # ATS via margin-vs-spread rule
        p_margin = p_team - p_opp
        pmvs_cv = p_margin - all_spreads[te_idx]
        non_push_cv = all_push[te_idx] == 0
        ats_pred_cv = (pmvs_cv[non_push_cv] > 0).astype(int)
        cv_ats_acc.append(accuracy_score(all_y_cov[te_idx][non_push_cv], ats_pred_cv))

        print(f"  Fold {fold+1}: Score MAE = {cv_mae[-1]:.2f}, ATS Acc = {cv_ats_acc[-1]:.1%}")

    print(f"\n  Average Score MAE: {np.mean(cv_mae):.2f} ± {np.std(cv_mae):.2f}")
    print(f"  Average ATS Acc:   {np.mean(cv_ats_acc):.1%} ± {np.std(cv_ats_acc):.1%}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Dataset: {len(df)} game logs across {df['Team'].nunique()} teams (2025-26 season)
Model games (with sufficient features): {len(model_df)}
Train/Test split: {len(train)}/{len(test)} (80/20 time-based)

SCORE PREDICTION (Gradient Boosting Regressor):
  Team Score  -> MAE: {mae_team:.2f}, RMSE: {rmse_team:.2f}
  Opp Score   -> MAE: {mae_opp:.2f}, RMSE: {rmse_opp:.2f}
  Margin      -> MAE: {mae_margin:.2f}
  Straight-Up -> {su_acc:.1%} accuracy

ATS (Margin vs Spread rule — no separate model):
  Accuracy: {ats_acc:.1%} (baseline ~50%)
  Method: If predicted margin > spread, take the cover.
  Confidence from edge size (how many points beyond the spread).
""")

    return {
        "model_team": model_team,
        "model_opp": model_opp,
        "scaler_score": scaler_s,
        "df": model_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    df = pd.read_csv(CSV_FILE)
    print(f"Loaded {len(df)} game records\n")

    print("Engineering features...")
    df = build_features(df)

    print("Training models and evaluating...\n")
    results = train_and_evaluate(df)
