"""
Run predictions for today's games (March 15, 2026) using the trained model.
ATS picks derived directly from predicted margin vs spread — no separate ATS model.
"""
import sys
import numpy as np
import pandas as pd
from ncaab_predict import build_features, SCORE_FEATURES, ATS_STRONG_THRESHOLD, ATS_LEAN_THRESHOLD
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ── Load and prepare data ──────────────────────────────────────────────────
df = pd.read_csv("ncaab_game_logs.csv")
df = build_features(df)

# Filter to usable rows for training
mask = (
    df["spread"].notna()
    & df["off_ppg"].notna()
    & df["avg_margin"].notna()
    & (df["Team_Record_games"] >= 3)
)
train_df = df[mask].copy().sort_values("Date").reset_index(drop=True)
for col in SCORE_FEATURES:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df[col].median())

print(f"Training on {len(train_df)} historical games...\n")

# ── Train score models on ALL available data ──────────────────────────────
scaler_s = StandardScaler()
X_s = scaler_s.fit_transform(train_df[SCORE_FEATURES].values)

model_team = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
model_team.fit(X_s, train_df["Team_Final_Score"].values)

model_opp = GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
model_opp.fit(X_s, train_df["Opp_Final_Score"].values)

# ── Helper: get latest stats for a team ────────────────────────────────────
def get_latest_team_stats(team_name):
    """Pull the most recent row for a team to get their current cumulative stats."""
    team_rows = df[df["Team"] == team_name].sort_values("Date")
    if team_rows.empty:
        return None
    return team_rows.iloc[-1]

# ── Today's games ──────────────────────────────────────────────────────────
# Format: (team, opponent, home_away, is_conf, closing_spread, opening_spread, closing_ml)
todays_games = [
    # Big Ten Tournament — Championship
    ("Purdue",   "Michigan", "neutral", "N", -3.5, -3.5, -165),
    ("Michigan", "Purdue",   "neutral", "N",  3.5,  3.5,  140),
]

print("=" * 85)
print("PREDICTIONS FOR MARCH 15, 2026 — CONFERENCE TOURNAMENT DAY")
print("=" * 85)

# Process in pairs (both sides of same game)
def ml_to_prob(ml_val):
    if ml_val < 0:
        return abs(ml_val) / (abs(ml_val) + 100)
    else:
        return 100 / (ml_val + 100)


def build_game_features(team_stats, opp_stats, game_info):
    """Build feature dict for one side of a game."""
    spread = game_info[4]
    spread_open = game_info[5]
    ml = game_info[6]

    def _get(row, col, default=0):
        v = row.get(col, default)
        return default if pd.isna(v) else v

    feat = {
        # --- Venue ---
        "is_home": 0,
        "is_away": 0,
        "is_neutral": 1,
        "is_conf": 0,
        # --- Market ---
        "spread": spread,
        "spread_move": spread - spread_open,
        "ml_implied_prob": ml_to_prob(ml),
        "spread_open": spread_open,
        # --- Team Offense ---
        "off_ppg": _get(team_stats, "off_ppg", 75),
        "off_ppg_conf": _get(team_stats, "off_ppg_conf", 75),
        "off_ppg_home": _get(team_stats, "off_ppg_home", 75),
        "off_ppg_away": _get(team_stats, "off_ppg_away", 75),
        # --- Team Defense ---
        "def_ppg": _get(team_stats, "def_ppg", 70),
        "def_ppg_conf": _get(team_stats, "def_ppg_conf", 70),
        "def_ppg_home": _get(team_stats, "def_ppg_home", 70),
        "def_ppg_away": _get(team_stats, "def_ppg_away", 70),
        # --- Win % ---
        "win_pct": _get(team_stats, "win_pct", 0.5),
        "win_pct_conf": _get(team_stats, "win_pct_conf", 0.5),
        "win_pct_home": _get(team_stats, "win_pct_home", 0.5),
        "win_pct_away": _get(team_stats, "win_pct_away", 0.5),
        # --- ATS % ---
        "ats_pct": _get(team_stats, "ats_pct", 0.5),
        "ats_pct_conf": _get(team_stats, "ats_pct_conf", 0.5),
        "ats_pct_home": _get(team_stats, "ats_pct_home", 0.5),
        "ats_pct_away": _get(team_stats, "ats_pct_away", 0.5),
        # --- Margin ---
        "avg_margin": _get(team_stats, "avg_margin", 0),
        "avg_margin_conf": _get(team_stats, "avg_margin_conf", 0),
        # --- Form rolling 5 ---
        "form5_off": _get(team_stats, "form5_off", 75),
        "form5_def": _get(team_stats, "form5_def", 70),
        "form5_margin": _get(team_stats, "form5_margin", 0),
        "form5_ats": _get(team_stats, "form5_ats", 0.5),
        "form5_win": _get(team_stats, "form5_win", 0.5),
        "form5_off_conf": _get(team_stats, "form5_off_conf", 75),
        "form5_def_conf": _get(team_stats, "form5_def_conf", 70),
        "form5_ats_conf": _get(team_stats, "form5_ats_conf", 0.5),
        # --- Form rolling 3 ---
        "form3_off": _get(team_stats, "form3_off", 75),
        "form3_def": _get(team_stats, "form3_def", 70),
        "form3_margin": _get(team_stats, "form3_margin", 0),
        "form3_ats": _get(team_stats, "form3_ats", 0.5),
        "form3_win": _get(team_stats, "form3_win", 0.5),
        # --- Opponent stats (from opp_stats row's team-side columns) ---
        "opp_off_ppg": _get(opp_stats, "off_ppg", 75),
        "opp_off_ppg_conf": _get(opp_stats, "off_ppg_conf", 75),
        "opp_off_ppg_home": _get(opp_stats, "off_ppg_home", 75),
        "opp_off_ppg_away": _get(opp_stats, "off_ppg_away", 75),
        "opp_def_ppg": _get(opp_stats, "def_ppg", 70),
        "opp_def_ppg_conf": _get(opp_stats, "def_ppg_conf", 70),
        "opp_def_ppg_home": _get(opp_stats, "def_ppg_home", 70),
        "opp_def_ppg_away": _get(opp_stats, "def_ppg_away", 70),
        "opp_win_pct": _get(opp_stats, "win_pct", 0.5),
        "opp_win_pct_conf": _get(opp_stats, "win_pct_conf", 0.5),
        "opp_win_pct_home": _get(opp_stats, "win_pct_home", 0.5),
        "opp_win_pct_away": _get(opp_stats, "win_pct_away", 0.5),
        "opp_ats_pct": _get(opp_stats, "ats_pct", 0.5),
        "opp_ats_pct_conf": _get(opp_stats, "ats_pct_conf", 0.5),
        "opp_ats_pct_home": _get(opp_stats, "ats_pct_home", 0.5),
        "opp_ats_pct_away": _get(opp_stats, "ats_pct_away", 0.5),
        "opp_avg_margin": _get(opp_stats, "avg_margin", 0),
        "opp_avg_margin_conf": _get(opp_stats, "avg_margin_conf", 0),
        "opp_form5_off": _get(opp_stats, "form5_off", 75),
        "opp_form5_def": _get(opp_stats, "form5_def", 70),
        "opp_form5_margin": _get(opp_stats, "form5_margin", 0),
        "opp_form5_ats": _get(opp_stats, "form5_ats", 0.5),
        "opp_form3_off": _get(opp_stats, "form3_off", 75),
        "opp_form3_def": _get(opp_stats, "form3_def", 70),
        "opp_form3_margin": _get(opp_stats, "form3_margin", 0),
        "opp_form3_ats": _get(opp_stats, "form3_ats", 0.5),
        "opp_form5_off_conf": _get(opp_stats, "form5_off_conf", 75),
        "opp_form5_def_conf": _get(opp_stats, "form5_def_conf", 70),
        "opp_form5_ats_conf": _get(opp_stats, "form5_ats_conf", 0.5),
    }

    # --- Matchup interactions (computed from the above) ---
    feat["off_vs_def"] = feat["off_ppg"] - feat["opp_def_ppg"]
    feat["off_vs_def_conf"] = feat["off_ppg_conf"] - feat["opp_def_ppg_conf"]
    feat["off_vs_def_form5"] = feat["form5_off"] - feat["opp_form5_def"]
    feat["off_vs_def_form3"] = feat["form3_off"] - feat["opp_form3_def"]
    feat["opp_off_vs_def"] = feat["opp_off_ppg"] - feat["def_ppg"]
    feat["opp_off_vs_def_conf"] = feat["opp_off_ppg_conf"] - feat["def_ppg_conf"]
    feat["ppg_diff"] = feat["off_ppg"] - feat["opp_off_ppg"]
    feat["ppg_diff_conf"] = feat["off_ppg_conf"] - feat["opp_off_ppg_conf"]

    # --- Conference-weighted blends ---
    CW = 0.7
    feat["blend_off"] = feat["off_ppg_conf"] * CW + feat["off_ppg"] * (1 - CW)
    feat["blend_def"] = feat["def_ppg_conf"] * CW + feat["def_ppg"] * (1 - CW)
    feat["blend_margin"] = feat["avg_margin_conf"] * CW + feat["avg_margin"] * (1 - CW)
    feat["blend_ats"] = feat["ats_pct_conf"] * CW + feat["ats_pct"] * (1 - CW)
    feat["blend_win"] = feat["win_pct_conf"] * CW + feat["win_pct"] * (1 - CW)
    feat["opp_blend_off"] = feat["opp_off_ppg_conf"] * CW + feat["opp_off_ppg"] * (1 - CW)
    feat["opp_blend_def"] = feat["opp_def_ppg_conf"] * CW + feat["opp_def_ppg"] * (1 - CW)
    feat["opp_blend_margin"] = feat["opp_avg_margin_conf"] * CW + feat["opp_avg_margin"] * (1 - CW)
    feat["opp_blend_ats"] = feat["opp_ats_pct_conf"] * CW + feat["opp_ats_pct"] * (1 - CW)
    feat["blend_off_vs_def"] = feat["blend_off"] - feat["opp_blend_def"]
    feat["blend_opp_off_vs_def"] = feat["opp_blend_off"] - feat["blend_def"]
    feat["blend_ppg_diff"] = feat["blend_off"] - feat["opp_blend_off"]
    feat["blend_margin_diff"] = feat["blend_margin"] - feat["opp_blend_margin"]
    # Handle NaNs
    for k, v in feat.items():
        if pd.isna(v):
            feat[k] = train_df[k].median() if k in train_df.columns else 0
    return feat


games_output = []
for i in range(0, len(todays_games), 2):
    fav_info = todays_games[i]
    dog_info = todays_games[i + 1]

    fav_name = fav_info[0]
    dog_name = dog_info[0]

    fav_stats = get_latest_team_stats(fav_name)
    dog_stats = get_latest_team_stats(dog_name)

    if fav_stats is None or dog_stats is None:
        print(f"\nSkipping {fav_name} vs {dog_name} — missing team data")
        continue

    # Build features for both perspectives
    fav_feat = build_game_features(fav_stats, dog_stats, fav_info)
    dog_feat = build_game_features(dog_stats, fav_stats, dog_info)

    # Predict scores from favorite's perspective
    X_fav_s = scaler_s.transform([[fav_feat[f] for f in SCORE_FEATURES]])
    fav_pred_team = model_team.predict(X_fav_s)[0]
    fav_pred_opp = model_opp.predict(X_fav_s)[0]

    # Predict scores from underdog's perspective
    X_dog_s = scaler_s.transform([[dog_feat[f] for f in SCORE_FEATURES]])
    dog_pred_team = model_team.predict(X_dog_s)[0]
    dog_pred_opp = model_opp.predict(X_dog_s)[0]

    # Predicted margin from favorite's perspective
    pred_margin = fav_pred_team - fav_pred_opp

    # Edge = how far predicted margin exceeds the spread
    fav_edge = pred_margin - fav_info[4]   # pred_margin - spread (spread is negative for fav)
    dog_edge = -pred_margin - dog_info[4]   # flip margin for dog perspective

    games_output.append({
        "fav": fav_name, "dog": dog_name,
        "fav_spread": fav_info[4], "dog_spread": dog_info[4],
        "pred_fav_score": fav_pred_team, "pred_dog_score": fav_pred_opp,
        "pred_margin": pred_margin,
        "fav_edge": fav_edge, "dog_edge": dog_edge,
    })

# Print results
for g in games_output:
    fav, dog = g["fav"], g["dog"]

    print(f"\n{'─' * 85}")
    print(f"  {fav.upper()} ({g['fav_spread']:+.1f}) vs {dog.upper()} ({g['dog_spread']:+.1f})")
    print(f"{'─' * 85}")
    print(f"  Predicted Score:  {fav} {g['pred_fav_score']:.0f}  —  {dog} {g['pred_dog_score']:.0f}")
    print(f"  Predicted Margin: {fav} by {g['pred_margin']:.1f} points")
    print(f"  Vegas Spread:     {fav} {g['fav_spread']:+.1f}")

    # ATS analysis using edge (predicted margin vs spread)
    print()
    print(f"  ATS Analysis (predicted margin vs spread):")
    print(f"    {fav:<15} edge: {g['fav_edge']:+.1f} pts beyond spread")
    print(f"    {dog:<15} edge: {g['dog_edge']:+.1f} pts beyond spread")

    # Determine pick based on edge thresholds
    fav_edge = g["fav_edge"]
    dog_edge = g["dog_edge"]

    if fav_edge >= ATS_STRONG_THRESHOLD:
        pick = f"{fav} {g['fav_spread']:+.1f}"
        strength = "STRONG"
        edge = fav_edge
    elif fav_edge >= ATS_LEAN_THRESHOLD:
        pick = f"{fav} {g['fav_spread']:+.1f}"
        strength = "LEAN"
        edge = fav_edge
    elif dog_edge >= ATS_STRONG_THRESHOLD:
        pick = f"{dog} {g['dog_spread']:+.1f}"
        strength = "STRONG"
        edge = dog_edge
    elif dog_edge >= ATS_LEAN_THRESHOLD:
        pick = f"{dog} {g['dog_spread']:+.1f}"
        strength = "LEAN"
        edge = dog_edge
    else:
        pick = "NO EDGE (Toss-up)"
        strength = "PASS"
        edge = max(abs(fav_edge), abs(dog_edge))

    winner = fav if g["pred_margin"] > 0 else dog
    print()
    print(f"  >> STRAIGHT-UP PICK: {winner}")
    print(f"  >> ATS PICK:         {pick}  [{strength}] ({edge:+.1f} pts edge)")

print(f"\n{'=' * 85}")
print("DISCLAIMER: Model predictions for informational purposes only.")
print(f"ATS method: predicted margin vs spread. Thresholds: LEAN={ATS_LEAN_THRESHOLD} pts, STRONG={ATS_STRONG_THRESHOLD} pts.")
print(f"{'=' * 85}")
