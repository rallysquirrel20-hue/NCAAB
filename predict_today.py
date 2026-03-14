"""
Run predictions for today's games (March 14, 2026) using the trained model.
"""
import sys
import numpy as np
import pandas as pd
from ncaab_predict import build_features, SCORE_FEATURES, ATS_FEATURES
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Load and prepare data ──────────────────────────────────────────────────
df = pd.read_csv("ncaab_game_logs.csv")
df = build_features(df)

# Filter to usable rows for training
mask = (
    df["spread"].notna()
    & df["Team_PPG"].notna()
    & df["team_avg_margin"].notna()
    & (df["Team_Record_games"] >= 3)
)
train_df = df[mask].copy().sort_values("Date").reset_index(drop=True)
for col in SCORE_FEATURES + ATS_FEATURES:
    if col in train_df.columns:
        train_df[col] = train_df[col].fillna(train_df[col].median())

print(f"Training on {len(train_df)} historical games...\n")

# ── Train models on ALL available data ─────────────────────────────────────
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

scaler_a = StandardScaler()
train_ats = train_df[train_df["ats_push"] == 0]
X_a = scaler_a.fit_transform(train_ats[ATS_FEATURES].values)

ats_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
ats_model.fit(X_a, train_ats["covered"].values)

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
    # Big Ten Tournament Semifinals
    ("Michigan",  "Wisconsin", "neutral", "N", -12.5, -12.5, -893),
    ("Wisconsin", "Michigan",  "neutral", "N",  12.5,  12.5,  581),
    ("Purdue",    "UCLA",      "neutral", "N",  -7.5,  -8.0, -310),
    ("UCLA",      "Purdue",    "neutral", "N",   7.5,   8.0,  290),
    # SEC Tournament Semifinal
    ("Florida",    "Vanderbilt", "neutral", "N", -8.5, -8.5, -395),
    ("Vanderbilt", "Florida",    "neutral", "N",  8.5,  8.5,  310),
    # ACC Tournament Championship
    ("Duke",     "Virginia", "neutral", "N", -8.5, -8.5, -365),
    ("Virginia", "Duke",     "neutral", "N",  8.5,  8.5,  281),
    # Big 12 Tournament Championship
    ("Arizona",  "Houston",  "neutral", "N", -2.5, -2.5, -150),
    ("Houston",  "Arizona",  "neutral", "N",  2.5,  2.5,  125),
    # Big East Tournament Championship
    ("UConn",      "St. John's", "neutral", "N", -3.0, -2.5, -140),
    ("St. John's", "UConn",      "neutral", "N",  3.0,  2.5,  116),
]

print("=" * 85)
print("PREDICTIONS FOR MARCH 14, 2026 — CONFERENCE TOURNAMENT DAY")
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
    feat = {
        "is_home": 0,
        "is_away": 0,
        "is_neutral": 1,
        "is_conf": 0,
        "spread": spread,
        "spread_move": spread - spread_open,
        "ml_implied_prob": ml_to_prob(ml),
        "Team_Record_pct": team_stats.get("Team_Record_pct", 0.5),
        "Team_Home_Record_pct": team_stats.get("Team_Home_Record_pct", 0.5),
        "Team_Away_Record_pct": team_stats.get("Team_Away_Record_pct", 0.5),
        "Team_Record_games": team_stats.get("Team_Record_games", 20),
        "Opp_Record_pct": opp_stats.get("Team_Record_pct", 0.5),
        "Opp_Home_Record_pct": opp_stats.get("Team_Home_Record_pct", 0.5),
        "Opp_Away_Record_pct": opp_stats.get("Team_Away_Record_pct", 0.5),
        "Team_PPG": team_stats.get("Team_PPG", 75),
        "Team_Home_PPG": team_stats.get("Team_Home_PPG", 75),
        "Team_Away_PPG": team_stats.get("Team_Away_PPG", 75),
        "Opp_PPG": opp_stats.get("Team_PPG", 70),
        "Opp_Home_PPG": opp_stats.get("Team_Home_PPG", 70),
        "Opp_Away_PPG": opp_stats.get("Team_Away_PPG", 70),
        "ppg_diff": team_stats.get("Team_PPG", 75) - opp_stats.get("Team_PPG", 70),
        "team_avg_margin": team_stats.get("team_avg_margin", 0),
        "team_avg_score": team_stats.get("team_avg_score", 75),
        "team_avg_opp_score": team_stats.get("team_avg_opp_score", 70),
        "team_recent_margin": team_stats.get("team_recent_margin", 0),
        "team_recent_score": team_stats.get("team_recent_score", 75),
        "Team_ATS_cover_pct": team_stats.get("Team_ATS_cover_pct", 0.5),
        "Team_Home_ATS_cover_pct": team_stats.get("Team_Home_ATS_cover_pct", 0.5),
        "Team_Away_ATS_cover_pct": team_stats.get("Team_Away_ATS_cover_pct", 0.5),
        "Opp_ATS_cover_pct": opp_stats.get("Team_ATS_cover_pct", 0.5),
        "spread_open": spread_open,
    }
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

    # Predict from favorite's perspective
    X_fav_s = scaler_s.transform([[fav_feat[f] for f in SCORE_FEATURES]])
    X_fav_a = scaler_a.transform([[fav_feat[f] for f in ATS_FEATURES]])
    fav_pred_team = model_team.predict(X_fav_s)[0]
    fav_pred_opp = model_opp.predict(X_fav_s)[0]
    fav_ats_prob = ats_model.predict_proba(X_fav_a)[0][1]

    # Predict from underdog's perspective
    X_dog_a = scaler_a.transform([[dog_feat[f] for f in ATS_FEATURES]])
    dog_ats_prob = ats_model.predict_proba(X_dog_a)[0][1]

    games_output.append({
        "fav": fav_name, "dog": dog_name,
        "fav_spread": fav_info[4], "dog_spread": dog_info[4],
        "pred_fav_score": fav_pred_team, "pred_dog_score": fav_pred_opp,
        "pred_margin": fav_pred_team - fav_pred_opp,
        "fav_ats_prob": fav_ats_prob, "dog_ats_prob": dog_ats_prob,
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
    print(f"  Model vs Vegas:   Model has {fav} winning by {abs(g['pred_margin']):.1f} vs line of {abs(g['fav_spread']):.1f}")

    # ATS call
    print()
    print(f"  ATS Analysis:")
    print(f"    {fav:<15} cover prob: {g['fav_ats_prob']:.1%}")
    print(f"    {dog:<15} cover prob: {g['dog_ats_prob']:.1%}")

    # Determine pick
    if g["fav_ats_prob"] > 0.55:
        pick = f"{fav} {g['fav_spread']:+.1f}"
        conf = g["fav_ats_prob"]
        strength = "STRONG" if g["fav_ats_prob"] > 0.6 else "LEAN"
    elif g["dog_ats_prob"] > 0.55:
        pick = f"{dog} {g['dog_spread']:+.1f}"
        conf = g["dog_ats_prob"]
        strength = "STRONG" if g["dog_ats_prob"] > 0.6 else "LEAN"
    else:
        pick = "NO EDGE (Toss-up)"
        conf = max(g["fav_ats_prob"], g["dog_ats_prob"])
        strength = "PASS"

    winner = fav if g["pred_margin"] > 0 else dog
    print()
    print(f"  >> STRAIGHT-UP PICK: {winner}")
    print(f"  >> ATS PICK:         {pick}  [{strength}] ({conf:.1%} confidence)")

print(f"\n{'=' * 85}")
print("DISCLAIMER: Model predictions for informational purposes only.")
print("ATS accuracy in backtesting: ~52%. High-confidence picks (>55%): ~67%.")
print(f"{'=' * 85}")
