# NCAAB March Madness Betting Model

## Overview
College basketball (NCAAB) data pipeline for the 2025-26 season. Builds comprehensive game logs with point-in-time (PIT) statistics and betting odds to power a profitable against-the-spread (ATS) prediction model for March Madness.

## Architecture
Multi-file pipeline with shared config:

- **`ncaab_config.py`** — shared constants: `TEAMS` (68), `CONFERENCE_MAP`, `ESPN_NAME_ALIASES`, `ODDS_NAME_MAP`, `HARDCODED_ESPN_IDS`, API base URLs, rate-limit settings.
- **`ncaab_daily_builder.py`** — primary incremental pipeline (run daily). 4-phase process:
  1. **Phase 1 — Fetch new games** (ESPN scoreboard, 1 call/date, skips already-stored events)
  2. **Phase 2 — Fetch odds** (The Odds API, 2 calls per NEW date only)
  3. **Phase 3 — Compute PIT stats** (pure math, zero API calls — always recomputed)
  4. **Phase 4 — Export CSV** (write final output)
- **`ncaab_data_builder.py`** — legacy full-season builder (single run, uses its own inline team list + gzipped JSON cache). Kept for reference/one-off rebuilds.

## Key Files
- `ncaab_config.py` — single source of truth for teams, conferences, and name mappings
- `ncaab_daily_builder.py` — primary entry point (`python ncaab_daily_builder.py`, supports `--backfill-odds`)
- `ncaab_data_builder.py` — legacy full-rebuild script (`python ncaab_data_builder.py`)
- `ncaab_game_logs.parquet` — Parquet store (primary data, used by daily builder)
- `ncaab_game_logs.csv` — CSV export (36 columns per game row)
- `espn_team_ids.json` — cached ESPN team ID resolution
- `ncaab_cache.json.gz` — legacy compressed API cache (used by data_builder only)
- `test_arizona_daily.py` — test script simulating daily runs for Arizona

## Dependencies
```
pip install requests python-dotenv pandas pyarrow
```
(`tqdm` needed only for legacy `ncaab_data_builder.py`)

## Environment Variables
Requires a `.env` file in the **parent** directory of the script (`PROJECT_DIR = SCRIPT_DIR.parent`), or in the script directory itself (daily builder checks both):
```
THE_ODDS_API_KEY=<your-key>
```

## Data Sources
- **ESPN API** (`site.api.espn.com`) — team directory, scoreboard, game summaries (half scores). No auth required.
- **The Odds API** (`api.the-odds-api.com`) — historical opening/closing odds. Requires API key. Each call costs ~20 credits.

## CSV Output Columns
`Team, Date, Opponent, Home_Away, Conference_Game, Team_Final_Score, Opp_Final_Score, W_L, Opening_FG_ML, Closing_FG_ML, Opening_FG_Spread, Closing_FG_Spread, Team_Record, Team_Home_Record, Team_Neutral_Record, Team_Away_Record, Opp_Record, Opp_Home_Record, Opp_Neutral_Record, Opp_Away_Record, Team_ATS, Team_Home_ATS, Team_Neutral_ATS, Team_Away_ATS, Opp_ATS, Opp_Home_ATS, Opp_Neutral_ATS, Opp_Away_ATS, Team_PPG, Team_Home_PPG, Team_Neutral_PPG, Team_Away_PPG, Opp_PPG, Opp_Home_PPG, Opp_Neutral_PPG, Opp_Away_PPG`

## Team Coverage
68 teams: the confirmed 2025-26 NCAA Tournament field (including First Four). Defined in `ncaab_config.py` → `TEAMS`, organized by region (East, South, Midwest, West). Conference affiliations in `CONFERENCE_MAP`. Opponent PIT stats are only populated when the opponent is one of these 68 teams.

## Caching & Storage
- **Daily builder**: Parquet file (`ncaab_game_logs.parquet`) is the primary store. `espn_team_ids.json` caches team ID resolution. Re-running skips already-stored events.
- **Legacy builder**: Uses `ncaab_cache.json.gz` for all API responses.
- Rate-limited at 1 req/sec with exponential backoff on 429s.

## Important Patterns
- **PIT stats are pre-game**: each row's records/ATS/PPG reflect the team's stats *before* that game, not after.
- **Neutral site handling**: `home_away` is overridden to `"neutral"` when ESPN flags `neutralSite: true`.
- **Name aliasing**: ESPN and Odds API use different team names. `ESPN_NAME_ALIASES` and `ODDS_NAME_MAP` in `ncaab_config.py` handle the translation.
- **Incremental design**: daily builder only fetches scoreboard data for dates not yet covered and odds only for newly added dates. `--backfill-odds` flag re-fetches odds for all dates with missing spread data.
