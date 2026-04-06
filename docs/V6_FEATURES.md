# V6 Model — Feature Reference

## Overview

V6 is the current production model powering daily game outcome forecasts.
It is a **pitcher-venue stacked ensemble** that extends V5 (matchup-aware)
by adding pitcher arsenal quality and batter venue history as new signals.

**Architecture:** LR + XGBoost + LightGBM → meta Logistic Regression  
**Model artifact:** `gs://hanks_tank_data/models/vertex/game_outcome_2026_v6/model.pkl`  
**Cloud Function:** `mlb-2026-daily-pipeline` (us-central1, 2nd gen) — updated 2026-04-03  
**Prediction entry point:** `predict_today_games.py` → `DailyPredictor`, using V6 by default

---

## Feature Groups

### Group 1 — Team Rolling Form (from V3)

Rolling and EMA-weighted win rates computed from recent game history.
Uses combined 2025 historical games + 2026 season games for rolling context.

| Feature | Description |
|---|---|
| `home_win_pct_10d` | Home team win % over last 10 games |
| `away_win_pct_10d` | Away team win % over last 10 games |
| `home_win_pct_30d` | Home team win % over last 30 games |
| `away_win_pct_30d` | Away team win % over last 30 games |
| `home_ema_form` | Home team exponentially-weighted form (span=10) |
| `away_ema_form` | Away team exponentially-weighted form (span=10) |
| `home_form_squared` | `home_ema_form²` — non-linear momentum |
| `away_form_squared` | `away_ema_form²` — non-linear momentum |
| `home_momentum` | `home_win_pct_10d - home_ema_form` (trend direction) |
| `away_momentum` | `away_win_pct_10d - away_ema_form` (trend direction) |
| `win_pct_diff` | `home_win_pct_30d - away_win_pct_30d` |
| `form_difference` | `home_ema_form - away_ema_form` |
| `form_interaction` | `home_ema_form × away_ema_form` |
| `trend_alignment` | Combined trend direction indicator |
| `home_trend_direction` | +1 if home team improving, -1 if declining |
| `away_trend_direction` | +1 if away team improving, -1 if declining |

---

### Group 2 — Rest, Travel & Fatigue (from V3)

| Feature | Description |
|---|---|
| `home_team_rest_days` | Days since home team's last game |
| `away_team_rest_days` | Days since away team's last game |
| `rest_balance` | `(home_rest - away_rest) / 5.0` |
| `is_back_to_back` | 1 if home team played yesterday |
| `fatigue_index` | `is_back_to_back × 2` (proxy for travel fatigue) |
| `travel_distance_km` | Approximate travel distance for away team |

---

### Group 3 — Pitcher Quality & Park (from V3)

| Feature | Description |
|---|---|
| `home_pitcher_quality` | `1 - (home_team_ERA / 10)` — team ERA proxy |
| `away_pitcher_quality` | `1 - (away_team_ERA / 10)` — team ERA proxy |
| `pitcher_quality_diff` | `home_pitcher_quality - away_pitcher_quality` |
| `home_park_run_factor` | Park run factor for home stadium |
| `away_park_run_factor` | Park run factor for away stadium |
| `park_advantage` | `home_park_run_factor / away_park_run_factor` |

---

### Group 4 — Composite Team Strength (from V3)

| Feature | Description |
|---|---|
| `home_composite_strength` | `0.4×win_pct_10d + 0.3×ema_form + 0.2×pitcher_quality + 0.1×win_pct_30d` |
| `away_composite_strength` | Same formula for away team |

---

### Group 5 — Temporal Encoding (from V3)

| Feature | Description |
|---|---|
| `month` | Calendar month (3–10) |
| `day_of_week` | Day of week (0=Monday) |
| `is_home` | Always 1 (home team indicator) |
| `month_3` … `month_11` | One-hot encoded month dummies |
| `dow_1` … `dow_7` | One-hot encoded day-of-week dummies |
| `season_phase_home_effect` | Month × home team interaction |
| `month_home_effect` | Derived seasonal home advantage modifier |

---

### Group 6 — Lineup vs Starter Platoon Splits (from V5)

Computed from historical Statcast data (2015–2025 + 2026 season).
Source: `mlb_historical_data.statcast_pitches` via `build_matchup_features.py`.
Requires confirmed lineup from `mlb_2026_season.lineups`.

| Feature | Description |
|---|---|
| `lineup_woba_differential` | Home lineup wOBA vs away starter hand − Away lineup wOBA vs home starter hand |
| `starter_woba_differential` | Away starter wOBA allowed − Home starter wOBA allowed |
| `matchup_advantage_home` | Composite platoon/H2H score differential (home minus away) |
| `home_pct_same_hand` | % of home batters facing same-hand pitcher (platoon disadvantage) |
| `away_pct_same_hand` | % of away batters facing same-hand pitcher (platoon disadvantage) |
| `h2h_woba_differential` | Home lineup H2H wOBA vs that specific starter − Away lineup H2H wOBA |
| `home_top3_woba_vs_hand` | Positions 1–3 aggregate wOBA vs starter handedness |
| `away_top3_woba_vs_hand` | Positions 1–3 aggregate wOBA vs starter handedness |
| `lineup_k_pct_differential` | Away lineup K% vs home starter − Home lineup K% vs away starter |
| `home_starter_k_pct` | Home starter K% (season-level from matchup features) |
| `away_starter_k_pct` | Away starter K% (season-level from matchup features) |

**Fallback:** Games without confirmed lineups receive median-imputed neutral values.

---

### Group 7 — Pitcher Arsenal & Stuff Quality ⬅ NEW in V6

Computed from `mlb_historical_data.pitcher_game_stats` using a **30-day rolling
lookback** before game date. The top-pitch-count starter per team type is used.  
Source: `build_v6_matchup_features.py`, output table: `mlb_2026_season.matchup_v6_features`.

| Feature | Description |
|---|---|
| `home_starter_fastball_pct` | % of pitches that are fastballs (home starter, rolling 30d) |
| `away_starter_fastball_pct` | % of pitches that are fastballs (away starter, rolling 30d) |
| `home_starter_breaking_pct` | % breaking balls (curveball + slider) |
| `away_starter_breaking_pct` | % breaking balls (curveball + slider) |
| `home_starter_offspeed_pct` | % offspeed pitches (changeup, splitter, etc.) |
| `away_starter_offspeed_pct` | % offspeed pitches |
| `home_starter_xwoba_allowed` | Expected wOBA allowed (pitch quality, rolling 30d) |
| `away_starter_xwoba_allowed` | Expected wOBA allowed (pitch quality, rolling 30d) |
| `home_starter_k_bb_pct` | K% minus BB% (command indicator) |
| `away_starter_k_bb_pct` | K% minus BB% (command indicator) |
| `home_starter_velo_norm` | Fastball velocity normalised: `(velo - 93.0) / 3.0` |
| `away_starter_velo_norm` | Fastball velocity normalised: `(velo - 93.0) / 3.0` |
| `home_starter_velo_trend` | Recent 2-start avg velo minus previous 3-start avg (fatigue/sharpness) |
| `away_starter_velo_trend` | Same for away starter |
| `starter_arsenal_advantage` | Composite quality differential (home minus away), weighted sum of xwOBA + K-BB% + velo_norm |

**Fallback:** Missing arsenal data → training-time median imputed values (not 0).

---

### Group 8 — Batter Venue History ⬅ NEW in V6

Per-batter career wOBA at the specific game venue, from Statcast play-by-play.  
Source: `build_player_venue_splits.py` → `mlb_historical_data.player_venue_splits`.  
Minimum 15 PA at venue required; otherwise falls back to season average.

| Feature | Description |
|---|---|
| `home_lineup_venue_woba` | Lineup-average wOBA of home batters **at this specific park** |
| `away_lineup_venue_woba` | Lineup-average wOBA of away batters **at this specific park** |
| `venue_woba_differential` | `home_lineup_venue_woba - away_lineup_venue_woba` |
| `home_venue_advantage` | `home_lineup_venue_woba - home_lineup_season_woba` (park comfort vs baseline) |
| `away_venue_disadvantage` | `away_lineup_venue_woba - away_lineup_season_woba` (familiar vs unfamiliar park) |

---

## Feature Count Summary

| Version | Feature Groups | Total Features | Added |
|---|---|---|---|
| V3 | Rolling form, rest, park, composite, temporal | 57 | baseline |
| V4 | + stacked ensemble meta-features | ~57 | architecture only |
| V5 | + platoon splits, H2H, lineup matchup | ~68 | +11 matchup |
| **V6** | + **pitcher arsenal + venue history** | **~88** | **+20 new** |

---

## Data Sources

| Data | BigQuery Table | Updated |
|---|---|---|
| 2026 game results | `mlb_2026_season.games` | Daily (pipeline) |
| Team stats / ERA | `mlb_2026_season.team_stats` | Daily |
| Standings / win% | `mlb_2026_season.standings` | Daily |
| Confirmed lineups | `mlb_2026_season.lineups` | Pre-game (~90 min before) |
| V5 matchup features | `mlb_2026_season.matchup_features` | Pre-game |
| **V6 matchup features** | `mlb_2026_season.matchup_v6_features` | Pre-game |
| Pitcher game stats | `mlb_historical_data.pitcher_game_stats` | Daily |
| Player venue splits | `mlb_historical_data.player_venue_splits` | Weekly rebuild |
| Player season splits | `mlb_historical_data.player_season_splits` | Weekly rebuild |
| Historical Statcast | `mlb_historical_data.statcast_pitches` | Static (2015–2025) |
| 2026 Statcast | `mlb_2026_season.statcast_pitches` | Daily |

---

## Daily Prediction Pipeline

```
10:00 AM ET  →  schedule_pregame_tasks        fetch today's schedule, enqueue Cloud Tasks
~90 min before first pitch (per game):
  1. lineups            fetch confirmed batting order from MLB API
  2. matchup_features   compute V5 platoon/H2H features
  3. build_v6_matchup_features  compute arsenal + venue features
  4. predict_today_games  run V6 model, write to mlb_2026_season.game_predictions

Nightly:
  season_2026_pipeline  collect game results, team stats, statcast
  build_2026_features   recompute rolling features for completed games

Weekly (Sunday):
  train_v6_models       retrain V6 on expanded 2026 season data, upload to GCS
```

---

## Model Files

| File | Purpose |
|---|---|
| `src/train_v6_models.py` | Train V6, serialize `StackedV6Model`, upload to GCS |
| `src/build_v6_matchup_features.py` | Build arsenal + venue features for today's games |
| `src/build_player_venue_splits.py` | One-time / periodic BQ SQL rebuild of venue split tables |
| `src/model_classes.py` | `StackedV5Model` + `StackedV6Model` (required for pickle) |
| `src/predict_today_games.py` | Daily inference — loads V6 from GCS, writes predictions |
| `src/cloud_function_main.py` | Cloud Function entry point (HTTP trigger) |
