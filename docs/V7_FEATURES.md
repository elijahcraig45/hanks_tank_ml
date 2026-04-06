# V7 Model — Feature Reference

## Overview

V7 is the next planned production model, extending V6 with three new signal groups:
bullpen health/availability, moon phase & circadian effects, and starter historical
performance at the specific ballpark. It also recalibrates the home-advantage signal
to reduce its previously outsized impact.

**Architecture:** LR + XGBoost + LightGBM → meta Logistic Regression (same as V5/V6)  
**Model artifact (target):** `gs://hanks_tank_data/models/vertex/game_outcome_2026_v7/model.pkl`  
**Training script:** `src/train_v7_models.py`  
**Feature builder:** `src/build_v7_features.py`  
**Deploy script:** `scripts/gcp/2026_season/deploy_v7.sh`

### V7 vs V6 Summary

| | V6 | V7 |
|---|---|---|
| Total features | ~88 | ~122 |
| Bullpen health | ❌ | ✅ |
| Moon / circadian | ❌ | ✅ |
| Starter venue history | ❌ | ✅ (pitcher splits) |
| Home advantage weight | Full (is_home = 1.0) | Recalibrated (×0.72) |
| BQ output table | `matchup_v6_features` | `matchup_v7_features` |

---

## All Inherited Features (V6)

See [V6_FEATURES.md](V6_FEATURES.md) for the full list. V7 carries all ~88 V6 features
unchanged with the exception of the `is_home` constant (down-weighted at training time).

---

## NEW — Group A: Bullpen Health & Availability

**Source:** `mlb_historical_data.pitcher_game_stats` joined to recent game log  
**Window:** 7 rolling days before game date  
**Script:** `build_v7_features.py → V7FeatureBuilder._bullpen_health_query()`  
**BQ table:** computed inline, written to `matchup_v7_features`

The fatigue score is a leverage-weighted, recency-decayed sum of relief pitches thrown.
Pitchers who threw high-stress appearances (>60 pitches in ≤2 innings) are weighted 1.5×.
Days further in the past are down-weighted by 10%/day so yesterday counts more than last week.

| Feature | Description |
|---|---|
| `home_bullpen_pitches_7d` | Total relief pitches thrown by home bullpen, last 7 days |
| `away_bullpen_pitches_7d` | Same for away team |
| `home_bullpen_games_7d` | Relief appearances (distinct game-days), last 7 days |
| `away_bullpen_games_7d` | Same for away team |
| `home_bullpen_ip_7d` | Relief innings pitched, last 7 days |
| `away_bullpen_ip_7d` | Same for away team |
| `home_bullpen_fatigue_score` | Leverage-weighted, recency-decayed stress score ÷ 100 |
| `away_bullpen_fatigue_score` | Same for away team |
| `home_closer_days_rest` | Days since highest-leverage reliever last pitched (proxy for closer availability) |
| `away_closer_days_rest` | Same for away closer |
| `home_bullpen_depth_score` | Fraction of bullpen pool NOT used in the last 1 day (0–1, higher = fresher) |
| `away_bullpen_depth_score` | Same for away team |
| `bullpen_fatigue_differential` | `away_fatigue_score − home_fatigue_score` (positive = home has fresher pen) |

**Hypothesis:** A team whose bullpen threw 400+ pitches in the last 7 days, with their closer
on consecutive days, faces significantly higher late-game collapse risk than a team with fresh depth.

---

## NEW — Group B: Moon Phase & Circadian Offset

**Source:** Computed locally via `ephem` library (or synodic approximation fallback)  
**Script:** `build_v7_features.py → compute_moon_features()`, `circadian_offset()`  
**No BQ query needed** — purely date/time math

### Moon Phase Features

The moon phase is calculated using astronomical ephemeris data (`ephem` library).
If `ephem` is not installed, a synodic formula (anchored at 2000-01-06 new moon,
29.53059-day cycle) is used as fallback — accurate to within ±0.5 days.

| Feature | Description |
|---|---|
| `moon_phase` | Continuous 0–1 (0 = new moon, 0.5 = full, 1 = new again) |
| `moon_illumination` | % of moon surface illuminated (0–100) |
| `is_full_moon` | 1 if phase within ±0.08 of 0.5 (~±2.4 days of full moon) |
| `is_new_moon` | 1 if phase ≤ 0.08 or ≥ 0.92 (~±2.4 days of new moon) |
| `moon_waxing` | 1 if phase < 0.5 (increasing), 0 if waning |

**Research basis:** `research/astrology_and_calendar_effects.md`,
`research/astrology_and_temporal_factors.md`. The "chaos theory" hypothesis — full
moons correlate with higher-variance games (more errors, extra innings, wild pitches).
The model treats this as a variance modulator rather than a directional predictor.

### Circadian Offset Features

Athletic performance peaks at ~18:00 local home-team time (core body temperature peak).
West Coast teams playing early East Coast starts are biologically behind their peak;
East Coast teams playing late West Coast games are past theirs.

| Feature | Description |
|---|---|
| `home_circadian_offset` | Hours from home team's local game time to their biological peak (18:00 home tz) |
| `away_circadian_offset` | Same for away team (uses their home city time zone) |
| `circadian_differential` | `away_offset − home_offset` — positive means away team is more disadvantaged |

**Research basis:** `research/circadian_rhythms_in_baseball.md`,
`research/astrology_and_calendar_effects.md` §3A.

---

## NEW — Group C: Starter Pitcher Venue History

**Source:** `mlb_historical_data.pitcher_game_stats` joined to `games_historical` for venue_id  
**Script:** `build_v7_features.py → V7FeatureBuilder._pitcher_venue_splits()`  
**Minimum sample:** 5 innings at venue to use actual data; below that, neutral values used  
**Covers:** 2015–2025 historical + 2026 season games before prediction date

This is the **pitcher's** historical performance at the specific ballpark, contrasted with
V6's *batter* venue history (`player_venue_splits`). Together they capture asymmetric
park effects for both sides of the battery.

| Feature | Description |
|---|---|
| `home_starter_venue_era` | Home starter's career ERA at this ballpark (IP ≥ 5 threshold) |
| `home_starter_venue_whip` | Home starter's career WHIP at this ballpark |
| `home_starter_venue_k9` | Home starter's career K/9 at this ballpark |
| `home_starter_venue_pa_total` | Total batters faced at venue (sample size indicator) |
| `away_starter_venue_era` | Away starter's career ERA at this ballpark |
| `away_starter_venue_whip` | Away starter career WHIP at this ballpark |
| `away_starter_venue_k9` | Away starter career K/9 at this ballpark |
| `away_starter_venue_pa_total` | Total batters faced at venue (sample size indicator) |
| `starter_venue_era_differential` | `away_venue_ERA − home_venue_ERA` (positive = home starter advantaged at this park) |

**Neutral fallback:** ERA=4.25, WHIP=1.30, K/9=8.0 (league average proxies) when<br>
insufficient historical PA at venue.

**Note on park familiarity:** Home starters pitch at their own ballpark all season
(high sample, low variance). Away starters have limited appearances (low sample, higher
uncertainty). The `venue_pa_total` feature lets the model learn to discount low-sample splits.

---

## Home Advantage Recalibration

V3–V6 all included `is_home = 1` as a raw constant feature, giving the model a fixed
"home field advantage" signal regardless of park, opponent, or context. Data suggests
this feature was carrying too much weight relative to what's actually explained by
park effects vs. scheduling / familiarity.

V7 makes two changes:

1. **`park_ha_recalibrated`** replaces the raw `is_home` constant:
   ```
   park_ha_recalibrated = park_run_factor × 0.72 × (1 − 0.05 × moon_phase)
   ```
   - `park_run_factor × 0.72` reduces the naive home advantage weight by 28%
   - `× (1 − 0.05 × moon_phase)` applies a small downward nudge on full moons
     (higher variance → less home predictability)

2. **`is_home` is also scaled by 0.72** at training time in `prepare_features()` so
   the gradient-boosted models don't learn to lean on it as a shortcut.

| Feature | Description |
|---|---|
| `park_ha_recalibrated` | Recalibrated home advantage: `park_run_factor × 0.72 × moon_factor` |

---

## Feature Count Summary

| Version | New Groups | Features Added | Total |
|---|---|---|---|
| V3 | Rolling form, rest, park, temporal | baseline | 57 |
| V5 | Platoon/H2H lineup matchup | +11 | ~68 |
| V6 | Pitcher arsenal, batter venue | +20 | ~88 |
| **V7** | **Bullpen health, moon/circadian, pitcher venue** | **+34** | **~122** |

---

## Data Sources

| Data | BigQuery Table | Updated |
|---|---|---|
| Relief pitcher usage | `mlb_historical_data.pitcher_game_stats` | Daily |
| 2026 game schedule | `mlb_2026_season.games` | Daily |
| V6 matchup features | `mlb_2026_season.matchup_v6_features` | Pre-game |
| **V7 matchup features** | `mlb_2026_season.matchup_v7_features` | Pre-game |
| Pitcher game stats (hist) | `mlb_historical_data.pitcher_game_stats` | Daily |
| Games historical | `mlb_historical_data.games_historical` | Static (2015–2025) |

Moon phase and circadian offset are computed locally — no BQ table needed.

---

## Deployment Sequence

To deploy V7 while keeping V6 running uninterrupted:

```bash
# From hanks_tank_ml repo root:
cd scripts/gcp/2026_season

# Full deploy: redeploy function + backfill V7 features + train
./deploy_v7.sh

# Options:
./deploy_v7.sh --skip-backfill      # Deploy + train only
./deploy_v7.sh --skip-train         # Deploy + backfill only
./deploy_v7.sh --dry-run            # Preview all steps
./deploy_v7.sh --only-function      # Just redeploy the Cloud Function
```

### What `deploy_v7.sh` does:
1. Redeploys `mlb-2026-daily-pipeline` Cloud Function with V7 source files
2. Adds two new Cloud Scheduler jobs (V7 weekly training, V7 daily feature build)
3. **Leaves all existing V6 scheduler jobs running** (daily pipeline, V6 train_weekly, etc.)
4. Backfills V7 features from 2026-03-27 through yesterday via Cloud Function HTTP call
5. Triggers a one-time V7 model training run, uploads model to GCS

### Post-deploy: Enable V7 in pre-game tasks
The backend schedules per-game Cloud Tasks ~90 min before first pitch. Update the task
payload mode from `"pregame"` → `"pregame_v7"` to include V7 features in the live pipeline:

```
hanks_tank_backend/src/controllers/lineup.controller.ts
```

---

## Cloud Scheduler Jobs After V7 Deploy

| Job | Schedule | Mode | Purpose |
|---|---|---|---|
| `mlb-2026-daily` | 4 AM ET daily (Mar–Nov) | `daily` | Data collection + V3 features |
| `mlb-2026-validate` | 6 AM ET daily | `validate` | Data quality check |
| `mlb-2026-v7-features-daily` | 5:30 AM ET daily | `matchup_v7_features` | Build V7 features for today |
| `mlb-2026-weekly-predict` | Friday 5 AM ET | `predict` | Batch weekly predictions |
| `mlb-2026-train-v7-weekly` | Sunday 7 AM ET | `train_weekly / v7` | Retrain V7 on expanded data |

Pre-game (per game, triggered by Cloud Tasks):
```
lineups → matchup_features (V5) → build_v6_matchup_features → build_v7_features → predict_today_games
```

---

## Research References

| Topic | File |
|---|---|
| Bullpen fatigue & workload | `research/PITCHER_FATIGUE_FEATURES.md` |
| Pitcher rest cycles, Dead Arm | `research/pitcher_rest_cycles.md` |
| Pitcher rest analysis | `research/pitcher_rest_analysis.md` |
| Moon phase & chaos effect | `research/astrology_and_calendar_effects.md` |
| Circadian rhythms, West Coast edge | `research/circadian_rhythms_in_baseball.md` |
| Temporal / calendar factors | `research/astrology_and_temporal_factors.md` |
| Park factors & geometry | `research/park_and_weather_factors.md` |
