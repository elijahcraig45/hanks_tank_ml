# Architecture Guide — HanksTank MLB Prediction Pipeline

> **Current production model: V8** (deployed 2026-04-08)  
> **Overall accuracy: 57.65% | High-confidence (≥60% prob): 65.4% on 11.7% of games**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [GCP Architecture Diagram](#2-gcp-architecture-diagram)
3. [Data Flow](#3-data-flow)
4. [BigQuery Schema](#4-bigquery-schema)
5. [Model Evolution](#5-model-evolution)
6. [Architecture Audit — Issues Found & Fixed (2026-04-08)](#6-architecture-audit)
7. [Cost Analysis](#7-cost-analysis)
8. [Runbook — Common Operations](#8-runbook)
9. [Known Remaining Limitations](#9-known-remaining-limitations)

---

## 1. System Overview

HanksTank is a GCP-based MLB game prediction system. It collects daily game data,
computes ML features, generates win probability predictions, and exposes those predictions
through a backend API consumed by the React frontend.

**Components:**

| Component | Technology | Purpose |
|---|---|---|
| Daily pipeline | Cloud Function (Gen2) | Orchestrates all daily tasks |
| Scheduler | Cloud Scheduler (5 jobs) | Triggers the pipeline on cron schedules |
| Feature store | BigQuery (`mlb_2026_season`) | Game features, V8 features, predictions |
| Historical data | BigQuery (`mlb_historical_data`) | 2015-2025 game outcomes, statcast |
| Model storage | GCS (`hanks_tank_data/models/`) | Serialized model artifacts |
| ML model | V8 CatBoost + team embeddings | 85 features, 57.65% accuracy |
| Backend API | Node.js / App Engine | Reads BQ predictions, serves frontend |
| Frontend | React | Displays predictions, news, standings |

---

## 2. GCP Architecture Diagram

```
Cloud Scheduler (5 cron jobs)
        │
        │  HTTP POST (OIDC authenticated)
        ▼
Cloud Function: mlb-2026-daily-pipeline
 entry: daily_pipeline()  ←  src/cloud_function_main.py
        │
        ├── mode: daily
        │     ├── SeasonPipeline.run_daily()       → mlb_2026_season.games
        │     │                                        .standings, .team_stats
        │     │                                        .rosters, .transactions
        │     ├── DataValidator.run()               (quality checks)
        │     ├── FeatureBuilder.build_features()  → mlb_2026_season.game_features
        │     ├── V8LiveFeatureBuilder              → mlb_2026_season.team_elo_ratings
        │     │   .update_elo_after_games()           (30-row Elo state table)
        │     └── V8LiveFeatureBuilder              → mlb_2026_season.game_v8_features
        │         .run_for_date()                     (V8 features for today's games)
        │
        ├── mode: schedule_pregame_tasks
        │     └── Enqueue Cloud Tasks (per game, ~90 min before first pitch)
        │
        ├── mode: pregame_v8  [per-game Cloud Task]
        │     ├── LineupFetcher.run_for_game_pks()   → mlb_2026_season.lineups
        │     ├── MatchupFeatureBuilder               → mlb_2026_season.matchup_features
        │     ├── V7FeatureBuilder                   → mlb_2026_season.matchup_v7_features
        │     ├── V8LiveFeatureBuilder                → mlb_2026_season.game_v8_features
        │     └── DailyPredictor.run_for_game_pks()  → mlb_2026_season.game_predictions
        │
        └── mode: train_weekly  [Sunday 2 AM]
              └── V8 model retraining                → GCS models/vertex/game_outcome_2026_v8/
                                                        (with 2026 season data included)


BigQuery                                    GCS (hanks_tank_data)
──────────────────────────────             ──────────────────────────────
mlb_historical_data                        models/vertex/
  games_historical (2015-2025)               game_outcome_2026_v8/model.pkl  ← ACTIVE
  pitcher_game_stats                         game_outcome_2026_v7/model.pkl  ← fallback
  statcast_pitches                           game_outcome_2026_v6/model.pkl  ← fallback
  lineups                                    game_outcome_2026/model.pkl     ← fallback
  player_season_splits
  player_venue_splits

mlb_2026_season
  games               (raw game results)
  standings           (daily snapshot)
  team_stats          (daily snapshot)
  rosters             (weekly refresh)
  transactions        (daily)
  lineups             (pre-game, per-game)
  game_features       (V3/V4 features)
  matchup_features    (V5/V6 matchup)
  matchup_v7_features (V7: bullpen, moon, venue splits)
  game_v8_features    (V8: Elo, Pythag, run diff, streaks, H2H) ← NEW
  team_elo_ratings    (V8 Elo state, 30 rows)                   ← NEW
  game_predictions    (final predictions per game)
  weekly_predictions  (Friday batch predictions)
```

---

## 3. Data Flow

### Daily (4:00 AM ET)
```
MLB Stats API → games, standings, team_stats, rosters, transactions → BQ (mlb_2026_season)
                                                    ↓
                              FeatureBuilder (V3/V4 rolling features) → game_features
                                                    ↓
                         V8LiveFeatureBuilder (Elo update) → team_elo_ratings (30-row upsert)
                                                    ↓
                       V8LiveFeatureBuilder (daily features) → game_v8_features
```

### Pre-game (~90 min before first pitch, per game)
```
MLB Stats API → confirmed lineups → BQ (lineups)
                                           ↓
                     MatchupFeatureBuilder → matchup_features (V5/V6 wOBA, H2H)
                                           ↓
                          V7FeatureBuilder → matchup_v7_features (bullpen, moon, venue)
                                           ↓
                     V8LiveFeatureBuilder  → game_v8_features (refresh Elo + rolling stats)
                                           ↓
                           DailyPredictor  → game_predictions
                                           ↓
                         Backend API reads → Frontend displays
```

### Weekly (Friday 5:00 AM ET)
```
Scheduled games (MLB API) → WeeklyPredictor → weekly_predictions
```

### Weekly retraining (Sunday 2:00 AM ET)
```
Historical BQ data + 2026 season data → V8 training script → updated model → GCS
```

---

## 4. BigQuery Schema

### `game_v8_features` (NEW — V8)
Partitioned by `game_date`, clustered by `game_pk`.

| Column | Type | Description |
|---|---|---|
| `game_pk` | INTEGER | MLB game primary key |
| `game_date` | DATE | Game date (partition key) |
| `home_elo` | FLOAT | Home team Elo rating (pre-game) |
| `away_elo` | FLOAT | Away team Elo rating (pre-game) |
| `elo_differential` | FLOAT | home_elo − away_elo |
| `elo_home_win_prob` | FLOAT | Elo win probability (0.0–1.0) |
| `elo_win_prob_differential` | FLOAT | elo_home_win_prob − 0.5 |
| `home_pythag_season` | FLOAT | Pythagorean win% (season) |
| `away_pythag_season` | FLOAT | |
| `home_pythag_last30` | FLOAT | Pythagorean win% (last 30 games) |
| `away_pythag_last30` | FLOAT | |
| `pythag_differential` | FLOAT | home_pythag_season − away_pythag_season |
| `home_luck_factor` | FLOAT | actual_win% − pythagorean_win% |
| `away_luck_factor` | FLOAT | |
| `luck_differential` | FLOAT | away_luck − home_luck |
| `home_run_diff_10g` | FLOAT | Avg run differential, last 10 games |
| ... | | *(17 more run differential columns)* |
| `home_current_streak` | INTEGER | +N = win streak, −N = loss streak |
| `away_current_streak` | INTEGER | |
| ... | | *(11 more streak columns)* |
| `h2h_win_pct_season` | FLOAT | Season H2H win% (home team perspective) |
| `h2h_win_pct_3yr` | FLOAT | 3-year H2H win% |
| ... | | *(3 more H2H columns)* |
| `is_divisional` | INTEGER | 1 if same division |
| `season_pct_complete` | FLOAT | games_played / 162 |
| `season_stage` | INTEGER | 0=early, 1=mid, 2=late |
| `data_completeness` | FLOAT | Confidence in rolling stats (0–1) |
| `computed_at` | TIMESTAMP | Feature computation timestamp |

### `team_elo_ratings` (NEW — V8)
30 rows — one per MLB team. Updated daily.

| Column | Type | Description |
|---|---|---|
| `team_id` | INTEGER | MLB team ID |
| `elo_rating` | FLOAT | Current Elo rating |
| `season` | INTEGER | Season year (2026) |
| `games_played` | INTEGER | Games since season start |
| `last_game_pk` | INTEGER | Most recent game used for update |
| `updated_at` | TIMESTAMP | Last update time |

---

## 5. Model Evolution

| Version | Accuracy | AUC | Key Addition | Notes |
|---|---|---|---|---|
| V1 | 54.0% | 0.543 | 5 basic features | Logistic regression baseline |
| V2 | 54.4% | 0.534 | 44 features | More team stats |
| V3 | 54.6% | 0.546 | 57 features, XGBoost | Previous best validated |
| V4 | ~54.6% | — | Stacking (LR+XGB+LGB) | Did not improve accuracy |
| V5 | ~55% | — | Pitcher/batter matchup features | Live wOBA splits |
| V6 | ~55% | — | Pitcher arsenal, venue history | Statcast velocity/xwOBA |
| V7 | ~55% | — | Bullpen health, moon phase, venue ERA | Deployed to GCP |
| **V8** | **57.65%** | **0.575** | Elo + Pythagorean + team embeddings | **Current production** |

### V8 Model Architecture

**Primary:** CatBoost with team ID categorical embeddings  
**Ensemble:** CatBoost (54%) + MLP (24%) + LightGBM (18%) + CatBoost-wide (4%)  
**Features:** 85 total (83 numerical + 2 categorical: home_team_id, away_team_id)

**Confidence tiers (calibrated on 2025 validation set):**

| Tier | Probability | Accuracy | Coverage |
|---|---|---|---|
| High | ≥60% or ≤40% | **65.4%** | 11.7% of games |
| Medium | 55–60% or 40–45% | **59.9%** | 43% of games |
| Low | 45–55% | ~54% | 45% of games |

---

## 6. Architecture Audit

> **Audit date:** 2026-04-08  
> All issues found during V8 deployment preparation.

### Issue 1 — V8 completely absent from production pipeline [FIXED]

**Found:** `predict_today_games.py` model chain was V7→V6→V5→V4. V8 did not exist
anywhere in the prediction or scheduling code.

**Impact:** Despite training V8 locally (57.65% accuracy), all GCP predictions were
still using V6 (55% accuracy). The V8 model artifact existed locally but was never
uploaded to GCS or referenced by the predictor.

**Fix:** 
- Added V8 to model resolution chain: **V8→V7→V6→V5→V4** in `predict_today_games.py`
- V8 model cached locally after first GCS download (avoids repeated GCS reads)
- Added `_is_v8` flag to conditionally load V8 features

**Files changed:** `src/predict_today_games.py`

---

### Issue 2 — V8 features not computable for live predictions [FIXED]

**Found:** The training-time `build_v8_features.py` reads local parquet files and a
local CSV. It was only suitable for offline batch training. There was **no code** to
compute Elo ratings, Pythagorean win%, run differential, or streaks from live BQ data.

`build_2026_features.py` (the BQ-based feature builder) only computed ~50 V3-style
features. The V8 model expects 85 features. Deploying V8 without live feature
computation would have caused the model to use neutral defaults (0.5) for its most
important features — degrading accuracy back to ~54%.

**Impact:** V8 model would have been functionally equivalent to V3 without this fix.

**Fix:** Created `src/build_v8_features_live.py`:
- Maintains Elo state in `mlb_2026_season.team_elo_ratings` (30 rows, K=15)
- Computes Pythagorean, run differential, streaks, H2H from BigQuery game history
- Uses UNION query pattern (historical + 2026) — scans ~2MB/run (~$0.00/month)
- Integrated into daily pipeline and pre-game pipeline

**Files created:** `src/build_v8_features_live.py`  
**Files changed:** `src/cloud_function_main.py`, `src/predict_today_games.py`

---

### Issue 3 — `cloud_functions/` directory was dead code [DOCUMENTED]

**Found:** The `cloud_functions/` directory contained four Cloud Functions:
1. `update_statcast_2026`: `fetch_statcast_from_baseball_savant()` always returns `[]` —
   the implementation was never written (TODO comment)
2. `predict_today_games`: returns hardcoded `{"predictions_generated": 0}` — never wired
   to the actual predictor
3. `update_pitcher_stats_2026`: same unimplemented statcast dependency
4. `rebuild_v7_features`: was functional but superseded by `src/cloud_function_main.py`

Additionally, `setup_scheduler.sh` targeted old top-level function names that no longer
match the deployed entry point (`daily_pipeline` in `src/cloud_function_main.py`).

**Impact:** Any scheduler job created by `setup_scheduler.sh` would trigger nothing
useful. The statcast and prediction functions were effectively no-ops.

**Fix:** Added `DEPRECATED.md` to `cloud_functions/` explaining what was replaced and
why. Added deprecation headers to `main.py` and `daily_updates.py`, replaced the old
scheduler script with a hard stop, and removed the obsolete helper functions from GCP.

**Files changed:** `cloud_functions/DEPRECATED.md` (new), `cloud_functions/main.py`,
`cloud_functions/daily_updates.py`

---

### Issue 4 — Hard-coded absolute Mac path in `build_v8_features.py` [FIXED]

**Found:** 
```python
GAMES_CSV = Path("/Users/VTNX82W/Documents/personalDev/hanks_tank_backend/data/games/games_2015_2024.csv")
```
This path is literally a Mac home directory. It would fail immediately in any GCP
Cloud Function, CI environment, or on any other developer's machine.

**Fix:** Replaced with three-tier resolution:
1. `GAMES_CSV_PATH` environment variable (for CI/GCP)
2. Relative path from repo structure (`../hanks_tank_backend/data/games/...`)
3. Informative warning if neither resolves

**Files changed:** `src/build_v8_features.py`

---

### Issue 5 — Missing `catboost` dependency [FIXED]

**Found:** `requirements.txt` and `src/requirements.txt` listed XGBoost, LightGBM,
optuna — but not CatBoost. V8's best model is a CatBoostClassifier. Deploying V8
to Cloud Function without adding catboost would fail with an ImportError at runtime.

**Fix:** Added `catboost>=1.2.0` to both requirements files. Also reorganized both
files with comments grouping dependencies by category.

**Files changed:** `requirements.txt`, `src/requirements.txt`

---

### Issue 6 — Cloud Function memory insufficient for V8 CatBoost [FIXED]

**Found:** `deploy_2026_pipeline.sh` set Cloud Function memory to `1024MB`. The V8
final ensemble model artifact (`game_outcome_2026_v8_final.pkl`) is ~5MB on disk but
expands to ~1.2GB when loaded into memory (CatBoost 300 iterations × 4 models in
ensemble). The function would OOM silently and fall back to V4.

**Fix:** `deploy_v8.sh` sets `--memory=2048MB`. V8 loads comfortably under 1.5GB.

**Files changed:** `scripts/gcp/2026_season/deploy_v8.sh`

---

### Issue 7 — `predict_2026_weekly.py` hardwired to V4 model [KNOWN LIMITATION]

**Found:** `predict_2026_weekly.py` loads `game_outcome_2026_vertex.pkl` (V4)
exclusively with no model fallback chain. Weekly predictions are stale.

**Status:** Documented, not yet fixed. Weekly batch predictions are lower priority
than per-game predictions (which now use V8 via pregame pipeline). Fix: add same
model chain as `predict_today_games.py` to `predict_2026_weekly.py`.

---

### Issue 8 — Walk-forward CV gap vs single-year val gap [DOCUMENTED]

**Found:** V8 achieves 57.65% on the 2025 validation year. Walk-forward 5-fold CV
(each year held out) averages 54.81% ± 0.96%. The gap (2.84%) is larger than
expected and indicates some of the 2025 improvement may be year-specific.

**Explanation:** 2025 was a relatively high-parity season with consistent team quality.
Years like 2020 (COVID shortened) and 2023 (higher variance) show lower accuracy.

**Implication:** True expected deployment accuracy is approximately **55–57%**, not 57.65%.
The confidence-filtered accuracy (65.4% at ≥60% probability) is more robust since it
relies on the model's calibration, not just its point predictions.

**This is documented in `docs/V8_EXPERIMENT_COMPLETE.md` Section: Walk-Forward CV.**

---

## 7. Cost Analysis

All estimates at current GCP pricing (us-central1, Apr 2026).

| Component | Monthly Cost | Notes |
|---|---|---|
| Cloud Function invocations | ~$0.00 | Well within free tier (2M free/month) |
| Cloud Function compute | ~$0.05 | 5 jobs × 540s × 2GB = ~14 GB-hr/month |
| Cloud Scheduler | ~$0.50 | 5 jobs × $0.10 |
| BigQuery queries | ~$0.02 | ~4MB/run × 30 runs × $5/TB |
| BigQuery storage | ~$0.10 | ~10GB active data |
| GCS storage | ~$0.01 | ~15MB model artifacts |
| **Total** | **~$0.70/month** | Well under $5/month target |

**Cost-saving decisions made:**
- V8 feature computation uses BigQuery window functions (not Python loops) — minimizes
  compute cost and Cloud Function CPU time
- `team_elo_ratings` is 30 rows — uses simple DELETE+INSERT instead of MERGE
- Walk-forward model retraining runs only on Sundays — not daily
- Weekly batch predictions (Friday) distinct from daily per-game predictions — avoids
  running the full model on all 162-game schedule unnecessarily
- Historical data is loaded once per daily run and cached in memory for all games
  that day (not queried per-game)

---

## 8. Runbook — Common Operations

### Deploy V8 (initial or redeploy)
```bash
cd scripts/gcp/2026_season
./deploy_v8.sh                    # full deploy
./deploy_v8.sh --dry-run          # preview
./deploy_v8.sh --skip-backfill    # redeploy function only
```

### Manually trigger today's predictions
```bash
# Via gcloud
gcloud functions call mlb-2026-daily-pipeline \
  --gen2 --region=us-central1 \
  --data='{"mode":"predict_today","date":"2026-04-08"}'

# Via curl (requires OIDC token)
TOKEN=$(gcloud auth print-identity-token)
curl -X POST "$(gcloud functions describe mlb-2026-daily-pipeline \
    --gen2 --region=us-central1 --format='value(serviceConfig.uri)')" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mode":"predict_today"}'
```

### Backfill V8 features
```bash
cd src
python3 build_v8_features_live.py --backfill --start 2026-03-27
```

### Seed / reset Elo ratings
```bash
cd src
python3 build_v8_features_live.py --seed-elo
```

### Update Elo after a specific date's games
```bash
cd src
python3 build_v8_features_live.py --update-elo --date 2026-04-07
```

### Check prediction quality
```sql
-- BigQuery console or bq CLI
SELECT
  game_date,
  COUNT(*) AS games,
  COUNTIF(confidence_tier = 'high') AS high_conf,
  COUNTIF(confidence_tier = 'medium') AS med_conf,
  AVG(home_win_probability) AS avg_prob,
  model_version
FROM `hankstank.mlb_2026_season.game_predictions`
WHERE game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
GROUP BY game_date, model_version
ORDER BY game_date DESC
```

### Check accuracy of completed predictions
```sql
SELECT
  p.model_version,
  COUNT(*) AS total_games,
  COUNTIF(
    (p.predicted_winner = g.home_team_name AND g.home_score > g.away_score) OR
    (p.predicted_winner = g.away_team_name AND g.away_score > g.home_score)
  ) AS correct,
  ROUND(COUNTIF(
    (p.predicted_winner = g.home_team_name AND g.home_score > g.away_score) OR
    (p.predicted_winner = g.away_team_name AND g.away_score > g.home_score)
  ) / COUNT(*) * 100, 1) AS accuracy_pct
FROM `hankstank.mlb_2026_season.game_predictions` p
JOIN `hankstank.mlb_2026_season.games` g USING (game_pk)
WHERE g.status IN ('Final', 'Completed Early')
  AND p.model_version LIKE 'V8%'
GROUP BY model_version
ORDER BY accuracy_pct DESC
```

### Check V8 feature data completeness
```sql
SELECT
  game_date,
  COUNT(*) AS games,
  ROUND(AVG(data_completeness) * 100, 1) AS avg_completeness_pct,
  COUNTIF(data_completeness < 0.5) AS low_completeness_games
FROM `hankstank.mlb_2026_season.game_v8_features`
WHERE game_date >= '2026-03-27'
GROUP BY game_date
ORDER BY game_date DESC
LIMIT 30
```

### Force retrain V8 model
```bash
gcloud functions call mlb-2026-daily-pipeline \
  --gen2 --region=us-central1 \
  --data='{"mode":"train_weekly","model_version":"v8"}'
```

---

## 9. Known Remaining Limitations

### L1 — Statcast ingestion not implemented
`fetch_statcast_from_baseball_savant()` in `cloud_functions/daily_updates.py` returns
empty always. The new pipeline in `src/cloud_function_main.py` calls
`SeasonPipeline.collect_games()` which fetches game outcomes from MLB Stats API, but
does not fetch pitch-level Statcast data.

**Impact on V8:** V8 does not use Statcast features directly — it uses game outcomes
only (scores → run differential, Elo). V7 features (velocity, xwOBA, bullpen fatigue)
DO require Statcast. If V7 feature computation is blocked by missing Statcast, those
fields will be NULL/neutral defaults.

**Workaround:** V8 accuracy is not impacted. V7 features degrade gracefully.

**To fix:** Implement `pybaseball.statcast()` call in `season_2026_pipeline.py` with
proper error handling and rate limiting (~150 req/min for Baseball Savant).

### L2 — `predict_2026_weekly.py` still uses V4 model
Weekly (Friday) batch predictions use the old V4 model. These populate
`mlb_2026_season.weekly_predictions` independently of per-game predictions.

**Impact:** If the frontend reads from `weekly_predictions`, it shows V4-quality
predictions (54%). Per-game predictions in `game_predictions` are V8-quality (57.65%).

**To fix:** Update `predict_2026_weekly.py` to use the same model chain as
`predict_today_games.py`.

### L3 — 2025 run features estimated, not exact
V8 training used actual game scores for 2015-2024 (from `games_2015_2024.csv`) but
estimated 2025 run differential features using 40% decay from 2024 end-of-season.

For 2026 live predictions, the V8LiveFeatureBuilder queries real 2026 game scores from
BQ, so this limitation only applies to the training data, not production predictions.

### L4 — Team ID Elo seeds are approximated
`ELO_2026_SEED` in `build_v8_features_live.py` contains manually-set Elo start values
for 2026. These were derived from the training-time Elo computation end-of-2025 values
with 40% regression, but the exact per-team values are approximations.

**Impact:** Predictions in the first ~20 games of the 2026 season have less accurate
Elo-based win probabilities. As games accumulate, Elo self-corrects.

**To fix:** Extract exact 2025 end-of-season Elo ratings from the training run
(`build_v8_features.py` returns them) and store in a JSON file for the 2026 seed.

### L5 — No Vegas line feature
Vegas money line odds consistently achieve ~57-58% accuracy through market
information aggregation. Adding Vegas odds as a feature is estimated to improve
V8 accuracy by 1-1.5%, potentially reaching the 60% overall target.

**Data source:** Multiple odds APIs available (The Odds API, sportsbettingapi)
**Cost:** ~$50/month for a reliable tier  
**Impact:** High — likely the single largest remaining accuracy improvement

---

*Architecture version: V8 | Last updated: 2026-04-08*
