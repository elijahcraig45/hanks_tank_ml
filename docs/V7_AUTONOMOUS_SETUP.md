# Hank's Tank V7 Model Setup & Autonomous Prediction Guide

## Overview

The V7 model extends the stacked ensemble architecture with three new feature categories:

### V7 Features (113 total)
- **Bullpen Health** (9 features): 7-day rolling pitch counts, fatigue scores, closer rest, depth
- **Moon Phase & Circadian** (6 features): Moon illumination, phase state, home/away circadian offset
- **Pitcher Venue Splits** (8 features): Starter ERA/WHIP/K9 at specific ballpark
- **V6 Core** (82 features): Rolled form, rest, travel, park, H2H records

## Model Training

### Prerequisites
```bash
cd ~/hanks_tank_ml
pip install -r requirements.txt
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcloud/application_default_credentials.json
```

### Training V7 Model (Weekly, Sundays)

**Local training** (with GCS upload):
```bash
python src/train_v7_models.py
```

Output:
- Local: `models/game_outcome_2026_v7.pkl`
- GCS: `gs://hanks_tank_data/models/vertex/game_outcome_2026_v7/model.pkl`

**Important**: The model pickle must include the `"features"` key for the predictor to load correctly:
```python
payload = {
    "model": stacked,
    "model_name": "V7_BullpenMoonVenueSplit",
    "version": "v7",
    "features": self.feature_cols,  # ← Required by predictor
    "metrics": result["metrics"],
    "trained_at": datetime.utcnow().isoformat(),
}
```

## Autonomous Prediction Pipeline

### Architecture

The prediction pipeline has two entry points:

1. **Cloud Function** (triggered by Cloud Scheduler)
   - Entry: `src/cloud_function_main.py`
   - Trigger: HTTP POST from Cloud Scheduler

2. **Local/Batch** (manual or cron)
   - Entry: `src/predict_today_games.py`
   - Command: `python predict_today_games.py --date YYYY-MM-DD`

### Daily Pipeline (Automatic)

**Cloud Scheduler Configuration**:
```
Name: hanks-tank-ml-daily
Frequency: 0 4 * * *  (4 AM UTC daily)
HTTP Target: https://[region]-[project].cloudfunctions.net/daily_pipeline
Auth: Service account with ML/BigQuery roles
Request Body: {"mode": "daily"}
```

**What happens at 4 AM**:
1. `season_2026_pipeline.py` — Collect yesterday's games/stats/standings
2. Validate data quality
3. Build matchup features (V5/V6)
4. **Friday only**: Weekly model training (V6 by default)
5. Daily predictions for today's games

### Per-Game Pre-Game Trigger (90 min before first pitch)

**Cloud Tasks are enqueued per game**:
```python
# Cloud Tasks sends HTTP request
POST /daily_pipeline
{
  "mode": "pregame_v7",
  "date": "2026-04-06",
  "game_pks": [824456, 823401, ...]
}
```

**This runs**:
1. Fetch lineups from MLB API
2. Build V5/V6 matchup features
3. Build V7 features (bullpen, moon, pitcher splits)
4. Predict all games with V7 model
5. Insert predictions into `mlb_2026_season.game_predictions`

### Switching to V7 in Daily Pipeline

**Option 1**: Update Cloud Scheduler request body
```json
{
  "mode": "daily",
  "model_version": "v7"
}
```

This only affects weekly training. To make predictions use V7 automatically, ensure the V7 model file exists:
- Local: `models/game_outcome_2026_v7.pkl`
- GCS: `gs://hanks_tank_data/models/vertex/game_outcome_2026_v7/model.pkl`

The predictor automatically tries V7 → V6 → V5 → V4 in order.

**Option 2**: Force V7 in predictions (Cloud Function mode override)
```json
{
  "mode": "predict_today",
  "date": "2026-04-06"
}
```

Will load the best available model (tries V7 first).

## Manual Prediction Execution

### Today's games
```bash
python src/predict_today_games.py
```

### Specific date
```bash
python src/predict_today_games.py --date 2026-04-06
```

### Specific game PKs
```bash
python src/predict_today_games.py --game-pk 824456,823401
```

### Dry-run (no database write)
```bash
python src/predict_today_games.py --dry-run
```

## Backfill Predictions (Missing Dates)

If predictions were missed for a date range, use the backfill script:

```bash
python src/backfill_v7_predictions.py --start 2026-03-27 --end 2026-04-06
```

This will:
1. Load V7 model (from local or GCS)
2. Fetch scheduled games for each date
3. Compute features and generate predictions
4. Append to `mlb_2026_season.game_predictions`

## Troubleshooting

### Issue: "KeyError: 'features'"
**Cause**: Model pickle missing the `"features"` key.
**Fix**: Retrain with the correct payload structure (see above).

### Issue: V7 model fails to load from GCS
**Cause**: Model file not uploaded or permissions issue.
**Check**:
```bash
gsutil ls -l gs://hanks_tank_data/models/vertex/game_outcome_2026_v7/
```

**Fix**: Retrain and ensure upload completes:
```bash
python src/train_v7_models.py  # (don't use --skip-upload)
```

### Issue: "No upcoming games" on a date
**Cause**: MLB API has no games scheduled or returned empty schedule.
**Check**: Verify date is during season and API is accessible:
```bash
curl "https://statsapi.mlb.com/api/v1/schedule?date=2026-04-06&sportId=1"
```

### Issue: Predictions have low confidence
**Review**:
- Bullpen depth scores (check if rosters are fresh in `mlb_2026_season.rosters`)
- Moon phase calculation (verify `game_date` in features)
- Pitcher venue splits (ensure starter history exists in features)

## BigQuery Tables (Input & Output)

**Prediction Inputs**:
- `mlb_2026_season.game_features` — Team rolling stats, rest, park factors
- `mlb_2026_season.matchup_features` — V5/V6 matchup deltas
- `mlb_2026_season.matchup_v7_features` — V7-specific features
- `mlb_2026_season.lineups` — Player lineups (populated 90 min before first pitch)
- `mlb_2026_season.rosters` — Team rosters (daily refresh)

**Prediction Output**:
- `mlb_2026_season.game_predictions` — Per-game predictions with confidence tiers

**Schema**:
```python
game_pk (INT)
game_date (DATE)
home_team_id, home_team_name (INT, STRING)
away_team_id, away_team_name (INT, STRING)
home_starter_id, home_starter_name (INT, STRING)
away_starter_id, away_starter_name (INT, STRING)
home_win_probability, away_win_probability (FLOAT)
predicted_winner (STRING)
confidence_tier (STRING)  # "high" (>62%), "medium" (57-62%), "low" (<57%)
model_version (STRING)    # "V7_BullpenMoonVenueSplit" etc
lineup_confirmed (BOOL)
matchup_advantage_home (FLOAT)
predicted_at (TIMESTAMP)
```

## Cloud Scheduler Configuration (GCP Console)

1. Go to **Cloud Scheduler** in GCP Console
2. Create/edit job `hanks-tank-ml-daily`
   - **Frequency**: `0 4 * * *` (4 AM UTC)
   - **HTTP**: POST
   - **URL**: `https://[region]-[project].cloudfunctions.net/daily_pipeline`
   - **Service Account**: `[project]@appspot.gserviceaccount.com` (with roles: `Cloud Functions Invoker`, `BigQuery Editor`)
   - **Auth header**: Add OIDC token
   - **Request body**:
     ```json
     {
       "mode": "daily",
       "model_version": "v7"
     }
     ```

3. Test with **Force run** button (check logs in Cloud Logging)

## Model Performance Summary

**V7 (Validation 2025)**:
- Accuracy: 56.31%
- AUC: 0.5665
- Brier Score: 0.2450
- Home win rate: 53.4% (actual 54.3%)

This is trained on 2015-2024 historical data with temporal expansion-year CV to prevent leakage.

## Next Steps

- Monitor daily prediction quality in Q2 2026
- Adjust moon phase / circadian offset weights based on live results
- A/B test V7 vs V6 on holdout 2026 data
- Retrain weekly (Sundays) to incorporate 0-week-old games
