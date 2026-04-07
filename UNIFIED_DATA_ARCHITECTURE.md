# Unified Data Architecture: Historical + 2026 Season

## Overview

This document describes the unified approach for combining historical (2015-2025) and 2026 season data across all key statistics tables. Rather than maintaining separate code paths or duplicate tables, we use BigQuery UNION queries to seamlessly access both datasets with automatic fallback to historical data when current season data is incomplete.

## Core Principle

**Single Query Pattern**: All feature builders and data consumers use queries that UNION historical and 2026 tables:

```sql
SELECT * FROM `mlb_historical_data.TABLE_NAME`
WHERE conditions...
UNION ALL
SELECT * FROM `mlb_2026_season.TABLE_NAME`
WHERE conditions...
```

**Benefits:**
- ✅ Single code path (no if/else for data source)
- ✅ Automatic fallback when current season is incomplete
- ✅ Seamless transition as 2026 season progresses
- ✅ Deduplication via `QUALIFY ROW_NUMBER()` prevents double-counting
- ✅ Future-proof: Same code works for 2027, 2028, etc.

## All Statistics Tables to Unify

### 1. **Pitcher Game Stats** (PRIMARY)
- **Tables**: 
  - `mlb_historical_data.pitcher_game_stats` (2015-2025, 2M+ rows)
  - `mlb_2026_season.pitcher_game_stats` (2026, through Apr 2)
- **Key Columns**: pitcher, game_date, mean_fastball_velo, k_bb_pct, xwoba_allowed, IP, ER, SO, BB
- **V7 Use**: `_pitcher_arsenal_from_game_stats()` - rolling 30-day average with velocity trend
- **Coverage**: Up to Apr 2, 2026 (will auto-extend as daily updates run)

### 2. **Statcast Pitches** (SECONDARY - FALLBACK)
- **Tables**:
  - `mlb_historical_data.statcast_pitches` (2015-2025)
  - `mlb_2026_season.statcast_pitches` (2026, through Apr 6)
- **Key Columns**: pitcher, pitcher_id, velo, spin_rate, pitch_type, release_extension
- **V7 Use**: `_pitcher_arsenal_2026_statcast()` - fallback for per-pitch stats (currently unused)
- **Coverage**: Up to Apr 6, 2026
- **Note**: More granular (pitch-level) but less directly useful for game-level features

### 3. **Lineups**
- **Tables**:
  - `mlb_historical_data.lineups` (2015-2025)
  - `mlb_2026_season.lineups` (2026, incomplete for Apr 6+)
- **Key Columns**: game_pk, game_date, home_team, away_team, starter_pitcher_id, lineup_batter_id
- **V7 Use**: Extract starter pitcher IDs for feature matching
- **Coverage**: Through Apr 2, 2026 (starters not confirmed until day-of)
- **BLOCKER**: Starter confirmation required before pitcher data can be matched

### 4. **Games**
- **Tables**:
  - `mlb_historical_data.games` (2015-2025)
  - `mlb_2026_season.games` (2026, through Apr 6)
- **Key Columns**: game_pk, game_date, home_team, away_team, venue_name, temperature, wind
- **V7 Use**: Join to extract venue/weather context for features
- **Coverage**: Up to Apr 6, 2026 (schedule released in advance)

### 5. **Player Season Splits**
- **Tables**:
  - `mlb_historical_data.player_season_splits` (2015-2025)
  - `mlb_2026_season.player_season_splits` (2026, updated daily)
- **Key Columns**: player_id, season, splits (vs. L/R, home/away, etc.), AVG, OBP, SLG
- **V7 Use**: Batter performance metrics against starter types
- **Coverage**: Through yesterday for 2026 season

### 6. **Player Venue Splits**
- **Tables**:
  - `mlb_historical_data.player_venue_splits` (2015-2025)
  - `mlb_2026_season.player_venue_splits` (2026, updated periodically)
- **Key Columns**: player_id, venue_id, venue_name, G, AVG, OBP, SLG
- **V7 Use**: Batter performance in specific stadiums (crucial for home teams)
- **Coverage**: Through most recent update

## Implementation Pattern

### Single Source of Truth: UNION Query

**Example: Pitcher Arsenal**
```python
def _pitcher_arsenal_from_game_stats(self, pitcher_id, game_date):
    """Query both historical and 2026 pitcher_game_stats via UNION"""
    gd_str = game_date.strftime('%Y-%m-%d')
    sql = f"""
    WITH recent_starts AS (
        SELECT pitcher, game_date, mean_fastball_velo, k_bb_pct, 
               xwoba_allowed, SO, BB
        FROM (
            -- Historical data (2015-2025)
            SELECT * FROM `{PITCHER_STATS_TABLE}`
            WHERE pitcher = {pitcher_id} AND game_date < '{gd_str}'
            UNION ALL
            -- 2026 season data
            SELECT * FROM `{PITCHER_STATS_SEASON}`
            WHERE pitcher = {pitcher_id} AND game_date < '{gd_str}'
        ) all_data
        -- Deduplicate if same pitcher pitched multiple times on same date
        QUALIFY ROW_NUMBER() OVER (PARTITION BY game_date ORDER BY 1) = 1
        ORDER BY game_date DESC
        LIMIT 30  -- 30-day lookback
    )
    SELECT 
        ROUND(AVG(mean_fastball_velo), 2) as mean_velo,
        ROUND(STDDEV(mean_fastball_velo), 2) as velo_trend,
        ROUND(AVG(k_bb_pct), 4) as k_bb_pct,
        ROUND(AVG(xwoba_allowed), 3) as xwoba_allowed
    FROM recent_starts
    """
    # Execute and return results
```

### Daily Update Pattern: DELETE-Before-INSERT

All data fetch/update functions follow this pattern to prevent duplicates:

```python
def update_pitcher_stats_2026(request):
    """Cloud Function to fetch and update pitcher stats for 2026"""
    from google.cloud import bigquery
    import datetime
    
    client = bigquery.Client()
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    
    # Step 1: DELETE old data for yesterday
    delete_query = f"""
    DELETE FROM `hankstank.mlb_2026_season.pitcher_game_stats`
    WHERE DATE(game_date) = '{yesterday}'
    """
    client.query(delete_query).result()
    
    # Step 2: FETCH new data from source
    # TODO: Implement actual fetch (Baseball Savant, MLB API, etc.)
    new_data = fetch_pitcher_stats_from_source(yesterday)
    
    # Step 3: INSERT fresh data
    table = client.get_table(f"hankstank.mlb_2026_season.pitcher_game_stats")
    errors = client.insert_rows_json(table, new_data)
    
    return {
        "status": "success",
        "date": yesterday,
        "rows_inserted": len(new_data),
        "errors": errors
    }
```

## Cloud Scheduler Orchestration

Daily update jobs run in sequence to ensure data dependencies are met:

| Time (UTC) | Job | Table Updated | Purpose |
|-----------|-----|---------------|---------|
| **06:00** | fetch-statcast-2026 | `mlb_2026_season.statcast_pitches` | Get pitch-level data from MLB API |
| **06:30** | fetch-pitcher-stats-2026 | `mlb_2026_season.pitcher_game_stats` | Aggregate pitcher stats from games |
| **08:00** | rebuild-v7-features | `matchup_v7_features` | Build features after all data loaded |
| **12:00** | predict-today-games | `game_predictions` | Generate predictions after lineups confirmed |

### Rationale:
1. **06:00-06:30**: Fetch raw data from external sources (parallel-safe)
2. **08:00**: Wait for lineups to stabilize, then build features
3. **12:00**: Starters should be confirmed, generate predictions before gates open

## Data Availability Timeline

### Current Status (as of Apr 7, 2026):
- ✅ pitcher_game_stats: through Apr 2
- ✅ statcast_pitches: through Apr 6  
- ⏳ lineups: through Apr 2 (starters not confirmed for Apr 6+ until day-of)
- ✅ games: through Apr 6 (schedule released in advance)
- ✅ player_season_splits: updated daily
- ✅ player_venue_splits: updated daily/weekly

### Blockers & Timeline:
- **Blocker**: Lineups starters confirmation (external - MLB schedule dependent)
- **Solution**: Graceful NULL handling - pitcher features will populate once lineups confirmed
- **Alternative**: Use previous season starters from `historical_lineups` if needed

## Code Changes Required

### 1. **src/build_v7_features.py**
- ✅ **DONE**: `_pitcher_arsenal_from_game_stats()` method added (UNION queries both sources)
- ✅ **DONE**: Call site updated to use combined method
- ✅ **DONE**: DELETE-before-INSERT pattern in upsert logic

### 2. **cloud_functions/daily_updates.py** (NEW)
- Functions for each update job:
  - `update_pitcher_stats_2026()` - fetch and load pitcher stats
  - `update_statcast_2026()` - fetch and load statcast data
  - `rebuild_v7_features()` - call V7 builder
  - `predict_today_games()` - call prediction engine

### 3. **cloud_functions/setup_scheduler.sh** (NEW)
- Bash script to deploy Cloud Scheduler jobs via gcloud CLI
- Creates 4 scheduled HTTP triggers (times listed above)
- Updates Cloud Function environments/settings

### 4. **cloud_functions/requirements.txt** (NEW)
- Dependencies for Cloud Function runtimes

## Future Work

### Phase 2: Expand to Other Tables
When ready, apply same pattern to:
- Bullpen stats (historical + 2026)
- Advanced metrics (WAR, etc.) - when available
- Player injury data
- Weather data

### Phase 3: Real-time Updates
- Move from daily batch to near-real-time streaming
- Use Pub/Sub triggers for statcast pitch events
- Update V7 features incrementally as games progress

### Phase 4: Archive & Housekeeping
- Archive completed seasons to cost-optimized storage
- Implement retention policies
- Separate hot (current season) from cold (historical) storage

## Validation Queries

Check unified data is working:

```sql
-- Verify UNION works and shows both historical + 2026
SELECT 
  COUNT(DISTINCT CAST(game_date AS DATE)) as distinct_dates,
  MIN(game_date) as earliest,
  MAX(game_date) as latest,
  COUNT(*) as total_records
FROM (
  SELECT game_date FROM `mlb_historical_data.pitcher_game_stats`
  UNION ALL
  SELECT game_date FROM `mlb_2026_season.pitcher_game_stats`
);
```

Expected result: Should show dates from 2015 through Apr 2, 2026.

```sql
-- Verify deduplication works (no pitcher pitched twice same date in combined query)
SELECT pitcher, game_date, COUNT(*) as appearances
FROM (
  SELECT pitcher, game_date FROM `mlb_historical_data.pitcher_game_stats`
  UNION ALL
  SELECT pitcher, game_date FROM `mlb_2026_season.pitcher_game_stats`
)
GROUP BY pitcher, game_date
HAVING COUNT(*) > 1
LIMIT 10;
```

Expected result: Should return 0 rows (or very few edge cases like same pitcher pitching for two teams in different time periods).

## Deployment Checklist

- [ ] Cloud Functions created and tested locally
- [ ] Cloud Scheduler jobs deployed
- [ ] Service account has BigQuery write access to 2026_season dataset
- [ ] Error logging configured (Cloud Logging)
- [ ] Monitoring/alerts set up (Cloud Monitoring)
- [ ] First daily run successful
- [ ] V7 features populated correctly with 2026 data
- [ ] Predictions appear on frontend with pitcher velo/stats
- [ ] Historical fallback tested (e.g., query 2025 data)

## References

- Cloud Scheduler docs: https://cloud.google.com/scheduler/docs
- Cloud Functions: https://cloud.google.com/functions/docs
- BigQuery best practices: https://cloud.google.com/bigquery/docs/best-practices
        "current": "hankstank.mlb_2026_season.pitcher_game_stats",
        "unified_view": "hankstank.mlb_2026_season.pitcher_game_stats_combined",
        "daily_updater": "cloud_functions/update_pitcher_stats_2026.py",
        "update_schedule": "0 6:30 * * * UTC",
    },
    "lineups": {
        "archive": "hankstank.mlb_historical_data.lineups",
        "current": "hankstank.mlb_2026_season.lineups",
        "unified_view": "hankstank.mlb_2026_season.lineups_combined",
        "daily_updater": "cloud_functions/update_lineups_2026.py",
        "update_schedule": "0 7 * * * UTC",
    },
    "games": {
        "archive": "hankstank.mlb_historical_data.games_historical",
        "current": "hankstank.mlb_2026_season.games",
        "unified_view": "hankstank.mlb_2026_season.games_combined",
        "daily_updater": "cloud_functions/update_games_2026.py",
        "update_schedule": "0 7 * * * UTC",
    },
    "player_season_splits": {
        "archive": "hankstank.mlb_historical_data.player_season_splits",
        "current": "hankstank.mlb_2026_season.player_season_splits",
        "unified_view": "hankstank.mlb_2026_season.player_season_splits_combined",
        "daily_updater": "cloud_functions/update_player_splits_2026.py",
        "update_schedule": "0 6 * * * UTC",
    },
    "player_venue_splits": {
        "archive": "hankstank.mlb_historical_data.player_venue_splits",
        "current": "hankstank.mlb_2026_season.player_venue_splits",
        "unified_view": "hankstank.mlb_2026_season.player_venue_splits_combined",
        "daily_updater": "cloud_functions/update_player_venue_splits_2026.py",
        "update_schedule": "0 6 * * * UTC",
    },
}

# Cloud Function Orchestration (Cloud Scheduler triggers):
CLOUD_SCHEDULER_JOBS = [
    {
        "name": "fetch-statcast-2026-daily",
        "schedule": "0 6 * * *",  # 6 AM UTC
        "timezone": "UTC",
        "http_target": "https://region-project.cloudfunctions.net/update_statcast_2026",
        "description": "Fetch and append new 2026 statcast data daily",
    },
    {
        "name": "update-features-daily",
        "schedule": "0 8 * * *",  # 8 AM UTC (after all data loaded)
        "timezone": "UTC",
        "http_target": "https://region-project.cloudfunctions.net/rebuild_v7_features",
        "description": "Rebuild V7 matchup features after data updates",
    },
    {
        "name": "predict-today-games",
        "schedule": "0 12 * * *",  # 12 PM UTC (after lineups confirmed)
        "timezone": "UTC",
        "http_target": "https://region-project.cloudfunctions.net/predict_today_games",
        "description": "Generate predictions for today's games",
    },
]

if __name__ == "__main__":
    print("UNIFIED DATA ARCHITECTURE PLAN")
    print("=" * 100)
    for table, config in UNIFICATION_PLAN.items():
        print(f"\n{table.upper()}")
        print(f"  Archive: {config['archive']}")
        print(f"  Current: {config['current']}")
        print(f"  Unified: {config['unified_view']}")
        print(f"  Updater: {config['daily_updater']}")
        print(f"  Schedule: {config['update_schedule']}")
    
    print("\n\nCLOUD SCHEDULER JOBS")
    print("=" * 100)
    for job in CLOUD_SCHEDULER_JOBS:
        print(f"\n{job['name']}")
        print(f"  Schedule: {job['schedule']} {job['timezone']}")
        print(f"  Endpoint: {job['http_target']}")
        print(f"  Description: {job['description']}")
