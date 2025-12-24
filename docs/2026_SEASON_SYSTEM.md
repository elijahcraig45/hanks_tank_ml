# 2026 Season Real-Time Tracking System

Complete automated system for collecting, syncing, validating, and analyzing 2026 MLB season data in real-time.

## System Overview

```
┌─────────────────┐
│  MLB Stats API  │  (Live 2026 data)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Collector     │  season_2026_collector.py
│  - Games        │  Fetches daily games, stats, standings
│  - Stats        │
│  - Statcast     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Local Cache   │  data/2026/*
│  (JSON files)   │  Temporary storage
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BigQuery Sync  │  bigquery_sync.py
│  - Upsert data  │  Safe incremental updates
│  - No duplicates│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   BigQuery DB   │  hankstank.mlb_historical_data
│  - Historical   │  2015-2026 unified storage
│  - Validated    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Validation    │  data_validation.py
│  - Completeness │  Automated quality checks
│  - Quality      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Models     │  Ready for analysis
│  - Training     │
│  - Prediction   │
└─────────────────┘
```

## Quick Start

### 1. Setup (One-Time)

```bash
cd /Users/VTNX82W/Documents/personalDev/hanks_tank_ml
./setup_2026_automation.sh
```

This will:
- Install Python dependencies
- Create data directories
- Configure GCP credentials
- Test BigQuery connection

### 2. Test the Pipeline

```bash
# Dry run (no BigQuery writes)
python3 season_2026_pipeline.py --dry-run

# Test with actual data (when season starts)
python3 season_2026_pipeline.py
```

### 3. Backfill Season Data

When the 2026 season begins:

```bash
# Backfill from opening day to today
python3 season_2026_pipeline.py --backfill --start 2026-03-27
```

### 4. Schedule Daily Automation

Add to crontab for automatic daily runs:

```bash
crontab -e

# Add this line (runs at 4 AM daily):
0 4 * * * cd /Users/VTNX82W/Documents/personalDev/hanks_tank_ml && ./run_daily_2026.sh
```

## Core Components

### season_2026_collector.py
Fetches live data from MLB Stats API:
- **Games**: All completed games with scores, teams, venues
- **Statcast**: Pitch-by-pitch data for every game
- **Player Stats**: Cumulative season stats for all players
- **Team Stats**: Batting and pitching stats for all teams
- **Standings**: Current division standings

```bash
# Collect today's data
python3 season_2026_collector.py

# Collect specific date
python3 season_2026_collector.py --date 2026-04-15

# Backfill range
python3 season_2026_collector.py --backfill --start-date 2026-03-27
```

### bigquery_sync.py
Syncs collected data to BigQuery:
- **Upsert logic**: Updates existing records, inserts new ones
- **No duplicates**: Key-based deduplication
- **Safe updates**: Atomic transactions
- **Incremental sync**: Only syncs new/changed data

```bash
# Sync all new data
python3 bigquery_sync.py

# Sync specific date
python3 bigquery_sync.py --date 2026-04-15

# Sync and validate
python3 bigquery_sync.py --validate
```

### season_2026_pipeline.py
Orchestrates the complete workflow:
1. Collect data from MLB API
2. Sync to BigQuery
3. Validate data quality
4. Generate report

```bash
# Run full pipeline
python3 season_2026_pipeline.py

# Dry run (no BigQuery writes)
python3 season_2026_pipeline.py --dry-run

# Specific date
python3 season_2026_pipeline.py --date 2026-04-15

# Backfill
python3 season_2026_pipeline.py --backfill --start 2026-03-27
```

### data_validation.py
Validates data quality (already exists, now supports 2026):
- Record counts
- NULL value checks
- Data consistency
- Schema validation

```bash
# Validate 2026 data
python3 data_validation.py --year 2026

# Quick 2026 check
./validate_2026.sh
```

## Daily Automation

The system runs automatically when scheduled via cron:

**What happens each day at 4 AM:**
1. Collect previous day's completed games
2. Fetch statcast data for those games
3. Update cumulative player/team stats
4. Update standings
5. Sync everything to BigQuery (upsert - safe)
6. Run data quality validation
7. Log results to `logs/2026/daily_YYYYMMDD.log`

**Manual run:**
```bash
./run_daily_2026.sh
```

**Monitor:**
```bash
tail -f logs/2026/daily_$(date +%Y%m%d).log
```

## Data Storage

### Local Cache (Temporary)
```
data/2026/
├── games/              # Individual game files
│   ├── 123456.json
│   └── 123457.json
├── stats/              # Cumulative stats
│   ├── team_stats.json
│   ├── player_stats.json
│   └── standings.json
├── statcast/           # Pitch data
│   ├── 123456_pitches.json
│   └── 123457_pitches.json
├── collection_log.jsonl  # Collection history
├── sync_log.jsonl        # Sync history
└── pipeline_log.jsonl    # Pipeline runs
```

### BigQuery (Permanent)
```
hankstank.mlb_historical_data
├── games_historical          # 2015-2026
├── team_stats_historical     # 2015-2026
├── player_stats_historical   # 2015-2026
├── standings_historical      # 2015-2026
└── statcast_pitches         # 2015-2026
```

## Data Quality Standards

All 2026 data is validated against the same standards as historical data:

- **Games**: 2400-2900 per season (regular + postseason + spring)
- **Teams**: 60-120 team stat records (30 teams × batting/pitching)
- **Players**: 120-250 player records (varies with call-ups)
- **Rosters**: 26-70 players per team (active + 40-man)
- **Statcast**: Pitch-level data for all completed games

Critical failures trigger alerts in pipeline logs.

## Season Timeline

### Pre-Season (Now - March 2026)
- ✅ System setup complete
- ✅ Validation ready for 2026
- ⏳ Waiting for season start

### Opening Day (March 27, 2026)
```bash
# Backfill opening day
python3 season_2026_pipeline.py --backfill --start 2026-03-27
```

### During Season (Daily - Automated)
- Cron runs at 4 AM daily
- Collects previous day's completed games
- Syncs to BigQuery
- Validates quality
- Logs results

### End of Season (October 2026)
- All 2026 data in BigQuery
- Validated and ready for ML models
- Historical dataset now spans 2015-2026

## Troubleshooting

### No games collected
- Season may not have started yet
- Check date: games are only collected after they're completed
- Verify MLB API is accessible

### BigQuery sync fails
- Check GCP credentials: `gcloud auth application-default login`
- Verify project access: `bq ls hankstank:mlb_historical_data`
- Check logs: `logs/2026/daily_*.log`

### Validation failures
- Review: `validation_report_*.json`
- Early in season: Low counts are expected
- Check specific errors in pipeline log

### Manual intervention
```bash
# Check what was collected
ls -la data/2026/games/

# Check sync status
cat data/2026/sync_log.jsonl | tail -1 | python3 -m json.tool

# Re-sync specific date
python3 bigquery_sync.py --date 2026-04-15

# Re-validate
./validate_2026.sh
```

## Architecture Features

✅ **Automated**: Runs daily via cron, zero manual intervention  
✅ **Safe**: Upsert logic prevents duplicates  
✅ **Validated**: Quality checks on every run  
✅ **Monitored**: Detailed logging and error tracking  
✅ **Incremental**: Only syncs new/changed data  
✅ **Complete**: Games, stats, standings, statcast - all in one system  
✅ **Unified**: 2026 data joins 2015-2025 in same BigQuery tables  
✅ **ML-Ready**: Data validated and formatted for model training

## Commands Reference

```bash
# Setup (one-time)
./setup_2026_automation.sh

# Daily run (manual)
./run_daily_2026.sh

# Pipeline operations
python3 season_2026_pipeline.py                    # Today's data
python3 season_2026_pipeline.py --dry-run          # Test mode
python3 season_2026_pipeline.py --date 2026-04-15  # Specific date
python3 season_2026_pipeline.py --backfill         # Fill gaps

# Individual components
python3 season_2026_collector.py                   # Just collect
python3 bigquery_sync.py                           # Just sync
python3 data_validation.py --year 2026             # Just validate

# Monitoring
tail -f logs/2026/daily_$(date +%Y%m%d).log       # Watch today's log
./validate_2026.sh                                 # Quick validation check
```

## Integration with Existing Systems

This system is **separate but compatible** with:

- **hanks_tank_backend**: Backend serves historical data (2015-2025) from BigQuery, will automatically include 2026 as it arrives
- **hanks_tank**: Frontend gets data from backend API
- **ML models**: Train on unified 2015-2026 dataset from BigQuery

No changes needed to backend/frontend - they'll automatically see 2026 data once synced to BigQuery.
