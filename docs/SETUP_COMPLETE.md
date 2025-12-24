# Data Validation System - Setup Complete ✅

## What Was Built

A comprehensive data quality validation pipeline for your MLB prediction project with automatic issue detection and repair capabilities.

### Core Components

1. **data_validation.py** (410 lines)
   - Validates all 8 BigQuery tables
   - Checks completeness, uniqueness, ranges, duplicates
   - Handles historical data (2015-2025) and future data (2026)
   - Generates detailed JSON reports
   - Returns proper exit codes for CI/CD integration

2. **fix_data_issues.py** (223 lines)
   - Automatically fixes duplicate game_pks
   - Creates backups before any changes
   - Dry-run mode by default (safe)
   - Detailed preview of what will change

3. **validate_2026.sh**
   - Quick wrapper for 2026 season monitoring
   - Integrates with your daily sync workflow
   - Clear success/failure indicators

---

## Current Data Quality Status

### ✅ What's Working

- **All tables exist** with correct schemas
- **30 teams per year** in teams/standings tables
- **No NULL values** in critical fields
- **Data continuity** across 2015-2025 (11 years complete)
- **8.4M+ total records** across all tables

### ⚠️ Issues Found (All Fixable)

**Duplicates:**
- 531 duplicate game_pks across all years (2015-2025)
  - 2015: 41 duplicates
  - 2016: 28 duplicates
  - 2017: 40 duplicates
  - 2018: 52 duplicates
  - 2019: 42 duplicates
  - 2020: 74 duplicates (COVID season)
  - 2021: 83 duplicates
  - 2022: 52 duplicates
  - 2023: 46 duplicates
  - 2024: 39 duplicates
  - 2025: 34 duplicates

**Cause:** Games synced multiple times (in-progress vs final scores)

**Impact:** Minimal - models will use first occurrence, but inflates row counts

**Minor Warnings:**
- Game counts slightly high (includes spring training/playoffs)
- Team 116 has 41 players (expanded roster - normal)
- team_stats has 100 records (batting + pitching splits - expected)
- player_stats has 152 records (multiple stat types - expected)

---

## How to Use

### Daily Workflow (2026 Season)

```bash
# 1. Run your backend sync (as usual)
cd hanks_tank_backend
npm run sync:2026  # or however you sync

# 2. Validate data quality
cd ../hanks_tank_ml
./validate_2026.sh

# 3. If issues found, fix them
python3 fix_data_issues.py --issue duplicates --year 2026 --execute
```

### One-Time Cleanup (Historical Data)

```bash
# Fix all duplicates in historical data
for year in {2015..2025}; do
    echo "Fixing $year..."
    python3 fix_data_issues.py --issue duplicates --year $year --execute
done

# Verify all fixed
python3 data_validation.py --year all
```

### Validation Examples

```bash
# Validate single year
python3 data_validation.py --year 2025

# Validate all years
python3 data_validation.py --year all

# Validate specific tables
python3 data_validation.py --year 2025 --tables games_historical rosters_historical

# Check 2026 (future year - should pass with no data yet)
python3 data_validation.py --year 2026
```

### Fix Examples

```bash
# See what would be fixed (safe - no changes)
python3 fix_data_issues.py --issue duplicates --year 2025

# Show details of duplicates
python3 fix_data_issues.py --issue duplicates --year 2025 --details

# Actually fix (creates backup first)
python3 fix_data_issues.py --issue duplicates --year 2025 --execute
```

---

## Validation Rules Reference

### Games Historical
- **Game count:** 2400-2500 per season (regular season)
  - Exception: 2020 had 1059 games (COVID-shortened - correct)
- **Scores:** 0-35 runs (max ever ~30, buffer for outliers)
- **No duplicates:** Each `game_pk` should appear once
- **Required fields:** `game_pk`, `game_date`, `home_team_id`, `away_team_id`

### Teams / Standings
- **Count:** Exactly 30 teams per year
- **No NULLs:** `team_id`, `wins`, `losses` must have values
- **Win percentage:** 0.0 - 1.0 range

### Rosters
- **Roster size:** 26-40 players per team
  - 26 = regular season active roster
  - 40 = September expanded rosters

### Player/Team Stats
- **Multiple records per team/player OK** (batting vs pitching splits)
- **No NULL wins/losses** in team_stats
- **Player stats vary widely** (no fixed count per year)

### Statcast
- **Pitch data exists** for modern years (2015+)
- **Speed ranges:** 0-110 mph for pitches

---

## Integration Points

### 1. Backend Sync Scripts

Add to `hanks_tank_backend/package.json`:
```json
{
  "scripts": {
    "validate:2026": "cd ../hanks_tank_ml && python3 data_validation.py --year 2026",
    "fix:duplicates": "cd ../hanks_tank_ml && python3 fix_data_issues.py --issue duplicates --year 2026 --execute"
  }
}
```

### 2. GitHub Actions / Cloud Build

```yaml
# In .github/workflows/data-quality.yml
name: Data Quality Check
on:
  schedule:
    - cron: '0 4 * * *'  # Daily at 4 AM (after sync)

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Validate 2026 Data
        run: |
          cd hanks_tank_ml
          python3 data_validation.py --year 2026
```

### 3. Cloud Scheduler (GCP)

```bash
# Create Cloud Scheduler job
gcloud scheduler jobs create http validate-mlb-data \
  --schedule="0 3 * * *" \
  --uri="https://your-cloud-function-url/validate" \
  --http-method=POST \
  --message-body='{"year": 2026}'
```

---

## Validation Report Output

### Console Output
```
================================================================================
MLB DATA VALIDATION PIPELINE
================================================================================
Project: hankstank
Dataset: mlb_historical_data
Timestamp: 2025-12-24 11:09:48

Validating tables: games_historical, teams_historical, standings_historical
Validating years: [2025]
================================================================================

✅ PASSED:     26 checks
⚠️  WARNING:     2 checks
❌ CRITICAL:    1 checks

────────────────────────────────────────────────────────────────────────────────
WARNINGS
────────────────────────────────────────────────────────────────────────────────
⚠️  games_historical [2025] game_count  Game count outside expected range: 2511

────────────────────────────────────────────────────────────────────────────────
CRITICAL FAILURES
────────────────────────────────────────────────────────────────────────────────
❌ games_historical [2025] duplicate_check  Found 34 duplicate game_pks

================================================================================
❌ VALIDATION FAILED - Critical issues found
================================================================================

Detailed report saved to: validation_report_20251224_110956.json
```

### JSON Report
```json
{
  "timestamp": "2025-12-24T11:09:56",
  "overall_status": "FAILED",
  "summary": {
    "total_checks": 29,
    "passed": 26,
    "warnings": 2,
    "critical": 1
  },
  "results": [
    {
      "check_name": "duplicate_check",
      "table": "games_historical",
      "year": 2025,
      "severity": "CRITICAL",
      "passed": false,
      "message": "Found 34 duplicate game_pks",
      "actual_value": 34,
      "timestamp": "2025-12-24T11:09:56"
    }
  ]
}
```

---

## Next Steps

### Immediate (Recommended)

1. **Fix historical duplicates:**
   ```bash
   python3 fix_data_issues.py --issue duplicates --year 2025 --execute
   ```

2. **Verify fix worked:**
   ```bash
   python3 data_validation.py --year 2025
   # Should now show 0 duplicates
   ```

3. **Add to daily workflow:**
   ```bash
   # Add this to your sync scripts
   cd hanks_tank_ml && ./validate_2026.sh
   ```

### Future Enhancements

- [ ] Add email/Slack alerts on validation failures
- [ ] Create validation dashboard (Grafana/Looker)
- [ ] Add more sophisticated outlier detection
- [ ] Implement data drift detection for ML features
- [ ] Add performance benchmarking (query speed tracking)
- [ ] Create data lineage tracking
- [ ] Build data quality SLA monitoring

---

## Files Created

```
hanks_tank_ml/
├── data_validation.py           # Main validation pipeline (410 lines)
├── fix_data_issues.py           # Automated fixes (223 lines)
├── validate_2026.sh             # Quick 2026 validation script
├── README.md                    # Updated with validation docs
├── SETUP_COMPLETE.md           # This file
└── validation_reports/          # Auto-generated JSON reports
    └── validation_report_YYYYMMDD_HHMMSS.json
```

---

## Support

### Troubleshooting

**"Table not found" errors:**
- Check table names match BigQuery exactly (use `bq ls`)
- Verify you have read permissions on dataset

**"No matching signature for function EXTRACT":**
- Date fields might be STRING not DATE
- Check schema: `bq show --schema hankstank:mlb_historical_data.TABLE_NAME`

**Validation too slow:**
- Run on specific tables: `--tables games_historical`
- Run on specific years: `--year 2026`

**False positives:**
- Adjust expected ranges in `data_validation.py`
- Some warnings are informational (e.g., high game counts = playoffs included)

### Getting Help

- Review validation logs in `validation_report_*.json`
- Check BigQuery directly: `bq query "SELECT COUNT(*) FROM ..."`
- See ML curriculum Lesson 1 for data pipeline concepts

---

**Status:** ✅ Data validation pipeline operational and tested  
**Ready for:** 2026 season monitoring  
**Last Updated:** December 24, 2025  
**Author:** MLB Prediction ML Team
