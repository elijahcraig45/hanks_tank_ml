# MLB Prediction ML Pipeline

Machine learning pipeline for predicting MLB game outcomes and player performance.

## Directory Structure

```
hanks_tank_ml/
├── data_validation.py          # Data quality validation pipeline
├── fix_data_issues.py          # Automated data quality fixes
├── ml_curriculum/              # Learning materials
│   ├── CURRICULUM.md           # Complete course outline
│   └── LESSON_01_DATA_PIPELINES.md
├── BIGQUERY_DATA_SCHEMA.md     # BigQuery table documentation
├── FEATURE_ENGINEERING_PLAN.md # ML feature engineering strategy
└── validation_reports/         # Generated validation reports (JSON)
```

## Quick Start

**1. Validate your current data (2015-2025):**
```bash
cd hanks_tank_ml
python3 data_validation.py --year 2025
```

**2. Fix any issues found:**
```bash
# Preview fixes
python3 fix_data_issues.py --issue duplicates --year 2025

# Apply fixes (creates automatic backup)
python3 fix_data_issues.py --issue duplicates --year 2025 --execute
```

**3. Set up for 2026 season:**
```bash
# Test validation on future year (should pass with no data)
./validate_2026.sh
```

**4. Integrate with your daily sync:**
Add to your backend sync workflow:
```bash
# After successful data sync
cd /path/to/hanks_tank_ml && ./validate_2026.sh
```

---

## Data Validation Pipeline

### Overview

Comprehensive data quality checks for all MLB historical data in BigQuery. Validates:
- ✅ Table existence and schema integrity
- ✅ Record counts and completeness
- ✅ Data ranges (scores, roster sizes, etc.)
- ✅ Duplicate detection
- ✅ NULL value checks
- ✅ Year-over-year continuity

### Usage

**Validate all data (all tables, all years):**
```bash
python data_validation.py
```

**Validate specific year:**
```bash
python data_validation.py --year 2025
```

**Validate multiple years:**
```bash
python data_validation.py --year 2024 2025
```

**Validate 2026 season data (as it comes in):**
```bash
python data_validation.py --year 2026
```

**Validate specific tables for 2026:**
```bash
python data_validation.py --year 2026 --tables games_historical rosters_historical
```

### Exit Codes

- `0`: All checks passed ✅
- `1`: Critical failures found ❌
- `2`: Warnings only ⚠️

### Integration with Pipeline

**Add to your daily sync (recommended):**
```bash
# In hanks_tank_backend after sync completes
cd ../hanks_tank_ml
python data_validation.py --year 2026 || echo "Data quality issues detected!"
```

**Use in CI/CD:**
```yaml
# In GitHub Actions or Cloud Build
- name: Validate Data Quality
  run: |
    cd hanks_tank_ml
    python data_validation.py --year 2026
  continue-on-error: false  # Fail pipeline if validation fails
```

### Output

The validation pipeline generates:

1. **Console output** with real-time validation results
2. **JSON report** saved to `validation_report_YYYYMMDD_HHMMSS.json`

**Example console output:**
```
================================================================================
MLB DATA VALIDATION PIPELINE
================================================================================
Project: hankstank
Dataset: mlb_historical_data
Timestamp: 2025-12-24 10:30:00

Validating tables: games_historical, teams, standings_historical
Validating years: [2025]
================================================================================

────────────────────────────────────────────────────────────────────────────────
TABLE: games_historical
────────────────────────────────────────────────────────────────────────────────
✅ table_exists: Table exists with 27,703 rows
✅ schema_integrity: All required columns present (19 total columns)
✅ game_count [2025]: Game count within expected range: 2511
✅ duplicate_check [2025]: No duplicate game_pks found
✅ score_validity [2025]: All scores valid (max: home=16, away=14)
✅ null_check [2025]: No NULL values in critical fields

================================================================================
VALIDATION SUMMARY
================================================================================

✅ PASSED:     42 checks
⚠️  WARNING:    3 checks
❌ CRITICAL:   0 checks

────────────────────────────────────────────────────────────────────────────────
WARNINGS
────────────────────────────────────────────────────────────────────────────────
⚠️  rosters_historical      [2025]   roster_size     Team 110: 45 players

================================================================================
✅ VALIDATION PASSED - All checks successful
================================================================================

Detailed report saved to: validation_report_20251224_103015.json
```

### Validation Rules

#### Games Historical
- Game count: 2400-2500 per season (regular season)
- Scores: 0-35 runs (max ever ~30)
- No duplicate `game_pk` values
- Required fields: `game_pk`, `game_date`, `home_team_id`, `away_team_id`

#### Teams / Standings
- Exactly 30 teams per year (MLB constant)
- No NULL values in `team_id`, `wins`, `losses`
- Win percentages between 0.0 and 1.0

#### Rosters
- Roster size: 26-40 players per team
- 26 = regular season roster
- 40 = September expanded rosters

#### Statcast
- Pitch data exists for modern years (2015+)
- Speed ranges: 0-110 mph (reasonable for pitches)

### 2026 Season Monitoring

As the 2026 season progresses, run validation after each daily sync:

**Quick validation script:**
```bash
./validate_2026.sh
```

**Or manual:**
```bash
python3 data_validation.py --year 2026
```

**By season phase:**

**Early Season (March-April):**
```bash
# Expect lower game counts
python3 data_validation.py --year 2026
# Will show warnings for low game count - this is normal
```

**Mid Season (May-August):**
```bash
# Game count should be growing
python3 data_validation.py --year 2026
```

**End of Season (September-October):**
```bash
# Should approach full 2400-2500 games
python3 data_validation.py --year 2026
```

**Integration with your backend sync:**
```bash
# In hanks_tank_backend after successful sync
cd ../hanks_tank_ml
./validate_2026.sh || echo "⚠️ Data quality issues detected!"
```

### Automated Monitoring (Future Enhancement)

```python
# Add to Cloud Scheduler
# Run daily at 3 AM (after sync completes)
0 3 * * * cd /path/to/hanks_tank_ml && python3 data_validation.py --year 2026 && \
  python3 send_validation_alert.py
```

## Data Issue Fixes

### Fix Duplicate Games

The fix_data_issues.py script can automatically repair common data quality problems.

**Preview what would be fixed (dry-run):**
```bash
python3 fix_data_issues.py --issue duplicates --year 2025
```

**Show details of duplicate games:**
```bash
python3 fix_data_issues.py --issue duplicates --year 2025 --details
```

**Execute the fix (creates backup automatically):**
```bash
python3 fix_data_issues.py --issue duplicates --year 2025 --execute
```

**How it works:**
- Identifies duplicate `game_pk` entries
- Keeps the first occurrence (by `game_date`)
- Creates automatic backup table before modifying data
- Provides rollback command if needed

**Example output:**
```
================================================================================
FIXING DUPLICATE GAMES FOR 2025
================================================================================

Found 34 duplicate game_pks:
  • game_pk 778431: 2 occurrences
  ...

🔧 EXECUTING FIX...

✅ Created deduplicated table: games_historical_deduped
Old table: 27,737 rows
New table: 27,703 rows
Removed: 34 duplicates

📦 Creating backup: games_historical_backup_20251224_110500
🔄 Replacing original table...

✅ COMPLETE! Duplicates removed.
   Backup saved to: games_historical_backup_20251224_110500
   You can restore with:
   bq cp hankstank:mlb_historical_data.games_historical_backup_20251224_110500 hankstank:mlb_historical_data.games_historical
```

## ML Curriculum

Comprehensive learning materials for data engineering and ML:
- 20 lessons across 5 modules
- Hands-on exercises using your MLB data
- 16-week timeline to production-ready models

Start here: [ml_curriculum/CURRICULUM.md](ml_curriculum/CURRICULUM.md)

## Data Schema

Complete documentation of all BigQuery tables: [BIGQUERY_DATA_SCHEMA.md](BIGQUERY_DATA_SCHEMA.md)

## Feature Engineering

Advanced ML strategy for 2026 predictions: [FEATURE_ENGINEERING_PLAN.md](FEATURE_ENGINEERING_PLAN.md)

---

**Last Updated:** December 24, 2025
