# Hank's Tank ML - 2026 Season Tracking

Automated MLB data pipeline for real-time 2026 season tracking, validation, and ML model training.

## 📁 Project Structure

```
hanks_tank_ml/
├── src/                          # Source code
│   ├── season_2026_collector.py  # MLB API data collection
│   ├── bigquery_sync.py          # BigQuery sync with upsert
│   ├── season_2026_pipeline.py   # Complete workflow orchestration
│   ├── data_validation.py        # Data quality validation
│   └── fix_data_issues.py        # One-time data fixes
│
├── scripts/                      # Automation scripts
│   ├── setup_2026_automation.sh  # One-time setup
│   ├── run_daily_2026.sh         # Daily cron job
│   ├── validate_2026.sh          # Quick validation
│   └── run_full_validation.sh    # Full historical validation
│
├── docs/                         # Documentation
│   ├── 2026_SEASON_SYSTEM.md     # Complete system guide
│   ├── BIGQUERY_DATA_SCHEMA.md   # Database schema
│   ├── FEATURE_ENGINEERING_PLAN.md  # ML features
│   └── SETUP_COMPLETE.md         # Setup history
│
├── ml_curriculum/                # Learning materials
│   ├── CURRICULUM.md             # Course outline
│   ├── LESSON_01_DATA_PIPELINES.md  # Lesson 1: Pipelines & ETL
│   ├── LESSON_02_BIGQUERY_DEEP_DIVE.md # Lesson 2: BigQuery & SQL
│   └── LESSON_03_DATA_MODELING.md   # Lesson 3: Data Modeling
│
├── research/                     # Research & Feature Engineering
│   ├── moneyball_principles.md   # Moneyball concepts
│   ├── advanced_sabermetrics.md  # Modern metrics (wRC+, FIP)
│   ├── park_and_weather_factors.md # Environmental factors
│   └── competitor_analysis.md    # Benchmarks & Competitors
│
├── data/                         # Local data cache
│   └── 2026/                     # 2026 season data
│       ├── games/
│       ├── stats/
│       └── statcast/
│
├── logs/                         # Execution logs
│   ├── validation/               # Validation reports
│   └── 2026/                     # Daily pipeline logs
│
└── requirements.txt              # Python dependencies
```

## 🚀 Quick Start

### 1. Setup (One-Time)

```bash
cd /Users/VTNX82W/Documents/personalDev/hanks_tank_ml
./scripts/setup_2026_automation.sh
```

### 2. Test the Pipeline

```bash
# Dry run (no BigQuery writes)
python3 src/season_2026_pipeline.py --dry-run

# Validate existing data
./scripts/validate_2026.sh
```

### 3. When Season Starts (March 27, 2026)

```bash
# Backfill from opening day
python3 src/season_2026_pipeline.py --backfill --start 2026-03-27
```

### 4. Daily Automation

```bash
# Manual run
./scripts/run_daily_2026.sh

# Schedule with cron (runs at 4 AM daily)
crontab -e
# Add: 0 4 * * * cd /path/to/hanks_tank_ml && ./scripts/run_daily_2026.sh
```

## 📊 System Overview

### Data Flow

```
MLB Stats API → Collector → Local Cache → BigQuery Sync → BigQuery
                                              ↓
                                          Validation
                                              ↓
                                          ML Models
```

### Daily Workflow (Automated)

1. **Collect** - Fetch previous day's games, stats, standings, statcast data
2. **Sync** - Upsert to BigQuery (no duplicates, safe updates)
3. **Validate** - Check data quality (completeness, consistency)
4. **Log** - Record results for monitoring

## 🎯 Alignment with Lesson 1 Best Practices

### ✅ ELT Pattern (Extract → Load → Transform)
- **Extract**: `season_2026_collector.py` pulls from MLB API
- **Load**: `bigquery_sync.py` loads to BigQuery (raw-ish data)
- **Transform**: BigQuery SQL for feature engineering (future)
- **Benefit**: Can re-transform without re-extracting

### ✅ Batch Processing
- Daily scheduled runs (efficient for non-real-time needs)
- Processes all games from previous day
- Lower cost than streaming for historical data

### ✅ Data Quality Validation
- **Completeness**: Check for NULL values in required fields
- **Uniqueness**: Prevent duplicates via upsert logic
- **Range checks**: Validate scores, dates, counts
- **Freshness**: Verify data is recent
- **Schema validation**: Type checking, consistency

### ✅ Incremental Loads
- Upsert pattern (update existing, insert new)
- Date-based filtering (only new data)
- No full table reloads

### ✅ Error Handling & Monitoring
- Try/catch blocks with specific error messages
- Detailed logging to `logs/2026/`
- Pipeline exit codes for alerting
- Validation reports saved as JSON

### ✅ Separation of Concerns
- **Collector**: Only API interaction
- **Sync**: Only BigQuery operations
- **Validation**: Only quality checks
- **Pipeline**: Orchestration layer

### ✅ Idempotency
- Can re-run same date multiple times safely
- Upsert prevents duplicate records
- Atomic operations

## 🔧 Components

### src/season_2026_collector.py
Fetches live MLB data:
- Games (scores, teams, venues)
- Statcast (pitch-by-pitch)
- Player stats (cumulative)
- Team stats (batting, pitching)
- Standings

### src/bigquery_sync.py
Syncs to BigQuery:
- Upsert logic (key-based deduplication)
- Batch inserts
- Error handling
- Sync logging

### src/season_2026_pipeline.py
Orchestrates workflow:
- Collect → Sync → Validate
- Error recovery
- Dry-run mode
- Backfill capability

### src/data_validation.py
Validates data quality:
- 189 checks across 2015-2026
- Completeness, uniqueness, ranges
- Critical vs warning severity
- JSON reports

## 📚 Documentation

- **[docs/2026_SEASON_SYSTEM.md](docs/2026_SEASON_SYSTEM.md)** - Complete system guide
- **[docs/BIGQUERY_DATA_SCHEMA.md](docs/BIGQUERY_DATA_SCHEMA.md)** - Database schema
- **[ml_curriculum/LESSON_01_DATA_PIPELINES.md](ml_curriculum/LESSON_01_DATA_PIPELINES.md)** - Pipeline fundamentals

## 🎓 Learning Path

Following the ML curriculum in `ml_curriculum/`:

1. **Lesson 1: Data Pipelines** ← Current system implements these concepts
   - ETL vs ELT patterns
   - Batch vs streaming
   - Data quality validation
   - Pipeline architecture

2. **Future Lessons**:
   - Feature engineering
   - Model training
   - Prediction serving
   - Model monitoring

## � Research & Analysis

We have conducted extensive research to guide our feature engineering and modeling strategy.

*   **[Moneyball Principles](research/moneyball_principles.md)**: Core concepts like Market Inefficiency and Pythagorean Expectation.
*   **[Advanced Sabermetrics](research/advanced_sabermetrics.md)**: Modern metrics (wRC+, FIP, SIERA) that isolate skill from luck.
*   **[Park & Weather](research/park_and_weather_factors.md)**: How environment (Coors Field, Wind, Temp) affects game outcomes.
*   **[Competitor Analysis](research/competitor_analysis.md)**: Benchmarks (aim for 55% accuracy) and lessons from PECOTA/ZiPS.
*   **[Fringe Factors](research/astrology_and_calendar_effects.md)**: Circadian rhythms, jet lag, and other hidden variables.

## �💡 Key Features

✅ **Automated** - Runs daily via cron, zero manual work  
✅ **Safe** - Upsert prevents duplicates  
✅ **Validated** - Quality checks every run  
✅ **Complete** - Games, stats, standings, statcast  
✅ **Controlled** - Dry-run testing, error handling  
✅ **Backed Up** - All data in BigQuery  
✅ **ML-Ready** - Unified 2015-2026 dataset  
✅ **Monitored** - Detailed logs and reports  
✅ **Lesson-Aligned** - Follows ETL best practices

## 🔍 Commands

```bash
# Setup
./scripts/setup_2026_automation.sh

# Daily run
./scripts/run_daily_2026.sh

# Pipeline
python3 src/season_2026_pipeline.py                    # Today's data
python3 src/season_2026_pipeline.py --dry-run          # Test mode
python3 src/season_2026_pipeline.py --date 2026-04-15  # Specific date
python3 src/season_2026_pipeline.py --backfill         # Fill gaps

# Validation
./scripts/validate_2026.sh                             # 2026 only
./scripts/run_full_validation.sh                       # All years (2015-2026)

# Monitoring
tail -f logs/2026/daily_$(date +%Y%m%d).log           # Watch logs
cat logs/validation/validation_report_*.json | jq     # View reports
```

## 🏗️ Next Steps

1. **When season starts**: Run backfill for opening day
2. **Schedule cron**: Automate daily runs
3. **Build features**: Use validated data for ML features (Lesson 2)
4. **Train models**: Game outcome and player performance prediction
5. **Deploy**: Serve predictions via API

---

**Status**: ✅ System ready, waiting for 2026 season start  
**Data**: 2015-2025 historical + 2026 real-time incoming  
**Quality**: 189/189 validation checks passing
