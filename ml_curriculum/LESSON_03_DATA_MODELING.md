# Lesson 3: Data Modeling & Schema Design

**Prerequisites:** Lesson 1 (Data Pipelines), Lesson 2 (BigQuery)  
**Duration:** 2-3 hours  
**Focus:** Design optimal database schemas for ML feature engineering

---

## Learning Objectives

By the end of this lesson, you will:
- Understand star schema vs snowflake schema design patterns
- Know when to normalize vs denormalize data
- Design fact and dimension tables for analytics
- Handle slowly changing dimensions (SCD)
- Create denormalized feature tables optimized for ML queries

---

## 1. Normalization vs Denormalization

### Normalization (OLTP - Traditional Databases)

**Goal:** Eliminate data redundancy, ensure data integrity

**Pros:**
- No duplicate data (saves storage)
- Single source of truth for updates
- Referential integrity through foreign keys

**Cons:**
- Requires many JOINs for queries
- Slower for analytics workloads
- Complex queries

**Example: Normalized Baseball Schema**
```sql
-- Teams table
CREATE TABLE teams (
  team_id STRING,
  team_name STRING,
  city STRING,
  division STRING,
  league STRING
);

-- Games table (references teams)
CREATE TABLE games (
  game_pk INT64,
  game_date DATE,
  home_team_id STRING,  -- References teams.team_id
  away_team_id STRING,  -- References teams.team_id
  home_score INT64,
  away_score INT64
);

-- To get team names, you MUST join:
SELECT 
  g.game_date,
  ht.team_name AS home_team_name,
  at.team_name AS away_team_name,
  g.home_score,
  g.away_score
FROM games g
JOIN teams ht ON g.home_team_id = ht.team_id
JOIN teams at ON g.away_team_id = at.team_id;
```

### Denormalization (OLAP - Analytics/ML)

**Goal:** Optimize for fast reads and analytics

**Pros:**
- Fewer JOINs = faster queries
- Simpler queries
- Lower query costs in BigQuery
- Perfect for ML feature engineering

**Cons:**
- Data duplication (storage cost)
- Multiple places to update
- Potential data inconsistency

**Example: Denormalized Baseball Schema**
```sql
-- Denormalized games table
CREATE TABLE games_denormalized (
  game_pk INT64,
  game_date DATE,
  season INT64,
  
  -- Home team info (denormalized)
  home_team_id STRING,
  home_team_name STRING,
  home_team_city STRING,
  home_division STRING,
  home_league STRING,
  home_score INT64,
  
  -- Away team info (denormalized)
  away_team_id STRING,
  away_team_name STRING,
  away_team_city STRING,
  away_division STRING,
  away_league STRING,
  away_score INT64,
  
  -- Computed fields
  winning_team_id STRING,
  score_differential INT64,
  is_home_win BOOL,
  game_type STRING
);

-- No joins needed!
SELECT 
  game_date,
  home_team_name,
  away_team_name,
  home_score,
  away_score
FROM games_denormalized
WHERE season = 2026;
```

**💡 Rule of Thumb for ML:**
- **Normalize** for data ingestion and storage
- **Denormalize** for feature engineering and ML queries

---

## 2. Star Schema vs Snowflake Schema

### Star Schema

**Structure:** One central fact table surrounded by dimension tables

**Characteristics:**
- Dimensions are denormalized
- Optimized for query performance
- Simple, intuitive structure
- Recommended for BigQuery

**Example: MLB Star Schema**
```
         ┌─────────────┐
         │   Teams     │
         │ Dimension   │
         └─────────────┘
                │
                │
┌───────────────┼───────────────┐
│               │               │
│         ┌─────▼──────┐        │
│         │   Games    │        │
│         │    Fact    │        │
│         └─────┬──────┘        │
│               │               │
│               │               │
▼               ▼               ▼
┌──────────┐  ┌──────────┐  ┌──────────┐
│  Dates   │  │ Stadiums │  │  Weather │
│Dimension │  │Dimension │  │Dimension │
└──────────┘  └──────────┘  └──────────┘
```

**SQL Implementation:**
```sql
-- Fact Table: Games
CREATE TABLE fact_games (
  game_pk INT64,
  date_key DATE,
  home_team_key STRING,
  away_team_key STRING,
  stadium_key STRING,
  
  -- Measures/Metrics
  home_score INT64,
  away_score INT64,
  attendance INT64,
  duration_minutes INT64,
  total_runs INT64,
  total_hits INT64
);

-- Dimension Table: Teams
CREATE TABLE dim_teams (
  team_key STRING PRIMARY KEY,
  team_name STRING,
  city STRING,
  state STRING,
  division STRING,
  league STRING,
  established_year INT64
);

-- Dimension Table: Dates
CREATE TABLE dim_dates (
  date_key DATE PRIMARY KEY,
  year INT64,
  month INT64,
  day INT64,
  quarter INT64,
  day_of_week STRING,
  is_weekend BOOL,
  is_opening_day BOOL
);

-- Dimension Table: Stadiums
CREATE TABLE dim_stadiums (
  stadium_key STRING PRIMARY KEY,
  stadium_name STRING,
  city STRING,
  state STRING,
  capacity INT64,
  surface_type STRING,
  roof_type STRING,
  altitude_feet INT64
);

-- Query: Join fact with dimensions
SELECT 
  d.year,
  d.month,
  ht.team_name AS home_team,
  s.stadium_name,
  f.home_score,
  f.away_score
FROM fact_games f
JOIN dim_dates d ON f.date_key = d.date_key
JOIN dim_teams ht ON f.home_team_key = ht.team_key
JOIN dim_stadiums s ON f.stadium_key = s.stadium_key
WHERE d.year = 2026;
```

### Snowflake Schema

**Structure:** Dimensions are normalized into sub-dimensions

**Characteristics:**
- More normalized than star schema
- Saves storage space
- More complex queries (more joins)
- Generally avoid in BigQuery

**Example: Snowflake Schema**
```
┌──────────────┐
│  Divisions   │
└──────┬───────┘
       │
       │
┌──────▼───────┐
│    Teams     │
└──────┬───────┘
       │
       │
   ┌───▼────┐
   │ Games  │
   │  Fact  │
   └────────┘
```

**When to use each:**
- ✅ **Star Schema**: Almost always for BigQuery/ML
- ⚠️ **Snowflake Schema**: Only if storage costs are critical

---

## 3. Fact and Dimension Tables

### Fact Tables

**Purpose:** Store measurable events or transactions

**Characteristics:**
- Large number of rows
- Contains foreign keys to dimensions
- Contains numeric measures/metrics
- Grows continuously over time

**MLB Examples:**
- `fact_games` - One row per game
- `fact_at_bats` - One row per at-bat
- `fact_pitches` - One row per pitch
- `fact_player_game_stats` - One row per player per game

**Fact Table Design Pattern:**
```sql
CREATE TABLE fact_player_batting_games (
  -- Keys (who, what, when, where)
  game_pk INT64,
  player_id INT64,
  team_id STRING,
  date_key DATE,
  
  -- Measures (quantifiable metrics)
  at_bats INT64,
  hits INT64,
  runs INT64,
  rbis INT64,
  home_runs INT64,
  strikeouts INT64,
  walks INT64,
  
  -- Computed measures
  batting_avg FLOAT64,
  on_base_pct FLOAT64,
  slugging_pct FLOAT64,
  
  -- Metadata
  inserted_at TIMESTAMP
);
```

### Dimension Tables

**Purpose:** Provide context and descriptive attributes

**Characteristics:**
- Relatively small number of rows
- Descriptive, textual information
- Changes slowly over time
- Wide tables (many columns)

**MLB Examples:**
- `dim_players` - Player information
- `dim_teams` - Team information
- `dim_stadiums` - Stadium information
- `dim_dates` - Date attributes

**Dimension Table Design Pattern:**
```sql
CREATE TABLE dim_players (
  player_id INT64 PRIMARY KEY,
  
  -- Descriptive attributes
  full_name STRING,
  first_name STRING,
  last_name STRING,
  birth_date DATE,
  birth_city STRING,
  birth_country STRING,
  
  -- Categorical attributes
  bats STRING,  -- L, R, S
  throws STRING,  -- L, R
  position STRING,
  
  -- Derived attributes
  age INT64,
  years_experience INT64,
  
  -- Current status
  current_team_id STRING,
  is_active BOOL,
  
  -- Metadata
  valid_from DATE,
  valid_to DATE,
  is_current BOOL
);
```

---

## 4. Slowly Changing Dimensions (SCD)

### The Problem

Dimension data changes over time:
- Players get traded to new teams
- Team names change
- Stadiums get renovated
- Player positions change

**How do we handle historical accuracy?**

### SCD Type 1: Overwrite

**Strategy:** Just update the record

**Pros:** Simple, saves space
**Cons:** Lose historical data

```sql
-- Player gets traded
UPDATE dim_players
SET current_team_id = 'NYY'
WHERE player_id = 660271;

-- Historical data is LOST!
-- Can't reconstruct "what team was he on in April?"
```

**Use when:** Historical values don't matter

### SCD Type 2: Add New Row (Recommended for ML)

**Strategy:** Keep all versions with validity dates

**Pros:** Complete history, accurate point-in-time queries
**Cons:** More storage, slightly complex queries

```sql
CREATE TABLE dim_players_scd2 (
  player_key INT64,  -- Surrogate key (unique per version)
  player_id INT64,   -- Natural key (same across versions)
  
  full_name STRING,
  current_team_id STRING,
  position STRING,
  
  -- SCD Type 2 columns
  valid_from DATE,
  valid_to DATE,
  is_current BOOL
);

-- Player starts with ATL
INSERT INTO dim_players_scd2 VALUES
(1, 660271, 'Spencer Strider', 'ATL', 'P', '2020-01-01', '9999-12-31', TRUE);

-- Player traded to NYY on 2026-07-15
-- 1. Close out old record
UPDATE dim_players_scd2
SET valid_to = '2026-07-14', is_current = FALSE
WHERE player_key = 1;

-- 2. Insert new record
INSERT INTO dim_players_scd2 VALUES
(2, 660271, 'Spencer Strider', 'NYY', 'P', '2026-07-15', '9999-12-31', TRUE);

-- Query for point-in-time accuracy
SELECT *
FROM fact_games f
JOIN dim_players_scd2 p 
  ON f.player_id = p.player_id
  AND f.game_date BETWEEN p.valid_from AND p.valid_to
WHERE f.game_date = '2026-05-01';
-- Returns team = 'ATL' (correct!)
```

### SCD Type 3: Add New Column

**Strategy:** Store both current and previous value

**Pros:** Simple queries
**Cons:** Limited history (only 2 versions)

```sql
CREATE TABLE dim_players_scd3 (
  player_id INT64,
  full_name STRING,
  current_team_id STRING,
  previous_team_id STRING,
  team_change_date DATE
);

-- Only tracks last change, not full history
```

**Use when:** You only care about current vs previous

### Which SCD Type to Use?

| Type | Use Case |
|------|----------|
| Type 1 | Corrections, unimportant changes |
| Type 2 | **ML/Analytics** - need complete history |
| Type 3 | Simple "before/after" comparisons |

---

## 5. Designing for ML Feature Engineering

### Anti-Pattern: Many Joins in Production

**❌ Bad for ML:**
```sql
-- This query runs EVERY prediction!
SELECT 
  g.game_pk,
  ht.team_name,
  ht.wins AS home_wins,
  ht.losses AS home_losses,
  at.wins AS away_wins,
  s.altitude_feet,
  w.temperature,
  w.wind_speed
FROM fact_games g
JOIN dim_teams ht ON g.home_team_id = ht.team_id
JOIN dim_teams at ON g.away_team_id = at.team_id
JOIN dim_stadiums s ON g.stadium_id = s.stadium_id
JOIN dim_weather w ON g.game_pk = w.game_pk
WHERE g.game_date = CURRENT_DATE();
```

**Problems:**
- Expensive (5 tables joined every time)
- Slow (100ms+ query time)
- Hard to version features
- Prone to errors

### Best Practice: Denormalized Feature Table

**✅ Good for ML:**
```sql
-- Create once, query many times
CREATE TABLE ml_game_features AS
SELECT 
  g.game_pk,
  g.game_date,
  g.season,
  
  -- Home team features (denormalized)
  g.home_team_id,
  ht.team_name AS home_team_name,
  ht.wins AS home_season_wins,
  ht.losses AS home_season_losses,
  ht.wins / (ht.wins + ht.losses) AS home_win_pct,
  
  -- Away team features (denormalized)
  g.away_team_id,
  at.team_name AS away_team_name,
  at.wins AS away_season_wins,
  at.losses AS away_season_losses,
  at.wins / (at.wins + at.losses) AS away_win_pct,
  
  -- Stadium features (denormalized)
  s.stadium_name,
  s.altitude_feet,
  s.surface_type,
  s.roof_type,
  
  -- Weather features (denormalized)
  w.temperature,
  w.wind_speed,
  w.humidity,
  
  -- Target variable
  CASE 
    WHEN g.home_score > g.away_score THEN 1 
    ELSE 0 
  END AS home_team_won,
  
  -- Metadata
  CURRENT_TIMESTAMP() AS feature_created_at
FROM fact_games g
JOIN dim_teams ht ON g.home_team_id = ht.team_id
JOIN dim_teams at ON g.away_team_id = at.team_id
JOIN dim_stadiums s ON g.stadium_id = s.stadium_id
LEFT JOIN dim_weather w ON g.game_pk = w.game_pk;

-- Now predictions are simple and fast!
SELECT * FROM ml_game_features
WHERE game_date = CURRENT_DATE();
```

**Benefits:**
- ⚡ Fast queries (no joins)
- 💰 Cheap (scan one table)
- 🔒 Feature versioning (snapshot at creation time)
- 🐛 Easier to debug
- 📊 Self-documenting (all features in one place)

---

## 6. Practical Schema Design for MLB ML

### Recommended Architecture

```
┌─────────────────────────────────────────────┐
│         Raw Data (Normalized)               │
│  - games_historical                         │
│  - team_stats_historical                    │
│  - player_stats_historical                  │
│  - statcast_pitches                         │
└─────────────────┬───────────────────────────┘
                  │
                  │ ETL / Feature Engineering
                  │
                  ▼
┌─────────────────────────────────────────────┐
│    ML Feature Tables (Denormalized)         │
│  - ml_game_prediction_features              │
│  - ml_player_performance_features           │
│  - ml_pitcher_matchup_features              │
└─────────────────────────────────────────────┘
```

### Example: Game Prediction Feature Table

```sql
CREATE TABLE ml_game_prediction_features
PARTITION BY game_date
CLUSTER BY home_team_id, away_team_id
AS
WITH home_rolling AS (
  SELECT 
    team_id,
    game_date,
    AVG(batting_avg) OVER w10 AS l10_batting_avg,
    AVG(era) OVER w10 AS l10_era,
    SUM(wins) OVER season AS season_wins
  FROM team_game_stats
  WINDOW 
    w10 AS (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
    season AS (PARTITION BY team_id, season ORDER BY game_date)
),
away_rolling AS (
  -- Same as home_rolling
  SELECT 
    team_id,
    game_date,
    AVG(batting_avg) OVER w10 AS l10_batting_avg,
    AVG(era) OVER w10 AS l10_era,
    SUM(wins) OVER season AS season_wins
  FROM team_game_stats
  WINDOW 
    w10 AS (PARTITION BY team_id ORDER BY game_date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW),
    season AS (PARTITION BY team_id, season ORDER BY game_date)
)
SELECT 
  g.game_pk,
  g.game_date,
  g.season,
  
  -- Home team features
  g.home_team_id,
  h.l10_batting_avg AS home_l10_batting_avg,
  h.l10_era AS home_l10_era,
  h.season_wins AS home_season_wins,
  
  -- Away team features
  g.away_team_id,
  a.l10_batting_avg AS away_l10_batting_avg,
  a.l10_era AS away_l10_era,
  a.season_wins AS away_season_wins,
  
  -- Matchup features
  h.l10_batting_avg - a.l10_batting_avg AS batting_diff,
  a.l10_era - h.l10_era AS era_diff,
  
  -- Target
  CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_won,
  
  -- Metadata
  CURRENT_TIMESTAMP() AS feature_version
FROM games_historical g
JOIN home_rolling h ON g.home_team_id = h.team_id AND g.game_date = h.game_date
JOIN away_rolling a ON g.away_team_id = a.team_id AND g.game_date = a.game_date;
```

---

## Hands-On Exercise: Redesign Your Schema

### Task 1: Analyze Current Schema

Examine your current `games_historical` table:
```sql
SELECT 
  column_name,
  data_type,
  is_nullable
FROM `your-project.mlb_historical_data.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'games_historical'
ORDER BY ordinal_position;
```

**Questions to ask:**
- Is this normalized or denormalized?
- How many joins would a typical query require?
- Are there repeated/redundant columns?

### Task 2: Create a Star Schema

Design a star schema with:
- **Fact table:** `fact_games`
- **Dimensions:** `dim_teams`, `dim_dates`, `dim_stadiums`

```sql
-- Example structure
CREATE TABLE fact_games (
  game_pk INT64,
  date_key DATE,
  home_team_key STRING,
  away_team_key STRING,
  stadium_key STRING,
  home_score INT64,
  away_score INT64,
  attendance INT64
);

CREATE TABLE dim_teams (
  team_key STRING,
  team_name STRING,
  city STRING,
  division STRING,
  league STRING
);

CREATE TABLE dim_dates (
  date_key DATE,
  year INT64,
  month INT64,
  day_of_week STRING,
  is_weekend BOOL
);
```

### Task 3: Create Denormalized Feature Table

Build a denormalized table optimized for ML:

```sql
CREATE OR REPLACE TABLE ml_game_features AS
SELECT 
  g.game_pk,
  g.game_date,
  
  -- Denormalize team info
  g.home_team_id,
  ht.team_name AS home_team_name,
  ht.division AS home_division,
  
  g.away_team_id,
  at.team_name AS away_team_name,
  at.division AS away_division,
  
  -- Add rolling stats (window functions)
  AVG(g_hist.home_score) OVER (
    PARTITION BY g.home_team_id 
    ORDER BY g.game_date 
    ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
  ) AS home_l10_avg_runs,
  
  -- Target
  g.home_score > g.away_score AS home_won
FROM games_historical g
JOIN teams ht ON g.home_team_id = ht.team_id
JOIN teams at ON g.away_team_id = at.team_id
LEFT JOIN games_historical g_hist 
  ON g.home_team_id = g_hist.home_team_id 
  AND g_hist.game_date < g.game_date;
```

---

## Key Takeaways

✅ **Normalize** for data ingestion, **denormalize** for ML queries  
✅ **Star schema** is best for BigQuery analytics  
✅ **Fact tables** store events, **dimension tables** store context  
✅ **SCD Type 2** preserves historical accuracy for ML  
✅ **Denormalized feature tables** are 10-100x faster than joins  
✅ **Partition and cluster** feature tables for best performance  

---

## 7. Production Architecture: Daily Workflow

### The Three-Layer Approach

When building production ML systems, use a three-layer data architecture:

```
┌─────────────────────────────────────────────────────────────┐
│  LAYER 1: Raw Data (Normalized - Source of Truth)          │
│  - games_historical                                         │
│  - team_stats_historical                                    │
│  - player_stats_historical                                  │
│  - statcast_pitches                                         │
│                                                             │
│  Purpose: Single source of truth, append-only              │
│  Updated: Throughout the day (as games finish)              │
│  Never modified, only new rows added                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Nightly rebuild (2-3 AM)
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 2: Dimension/Fact Tables (Optional)                 │
│  - dim_teams, dim_dates, dim_players                        │
│  - fact_games, fact_at_bats                                 │
│                                                             │
│  Purpose: Star schema for BI/analytics tools               │
│  Updated: Daily rebuild or incremental                      │
│  Skip this layer if only doing ML (go to Layer 3)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Nightly rebuild
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  LAYER 3: ML Feature Tables (Denormalized)                 │
│  - ml_game_prediction_features                             │
│  - ml_player_batting_features                              │
│  - ml_pitcher_matchup_features                             │
│                                                             │
│  Purpose: Fast ML queries, no joins needed                 │
│  Updated: Full nightly rebuild via CREATE OR REPLACE       │
│  This is your ML production layer!                         │
└─────────────────────────────────────────────────────────────┘
```

### Recommended Daily Workflow

**Throughout the Day:**
1. Games finish across MLB
2. Your data pipeline runs (season_2026_collector.py)
3. Raw tables updated with new data
4. Raw data accumulates but feature tables stay static

**Nightly (2-3 AM):**
1. Nightly job kicks off (cron or Cloud Scheduler)
2. Run `CREATE OR REPLACE TABLE ml_game_prediction_features AS ...`
3. Feature tables rebuilt from scratch using ALL raw data
4. Takes 5-30 minutes depending on data volume
5. New rolling statistics calculated with updated data

**Morning (Before Games):**
1. Feature tables are fresh with yesterday's data
2. Fast ML queries (no joins, pre-computed stats)
3. Generate predictions for today's games
4. Deploy predictions to your app

### Why Rebuild Nightly (vs Incremental Updates)?

**✅ Pros of Full Rebuild:**
- Simple to understand and debug
- No complex state management
- Can't have partial/corrupt data
- Easy to version control (one SQL query)
- Can recreate any feature table anytime
- Guarantees consistency

**⚠️ Cons of Full Rebuild:**
- Processes all data every time
- Higher compute costs (but negligible for MLB data)
- Slightly longer runtime

**When to use incremental updates instead:**
- Dataset > 10 million rows
- Need updates multiple times per day
- Compute costs become significant
- Have strong engineering team for complexity

**Recommendation:** Start with nightly rebuilds. They're simpler and perfectly adequate for MLB data (< 500K games total). Optimize later only if needed.

### Implementation: Nightly Feature Rebuild Script

Create `scripts/nightly_feature_rebuild.py`:

```python
#!/usr/bin/env python3
"""
Nightly ML Feature Rebuild

Rebuilds all denormalized ML feature tables from raw data.
Run this nightly via cron or Cloud Scheduler.

Usage:
    python scripts/nightly_feature_rebuild.py
"""

from google.cloud import bigquery
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "hankstank"
DATASET = "mlb_historical_data"

def rebuild_game_prediction_features():
    """Rebuild ml_game_prediction_features table"""
    client = bigquery.Client(project=PROJECT_ID)
    
    logger.info("Starting rebuild of ml_game_prediction_features...")
    
    query = f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET}.ml_game_prediction_features`
    PARTITION BY game_date
    CLUSTER BY home_team_id, away_team_id
    AS
    WITH team_games AS (
      SELECT 
        game_pk, game_date, season,
        home_team_id AS team_id,
        home_score AS runs_scored,
        away_score AS runs_allowed,
        CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS won
      FROM `{PROJECT_ID}.{DATASET}.games_historical`
      
      UNION ALL
      
      SELECT 
        game_pk, game_date, season,
        away_team_id AS team_id,
        away_score AS runs_scored,
        home_score AS runs_allowed,
        CASE WHEN away_score > home_score THEN 1 ELSE 0 END AS won
      FROM `{PROJECT_ID}.{DATASET}.games_historical`
    ),
    
    team_rolling_stats AS (
      SELECT 
        team_id, game_date, game_pk,
        
        -- Last 10 games rolling stats
        AVG(won) OVER w10 AS l10_win_pct,
        AVG(runs_scored) OVER w10 AS l10_runs_scored,
        AVG(runs_allowed) OVER w10 AS l10_runs_allowed,
        
        -- Season-to-date stats
        SUM(won) OVER season AS season_wins,
        COUNT(*) OVER season AS games_played,
        SUM(runs_scored) OVER season AS season_runs_scored,
        SUM(runs_allowed) OVER season AS season_runs_allowed
        
      FROM team_games
      WINDOW 
        w10 AS (
          PARTITION BY team_id, season 
          ORDER BY game_date 
          ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        ),
        season AS (
          PARTITION BY team_id, season 
          ORDER BY game_date
          ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
    )
    
    -- Join everything together - fully denormalized
    SELECT 
      g.game_pk,
      g.game_date,
      g.season,
      
      -- Home team features
      g.home_team_id,
      h.l10_win_pct AS home_l10_win_pct,
      h.l10_runs_scored AS home_l10_runs_scored,
      h.l10_runs_allowed AS home_l10_runs_allowed,
      h.season_wins AS home_season_wins,
      h.games_played AS home_games_played,
      
      -- Away team features
      g.away_team_id,
      a.l10_win_pct AS away_l10_win_pct,
      a.l10_runs_scored AS away_l10_runs_scored,
      a.l10_runs_allowed AS away_l10_runs_allowed,
      a.season_wins AS away_season_wins,
      a.games_played AS away_games_played,
      
      -- Derived matchup features
      h.l10_win_pct - a.l10_win_pct AS win_pct_diff,
      h.l10_runs_scored - a.l10_runs_scored AS offense_diff,
      a.l10_runs_allowed - h.l10_runs_allowed AS pitching_diff,
      
      -- Target variables
      g.home_score,
      g.away_score,
      CASE WHEN g.home_score > g.away_score THEN 1 ELSE 0 END AS home_won,
      
      -- Metadata
      CURRENT_TIMESTAMP() AS feature_created_at,
      '{datetime.now().strftime("%Y-%m-%d")}' AS feature_version
      
    FROM `{PROJECT_ID}.{DATASET}.games_historical` g
    LEFT JOIN team_rolling_stats h 
      ON g.game_pk = h.game_pk AND g.home_team_id = h.team_id
    LEFT JOIN team_rolling_stats a 
      ON g.game_pk = a.game_pk AND g.away_team_id = a.team_id
    WHERE g.game_date IS NOT NULL
    """
    
    job = client.query(query)
    result = job.result()  # Wait for completion
    
    bytes_processed = job.total_bytes_processed / 1e9
    logger.info(f"✅ ml_game_prediction_features rebuilt successfully")
    logger.info(f"   Processed: {bytes_processed:.2f} GB")
    logger.info(f"   Cost: ${bytes_processed / 1000 * 5:.4f}")
    
    return True


def rebuild_all_features():
    """Rebuild all ML feature tables"""
    start_time = datetime.now()
    logger.info(f"🔄 Starting nightly feature rebuild at {start_time}")
    
    try:
        # Rebuild each feature table
        rebuild_game_prediction_features()
        
        # Add more feature tables here as you build them:
        # rebuild_player_batting_features()
        # rebuild_pitcher_matchup_features()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"✅ All features rebuilt successfully in {duration:.1f} seconds")
        
    except Exception as e:
        logger.error(f"❌ Feature rebuild failed: {e}")
        raise


if __name__ == "__main__":
    rebuild_all_features()
```

### Schedule with Cron

Add to your crontab (runs at 2 AM daily):

```bash
# Edit crontab
crontab -e

# Add this line:
0 2 * * * cd /Users/VTNX82W/Documents/personalDev/hanks_tank_ml && /Users/VTNX82W/Documents/personalDev/hanks_tank_ml/.venv/bin/python scripts/nightly_feature_rebuild.py >> logs/feature_rebuild.log 2>&1
```

Or use Cloud Scheduler for cloud deployments:

```bash
gcloud scheduler jobs create http nightly-feature-rebuild \
  --schedule="0 2 * * *" \
  --uri="https://your-cloud-function-url/rebuild-features" \
  --http-method=POST \
  --time-zone="America/New_York"
```

### Key Principles for Production

1. **Raw Data is Sacred**
   - Never delete or modify historical raw data
   - Always append-only (INSERT only, no UPDATE/DELETE)
   - This is your single source of truth
   - Can always rebuild everything from raw data

2. **Feature Tables are Disposable**
   - Can be deleted and rebuilt anytime
   - Just re-run the `CREATE OR REPLACE` query
   - No data loss (rebuilt from raw data)
   - Treat them as cache, not storage

3. **Full Rebuild > Incremental (to start)**
   - Simpler to understand and debug
   - Less code = fewer bugs
   - Easier to version control
   - Optimize later only if needed

4. **Version Your Features**
   - Add `feature_created_at` timestamp to every row
   - Add `feature_version` string (e.g., "2026-03-15")
   - Can compare model performance across versions
   - Know exactly when features were last updated
   - Critical for ML debugging

5. **One Query Per Feature Table**
   - Each feature table = one `CREATE OR REPLACE` statement
   - Easy to version control in git
   - Easy to test and validate
   - Can run manually for debugging

6. **Test Before Production**
   - Create test table first: `ml_game_prediction_features_test`
   - Validate row counts, data quality
   - Compare to previous version
   - Then replace production table

### Monitoring Your Feature Pipeline

Add monitoring to your rebuild script:

```python
def validate_feature_table(table_name):
    """Validate feature table after rebuild"""
    client = bigquery.Client(project=PROJECT_ID)
    
    # Check row count
    count_query = f"SELECT COUNT(*) as cnt FROM `{PROJECT_ID}.{DATASET}.{table_name}`"
    count = client.query(count_query).to_dataframe()['cnt'][0]
    
    # Check for nulls in critical columns
    null_query = f"""
    SELECT 
      COUNTIF(home_l10_win_pct IS NULL) AS null_home_win_pct,
      COUNTIF(away_l10_win_pct IS NULL) AS null_away_win_pct
    FROM `{PROJECT_ID}.{DATASET}.{table_name}`
    WHERE game_date >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
    """
    nulls = client.query(null_query).to_dataframe()
    
    logger.info(f"Validation for {table_name}:")
    logger.info(f"  Total rows: {count:,}")
    logger.info(f"  Null home_l10_win_pct: {nulls['null_home_win_pct'][0]}")
    logger.info(f"  Null away_l10_win_pct: {nulls['null_away_win_pct'][0]}")
    
    # Alert if problems
    if nulls['null_home_win_pct'][0] > 100:
        logger.warning(f"⚠️  High null count in {table_name}")
```

---

## Next Steps

1. Analyze your current schema structure
2. Design a star schema for your data
3. Create a denormalized ML feature table
4. Set up nightly feature rebuild script
5. Test the rebuild process manually
6. Schedule with cron or Cloud Scheduler
7. Add monitoring and alerts
8. **Move to Lesson 4:** Workflow Orchestration

---

## Additional Resources

- [Kimball's Star Schema](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/)
- [BigQuery Schema Design Best Practices](https://cloud.google.com/bigquery/docs/best-practices-performance-patterns)
- [SCD Types Explained](https://www.sqlshack.com/implementing-slowly-changing-dimensions-scds-in-data-warehouses/)
- [Denormalization for Performance](https://cloud.google.com/bigquery/docs/best-practices-performance-compute)

---

**Ready to design schemas that make your ML queries 100x faster? Start with the hands-on exercises!** 🚀
